package unipd.se;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Pseudo Relevance Feedback (PRF) for improving search recall.
 * <p>
 * PRF is a classic IR technique that assumes the top-K results from an initial search
 * are relevant and extracts their most frequent non-stopword terms to expand the query.
 * A second search pass is performed with the expanded query to find documents containing
 * related concepts and technical synonyms.
 * </p>
 * <p>
 * This is particularly useful in medical/scientific domains where terminology variations
 * are common (e.g., "vaccine" vs "immunization").
 * </p>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class PseudoRelevanceFeedback {

    private static final MyEnglishAnalyzer_V2 ANALYZER = new MyEnglishAnalyzer_V2();

    /**
     * Represents the result of a single PRF iteration.
     */
    public static class PRFResult {
        public final List<String> topDocIds;      // Top documents from search
        public final List<String> expandedTerms;  // Terms extracted from top documents
        public final String expandedQuery;        // Original query + expanded terms

        /**
         * Creates a container for the intermediate artifacts produced by a PRF iteration.
         *
         * @param topDocIds the identifiers returned by the first-pass search
         * @param expandedTerms the feedback terms extracted from the top-ranked documents
         * @param expandedQuery the original query augmented with the extracted terms
         */
        public PRFResult(List<String> topDocIds, List<String> expandedTerms, String expandedQuery) {
            this.topDocIds = topDocIds;
            this.expandedTerms = expandedTerms;
            this.expandedQuery = expandedQuery;
        }
    }

    /**
     * Performs a single PRF iteration: search → extract top terms → expand query.
     * <p>
     * Steps:
     * 1. Execute the initial search query
     * 2. Take the top-K results
     * 3. Extract and tokenize text from title + abstract of top results
     * 4. Count term frequencies (excluding stopwords)
     * 5. Select the top N most frequent terms
     * 6. Combine original query with expanded terms using OR clauses
     * </p>
     *
     * @param dir              the Lucene index directory
     * @param originalQuery    the original search query string
     * @param fields           field weights for multi-field search (e.g., title → 2.0, abstract → 1.0)
     * @param topKRelevant     number of top documents to consider for term extraction (typically 3-5)
     * @param topNTerms        number of terms to extract from top documents (typically 10-20)
     * @param searchTopK       number of documents to retrieve in the first search
     * @return PRFResult containing top doc IDs, extracted terms, and expanded query
     * @throws IOException if index operations fail
     */
    public static PRFResult performPRF(
            Directory dir,
            String originalQuery,
            Map<String, Float> fields,
            int topKRelevant,
            int topNTerms,
            int searchTopK
    ) throws IOException {

        try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity());

            // Step 1: Perform initial search with original query
            SimpleQueryParser parser = new SimpleQueryParser(ANALYZER, fields);
            Query query = parser.parse(originalQuery);
            TopDocs topDocs = searcher.search(query, Math.max(searchTopK, topKRelevant));

            // Step 2: Collect top-K document IDs for output
            int limit = Math.min(topKRelevant, topDocs.scoreDocs.length);
            List<String> topDocIds = new ArrayList<>();
            for (int i = 0; i < limit; i++) {
                Document doc = searcher.storedFields().document(topDocs.scoreDocs[i].doc);
                topDocIds.add(doc.get("pubkey"));
            }

            // Step 3: Extract and analyze terms from top documents
            Map<String, Integer> termFrequencies = extractTermsFromTopDocs(
                    searcher, topDocs.scoreDocs, limit, topKRelevant
            );

            // Step 4: Select top N terms by frequency
            List<String> expandedTerms = selectTopTerms(termFrequencies, topNTerms);

            // Step 5: Create expanded query
            String expandedQuery = buildExpandedQuery(originalQuery, expandedTerms);

            return new PRFResult(topDocIds, expandedTerms, expandedQuery);
        }
    }

    /**
     * Extracts and counts term frequencies from the top-K retrieved documents.
     * <p>
     * For each of the top-K documents, the title and abstract are tokenized
     * using the same analyzer used for indexing. Token frequencies are accumulated.
     * </p>
     *
     * @param searcher     the IndexSearcher
     * @param scoreDocs    array of scored documents from initial search
     * @param limit        number of documents to consider
     * @param topKRelevant number of top documents to analyze
     * @return map of term → frequency
     * @throws IOException if document retrieval fails
     */
    private static Map<String, Integer> extractTermsFromTopDocs(
            IndexSearcher searcher,
            ScoreDoc[] scoreDocs,
            int limit,
            int topKRelevant
    ) throws IOException {

        Map<String, Integer> termFrequencies = new HashMap<>();
        int docsToAnalyze = Math.min(limit, topKRelevant);

        for (int i = 0; i < docsToAnalyze; i++) {
            Document doc = searcher.storedFields().document(scoreDocs[i].doc);

            // Extract terms from title
            String title = doc.get("title");
            if (title != null && !title.isEmpty()) {
                extractTermsFromText(title, termFrequencies);
            }

            // Extract terms from abstract
            String abstractText = doc.get("abstract");
            if (abstractText != null && !abstractText.isEmpty()) {
                extractTermsFromText(abstractText, termFrequencies);
            }
        }

        return termFrequencies;
    }

    /**
     * Tokenizes text and accumulates term frequencies.
     * <p>
     * Uses the same analyzer as the index to ensure consistent tokenization,
     * stemming, stopword removal, and normalization.
     * </p>
     *
     * @param text             the text to tokenize
     * @param termFrequencies  map to accumulate term counts
     * @throws IOException if tokenization fails
     */
    private static void extractTermsFromText(String text, Map<String, Integer> termFrequencies) throws IOException {
        try (TokenStream stream = ANALYZER.tokenStream("content", text)) {
            CharTermAttribute charTermAttr = stream.addAttribute(CharTermAttribute.class);
            stream.reset();

            while (stream.incrementToken()) {
                String term = charTermAttr.toString();
                // Only add non-empty terms
                if (!term.isEmpty()) {
                    termFrequencies.merge(term, 1, Integer::sum);
                }
            }

            stream.end();
        }
    }

    /**
     * Selects the top-N most frequent terms.
     * <p>
     * Terms are sorted by frequency (descending) and then alphabetically
     * for deterministic behavior.
     * </p>
     *
     * @param termFrequencies map of term → frequency
     * @param topN            number of top terms to select
     * @return list of top terms in order of frequency (descending)
     */
    private static List<String> selectTopTerms(Map<String, Integer> termFrequencies, int topN) {
        return termFrequencies.entrySet().stream()
                .sorted((a, b) -> {
                    int freqCmp = b.getValue().compareTo(a.getValue()); // Descending by frequency
                    return freqCmp != 0 ? freqCmp : a.getKey().compareTo(b.getKey()); // Ascending alphabetically
                })
                .limit(topN)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }

    /**
     * Builds an expanded query by combining the original query with extracted terms.
     * <p>
     * Format: "original_query term1 OR term2 OR term3 ..."
     * This allows Lucene's SimpleQueryParser to handle OR logic with default SHOULD clauses.
     * </p>
     *
     * @param originalQuery  the original query string
     * @param expandedTerms  list of terms to add
     * @return expanded query string
     */
    private static String buildExpandedQuery(String originalQuery, List<String> expandedTerms) {
        if (expandedTerms.isEmpty()) {
            return originalQuery;
        }
        return originalQuery + " " + String.join(" OR ", expandedTerms);
    }

    /**
     * Performs two-pass search with automatic PRF expansion.
     * <p>
     * Pass 1: Search with original query
     * Pass 2: Search with expanded query containing top frequent terms from Pass 1 results
     * </p>
     *
     * @param dir             the Lucene index directory
     * @param originalQuery   the original search query
     * @param fields          field weights for multi-field search
     * @param topKRelevant    number of top results from Pass 1 to use for expansion (default: 5)
     * @param topNTerms       number of expansion terms to extract (default: 10)
     * @param finalTopK       number of documents to retrieve in final results (default: 100)
     * @return map containing "pass1_docs", "pass2_docs", "expanded_terms", and "expanded_query"
     * @throws IOException if index operations fail
     */
    public static Map<String, Object> twoPassSearchWithPRF(
            Directory dir,
            String originalQuery,
            Map<String, Float> fields,
            int topKRelevant,
            int topNTerms,
            int finalTopK
    ) throws IOException {

        Map<String, Object> result = new HashMap<>();

        // Pass 1: Initial search
        PRFResult prfResult = performPRF(dir, originalQuery, fields, topKRelevant, topNTerms, finalTopK);
        result.put("pass1_docs", prfResult.topDocIds);
        result.put("expanded_terms", prfResult.expandedTerms);
        result.put("expanded_query", prfResult.expandedQuery);

        // Pass 2: Search with expanded query
        try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity());

            SimpleQueryParser parser = new SimpleQueryParser(ANALYZER, fields);
            Query expandedQueryObj = parser.parse(prfResult.expandedQuery);
            TopDocs topDocs = searcher.search(expandedQueryObj, finalTopK);

            List<String> pass2Docs = new ArrayList<>();
            for (ScoreDoc sd : topDocs.scoreDocs) {
                Document doc = searcher.storedFields().document(sd.doc);
                pass2Docs.add(doc.get("pubkey"));
            }

            result.put("pass2_docs", pass2Docs);
        }

        return result;
    }
}
