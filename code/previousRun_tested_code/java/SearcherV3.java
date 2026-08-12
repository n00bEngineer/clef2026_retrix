import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.similarities.BM25Similarity;
import unipd.se.model.ExpandedQueryDoc;
import unipd.se.model.QueryDoc;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

/**
 * Utility class for performing searches on a Lucene index of papers.
 * <p>
 * Supports both QueryDoc and subclasses (e.g., ExpandedQueryDoc).
 * Uses BM25 similarity and a weighted multi-field query (title + abstract).
 * </p>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class SearcherV3 {
    /** Shared custom analyzer for parsing queries. */
    private static final MyEnglishAnalyzer_TEX ANALYZER = new MyEnglishAnalyzer_TEX();
    private static final Pattern VENUE_PATTERN = Pattern.compile("\\bvenue:\\s*(\\S+)");
    private static final Pattern AUTHOR_PATTERN = Pattern.compile("\\bauthor:\\s*(\\S+(?:\\s+\\S+)*)");

    /** Weights used for the expansion of the queries */
    private static final float ORIGINAL_TEXT_BOOST = 0.3f;
    private static final float EXPANDED_TEXT_BOOST = 12.6f;
    private static final float SPARSE_BOOST = 20f;

    /**
     * Search the index with a configurable title boost.
     * Queries are processed in parallel on a dedicated ForkJoinPool.
     * IndexSearcher is thread-safe for concurrent reads (as guaranteed by Lucene).
     *
     * @param dir        the Lucene index directory
     * @param queries    list of queries (QueryDoc or subclasses)
     * @param titleBoost boost applied to the title field
     * @param topK       number of top documents to retrieve
     * @return map from query index → ranked list of pubkeys
     *
     * @throws IOException if the index cannot be opened or if the search fails
     */
    public static Map<String, List<String>> search(
            Directory dir,
            List<? extends QueryDoc> queries,
            float titleBoost,
            int topK
    ) throws IOException {
        return search(dir, queries, titleBoost, topK, ORIGINAL_TEXT_BOOST, EXPANDED_TEXT_BOOST, SPARSE_BOOST);
    }

    /**
     * Search the index with a configurable title boost and weights for different query representations.
     * Queries are processed in parallel on a dedicated ForkJoinPool.
     * IndexSearcher is thread-safe for concurrent reads (as guaranteed by Lucene).
     *
     * @param dir        the Lucene index directory
     * @param queries    list of queries (QueryDoc or subclasses)
     * @param titleBoost boost applied to the title field
     * @param topK       number of top documents to retrieve
     * @param w1         weight of the original text
     * @param w2         weight of the expanded text
     * @param w3         weight of the sparse representation
     *
     * @return map from query index → ranked list of pubkeys
     *
     * @throws IOException if the index cannot be opened or if the search fails
     */
    public static Map<String, List<String>> search(
            Directory dir,
            List<? extends QueryDoc> queries,
            float titleBoost,
            int topK,
            float w1,
            float w2,
            float w3
    ) throws IOException {
        Map<String, Float> fields = new HashMap<>();
        fields.put("title", titleBoost);
        fields.put("abstract", 1.0f);

        try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity());

            int cores = Runtime.getRuntime().availableProcessors();

            try (ForkJoinPool pool = new ForkJoinPool(cores)) {
                return pool.submit(() ->
                        queries.parallelStream().collect(Collectors.toMap(
                                q -> q.index,
                                q -> searchSingle(q, searcher, fields, topK, w1, w2, w3),
                                (a, _) -> a,
                                LinkedHashMap::new
                        ))
                ).get();

            } catch (InterruptedException | ExecutionException e) {
                throw new IOException("Search execution failed", e);
            }
        }
    }

    /**
     * Utility method for searching a single query
     *
     * @param q the query to search for
     * @param searcher the searcher
     * @param fields the fields weight
     * @param topK max hits
     * @param w1         weight of the original text
     * @param w2         weight of the expanded text
     * @param w3         weight of the sparse representation
     *
     * @return the list of retrieved documents (pubkey)
     */
    private static List<String> searchSingle(
            QueryDoc q,
            IndexSearcher searcher,
            Map<String, Float> fields,
            int topK,
            float w1,
            float w2,
            float w3
    ) {
        try {
            SimpleQueryParser parser = new SimpleQueryParser(ANALYZER, fields);
            Map<String, String> filters = new HashMap<>();
            Query finalQuery;

            if (q instanceof ExpandedQueryDoc eq) {
                // Costruisce la query pesata (Originale vs Espansione)
                finalQuery = buildWeightedExpandedQuery(eq, parser, filters, w1, w2, w3);
            } else {
                String text = q.getSearchText();
                if (text == null || text.isBlank()) return Collections.emptyList();
                finalQuery = buildSimpleQuery(text, parser, filters);
            }

            // Applica filtri venue/author se presenti
            finalQuery = applyFilters(finalQuery, filters);

            TopDocs topDocs = searcher.search(finalQuery, topK);
            List<String> topIds = new ArrayList<>(topDocs.scoreDocs.length);
            for (ScoreDoc sd : topDocs.scoreDocs) {
                Document doc = searcher.storedFields().document(sd.doc);
                topIds.add(doc.get("pubkey"));
            }
            return topIds;

        } catch (IOException e) {
            System.err.println("Search failed for query " + q.index + ": " + e.getMessage());
            return Collections.emptyList();
        }
    }

    /**
     * Build a BooleanQuery where the original text and the expansion have different weights.
     *
     * @param eq the expanded query doc
     * @param parser the parser used
     * @param filters the filters used
     * @param w1        weight of the original text
     * @param w2        weight of the expanded text
     * @param w3        weight of the sparse representation
     *
     * @return the built Query
     */
    private static Query buildWeightedExpandedQuery(
            ExpandedQueryDoc eq,
            SimpleQueryParser parser,
            Map<String, String> filters,
            float w1,
            float w2,
            float w3
    ) {
        String original = eq.getOriginal();
        String expanded = eq.getExpanded();
        String sparse = eq.getSparse();

        BooleanQuery.Builder mainBuilder = new BooleanQuery.Builder();

        // 1. Parte Originale (MUST o SHOULD con alto Boost)
        if (original != null && !original.isBlank()) {
            String cleanOriginal = extractFilters(original, filters);
            Query originalQuery = parser.parse(cleanOriginal);
            // Boost 5.0: Le parole dell'utente sono il segnale principale
            mainBuilder.add(new BoostQuery(originalQuery, w1), BooleanClause.Occur.SHOULD);
        }

        if (expanded != null && !expanded.isBlank()) {
            String cleanExpanded = extractFilters(expanded, filters);
            Query expandedQuery = parser.parse(cleanExpanded);
            mainBuilder.add(new BoostQuery(expandedQuery, w2), BooleanClause.Occur.SHOULD);
        }

        // 2. Parte Espansione (SHOULD con basso Boost)
        if (sparse != null && !sparse.isBlank()) {
            Query sparseQuery = parser.parse(sparse);
            // Boost 1.0: L'espansione aiuta solo se l'originale non è sufficiente
            mainBuilder.add(new BoostQuery(sparseQuery, w3), BooleanClause.Occur.SHOULD);
        }

        BooleanQuery bq = mainBuilder.build();
        // Fallback se entrambe le parti sono vuote
        return bq.clauses().isEmpty() ? parser.parse("") : bq;
    }

    /**
     * Builds a free-text query after removing inline structured filters from the raw input.
     *
     * @param text the raw query text
     * @param parser the Lucene parser used to create the query
     * @param filters the mutable map that collects extracted filters
     * @return the parsed Lucene query for the remaining free-text portion
     */
    private static Query buildSimpleQuery(String text, SimpleQueryParser parser, Map<String, String> filters) {
        String searchText = extractFilters(text, filters);
        return parser.parse(searchText);
    }

    /**
     * Extracts supported inline filters and returns the query text without the filter clauses.
     *
     * @param text the raw query text that may contain {@code venue:} or {@code author:} filters
     * @param filters the map where extracted filter values are stored
     * @return the cleaned query text with the structured filters removed
     */
    private static String extractFilters(String text, Map<String, String> filters) {
        Matcher venueM = VENUE_PATTERN.matcher(text);
        if (venueM.find()) {
            filters.put("venue", venueM.group(1));
            text = text.replaceFirst("\\bvenue:\\s*\\S+", "").trim();
        }

        Matcher authorM = AUTHOR_PATTERN.matcher(text);
        if (authorM.find()) {
            filters.put("authors", authorM.group(1));
            text = text.replaceFirst("\\bauthor:\\s*(\\S+(?:\\s+\\S+)*)", "").trim();
        }
        return text;
    }

    /**
     * Combines a free-text query with structured field filters when they are available.
     *
     * @param query the base Lucene query
     * @param filters the structured filters extracted from the raw query text
     * @return the original query if no filters are present, otherwise a boolean query with mandatory filters
     */
    private static Query applyFilters(Query query, Map<String, String> filters) {
        if (filters.isEmpty()) return query;
        BooleanQuery.Builder bq = new BooleanQuery.Builder();
        bq.add(query, BooleanClause.Occur.MUST);
        for (Map.Entry<String, String> f : filters.entrySet()) {
            bq.add(new TermQuery(new Term(f.getKey(), f.getValue().toLowerCase())), BooleanClause.Occur.MUST);
        }
        return bq.build();
    }
}
