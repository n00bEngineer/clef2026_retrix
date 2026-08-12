package unipd.se;

import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.similarities.BM25Similarity;
import unipd.se.model.QueryDoc;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

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
public class SearcherV1 {

    /**
     * Shared custom analyzer for parsing queries.
     */
    private static final MyEnglishAnalyzer_V2 ANALYZER = new MyEnglishAnalyzer_V2();

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
     * @throws IOException if the index cannot be opened or if the parallel search fails
     */
    public static Map<String, List<String>> search(
            Directory dir,
            List<? extends QueryDoc> queries,
            float titleBoost,
            int topK
    ) throws IOException {

        // Field weights (condivisi, immutabili)
        Map<String, Float> fields = new HashMap<>();
        fields.put("title",    titleBoost);
        fields.put("abstract", 1.0f);

        // IndexReader e IndexSearcher vengono aperti una volta sola e condivisi tra tutti i thread
        // DirectoryReader è thread-safe per letture concorrenti
        try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity());

            // Usa un ForkJoinPool con tanti thread quanti i core logici disponibili
            int cores = Runtime.getRuntime().availableProcessors();
            ForkJoinPool pool = new ForkJoinPool(cores);

            try {
                // Ogni query viene elaborata indipendentemente in parallelo
                // SimpleQueryParser viene creato per ogni thread (non è thread-safe)
                Map<String, List<String>> results = pool.submit(() ->
                    queries.parallelStream().collect(Collectors.toMap(
                        q -> q.index,
                        q -> {
                            String text = q.getSearchText();
                            if (text == null || text.isEmpty()) {
                                return Collections.<String>emptyList();
                            }
                            try {
                                // SimpleQueryParser non è thread-safe: istanza locale per thread
                                SimpleQueryParser parser = new SimpleQueryParser(ANALYZER, fields);
                                Query query = parser.parse(text);
                                TopDocs topDocs = searcher.search(query, topK);

                                List<String> topIds = new ArrayList<>(topDocs.scoreDocs.length);
                                for (ScoreDoc sd : topDocs.scoreDocs) {
                                    Document doc = searcher.storedFields().document(sd.doc);
                                    topIds.add(doc.get("pubkey"));
                                }
                                return topIds;
                            } catch (IOException e) {
                                System.err.println("Search failed for query " + q.index + ": " + e.getMessage());
                                return Collections.<String>emptyList();
                            }
                        },
                        (a, b) -> a,
                        LinkedHashMap::new   // mantieni ordine di inserimento
                    ))
                ).get();

                return results;

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Search interrupted", e);
            } catch (ExecutionException e) {
                throw new IOException("Search execution failed", e.getCause());
            } finally {
                pool.shutdown();
            }
        }
    }
}
