package unipd.se;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import unipd.se.model.ExpandedQueryDoc;
import unipd.se.model.Paper;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

/*
 * Entry point for the Information Retrieval pipeline.
 *
 * Modalità 1 — BM25 only (default):
 *   java Main
 *   → esegue retrieval BM25, salva results/bm25_results.json, valuta e salva metriche
 *
 * Modalità 2 — BM25 + re-ranking neurale:
 *   java Main [papersPath] [queriesPath] [rerankedResultsPath]
 *   → se rerankedResultsPath è fornito, valuta QUELLO invece dei risultati BM25
 *
 * Flusso completo consigliato:
 *   1. java Main                          → produce results/bm25_results.json
 *   2. python Reranker.py                 → produce results/reranked_results.json
 *   3. java Main _ _ results/reranked_results.json   → valuta il re-ranking
 */

/**
 * Main entry point of the information retrieval pipeline.
 * This class loads the scientific paper collection and the query set,
 * executes BM25 retrieval or evaluates previously re-ranked results,
 * and stores the evaluation metrics in JSON format.
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class Main {

    /**
     * Runs the retrieval and evaluation pipeline.
     * If no arguments are provided, the method loads the default paper
     * collection and expanded queries, performs BM25 retrieval, stores the
     * ranked results, and evaluates them.
     * If a third argument is provided, it is interpreted as the path to a
     * file containing re-ranked results, which are loaded and evaluated
     * directly instead of running BM25 again.
     *
     * @param args command-line arguments:
     *             args[0] = path to the paper collection JSON file,
     *             args[1] = path to the query JSON file,
     *             args[2] = optional path to a JSON file containing
     *             re-ranked results
     */
    static void main(String[] args) {
        int cores = Runtime.getRuntime().availableProcessors();
        System.out.println("Starting retrieval [cores=" + cores + "]");

        String papersPath  = args.length > 0 && !args[0].equals("_") ? args[0] : "code/data/collection_data.json";
        //String queriesPath = args.length > 1 && !args[1].equals("_") ? args[1] : "code/data/expanded_queries_multilingual_merged.json";

        //per run dev_set EN
        //String queriesPath = args.length > 1 && !args[1].equals("_") ? args[1] : "code/data/Test_set/expanded_queries_bge_largeDEV.json";
        String queriesPath = args.length > 1 && !args[1].equals("_") ? args[1] : "code/data/Test_set/expanded_queries_bge_large_deFINAL_en.json";

        // If rerankedResultPath is provided, skip BM25 and directly evaluate re-ranked results
        String rerankedPath = args.length > 2 ? args[2] : null;

        ObjectMapper mapper = new ObjectMapper();

        try (ExecutorService ioPool = Executors.newFixedThreadPool(2)) {
            // 1. Load data in parallel
            // Papers and queries are deserialized simultaneously using two I/O threads
            Future<List<Paper>> papersFuture =
                    ioPool.submit(() -> DataLoader.loadPapers(papersPath));
            Future<List<ExpandedQueryDoc>> queriesFuture =
                    ioPool.submit(() -> DataLoader.loadQueries(queriesPath, ExpandedQueryDoc[].class));

            List<Paper> papers             = papersFuture.get();
            List<ExpandedQueryDoc> queries = queriesFuture.get();

            Map<String, List<String>> results = Map.of();

            if (rerankedPath != null) {
                // Mode 2: load re-ranked results from file
                System.out.println("Loading re-ranked results from: " + rerankedPath);
                results = mapper.readValue(
                        new File(rerankedPath),
                        new TypeReference<>() {}
                );
                System.out.println("Loaded results for " + results.size() + " queries.");

            } /*else {
                // Mode 1: run BM25 and save results
                Directory index = IndexerV1.buildIndex(papers);

                // Parallel search (Searcher internally manages parallelism)
                results = SearcherV3.search(index, queries, 1.0f, 100);

                // Save BM25 results in the background while the main thread prepares the configuration
                Files.createDirectories(Paths.get("results"));

                File bm25File = new File("results/bm25_results.json");
                CompletableFuture<Void> saveFuture = CompletableFuture.runAsync(() -> {
                    try {
                        mapper.writerWithDefaultPrettyPrinter().writeValue(bm25File, results);
                        System.out.println("BM25 results saved to: " + bm25File.getPath());
                        System.out.println("→ Now run: python Reranker.py");
                        System.out.println("→ Then re-run Main with: java Main _ _ results/reranked_results.json");
                    } catch (Exception e) {
                        System.err.println("Failed to save BM25 results: " + e.getMessage());
                    }
                });

                // Build the configuration while the file is being written in the background
                ObjectNode config = mapper.createObjectNode();
                config.put("analyzer", "MyCustomAnalyzer");
                config.put("query_parser", "SBERT");
                config.put("top_n", 100);
                config.put("title_boost", 1.0);
                config.put("similarity", "BM25");
                config.put("reranker", "none");
                config.putPOJO("fields", new String[]{"title", "abstract"});

                // Ensure that the file has been written before proceeding
                saveFuture.join();

                String basePath = "results/evaluation_results";
                File file = new File(basePath + ".json");
                int counter = 1;
                while (file.exists()) {
                    file = new File(basePath + "_" + counter++ + ".json");
                }
                Evaluator.evaluate(results, queries, config, file.getPath());
                return;
            }
*/
            // 3. Configuration for reranked mode
            ObjectNode config = mapper.createObjectNode();
            config.put("analyzer",     "MyCustomAnalyzer");
            config.put("query_parser", "SBERT");
            config.put("top_n",        100);
            config.put("title_boost",  1.0);
            config.put("similarity",   "BM25");
            config.put("reranker",     "cross-encoder/ms-marco-MiniLM-L-6-v2");
            config.putPOJO("fields",   new String[]{"title", "abstract"});

            // 4. Save evaluation metrics to a progressively named file
            String basePath = "results/evaluation_results_reranked";
            File file = new File(basePath + ".json");
            int counter = 1;
            while (file.exists()) {
                file = new File(basePath + "_" + counter++ + ".json");
            }
            Evaluator.evaluate(results, queries, file.getPath());

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Pipeline interrupted: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Error running IR pipeline: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
