package unipd.se;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import unipd.se.model.ExpandedQueryDoc;
import unipd.se.model.Paper;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

/**
 * Main class for the information retrieval evaluation workflow.
 * This class loads the scientific paper collection and the query set,
 * evaluates a file of previously re-ranked results when provided, and
 * stores the resulting metrics in JSON format.
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class Main {

    /**
     * Runs the retrieval evaluation workflow.
     * The paper collection and query set are always loaded. If a third
     * argument is provided, it is interpreted as the path to a JSON file
     * containing re-ranked results, which are then loaded and evaluated
     * directly. If the third argument is missing or blank, the method exits
     * without running an evaluation. To evaluate a re-ranked results file
     * while keeping the default collection and query paths, pass
     * {@code _ _ results/reranked_results.json}.
     *
     * @param args command-line arguments where {@code args[0]} is the paper
     *             collection JSON path, {@code args[1]} is the query JSON
     *             path, and {@code args[2]} is the optional path to a JSON
     *             file containing re-ranked results required for evaluation
     */
    public static void main(String[] args) {
        int cores = Runtime.getRuntime().availableProcessors();
        System.out.println("Starting evaluation [cores=" + cores + "]");

        String papersPath  = args.length > 0 && !args[0].equals("_") ? args[0] : "code/data/collection_data.json";

        // Default development-set configuration for English queries.
        String queriesPath = args.length > 1 && !args[1].equals("_") ? args[1] : "code/data/Test_set/final_en_test.json";
        // String queriesPath = args.length > 1 && !args[1].equals("_") ? args[1] : "code/data/Dev_set/expanded_queries_bge_large_frDEV_en.json";

        // Optional path to the file containing re-ranked results to evaluate.
        String rerankedPath = args.length > 2 ? args[2] : null;

        ObjectMapper mapper = new ObjectMapper();

        try (ExecutorService ioPool = Executors.newFixedThreadPool(2)) {
            // Load papers and queries in parallel using two I/O threads.
            Future<List<Paper>> papersFuture =
                    ioPool.submit(() -> DataLoader.loadPapers(papersPath));
            Future<List<ExpandedQueryDoc>> queriesFuture =
                    ioPool.submit(() -> DataLoader.loadQueries(queriesPath, ExpandedQueryDoc[].class));

            List<Paper> papers             = papersFuture.get();
            List<ExpandedQueryDoc> queries = queriesFuture.get();

            Map<String, List<String>> results = null;

            if (rerankedPath != null && !rerankedPath.trim().isEmpty()) {
                // Load re-ranked results from the provided file.
                File rerankedFile = new File(rerankedPath);

                // Validate that the input file exists and is not empty.
                if (!rerankedFile.exists()) {
                    System.err.println("Error: Reranked results file does not exist: " + rerankedPath);
                    return;
                }

                if (rerankedFile.length() == 0) {
                    System.err.println("Error: Reranked results file is empty: " + rerankedPath);
                    return;
                }

                System.out.println("Loading re-ranked results from: " + rerankedPath);
                try {
                    results = mapper.readValue(
                            rerankedFile,
                            new TypeReference<>() {}
                    );
                    System.out.println("Loaded results for " + results.size() + " queries.");
                } catch (Exception e) {
                    System.err.println("Error reading reranked results file: " + e.getMessage());
                    return;
                }
            } else {
                System.out.println("No reranked results file provided or path is empty. Evaluation will not be performed.");
                return; // Exit because this workflow only evaluates precomputed re-ranked results.
            }

            // Ensure valid results are available before starting evaluation.
            if (results == null || results.isEmpty()) {
                System.err.println("Error: No valid results to evaluate.");
                return;
            }

            // Save metrics to the first available output file name.
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
