package prf;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import unipd.se.DataLoader;
import unipd.se.IndexerV1;
import unipd.se.PseudoRelevanceFeedback;
import unipd.se.PRFConfig;
import unipd.se.PRFIntegration;
import unipd.se.SearcherV2;
import unipd.se.model.Paper;
import unipd.se.model.QueryDoc;
import org.apache.lucene.store.Directory;

import java.io.File;
import java.util.*;
import java.util.concurrent.*;

/**
 * Advanced example: Using PRF with multiple queries and parallel processing.
 * <p>
 * This class demonstrates how to integrate PRF into a batch search pipeline,
 * comparing results from:
 * 1. Standard BM25 search
 * 2. PRF-based two-pass search
 * </p>
 * <p>
 * Usage:
 *   java PRFBatchExample [papersPath] [queriesPath]
 * </p>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class PRFBatchExample {

    /**
     * Runs the batch PRF comparison example from the command line.
     *
     * @param args command-line arguments containing the collection path and the queries path
     * @throws Exception if loading data, indexing, searching, or writing the output fails
     */
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Usage: java PRFBatchExample <papersPath> <queriesPath>");
            System.out.println("Example: java PRFBatchExample code/data/collection_data.json code/data/queries.json");
            System.exit(1);
        }

        String papersPath = args[0];
        String queriesPath = args[1];

        System.out.println("=== PRF Batch Processing Example ===");
        System.out.println("Papers: " + papersPath);
        System.out.println("Queries: " + queriesPath);
        System.out.println();

        // Load data
        System.out.println("Loading papers and queries...");
        List<Paper> papers = DataLoader.loadPapers(papersPath);
        List<QueryDoc> queries = DataLoader.loadQueries(queriesPath, QueryDoc[].class);
        System.out.println("Loaded " + papers.size() + " papers and " + queries.size() + " queries");

        // Build index
        System.out.println("Building index...");
        Directory index = IndexerV1.buildIndex(papers);

        // Setup search fields
        Map<String, Float> fields = new HashMap<>();
        fields.put("title", 2.0f);
        fields.put("abstract", 1.0f);

        // PRF parameters
        int topKRelevant = 5;
        int topNTerms = 10;
        int finalTopK = 100;

        // Perform comparisons
        ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> comparisonResults = new HashMap<>();
        comparisonResults.put("parameters", new HashMap<String, Integer>() {{
            put("topKRelevant", topKRelevant);
            put("topNTerms", topNTerms);
            put("finalTopK", finalTopK);
        }});

        List<Map<String, Object>> queryComparisons = new ArrayList<>();

        System.out.println();
        System.out.println("Processing " + queries.size() + " queries with PRF...");
        System.out.println();

        // Process first 3 queries as examples
        int queriesToProcess = Math.min(3, queries.size());
        for (int i = 0; i < queriesToProcess; i++) {
            QueryDoc query = queries.get(i);
            String queryText = query.getSearchText();

            System.out.println("Query " + (i + 1) + ": " + queryText);

            try {
                // Standard BM25 search
                List<String> bm25Results = SearcherV2.search(
                        index,
                        Collections.singletonList(query),
                        2.0f,
                        finalTopK
                ).get(query.index);

                // PRF two-pass search
                Map<String, Object> prfResult = PseudoRelevanceFeedback.twoPassSearchWithPRF(
                        index,
                        queryText,
                        fields,
                        topKRelevant,
                        topNTerms,
                        finalTopK
                );

                List<String> pass1Docs = (List<String>) prfResult.get("pass1_docs");
                List<String> pass2Docs = (List<String>) prfResult.get("pass2_docs");
                List<String> expandedTerms = (List<String>) prfResult.get("expanded_terms");
                String expandedQuery = (String) prfResult.get("expanded_query");

                // Comparison metrics
                Set<String> bm25Set = new HashSet<>(bm25Results);
                Set<String> pass1Set = new HashSet<>(pass1Docs);
                Set<String> pass2Set = new HashSet<>(pass2Docs);

                Set<String> onlyInBM25 = new HashSet<>(bm25Set);
                onlyInBM25.removeAll(pass2Set);

                Set<String> onlyInPRF = new HashSet<>(pass2Set);
                onlyInPRF.removeAll(bm25Set);

                Set<String> common = new HashSet<>(bm25Set);
                common.retainAll(pass2Set);

                // Store results
                Map<String, Object> queryResult = new HashMap<>();
                queryResult.put("query_index", query.index);
                queryResult.put("query_text", queryText);
                queryResult.put("expanded_query", expandedQuery);
                queryResult.put("expanded_terms", expandedTerms);
                queryResult.put("bm25_results_count", bm25Results.size());
                queryResult.put("prf_pass1_results", pass1Docs);
                queryResult.put("prf_pass2_results", pass2Docs);
                queryResult.put("common_results", common.size());
                queryResult.put("only_in_bm25", onlyInBM25.size());
                queryResult.put("only_in_prf", onlyInPRF.size());

                queryComparisons.add(queryResult);

                // Print summary
                System.out.println("  BM25 results: " + bm25Results.size());
                System.out.println("  PRF Pass 1 results: " + pass1Docs.size());
                System.out.println("  PRF Pass 2 results: " + pass2Docs.size());
                System.out.println("  Common results: " + common.size());
                System.out.println("  Expansion terms: " + expandedTerms.size());
                System.out.println("    Sample terms: " + (expandedTerms.isEmpty() ? "none" :
                        String.join(", ", expandedTerms.subList(0, Math.min(5, expandedTerms.size())))));
                System.out.println();

            } catch (Exception e) {
                System.err.println("  Error processing query: " + e.getMessage());
                e.printStackTrace();
            }
        }

        // Save results
        System.out.println("Saving comparison results...");
        new File("results").mkdirs();
        comparisonResults.put("query_comparisons", queryComparisons);
        File outputFile = new File("results/prf_batch_comparison.json");
        mapper.writerWithDefaultPrettyPrinter().writeValue(outputFile, comparisonResults);
        System.out.println("Results saved to: " + outputFile.getPath());

        // Print statistics
        System.out.println();
        System.out.println("=== Summary Statistics ===");
        int totalQueriesProcessed = queryComparisons.size();
        double avgBM25 = queryComparisons.stream()
                .mapToInt(q -> (int) q.get("bm25_results_count"))
                .average().orElse(0);
        double avgPRF = queryComparisons.stream()
                .mapToInt(q -> ((List<?>) q.get("prf_pass2_results")).size())
                .average().orElse(0);
        double avgExpansionTerms = queryComparisons.stream()
                .mapToInt(q -> ((List<?>) q.get("expanded_terms")).size())
                .average().orElse(0);

        System.out.println("Queries processed: " + totalQueriesProcessed);
        System.out.println("Average BM25 results: " + String.format("%.1f", avgBM25));
        System.out.println("Average PRF results: " + String.format("%.1f", avgPRF));
        System.out.println("Average expansion terms: " + String.format("%.1f", avgExpansionTerms));
    }
}
