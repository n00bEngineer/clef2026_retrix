package prf;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import unipd.se.DataLoader;
import unipd.se.IndexerV1;
import unipd.se.PseudoRelevanceFeedback;
import unipd.se.PRFConfig;
import unipd.se.model.Paper;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Validation and testing utilities for PRF implementation.
 * <p>
 * Includes methods to:
 * - Verify PRF produces valid results
 * - Test parameter sensitivity
 * - Validate term extraction consistency
 * </p>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class PRFValidator {

    /**
     * Validates PRF results meet basic requirements.
     *
     * @param result the PRF result to validate
     * @return true if result is valid
     */
    public static boolean validatePRFResult(PseudoRelevanceFeedback.PRFResult result) {
        if (result == null) {
            System.err.println("ERROR: PRF result is null");
            return false;
        }

        if (result.topDocIds == null || result.topDocIds.isEmpty()) {
            System.err.println("ERROR: No top documents found");
            return false;
        }

        if (result.expandedTerms == null) {
            System.err.println("ERROR: Expanded terms is null");
            return false;
        }

        if (result.expandedQuery == null || result.expandedQuery.isEmpty()) {
            System.err.println("ERROR: Expanded query is empty");
            return false;
        }

        // Verify expanded query contains original query elements
        if (!result.expandedQuery.contains("OR") && result.expandedTerms.size() > 0) {
            System.err.println("WARNING: Expanded query has no OR clauses but has expansion terms");
        }

        System.out.println("✓ PRF result validation passed");
        System.out.println("  Top documents: " + result.topDocIds.size());
        System.out.println("  Expanded terms: " + result.expandedTerms.size());
        System.out.println("  Expanded query: " + result.expandedQuery.substring(0,
            Math.min(80, result.expandedQuery.length())) + "...");

        return true;
    }

    /**
     * Tests PRF with several parameter combinations to observe sensitivity.
     *
     * @param dir the Lucene index directory
     * @param query the query used during the sensitivity experiment
     * @throws IOException if the index cannot be accessed
     */
    public static void testParameterSensitivity(Directory dir, String query) throws IOException {
        System.out.println("=== PRF Parameter Sensitivity Test ===");
        System.out.println("Query: " + query);
        System.out.println();

        Map<String, Float> fields = new HashMap<>();
        fields.put("title", 2.0f);
        fields.put("abstract", 1.0f);

        // Test different topKRelevant values
        System.out.println("Testing topKRelevant parameter (topNTerms=10, fixed):");
        int[] kValues = {3, 5, 10, 20};
        for (int k : kValues) {
            try {
                PseudoRelevanceFeedback.PRFResult result =
                    PseudoRelevanceFeedback.performPRF(dir, query, fields, k, 10, 100);
                System.out.println("  k=" + k + " → " + result.expandedTerms.size() +
                    " expansion terms from " + result.topDocIds.size() + " docs");
            } catch (Exception e) {
                System.err.println("  k=" + k + " → ERROR: " + e.getMessage());
            }
        }

        System.out.println();
        System.out.println("Testing topNTerms parameter (topKRelevant=5, fixed):");
        int[] nValues = {5, 10, 15, 20};
        for (int n : nValues) {
            try {
                PseudoRelevanceFeedback.PRFResult result =
                    PseudoRelevanceFeedback.performPRF(dir, query, fields, 5, n, 100);
                System.out.println("  n=" + n + " → " + result.expandedTerms.size() +
                    " expansion terms extracted");
                if (result.expandedTerms.size() > 0) {
                    System.out.println("    First 3: " + result.expandedTerms.subList(0,
                        Math.min(3, result.expandedTerms.size())));
                }
            } catch (Exception e) {
                System.err.println("  n=" + n + " → ERROR: " + e.getMessage());
            }
        }
    }

    /**
     * Tests whether PRF produces deterministic results across repeated runs.
     *
     * @param dir the Lucene index directory
     * @param query the query used during the consistency check
     * @throws IOException if the index cannot be accessed
     */
    public static void testConsistency(Directory dir, String query) throws IOException {
        System.out.println("=== PRF Consistency Test ===");
        System.out.println("Query: " + query);
        System.out.println();

        Map<String, Float> fields = new HashMap<>();
        fields.put("title", 2.0f);
        fields.put("abstract", 1.0f);

        System.out.println("Running PRF 3 times (should get identical results):");

        String[] expandedQueries = new String[3];
        String[] expandedTermsStr = new String[3];

        for (int i = 0; i < 3; i++) {
            PseudoRelevanceFeedback.PRFResult result =
                PseudoRelevanceFeedback.performPRF(dir, query, fields, 5, 10, 100);
            expandedQueries[i] = result.expandedQuery;
            expandedTermsStr[i] = result.expandedTerms.toString();
            System.out.println("  Run " + (i + 1) + ":");
            System.out.println("    Expanded terms: " + result.expandedTerms);
            System.out.println("    Query: " + result.expandedQuery.substring(0,
                Math.min(60, result.expandedQuery.length())) + "...");
        }

        System.out.println();
        boolean consistent = expandedQueries[0].equals(expandedQueries[1]) &&
                           expandedQueries[1].equals(expandedQueries[2]);
        if (consistent) {
            System.out.println("✓ Results are CONSISTENT across runs");
        } else {
            System.out.println("✗ Results are INCONSISTENT - check randomization!");
        }
    }

    /**
     * Runs a small suite of edge-case PRF queries.
     *
     * @param dir the Lucene index directory
     * @throws IOException if the index cannot be accessed
     */
    public static void testEdgeCases(Directory dir) throws IOException {
        System.out.println("=== PRF Edge Cases Test ===");
        System.out.println();

        Map<String, Float> fields = new HashMap<>();
        fields.put("title", 2.0f);
        fields.put("abstract", 1.0f);

        // Test 1: Single character query
        testQuery(dir, fields, "a");

        // Test 2: Very common query
        testQuery(dir, fields, "the");

        // Test 3: Non-existent query
        testQuery(dir, fields, "xyzabcnotreal");

        // Test 4: Empty results after stopword removal
        testQuery(dir, fields, "and or the");

        // Test 5: Very specific query
        testQuery(dir, fields, "SARS-CoV-2 pandemic 2020");
    }

    /**
     * Executes a single diagnostic PRF query and prints the observed outcome.
     *
     * @param dir the Lucene index directory
     * @param fields the field weights used by the query parser
     * @param query the query text to evaluate
     */
    private static void testQuery(Directory dir, Map<String, Float> fields, String query) {
        System.out.println("Query: \"" + query + "\"");
        try {
            PseudoRelevanceFeedback.PRFResult result =
                PseudoRelevanceFeedback.performPRF(dir, query, fields, 5, 10, 100);
            System.out.println("  ✓ Top docs: " + result.topDocIds.size());
            System.out.println("  ✓ Expansion terms: " + result.expandedTerms.size());
            if (!result.expandedTerms.isEmpty()) {
                System.out.println("    Sample: " + result.expandedTerms.subList(0,
                    Math.min(3, result.expandedTerms.size())));
            }
        } catch (IOException e) {
            System.out.println("  ✗ ERROR: " + e.getMessage());
        }
        System.out.println();
    }

    /**
     * Runs the full PRF validation suite from the command line.
     *
     * @param args ignored command-line arguments
     * @throws Exception if the index cannot be prepared or a validation step fails unexpectedly
     */
    public static void main(String[] args) throws Exception {
        System.out.println("PRF Validation Test Suite");
        System.out.println("========================");
        System.out.println();

        // Use existing index or build new one
        Directory index;
        try {
            System.out.println("Attempting to use existing index...");
            index = FSDirectory.open(Paths.get("index"));
        } catch (Exception e) {
            System.out.println("Building new index...");
            List<Paper> papers = DataLoader.loadPapers("code/data/collection_data.json");
            index = IndexerV1.buildIndex(papers);
        }

        // Run tests
        String testQuery = "vaccine effectiveness safety";

        testParameterSensitivity(index, testQuery);
        System.out.println();
        System.out.println();

        testConsistency(index, testQuery);
        System.out.println();
        System.out.println();

        testEdgeCases(index);
        System.out.println();
        System.out.println();

        System.out.println("=== All Tests Complete ===");
    }
}
