package prf;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import unipd.se.DataLoader;
import unipd.se.IndexerV1;
import unipd.se.PseudoRelevanceFeedback;
import unipd.se.model.Paper;
import org.apache.lucene.store.Directory;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Example usage of Pseudo Relevance Feedback (PRF) for query expansion.
 * <p>
 * This class demonstrates how to use PRF to improve search recall
 * by automatically expanding queries with related terms from top results.
 * </p>
 * <p>
 * Usage:
 *   java PRFExample [papersPath] [query]
 *
 * Example:
 *   java PRFExample code/data/collection_data.json "vaccine effectiveness"
 * </p>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class PRFExample {


    /**
     * Runs a single-query PRF demonstration from the command line.
     *
     * @param args command-line arguments containing the collection path and the query text
     * @throws Exception if loading data, indexing, searching, or writing the output fails
     */
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Usage: java PRFExample <papersPath> <query>");
            System.out.println("Example: java PRFExample code/data/collection_data.json \"vaccine effectiveness\"");
            System.exit(1);
        }

        String papersPath = args[0];
        String query = args[1];

        System.out.println("=== Pseudo Relevance Feedback (PRF) Demo ===");
        System.out.println("Papers: " + papersPath);
        System.out.println("Query: " + query);
        System.out.println();

        // Load papers and build index
        System.out.println("Loading papers...");
        List<Paper> papers = DataLoader.loadPapers(papersPath);
        System.out.println("Loaded " + papers.size() + " papers");

        System.out.println("Building index...");
        Directory index = IndexerV1.buildIndex(papers);

        // Setup search fields with weights
        Map<String, Float> fields = new HashMap<>();
        fields.put("title", 2.0f);      // Title boost: 2x
        fields.put("abstract", 1.0f);   // Abstract: 1x

        // Parameters
        int topKRelevant = 5;   // Take top 5 results from first pass
        int topNTerms = 10;     // Extract 10 most frequent terms
        int finalTopK = 100;    // Retrieve top 100 documents in final results

        // Perform PRF
        System.out.println();
        System.out.println("--- Pass 1: Initial Search ---");
        PseudoRelevanceFeedback.PRFResult prfResult = PseudoRelevanceFeedback.performPRF(
                index,
                query,
                fields,
                topKRelevant,
                topNTerms,
                finalTopK
        );

        System.out.println("Top " + topKRelevant + " results from initial search:");
        for (int i = 0; i < prfResult.topDocIds.size(); i++) {
            System.out.println("  " + (i + 1) + ". " + prfResult.topDocIds.get(i));
        }

        System.out.println();
        System.out.println("Extracted expansion terms (top " + topNTerms + " most frequent):");
        for (int i = 0; i < prfResult.expandedTerms.size(); i++) {
            System.out.println("  " + (i + 1) + ". " + prfResult.expandedTerms.get(i));
        }

        System.out.println();
        System.out.println("Original query:");
        System.out.println("  " + query);
        System.out.println();
        System.out.println("Expanded query:");
        System.out.println("  " + prfResult.expandedQuery);

        // Optional: Perform full two-pass search
        System.out.println();
        System.out.println("--- Pass 2: Search with Expanded Query ---");
        Map<String, Object> twoPassResult = PseudoRelevanceFeedback.twoPassSearchWithPRF(
                index,
                query,
                fields,
                topKRelevant,
                topNTerms,
                finalTopK
        );

        List<String> pass2Docs = (List<String>) twoPassResult.get("pass2_docs");
        System.out.println("Retrieved " + pass2Docs.size() + " documents in second pass");
        System.out.println("Top 10 results from expanded query:");
        for (int i = 0; i < Math.min(10, pass2Docs.size()); i++) {
            System.out.println("  " + (i + 1) + ". " + pass2Docs.get(i));
        }

        // Save results to JSON
        System.out.println();
        System.out.println("Saving results...");
        new File("results").mkdirs();
        ObjectMapper mapper = new ObjectMapper();

        ObjectNode output = mapper.createObjectNode();
        output.put("original_query", query);
        output.put("expanded_query", prfResult.expandedQuery);
        output.put("topKRelevant", topKRelevant);
        output.put("topNTerms", topNTerms);
        output.put("finalTopK", finalTopK);
        output.putPOJO("expanded_terms", prfResult.expandedTerms);
        output.putPOJO("pass1_results", prfResult.topDocIds);
        output.putPOJO("pass2_results", pass2Docs);

        File outputFile = new File("results/prf_demo_results.json");
        mapper.writerWithDefaultPrettyPrinter().writeValue(outputFile, output);
        System.out.println("Results saved to: " + outputFile.getPath());
    }
}
