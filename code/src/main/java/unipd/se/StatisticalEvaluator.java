package unipd.se;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import unipd.se.model.QueryDoc;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Standalone evaluator dedicated to statistical analysis.
 * <p>
 * Unlike {@link Evaluator}, this class exports both aggregate metrics and
 * per-query values so they can be reused in downstream ANOVA, Tukey HSD, and
 * boxplot analyses. The expected use case is to run it on a single, fixed query
 * split (for example English DEV) for several systems and then combine the
 * resulting JSON files in Python.
 * </p>
 *
 * <p>Usage:</p>
 * <pre>
 * mvn exec:java \
 *   -Dexec.mainClass=unipd.se.StatisticalEvaluator \
 *   -Dexec.args="code/data/Dev_set/en_dev.json \
 *                runs/dev/reranked/reranked_results_nemotron_topk1000BASELINE_DEV.json \
 *                results/statistics/nemotron_top1000_baseline_en_dev.json \
 *                nemotron_top1000_baseline_en_dev"
 * </pre>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public final class StatisticalEvaluator {

    /**
     * Shared mapper configured to accept query files with extra fields.
     * This makes the evaluator usable with both plain query files and expanded
     * query files that still contain the gold {@code pubkey}.
     */
    private static final ObjectMapper MAPPER = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    /**
     * Private constructor to prevent instantiation.
     */
    private StatisticalEvaluator() {}

    /**
     * CLI entry point.
     *
     * @param args command-line arguments:
     *             {@code <queries-json> <ranking-json> [output-json] [system-name]}
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            printUsage();
            return;
        }

        String queriesPath = args[0];
        String rankingPath = args[1];
        String outputPath = args.length >= 3 && !args[2].isBlank()
                ? args[2]
                : defaultOutputPath(rankingPath);
        String systemName = args.length >= 4 && !args[3].isBlank()
                ? args[3]
                : stripExtension(Path.of(rankingPath).getFileName().toString());

        try {
            List<QueryDoc> queries = loadQueries(queriesPath);
            Map<String, List<String>> results = loadResults(rankingPath);

            Path output = Path.of(outputPath);
            if (output.getParent() != null) {
                Files.createDirectories(output.getParent());
            }

            evaluate(results, queries, output, systemName, queriesPath, rankingPath);
        } catch (Exception e) {
            System.err.println("Statistical evaluation failed: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Evaluates ranked results and writes a JSON file containing both aggregate
     * metrics and per-query scores.
     *
     * @param results the retrieval results to evaluate
     * @param queries the query set with gold pubkeys
     * @param outputPath where to write the JSON output
     * @param systemName human-readable name of the system/run
     * @param queriesPath source query file path, stored as metadata
     * @param rankingPath source ranking file path, stored as metadata
     * @throws IOException if the output file cannot be written
     */
    public static void evaluate(
            Map<String, List<String>> results,
            List<? extends QueryDoc> queries,
            Path outputPath,
            String systemName,
            String queriesPath,
            String rankingPath
    ) throws IOException {

        int total = queries.size();

        double recall1 = 0.0, recall5 = 0.0, recall10 = 0.0, recall100 = 0.0;
        double precision1 = 0.0, precision5 = 0.0, precision10 = 0.0;
        double mrr5 = 0.0, map = 0.0, ndcg5 = 0.0, ndcg10 = 0.0, ndcg100 = 0.0;

        ArrayNode perQueryArray = MAPPER.createArrayNode();

        for (QueryDoc query : queries) {
            String qid = query.getIndex();
            String goldPubkey = query.getPubkey();

            Set<String> goldSet = new HashSet<>();
            if (goldPubkey != null) {
                goldSet.add(goldPubkey);
            }

            List<String> ranked = results.getOrDefault(qid, Collections.emptyList());
            int relCount = goldSet.size();

            int rank = firstRelevantRank(ranked, goldSet);

            double qRecall1 = rank == 1 ? 1.0 : 0.0;
            double qRecall5 = rank > 0 && rank <= 5 ? 1.0 : 0.0;
            double qRecall10 = rank > 0 && rank <= 10 ? 1.0 : 0.0;
            double qRecall100 = rank > 0 && rank <= 100 ? 1.0 : 0.0;

            double qPrecision1 = precisionAtK(ranked, goldSet, 1);
            double qPrecision5 = precisionAtK(ranked, goldSet, 5);
            double qPrecision10 = precisionAtK(ranked, goldSet, 10);

            double qF1At1 = qRecall1 + qPrecision1 > 0
                    ? 2.0 * qRecall1 * qPrecision1 / (qRecall1 + qPrecision1)
                    : 0.0;
            double qMrr5 = rank > 0 && rank <= 5 ? 1.0 / rank : 0.0;
            double qAp = averagePrecision(ranked, goldSet);
            double qNdcg5 = ndcgAtK(ranked, goldSet, relCount, 5);
            double qNdcg10 = ndcgAtK(ranked, goldSet, relCount, 10);
            double qNdcg100 = ndcgAtK(ranked, goldSet, relCount, 100);

            recall1 += qRecall1;
            recall5 += qRecall5;
            recall10 += qRecall10;
            recall100 += qRecall100;
            precision1 += qPrecision1;
            precision5 += qPrecision5;
            precision10 += qPrecision10;
            mrr5 += qMrr5;
            map += qAp;
            ndcg5 += qNdcg5;
            ndcg10 += qNdcg10;
            ndcg100 += qNdcg100;

            ObjectNode row = MAPPER.createObjectNode();
            row.put("qid", qid);
            row.put("gold_pubkey", goldPubkey == null ? "" : goldPubkey);
            row.put("first_relevant_rank", rank);
            row.put("hit@1", qRecall1);
            row.put("hit@5", qRecall5);
            row.put("hit@10", qRecall10);
            row.put("hit@100", qRecall100);
            row.put("precision@1", qPrecision1);
            row.put("precision@5", qPrecision5);
            row.put("precision@10", qPrecision10);
            row.put("f1@1", qF1At1);
            row.put("mrr@5", qMrr5);
            row.put("ap", qAp);
            row.put("ndcg@5", qNdcg5);
            row.put("ndcg@10", qNdcg10);
            row.put("ndcg@100", qNdcg100);
            perQueryArray.add(row);
        }

        double denom = total == 0 ? 1.0 : total;
        double avgRecall1 = recall1 / denom;
        double avgPrecision1 = precision1 / denom;
        double f1At1 = avgRecall1 + avgPrecision1 > 0
                ? 2.0 * avgRecall1 * avgPrecision1 / (avgRecall1 + avgPrecision1)
                : 0.0;

        ObjectNode root = MAPPER.createObjectNode();

        ObjectNode metadata = MAPPER.createObjectNode();
        metadata.put("system", systemName);
        metadata.put("query_set", queriesPath);
        metadata.put("ranking_file", rankingPath);
        root.set("metadata", metadata);

        ObjectNode metrics = MAPPER.createObjectNode();
        metrics.put("queries", total);
        metrics.put("recall@1", recall1 / denom);
        metrics.put("recall@5", recall5 / denom);
        metrics.put("recall@10", recall10 / denom);
        metrics.put("recall@100", recall100 / denom);
        metrics.put("precision@1", precision1 / denom);
        metrics.put("precision@5", precision5 / denom);
        metrics.put("precision@10", precision10 / denom);
        metrics.put("f1@1", f1At1);
        metrics.put("mrr@5", mrr5 / denom);
        metrics.put("map", map / denom);
        metrics.put("ndcg@5", ndcg5 / denom);
        metrics.put("ndcg@10", ndcg10 / denom);
        metrics.put("ndcg@100", ndcg100 / denom);
        root.set("metrics", metrics);

        root.set("per_query", perQueryArray);

        MAPPER.writerWithDefaultPrettyPrinter().writeValue(outputPath.toFile(), root);
        System.out.println("Statistical evaluation saved to: " + outputPath);
    }

    /**
     * Loads the query list while tolerating extra JSON fields.
     *
     * @param queriesPath query JSON path
     * @return the deserialized query list
     * @throws IOException if the file cannot be read
     */
    private static List<QueryDoc> loadQueries(String queriesPath) throws IOException {
        return Arrays.asList(MAPPER.readValue(new File(queriesPath), QueryDoc[].class));
    }

    /**
     * Loads a ranking JSON file in the {@code {qid: [pubkey, ...]}} format.
     *
     * @param rankingPath ranking JSON path
     * @return the deserialized ranking map
     * @throws IOException if the file cannot be read
     */
    private static Map<String, List<String>> loadResults(String rankingPath) throws IOException {
        return MAPPER.readValue(new File(rankingPath), new TypeReference<>() {});
    }

    /**
     * Returns the rank of the first relevant document.
     *
     * @param ranked ranked list of retrieved identifiers
     * @param goldSet set of relevant identifiers
     * @return one-based rank, or {@code -1} if no relevant document is found
     */
    private static int firstRelevantRank(List<String> ranked, Set<String> goldSet) {
        for (int i = 0; i < ranked.size(); i++) {
            if (goldSet.contains(ranked.get(i))) {
                return i + 1;
            }
        }
        return -1;
    }

    /**
     * Computes precision at cutoff {@code k}.
     *
     * @param ranked ranked result list
     * @param goldSet relevant identifiers
     * @param k cutoff
     * @return precision at {@code k}
     */
    private static double precisionAtK(List<String> ranked, Set<String> goldSet, int k) {
        if (ranked.isEmpty() || k <= 0) {
            return 0.0;
        }

        int limit = Math.min(k, ranked.size());
        int hits = 0;
        for (int i = 0; i < limit; i++) {
            if (goldSet.contains(ranked.get(i))) {
                hits++;
            }
        }
        return hits / (double) k;
    }

    /**
     * Computes average precision for a single ranked list.
     *
     * @param ranked ranked result list
     * @param goldSet relevant identifiers
     * @return average precision
     */
    private static double averagePrecision(List<String> ranked, Set<String> goldSet) {
        int relCount = goldSet.size();
        if (relCount == 0) {
            return 0.0;
        }

        double sum = 0.0;
        int hitCount = 0;
        for (int i = 0; i < ranked.size(); i++) {
            if (goldSet.contains(ranked.get(i))) {
                hitCount++;
                sum += hitCount / (double) (i + 1);
            }
        }
        return sum / relCount;
    }

    /**
     * Computes nDCG at cutoff {@code k} with binary relevance.
     *
     * @param ranked ranked result list
     * @param goldSet relevant identifiers
     * @param relCount number of relevant identifiers
     * @param k cutoff
     * @return nDCG at {@code k}
     */
    private static double ndcgAtK(List<String> ranked, Set<String> goldSet, int relCount, int k) {
        int limit = Math.min(k, ranked.size());
        double dcg = 0.0;
        double idcg = 0.0;

        for (int i = 0; i < limit; i++) {
            if (goldSet.contains(ranked.get(i))) {
                dcg += 1.0 / log2(i + 2);
            }
        }

        int idealLimit = Math.min(k, relCount);
        for (int i = 0; i < idealLimit; i++) {
            idcg += 1.0 / log2(i + 2);
        }

        return idcg == 0.0 ? 0.0 : dcg / idcg;
    }

    /**
     * Computes base-2 logarithm.
     *
     * @param x value
     * @return {@code log2(x)}
     */
    private static double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }

    /**
     * Builds the default output path under {@code results/statistics/}.
     *
     * @param rankingPath input ranking file path
     * @return default JSON output path
     */
    private static String defaultOutputPath(String rankingPath) {
        String stem = stripExtension(Path.of(rankingPath).getFileName().toString());
        return Path.of("results", "statistics", stem + "_stats.json").toString();
    }

    /**
     * Removes the last file extension from a filename.
     *
     * @param filename filename
     * @return filename without its last extension
     */
    private static String stripExtension(String filename) {
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }

    /**
     * Prints CLI usage instructions.
     */
    private static void printUsage() {
        System.out.println("""
                Usage:
                  mvn exec:java \\
                    -Dexec.mainClass=unipd.se.StatisticalEvaluator \\
                    -Dexec.args="<queries-json> <ranking-json> [output-json] [system-name]"

                Example:
                  mvn exec:java \\
                    -Dexec.mainClass=unipd.se.StatisticalEvaluator \\
                    -Dexec.args="code/data/Dev_set/en_dev.json \\
                                 runs/dev/reranked/reranked_results_nemotron_topk1000BASELINE_DEV.json \\
                                 results/statistics/nemotron_top1000_baseline_en_dev.json \\
                                 nemotron_top1000_baseline_en_dev"
                """);
    }
}
