package unipd.se;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import unipd.se.model.QueryDoc;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * Utility class that evaluates ranked information retrieval results.
 * It computes aggregate metrics such as Recall@k, Precision@k, F1@1, MRR@5,
 * MAP, and nDCG@k, and also derives per-query statistics for selected metrics.
 * The evaluation supports {@link QueryDoc} objects and subclasses, accepts
 * ranked results regardless of how they were produced, and parallelizes
 * per-query metric accumulation through thread-safe adders.
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public final class Evaluator {

    /**
     * Private constructor to prevent instantiation of this utility class.
     */
    private Evaluator() {}

    /**
     * Evaluates the ranked retrieval results for the given queries and writes
     * the computed metrics to both the console and a JSON output file.
     * The evaluation is performed in parallel and supports {@link QueryDoc}
     * objects as well as subclasses such as expanded queries.
     *
     * @param results the retrieval results, mapping each query identifier to
     *                the ranked list of retrieved document identifiers
     * @param queries the list of queries to evaluate
     * @param outputFilePath the path of the JSON file where evaluation results
     *                       will be saved
     * @throws IOException if the evaluation is interrupted, fails during
     *                     execution, or the output file cannot be written
     */
    public static void evaluate(
            Map<String, List<String>> results,
            List<? extends QueryDoc> queries,
            String outputFilePath
    ) throws IOException {

        int total = queries.size();

        // Thread-safe accumulators used for parallel aggregation.
        DoubleAdder hitAt1   = new DoubleAdder(), hitAt5  = new DoubleAdder();
        DoubleAdder hitAt10  = new DoubleAdder(), hitAt100 = new DoubleAdder();
        DoubleAdder precAt1  = new DoubleAdder(), precAt5 = new DoubleAdder();
        DoubleAdder precAt10 = new DoubleAdder();
        DoubleAdder mrrAcc   = new DoubleAdder();
        DoubleAdder mapAcc   = new DoubleAdder();
        DoubleAdder ndcg5Acc = new DoubleAdder(), ndcg10Acc = new DoubleAdder(), ndcg100Acc = new DoubleAdder();

        // Arrays storing per-query statistics; each thread writes to a distinct index.
        double[] perQueryMrrArr    = new double[total];
        double[] perQueryNdcg10Arr = new double[total];

        // Parallel per-query evaluation.
        int cores = Runtime.getRuntime().availableProcessors();
        try (ForkJoinPool pool = new ForkJoinPool(cores)) {
            pool.submit(() ->
                // Parallel iteration with explicit indexing through IntStream.
                java.util.stream.IntStream.range(0, total).parallel().forEach(idx -> {
                    QueryDoc q = queries.get(idx);
                    String qid = q.index;

                    Set<String> goldSet = new HashSet<>();
                    if (q.pubkey != null) goldSet.add(q.pubkey);

                    List<String> ranked = results.getOrDefault(qid, Collections.emptyList());
                    int relCount = goldSet.size();

                    // First relevant rank (1-based, -1 if absent).
                    int rank = -1;
                    for (int i = 0; i < ranked.size(); i++) {
                        if (goldSet.contains(ranked.get(i))) { rank = i + 1; break; }
                    }

                    // Recall@k
                    if (rank == 1)               hitAt1.add(1);
                    if (rank > 0 && rank <= 5)   hitAt5.add(1);
                    if (rank > 0 && rank <= 10)  hitAt10.add(1);
                    if (rank > 0 && rank <= 100) hitAt100.add(1);

                    // Precision@k
                    precAt1.add(precisionAtK(ranked, goldSet, 1));
                    precAt5.add(precisionAtK(ranked, goldSet, 5));
                    precAt10.add(precisionAtK(ranked, goldSet, 10));

                    // MRR@5
                    double qMrr = (rank > 0 && rank <= 5) ? 1.0 / rank : 0.0;
                    mrrAcc.add(qMrr);
                    perQueryMrrArr[idx] = qMrr;

                    // nDCG@k
                    double qNdcg5   = ndcgAtK(ranked, goldSet, relCount, 5);
                    double qNdcg10  = ndcgAtK(ranked, goldSet, relCount, 10);
                    double qNdcg100 = ndcgAtK(ranked, goldSet, relCount, 100);
                    ndcg5Acc.add(qNdcg5);
                    ndcg10Acc.add(qNdcg10);
                    ndcg100Acc.add(qNdcg100);
                    perQueryNdcg10Arr[idx] = qNdcg10;

                    // MAP
                    double avgPrecision = 0.0;
                    int hitCount = 0;
                    for (int i = 0; i < ranked.size(); i++) {
                        if (goldSet.contains(ranked.get(i))) {
                            hitCount++;
                            avgPrecision += hitCount / (double) (i + 1);
                        }
                    }
                    if (relCount > 0) mapAcc.add(avgPrecision / relCount);
                })
            ).get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Evaluation interrupted", e);
        } catch (ExecutionException e) {
            throw new IOException("Evaluation failed", e.getCause());
        }

        // Convert arrays to lists for statistics computation.
        List<Double> perQueryMrr    = new ArrayList<>(total);
        List<Double> perQueryNdcg10 = new ArrayList<>(total);
        for (int i = 0; i < total; i++) {
            perQueryMrr.add(perQueryMrrArr[i]);
            perQueryNdcg10.add(perQueryNdcg10Arr[i]);
        }

        double denom = total == 0 ? 1.0 : total;

        // F1@1
        double avgRecall1 = hitAt1.sum() / denom;
        double avgPrec1   = precAt1.sum() / denom;
        double f1At1 = (avgPrec1 + avgRecall1 > 0)
                ? 2.0 * avgPrec1 * avgRecall1 / (avgPrec1 + avgRecall1)
                : 0.0;

        // Per-query stats
        double medianMrr    = median(perQueryMrr);
        double medianNdcg10 = median(perQueryNdcg10);
        double minMrr       = perQueryMrr.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double maxMrr       = perQueryMrr.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        double minNdcg10    = perQueryNdcg10.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double maxNdcg10    = perQueryNdcg10.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);

        double mrr      = mrrAcc.sum();
        double map      = mapAcc.sum();
        double ndcgAt5  = ndcg5Acc.sum();
        double ndcgAt10 = ndcg10Acc.sum();
        double ndcgAt100= ndcg100Acc.sum();

        // Console output
        System.out.println("=== Evaluation Results ===");
        System.out.println("Queries:      " + total);
        System.out.println();
        System.out.printf("Recall@1:   %.4f%n", hitAt1.sum()  / denom);
        System.out.printf("Recall@5:   %.4f%n", hitAt5.sum()  / denom);
        System.out.printf("Recall@10:  %.4f%n", hitAt10.sum() / denom);
        System.out.printf("Recall@100: %.4f%n", hitAt100.sum()/ denom);
        System.out.printf("P@1:        %.4f%n", precAt1.sum() / denom);
        System.out.printf("P@5:        %.4f%n", precAt5.sum() / denom);
        System.out.printf("P@10:       %.4f%n", precAt10.sum()/ denom);
        System.out.printf("F1@1:      %.4f%n",  f1At1);
        System.out.printf("MRR@5:      %.4f  (min=%.4f, median=%.4f, max=%.4f)%n",
                mrr / denom, minMrr, medianMrr, maxMrr);
        System.out.printf("MAP:        %.4f%n", map      / denom);
        System.out.printf("nDCG@5:     %.4f%n", ndcgAt5  / denom);
        System.out.printf("nDCG@10:    %.4f  (min=%.4f, median=%.4f, max=%.4f)%n",
                ndcgAt10 / denom, minNdcg10, medianNdcg10, maxNdcg10);
        System.out.printf("nDCG@100:   %.4f%n", ndcgAt100/ denom);

        // JSON output
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode root = mapper.createObjectNode();

        ObjectNode metrics = mapper.createObjectNode();
        metrics.put("queries",        total);
        metrics.put("recall@1",       hitAt1.sum()  / denom);
        metrics.put("recall@5",       hitAt5.sum()  / denom);
        metrics.put("recall@10",      hitAt10.sum() / denom);
        metrics.put("recall@100",     hitAt100.sum()/ denom);
        metrics.put("precision@1",    precAt1.sum() / denom);
        metrics.put("precision@5",    precAt5.sum() / denom);
        metrics.put("precision@10",   precAt10.sum()/ denom);
        metrics.put("f1@1",           f1At1);
        metrics.put("mrr@5",          mrr      / denom);
        metrics.put("map",            map      / denom);
        metrics.put("ndcg@5",         ndcgAt5  / denom);
        metrics.put("ndcg@10",        ndcgAt10 / denom);
        metrics.put("ndcg@100",       ndcgAt100/ denom);

        ObjectNode stats = mapper.createObjectNode();
        ObjectNode mrrStats = mapper.createObjectNode();
        mrrStats.put("min",    minMrr);
        mrrStats.put("median", medianMrr);
        mrrStats.put("max",    maxMrr);
        stats.set("mrr@5", mrrStats);

        ObjectNode ndcg10Stats = mapper.createObjectNode();
        ndcg10Stats.put("min",    minNdcg10);
        ndcg10Stats.put("median", medianNdcg10);
        ndcg10Stats.put("max",    maxNdcg10);
        stats.set("ndcg@10", ndcg10Stats);

        metrics.set("per_query_stats", stats);
        root.set("metrics", metrics);

        mapper.writerWithDefaultPrettyPrinter().writeValue(new File(outputFilePath), root);
        System.out.println("\nEvaluation saved to: " + outputFilePath);
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    /**
     * Computes Precision@k for a ranked list of retrieved document identifiers.
     *
     * @param ranked the ranked list of retrieved document identifiers
     * @param goldSet the set of relevant document identifiers
     * @param k the cutoff rank
     * @return the precision value at rank {@code k}
     */
    private static double precisionAtK(List<String> ranked, Set<String> goldSet, int k) {
        if (ranked.isEmpty() || k == 0) return 0.0;
        int limit = Math.min(k, ranked.size());
        int hits = 0;
        for (int i = 0; i < limit; i++) {
            if (goldSet.contains(ranked.get(i))) hits++;
        }
        return hits / (double) k;
    }

    /**
     * Computes nDCG@k using binary relevance.
     *
     * @param ranked the ranked list of retrieved document identifiers
     * @param goldSet the set of relevant document identifiers
     * @param relCount the total number of relevant documents used to compute
     *                 the ideal DCG
     * @param k the cutoff rank
     * @return the normalized discounted cumulative gain at rank {@code k}
     */
    private static double ndcgAtK(List<String> ranked, Set<String> goldSet, int relCount, int k) {
        int limit = Math.min(k, ranked.size());
        double dcg  = 0.0;
        double idcg = 0.0;

        for (int i = 0; i < limit; i++) {
            if (goldSet.contains(ranked.get(i))) {
                dcg += 1.0 / log2(i + 2);
            }
        }

        int idealHits = Math.min(relCount, k);
        for (int i = 0; i < idealHits; i++) {
            idcg += 1.0 / log2(i + 2);
        }

        return (idcg > 0) ? dcg / idcg : 0.0;
    }

    /**
     * Computes the median value of a list of numbers without modifying the
     * original list.
     *
     * @param values the input values
     * @return the median of the list, or {@code 0.0} if the list is empty
     */
    private static double median(List<Double> values) {
        if (values.isEmpty()) return 0.0;
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int n = sorted.size();
        if (n % 2 == 1) return sorted.get(n / 2);
        return (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2.0;
    }

    /**
     * Computes the base-2 logarithm of a number.
     *
     * @param x the input value
     * @return the base-2 logarithm of {@code x}
     */
    private static double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }
}
