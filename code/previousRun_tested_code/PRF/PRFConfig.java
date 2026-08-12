package unipd.se;

import org.apache.lucene.store.Directory;
import java.util.HashMap;
import java.util.Map;

/**
 * PRF Configuration Presets
 *
 * This class provides predefined parameter configurations for different use cases.
 * Use these as starting points, then tune based on your evaluation metrics.
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class PRFConfig {
    /**
     * Represents a PRF configuration
     */
    public static class Config {
        public final String name;
        public final String description;
        public final int topKRelevant;
        public final int topNTerms;
        public final int finalTopK;

        /**
         * Creates a PRF configuration with explicit tuning parameters.
         *
         * @param name the configuration name
         * @param description the human-readable description of the configuration
         * @param topKRelevant the number of top documents used for feedback
         * @param topNTerms the number of expansion terms extracted from feedback documents
         * @param finalTopK the number of documents returned by the second-pass search
         */
        public Config(String name, String description, int topKRelevant, int topNTerms, int finalTopK) {
            this.name = name;
            this.description = description;
            this.topKRelevant = topKRelevant;
            this.topNTerms = topNTerms;
            this.finalTopK = finalTopK;
        }

        /**
         * Returns a compact textual representation of this configuration.
         *
         * @return a summary string containing the configuration parameters
         */
        @Override
        public String toString() {
            return String.format(
                "%s: k=%d, n=%d, topk=%d - %s",
                name, topKRelevant, topNTerms, finalTopK, description
            );
        }
    }

    // Predefined configurations
    public static final Config BALANCED = new Config(
        "BALANCED",
        "Default balanced configuration for general use",
        5, 10, 100
    );

    public static final Config HIGH_RECALL = new Config(
        "HIGH_RECALL",
        "Maximize recall - extract many expansion terms from many documents",
        10, 20, 1000
    );

    public static final Config HIGH_RECALL_AGGRESSIVE = new Config(
        "HIGH_RECALL_AGGRESSIVE",
        "Aggressive recall optimization",
        15, 30, 2000
    );

    public static final Config HIGH_PRECISION = new Config(
        "HIGH_PRECISION",
        "Maximize precision - selective term extraction",
        3, 5, 100
    );

    public static final Config HIGH_PRECISION_CONSERVATIVE = new Config(
        "HIGH_PRECISION_CONSERVATIVE",
        "Very conservative - minimal expansion",
        2, 3, 100
    );

    public static final Config SCIENTIFIC_MEDICAL = new Config(
        "SCIENTIFIC_MEDICAL",
        "Optimized for medical/scientific queries with technical synonyms",
        8, 15, 500
    );

    public static final Config SHORT_QUERY = new Config(
        "SHORT_QUERY",
        "For short 1-2 word queries",
        3, 8, 100
    );

    public static final Config LONG_QUERY = new Config(
        "LONG_QUERY",
        "For longer multi-word queries",
        5, 12, 150
    );

    public static final Config NICHE_DOMAIN = new Config(
        "NICHE_DOMAIN",
        "For niche/specialized collections with sparse terminology",
        8, 12, 200
    );

    public static final Config FAST = new Config(
        "FAST",
        "Optimized for speed - minimal computation",
        3, 5, 50
    );

    /**
     * Returns a predefined configuration by name.
     *
     * @param name the configuration name to resolve
     * @return the matching configuration, or {@link #BALANCED} when the name is unknown
     */
    public static Config getConfig(String name) {
        switch (name.toUpperCase()) {
            case "BALANCED": return BALANCED;
            case "HIGH_RECALL": return HIGH_RECALL;
            case "HIGH_RECALL_AGGRESSIVE": return HIGH_RECALL_AGGRESSIVE;
            case "HIGH_PRECISION": return HIGH_PRECISION;
            case "HIGH_PRECISION_CONSERVATIVE": return HIGH_PRECISION_CONSERVATIVE;
            case "SCIENTIFIC_MEDICAL": return SCIENTIFIC_MEDICAL;
            case "SHORT_QUERY": return SHORT_QUERY;
            case "LONG_QUERY": return LONG_QUERY;
            case "NICHE_DOMAIN": return NICHE_DOMAIN;
            case "FAST": return FAST;
            default: return BALANCED;
        }
    }

    /**
     * Prints all predefined configurations to standard output.
     */
    public static void printAllConfigs() {
        System.out.println("=== Available PRF Configurations ===");
        System.out.println();
        System.out.println(BALANCED);
        System.out.println(HIGH_RECALL);
        System.out.println(HIGH_RECALL_AGGRESSIVE);
        System.out.println(HIGH_PRECISION);
        System.out.println(HIGH_PRECISION_CONSERVATIVE);
        System.out.println(SCIENTIFIC_MEDICAL);
        System.out.println(SHORT_QUERY);
        System.out.println(LONG_QUERY);
        System.out.println(NICHE_DOMAIN);
        System.out.println(FAST);
        System.out.println();
    }

    /**
     * Creates a custom configuration.
     *
     * @param name the configuration name
     * @param description the human-readable configuration description
     * @param topKRelevant the number of top documents used for feedback
     * @param topNTerms the number of expansion terms to extract
     * @param finalTopK the number of documents returned by the final search
     * @return a configuration built from the provided parameters
     */
    public static Config custom(String name, String description, int topKRelevant, int topNTerms, int finalTopK) {
        return new Config(name, description, topKRelevant, topNTerms, finalTopK);
    }

    /**
     * Tunes a configuration by scaling its recall-oriented parameters.
     *
     * @param base the base configuration to adapt
     * @param recallScale the multiplicative factor applied to recall-oriented values
     * @return a new configuration derived from the base one
     */
    public static Config tuneForRecall(Config base, double recallScale) {
        return new Config(
            base.name + "_RECALL_" + String.format("%.1f", recallScale),
            "Recall-tuned version of " + base.name,
            (int) (base.topKRelevant * recallScale),
            (int) (base.topNTerms * recallScale),
            (int) (base.finalTopK * recallScale)
        );
    }

    /**
     * Tunes a configuration by scaling its precision-oriented parameters.
     *
     * @param base the base configuration to adapt
     * @param precisionScale the divisor used to make expansion more selective
     * @return a new configuration derived from the base one
     */
    public static Config tuneForPrecision(Config base, double precisionScale) {
        return new Config(
            base.name + "_PRECISION_" + String.format("%.1f", precisionScale),
            "Precision-tuned version of " + base.name,
            (int) (base.topKRelevant / precisionScale),
            (int) (base.topNTerms / precisionScale),
            base.finalTopK
        );
    }

    /**
     * Demonstrates how to inspect and use the available PRF configurations.
     *
     * @param args ignored command-line arguments
     */
    public static void main(String[] args) {
        System.out.println("PRF Configuration System");
        System.out.println("=======================");
        System.out.println();

        printAllConfigs();

        System.out.println("Usage Examples:");
        System.out.println();
        System.out.println("1. Use predefined configuration:");
        System.out.println("   Config config = PRFConfig.BALANCED;");
        System.out.println();

        System.out.println("2. Get configuration by name:");
        System.out.println("   Config config = PRFConfig.getConfig(\"HIGH_RECALL\");");
        System.out.println();

        System.out.println("3. Create custom configuration:");
        System.out.println("   Config config = PRFConfig.custom(\"MY_CONFIG\", \"My description\", 7, 12, 200);");
        System.out.println();

        System.out.println("4. Tune existing configuration for recall:");
        System.out.println("   Config tuned = PRFConfig.tuneForRecall(PRFConfig.BALANCED, 1.5);");
        System.out.println();

        System.out.println("5. Tune existing configuration for precision:");
        System.out.println("   Config tuned = PRFConfig.tuneForPrecision(PRFConfig.BALANCED, 1.5);");
        System.out.println();

        System.out.println("Then use with PRF:");
        System.out.println("   Config config = PRFConfig.HIGH_RECALL;");
        System.out.println("   Map<String, Object> result = PseudoRelevanceFeedback.twoPassSearchWithPRF(");
        System.out.println("       index, query, fields,");
        System.out.println("       config.topKRelevant,");
        System.out.println("       config.topNTerms,");
        System.out.println("       config.finalTopK");
        System.out.println("   );");
    }
}
