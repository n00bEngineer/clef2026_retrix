package unipd.se.model;

/**
 * Represents a query enriched with additional information generated during
 * the query expansion phase.
 * <p>
 * This class extends {@link QueryDoc} by storing the original query text
 * and the expansion terms for sparse retrieval.
 * </p>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class ExpandedQueryDoc_TEX extends QueryDoc {
    /**
     * Original text
     */
    public String original;

    /**
     * Expanded query using a LLM
     */
    public String expanded;

    /**
     * Extra terms used to improve retrieval
     */
    public String sparse;

    /**
     * Additional keyword-oriented expansion terms generated for the query.
     */
    public String keywords;

    /**
     * Returns the original, non-expanded query text.
     *
     * @return the original query text
     */
    public String getOriginal() {
        return original;
    }

    /**
     * Returns the expanded query.
     *
     * @return the expanded query
     */
    public String getExpanded() {
        return expanded;
    }

    /**
     * Returns the expansion terms to add to the query (sparse).
     *
     * @return the expansion terms
     */
    public String getSparse() {
        return sparse;
    }
}
