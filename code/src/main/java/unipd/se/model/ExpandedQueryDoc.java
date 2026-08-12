package unipd.se.model;

/**
 * Represents a query enriched with additional information generated during
 * the query expansion phase.
 * <p>
 * This class extends {@link QueryDoc} by storing the original query text,
 * plus three retrieval-oriented variants optimized for sparse, dense
 * embedding, and ColBERT-style late interaction search.
 * </p>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class ExpandedQueryDoc extends QueryDoc {

    /**
     * Original text of the query before expansion.
     */
    public String original;

    /**
     * Keywords extracted from the original query.
     */
    public String keywords;

    /**
     * Sparse-optimized query representation.
     */
    public String sparse;

    /**
     * Dense-embedding-optimized query representation.
     */
    public String embedding;

    /**
     * ColBERT / late-interaction optimized query representation.
     */
    public String colbert;

    /**
     * Language code associated with the query record, when available.
     */
    public String language;

    /**
     * Backward-compatible alias for the expanded sparse representation.
     */
    public String expanded;

    /**
     * Returns the original, non-expanded query text.
     *
     * @return the original query text
     */
    public String getOriginal() {
        return original;
    }

    /**
     * Returns the expanded query text.
     * If the legacy {@code expanded} field is empty, the sparse
     * representation is returned instead.
     *
     * @return the expanded query text
     */
    public String getExpanded() {
        if (expanded != null && !expanded.isEmpty()) {
            return expanded;
        }
        return sparse;
    }

    /**
     * Returns the sparse-optimized query text.
     * If the dedicated sparse field is empty, the legacy expanded value is
     * used as a fallback.
     *
     * @return the sparse query text
     */
    public String getSparse() {
        if (sparse != null && !sparse.isEmpty()) {
            return sparse;
        }
        return getExpanded();
    }

    /**
     * Returns the dense-embedding-optimized query text.
     * If no dense representation is available, the sparse query text is
     * returned instead.
     *
     * @return the dense query text
     */
    public String getEmbedding() {
        if (embedding != null && !embedding.isEmpty()) {
            return embedding;
        }
        return getSparse();
    }

    /**
     * Returns the ColBERT-optimized query text.
     * If no ColBERT representation is available, the dense representation is
     * returned instead.
     *
     * @return the ColBERT query text
     */
    public String getColbert() {
        if (colbert != null && !colbert.isEmpty()) {
            return colbert;
        }
        return getEmbedding();
    }

    /**
     * Returns the language code if present.
     *
     * @return the language code, or null if unavailable
     */
    public String getLanguage() {
        return language;
    }

    /**
     * Returns the text that must be used by the search engine.
     * <p>
     * In this implementation, the search text corresponds to the sparse
     * query representation, with fallback to the backward-compatible
     * expanded value when needed.
     * </p>
     *
     * @return the query text used for retrieval
     */
    @Override
    public String getSearchText() {
        return getSparse();
    }
}
