package unipd.se.model;

/**
 * Represents a generic query used in the retrieval pipeline.
 * Implementations provide the query identifier, the identifier of the
 * relevant document, and the text to be used during retrieval.
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public interface SearchQuery {

    /**
     * Returns the unique identifier of the query.
     *
     * @return the query identifier
     */
    String getIndex();

    /**
     * Returns the identifier of the relevant document associated with the query.
     *
     * @return the relevant document identifier
     */
    String getPubkey();

    /**
     * Returns the text that must be used for retrieval.
     *
     * @return the search text
     */
    String getSearchText();
}
