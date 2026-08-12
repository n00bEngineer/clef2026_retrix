package unipd.se.model;

/**
 * Represents a basic query record loaded from the dataset.
 * Each query stores its identifier, the original query text, and the
 * identifier of the relevant scientific paper.
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class QueryDoc implements SearchQuery {

    /**
     * Unique identifier of the query in the dataset.
     */
    public String index;

    /**
     * Original text of the query.
     */
    public String text;

    /**
     * Identifier of the relevant paper associated with the query.
     */
    public String pubkey;

    /**
     * Returns the identifier of the query.
     *
     * @return the query identifier
     */
    @Override
    public String getIndex() {
        return index;
    }

    /**
     * Returns the identifier of the relevant paper.
     *
     * @return the relevant paper identifier
     */
    @Override
    public String getPubkey() {
        return pubkey;
    }

    /**
     * Returns the original search text of the query.
     *
     * @return the query text
     */
    @Override
    public String getSearchText() {
        return text;
    }
}
