package unipd.se.model;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Represents a scientific paper in the document collection.
 * <p>
 * Each paper stores the identifier used in the dataset, together with
 * its title, abstract, publication venue, and authorship information.
 * </p>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class Paper {

    /**
     * Unique identifier of the paper in the dataset.
     */
    public String pubkey;

    /**
     * Title of the scientific paper.
     */
    public String title;

    /**
     * Abstract text of the scientific paper.
     *
     * It is mapped from the JSON field named {@code abstract}.
     */
    @JsonProperty("abstract")
    public String abstractText;

    /**
     * Publication venue of the paper, such as a journal or conference.
     */
    public String venue;

    /**
     * Authors of the scientific paper.
     */
    public String authors;
}
