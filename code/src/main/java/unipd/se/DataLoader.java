package unipd.se;

import unipd.se.model.Paper;
import unipd.se.model.QueryDoc;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Utility class responsible for loading domain data from JSON files.
 * <p>
 * It uses Jackson {@link ObjectMapper} to deserialize JSON arrays into
 * Java objects such as papers and queries.
 * </p>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public final class DataLoader {

    /**
     * Shared ObjectMapper instance (thread-safe after configuration).
     */
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * Private constructor to prevent instantiation of this utility class.
     */
    private DataLoader() {}

    /**
     * Loads a list of {@link Paper} objects from a JSON file.
     *
     * @param path the path to the JSON file
     * @return the list of loaded papers
     * @throws IOException if the file cannot be read or parsed
     */
    public static List<Paper> loadPapers(String path) throws IOException {
        return Arrays.asList(MAPPER.readValue(new File(path), Paper[].class));
    }

    /**
     * Loads a list of queries from a JSON file.
     * <p>
     * Works for both {@link QueryDoc} and subclasses such as
     * {@link unipd.se.model.ExpandedQueryDoc}.
     *
     * @param path the path to the JSON file
     * @param clazz the array class representing the concrete query type
     * @param <T> the query type extending {@link QueryDoc}
     * @return the list of loaded queries
     * @throws IOException if the file cannot be read or parsed
     */
    public static <T extends QueryDoc> List<T> loadQueries(String path, Class<T[]> clazz)
            throws IOException {
        return Arrays.asList(MAPPER.readValue(new File(path), clazz));
    }
}
