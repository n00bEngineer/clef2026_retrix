package unipd.se;

import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.similarities.BM25Similarity;
import unipd.se.model.QueryDoc;
import unipd.se.model.ExpandedQueryDoc;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
/**
 * Utility class for performing searches on a Lucene index of papers.
 * <p>
 * Supports both QueryDoc and subclasses (e.g., ExpandedQueryDoc).
 * Uses BM25 similarity and a weighted multi-field query (title + abstract).
 * </p>
 */
public class SearcherV2 {

    /** Shared custom analyzer for parsing queries. */
    private static final MyEnglishAnalyzer_TEX ANALYZER = new MyEnglishAnalyzer_TEX();

    /**
     * Search the index with a configurable title boost.
     * Queries are processed in parallel on a dedicated ForkJoinPool.
     * IndexSearcher is thread-safe for concurrent reads (as guaranteed by Lucene).
     *
     * @param dir        the Lucene index directory
     * @param queries    list of queries (QueryDoc or subclasses)
     * @param titleBoost boost applied to the title field
     * @param topK       number of top documents to retrieve
     * @return map from query index → ranked list of pubkeys
     *
     * @throws IOException if the index cannot be opened or if the search fails
     */
    public static Map<String, List<String>> search(
            Directory dir,
            List<? extends QueryDoc> queries,
            float titleBoost,
            int topK
    ) throws IOException {

        // Field weights (condivisi, immutabili)
        Map<String, Float> fields = new HashMap<>();
        fields.put("title",    titleBoost);
        fields.put("abstract", 1.0f);

        // IndexReader e IndexSearcher vengono aperti una volta sola e condivisi tra tutti i thread
        // DirectoryReader è thread-safe per letture concorrenti
        try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity());

            // Usa un ForkJoinPool con tanti thread quanti i core logici disponibili
            int cores = Runtime.getRuntime().availableProcessors();

            try (ForkJoinPool pool = new ForkJoinPool(cores)) {
                // Ogni query viene elaborata indipendentemente in parallelo
                // SimpleQueryParser viene creato per ogni thread (non è thread-safe)

                return pool.submit(() ->
                    queries.parallelStream().collect(Collectors.toMap(
                        q -> q.index,
                        q -> {
                            String text = q.getSearchText();
                            if (text == null || text.isEmpty()) {
                                return Collections.<String>emptyList();
                            }
                            try {
                                // SimpleQueryParser non è thread-safe: istanza locale per thread
                                SimpleQueryParser parser = new SimpleQueryParser(ANALYZER, fields);
                                Query query;

                                // Parse filters from query text
                                Map<String, String> filters = new HashMap<>();
                                Pattern venuePattern = Pattern.compile("\\bvenue:\\s*([^\\s]+)");
                                Pattern authorPattern = Pattern.compile("\\bauthor:\\s*([^\\s]+(?:\\s+[^\\s]+)*)");

                                if (q instanceof ExpandedQueryDoc eq) {
                                    String original = eq.getOriginal();
                                    String expanded = eq.getExpanded();

                                    // Parse filters from original
                                    String origSearch = original;
                                    Matcher mOrig = venuePattern.matcher(origSearch);
                                    if (mOrig.find()) {
                                        filters.put("venue", mOrig.group(1));
                                        origSearch = origSearch.replaceFirst("\\bvenue:\\s*[^\\s]+", "").trim();
                                    }
                                    Matcher m2Orig = authorPattern.matcher(origSearch);
                                    if (m2Orig.find()) {
                                        filters.put("authors", m2Orig.group(1));
                                        origSearch = origSearch.replaceFirst("\\bauthor:\\s*[^\\s]+(?:\\s+[^\\s]+)*", "").trim();
                                    }

                                    if (original != null && !original.isEmpty() && expanded != null && !expanded.isEmpty()) {
                                        // Create BooleanQuery with weighted clauses
                                        BooleanQuery.Builder bq = new BooleanQuery.Builder();

                                        Query origQuery = parser.parse(origSearch);
                                        origQuery = new BoostQuery(origQuery, 2.0f);
                                        bq.add(origQuery, BooleanClause.Occur.SHOULD);

                                        Query expQuery = parser.parse(expanded);
                                        expQuery = new BoostQuery(expQuery, 1.0f);
                                        bq.add(expQuery, BooleanClause.Occur.SHOULD);

                                        query = bq.build();
                                    } else if (expanded != null && !expanded.isEmpty()) {
                                        query = parser.parse(expanded);
                                    } else if (original != null && !original.isEmpty()) {
                                        query = parser.parse(origSearch);
                                    } else {
                                        query = parser.parse(text);
                                    }
                                } else {
                                    // Regular QueryDoc
                                    String searchText = text;
                                    Matcher m = venuePattern.matcher(searchText);
                                    if (m.find()) {
                                        filters.put("venue", m.group(1));
                                        searchText = searchText.replaceFirst("\\bvenue:\\s*[^\\s]+", "").trim();
                                    }
                                    Matcher m2 = authorPattern.matcher(searchText);
                                    if (m2.find()) {
                                        filters.put("authors", m2.group(1));
                                        searchText = searchText.replaceFirst("\\bauthor:\\s*[^\\s]+(?:\\s+[^\\s]+)*", "").trim();
                                    }
                                    query = parser.parse(searchText);
                                }

                                // Apply filters if any
                                if (!filters.isEmpty()) {
                                    BooleanQuery.Builder bq = new BooleanQuery.Builder();
                                    bq.add(query, BooleanClause.Occur.MUST);
                                    for (Map.Entry<String, String> f : filters.entrySet()) {
                                        TermQuery tq = new TermQuery(new Term(f.getKey(), f.getValue()));
                                        bq.add(tq, BooleanClause.Occur.MUST);
                                    }
                                    query = bq.build();
                                }

                                TopDocs topDocs = searcher.search(query, topK);

                                List<String> topIds = new ArrayList<>(topDocs.scoreDocs.length);
                                for (ScoreDoc sd : topDocs.scoreDocs) {
                                    Document doc = searcher.storedFields().document(sd.doc);
                                    topIds.add(doc.get("pubkey"));
                                }
                                return topIds;
                            } catch (IOException e) {
                                System.err.println("Search failed for query " + q.index + ": " + e.getMessage());
                                return Collections.<String>emptyList();
                            }
                        },
                        (a, b) -> a,
                        LinkedHashMap::new   // mantieni ordine di inserimento
                    ))
                ).get();

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Search interrupted", e);
            } catch (ExecutionException e) {
                throw new IOException("Search execution failed", e.getCause());
            }
        }
    }
}
