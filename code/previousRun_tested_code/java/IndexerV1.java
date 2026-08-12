package unipd.se;

import org.apache.lucene.store.FSDirectory;
import unipd.se.model.Paper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.Directory;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

/**
 * Utility class for building a Lucene index from a collection of papers.
 * <p>
 * This class creates a persistent, on-disk index using {@link FSDirectory}.
 * Each paper is stored as a {@link Document} with the following fields:
 * <ul>
 *     <li>{@code pubkey} – unique identifier, stored but not tokenized</li>
 *     <li>{@code title} – paper title, tokenized for full-text search</li>
 *     <li>{@code abstract} – paper abstract, tokenized for full-text search</li>
 * </ul>
 * <p>
 * The index is stored in the {@code index/} folder under the project root.
 * A shared {@link StandardAnalyzer} is used, and documents are buffered in memory
 * to improve indexing performance.
 * </p>
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class IndexerV1 {

    /**
     * Shared custom analyzer for tokenizing text fields.
     */
    private static final MyEnglishAnalyzer_TEX ANALYZER = new MyEnglishAnalyzer_TEX();

    /**
     * Builds a persistent Lucene index from the given list of {@link Paper} objects.
     * <p>
     * Each paper is converted into a Lucene {@link Document} with its {@code pubkey},
     * {@code title}, and {@code abstract} fields.
     * </p>
     *
     * @param papers the list of {@link Paper} objects to index
     * @return a {@link Directory} representing the persistent on-disk index
     * @throws IOException if an error occurs while writing the index to disk
     */
    public static Directory buildIndex(List<Paper> papers) throws IOException {
        Directory dir = FSDirectory.open(Paths.get("index"));

        IndexWriterConfig config = new IndexWriterConfig(ANALYZER);
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        // RAM buffer aumentato: riduce i flush intermedi su disco durante l'indicizzazione
        config.setRAMBufferSizeMB(512.0);
        // Usa tutti i core disponibili per il merge dei segmenti
        int cores = Runtime.getRuntime().availableProcessors();
        config.setMergeScheduler(new ConcurrentMergeScheduler());
        ((ConcurrentMergeScheduler) config.getMergeScheduler()).setMaxMergesAndThreads(cores, Math.max(1, cores / 2));

        try (IndexWriter writer = new IndexWriter(dir, config)) {
            // Aggiungi i documenti in batch per ridurre l'overhead per documento
            final int BATCH = 500;
            int i = 0;
            for (Paper p : papers) {
                Document doc = new Document();
                doc.add(new StringField("pubkey",    p.pubkey,       Field.Store.YES));
                doc.add(new TextField("title",       p.title,        Field.Store.YES));
                doc.add(new TextField("abstract",    p.abstractText, Field.Store.YES));
                writer.addDocument(doc);

                // Flush esplicito ogni BATCH documenti: mantiene l'heap sotto controllo
                // senza aspettare che RAMBuffer si riempia del tutto
                if (++i % BATCH == 0) {
                    writer.flush();
                }
            }
        }

        return dir;
    }
}
