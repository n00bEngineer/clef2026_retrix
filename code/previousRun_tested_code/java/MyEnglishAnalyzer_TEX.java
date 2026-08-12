package unipd.se;

import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.en.EnglishPossessiveFilter;
import org.apache.lucene.analysis.en.KStemFilter;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.pattern.PatternReplaceCharFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Custom analyzer for social-post-like queries and scientific documents.
 * <p>
 * Features:
 * - Removes URLs
 * - Removes mentions (@user → '')
 * - Splits compound terms (e.g., "SARS-CoV-2" → "SARS", "CoV", "2") while preserving the original
 * - Removes '#' from hashtags (#anxiety → anxiety)
 * - Lowercases text
 * - Removes stopwords
 * - Applies stemming (KStem stemmer)
 * - Normalizes accents
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class MyEnglishAnalyzer_TEX extends Analyzer {
    private static final Pattern URL_PATTERN = Pattern.compile("https?://\\S+\\s?");
    private static final Pattern MENTION_PATTERN = Pattern.compile("@\\w+\\s?");
    private static final Pattern HASHTAG_SYMBOL = Pattern.compile("#");

    /**
     * Preprocesses the incoming text before tokenization by removing URLs, mentions, and hashtag markers.
     *
     * @param fieldName the name of the field being analyzed
     * @param reader the original character reader
     * @return a reader wrapped with the configured character filters
     */
    @Override
    protected Reader initReader(String fieldName, Reader reader) {
        Reader filter = new PatternReplaceCharFilter(URL_PATTERN, "", reader);
        filter = new PatternReplaceCharFilter(MENTION_PATTERN, "", filter);
        filter = new PatternReplaceCharFilter(HASHTAG_SYMBOL, "", filter);

        return filter;
    }

    /**
     * Builds the tokenization pipeline used after character-level normalization.
     *
     * @param fieldName the name of the field being analyzed
     * @return the tokenizer together with the configured token filters
     */
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer tokenizer = new StandardTokenizer();
        TokenStream stream = tokenizer;

        CharArraySet stopWords = loadStopWords();

        stream = new ASCIIFoldingFilter(stream);
        stream = new LowerCaseFilter(stream);
        stream = new EnglishPossessiveFilter(stream);
        stream = new StopFilter(stream, stopWords);
        stream = new KStemFilter(stream);

        return new TokenStreamComponents(tokenizer, stream);
    }

    /**
     * Loads the custom stopword list used by this analyzer.
     *
     * @return a mutable stopword set, or an empty set when the custom stoplist is unavailable
     */
    private CharArraySet loadStopWords() {
        CharArraySet stopWords = new CharArraySet(32, true);
        try {
            Path path = Paths.get("code", "data", "stoplist_en_TEX.txt");
            if (Files.exists(path)) {
                List<String> lines = Files.readAllLines(path);
                stopWords.addAll(lines);
            }
        } catch (IOException e) {
            System.err.println("Warning: Could not load custom stoplist, using default empty set.");
        }
        return stopWords;
    }
}
