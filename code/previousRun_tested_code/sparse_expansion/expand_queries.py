"""
sparse_query_expander.py (with Porter Stemming)

Refines queries by applying tokenization, stopword removal, and Porter Stemming, followed by semantic expansion using a precomputed Word2Vec dictionary. This version is designed for systems where the index also uses stemmed tokens.

Workflow:

    Initializes MyEnglishAnalyzerNLTK to clean text (removes URLs, mentions, hashtags).

    Processes text through a standard NLP pipeline: Lowercasing -> Tokenization -> Stopword Removal -> Porter Stemming.

    Loads a Word2Vec expansion dictionary (must contain stemmed keys for matching).

    For each original token, it fetches up to N synonyms that exceed the similarity threshold.

    Combines original stemmed tokens and expansion terms into a single "sparse" string.

Key Components:

    PorterStemmer: Reduces words to their root form (e.g., "running" -> "run") to improve match consistency.

    RegexpTokenizer: Extracts alphanumeric tokens and dashes, discarding punctuation.

Parameters (Command Line):
threshold (argv[1])  Minimum similarity score for expansion terms (default: 0.8).
max_terms (argv[2])  Max synonyms to add per original stemmed token (default: 7).

Usage:
python sparse_query_expander.py [threshold] [max_terms]

Output:
expanded_queries_en.json  ← JSON enriched with the "sparse" field containing stemmed and expanded terms.
"""

import json
import re
import sys
import nltk
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)

class MyEnglishAnalyzerNLTK:
    def __init__(self, stopwords_file):
        self.url_pattern = re.compile(r"https?://\S+\s?")
        self.mention_pattern = re.compile(r"@\w+\s?")
        self.hashtag_symbol = re.compile(r"#")
        self.stemmer = PorterStemmer()

        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.stop_words = set(line.strip().lower() for line in f if line.strip())
        except FileNotFoundError:
            self.stop_words = set(stopwords.words('english'))

        self.tokenizer = RegexpTokenizer(r"[a-z0-9\-]+")

    def analyze(self, text):
        if not text:
            return []

        # Pre-processing
        text = self.url_pattern.sub("", text)
        text = self.mention_pattern.sub("", text)
        text = self.hashtag_symbol.sub("", text)

        text = text.lower()
        tokens = self.tokenizer.tokenize(text)

        cleaned_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                stemmed = self.stemmer.stem(token)
                cleaned_tokens.append(stemmed)

        return cleaned_tokens

def run_expansion_pipeline(input_path, model_path, output_path, stop_path, threshold=0.85, max_terms=3):
    analyzer = MyEnglishAnalyzerNLTK(stop_path)

    with open(model_path, 'r', encoding='utf-8') as f:
        expansion_lookup = json.load(f)

    with open(input_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    final_results = []

    for q in tqdm(queries, desc="Analysis and expansion"):
        original_text = q.get('original', '')


        analyzed_tokens = analyzer.analyze(original_text)

        combined_terms = set(analyzed_tokens)

        for token in analyzed_tokens:
            if token in expansion_lookup:
                similars = [item['term'] for item in expansion_lookup[token][:max_terms]
                            if item['score'] >= threshold]
                combined_terms.update(similars)

        output_obj = {
            "index": q.get("index"),
            "original": original_text,
            "expanded": q.get("expanded", ""),
            "sparse": " ".join(list(combined_terms)),
            "pubkey": q.get("pubkey")
        }

        final_results.append(output_obj)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
    MAX_TERMS = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    run_expansion_pipeline(
        input_path='../../data/expanded_queries_bge_large.json',
        model_path='../../src/main/java/unipd/se/expansion/word2vec_expansion.json',
        output_path='../../data/Train_set/expanded_queries_en.json',
        stop_path='../stoplist_en_TEX.txt',
        threshold=THRESHOLD,
        max_terms=MAX_TERMS
    )