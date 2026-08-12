"""
word2vec_expansion.py

Builds a domain-specific query expansion dictionary using Word2Vec embeddings trained on a medical corpus.

Workflow:

    Loads a custom stoplist to filter out non-informative terms.

    Preprocesses titles and abstracts (cleaning citations, URLs, and punctuation).

    Trains a Word2Vec model (Skip-gram/CBOW) to learn semantic relationships.

    Generates a JSON dictionary where each key is a term and the value is a list
    of semantically similar synonyms (cosine similarity > 0.6).

Word2Vec Configuration:
vector_size    200
window         10
min_count      3
epochs         10

Filters:

    Stopwords are never expanded.

    Stopwords are removed from the candidate synonym lists.

    Only terms with a similarity score > 0.6 are kept.

Usage:
python word2vec_expansion.py

Output:
word2vec_expansion.json  ← Expansion dictionary {"word": [{"term": str, "score": float}, ...]}
"""

import json
import re
from gensim.models import Word2Vec
from tqdm import tqdm

def load_stopwords(filepath):
    """Loads stopwords from a text file, one per row."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Legge, pulisce spazi e filtra righe vuote o commenti
            return set(line.strip().lower() for line in f if line.strip() and not line.startswith('#'))
    except FileNotFoundError:
        print(f"Warning: stoplist not found in {filepath}. Continuing without filter.")
        return set()

def clean_medical_text(text):
    if not text:
        return []
    text = re.sub(r'\[\d+]|\(\d+\)|Google Scholar|Crossref|PubMed', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\-]', ' ', text)
    return text.split()

def run_pipeline(input_file, output_file, stopwords_file):
    # 1. Loading stopwords
    stopwords = load_stopwords(stopwords_file)
    print(f"Loaded {len(stopwords)} stopwords.")

    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentences = []
    for paper in tqdm(data, desc="Preprocessing"):
        content = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        tokens = clean_medical_text(content)
        if tokens:
            sentences.append(tokens)

    print(f"Training Word2Vec on {len(sentences)} documents...")
    model = Word2Vec(
        sentences,
        vector_size=200,
        window=10,
        min_count=3,
        workers=4,
        epochs=10
    )

    print("Extracting synonyms with stopwords filter...")
    expansion_dict = {}
    vocab = model.wv.index_to_key

    for word in tqdm(vocab, desc="Generating Dictionary"):
        # Filtro 1: We do not expand stopwords
        if word in stopwords:
            continue

        similar = model.wv.most_similar(word, topn=10) # We take extra terms and then filter

        # Filtro 2: We only keep similar terms (except for stopwords)
        filtered_similar = [
            {"term": s[0], "score": round(s[1], 3)}
            for s in similar
            if s[0] not in stopwords and s[1] > 0.6
        ]

        if filtered_similar:
            expansion_dict[word] = filtered_similar

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expansion_dict, f, indent=2)
    print(f"Dictionary saved. Expanded terms: {len(expansion_dict)}")

if __name__ == "__main__":
    # Paths to be updated depending on the structure
    DATA_PATH = '../../../../../../data/collection_data.json'
    STOPLIST_PATH = '../../../../../../data/stoplist_en_TEX.txt'
    OUTPUT_FILE = 'word2vec_expansion.json'

    run_pipeline(DATA_PATH, OUTPUT_FILE, STOPLIST_PATH)