"""
Expand English queries with BGE-large semantic neighbors and noun keywords.

The script embeds the corpus and queries with `BAAI/bge-large-en-v1.5`, retrieves
the top semantic neighbors for each query, extracts high-frequency candidate
terms from those neighbors, and combines them with spaCy noun/proper-noun
keywords from the original query.

BGE-large requires the retrieval instruction prefix only for queries; corpus
documents are encoded without a prefix. The output JSON keeps the original query
identifier and adds `original`, `keywords`, and `expanded` fields consumed by
the downstream Lucene/BM25 stage.
"""

import json
import re
import torch
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import unicodedata
import emoji
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS_BATCH_SIZE = 256
QUERY_BATCH_SIZE  = 32
TOP_K_DOCS        = 5
TOP_K_TERMS       = 15
SPACY_BATCH_SIZE  = 128
SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[1] 
REPO_ROOT = SCRIPT_PATH.parents[2]
DATA_BASE = CODE_ROOT / "data"

MODEL_NAME = "BAAI/bge-large-en-v1.5"

# BGE uses this prefix for query embeddings only; corpus embeddings stay raw.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_NAME}")

# ──────────────────────────────────────────────────────────────
# Lucene stop words
# ──────────────────────────────────────────────────────────────
LUCENE_STOPS: set[str] = {
    "a","an","and","are","as","at","be","but","by","for","if","in","into",
    "is","it","no","not","of","on","or","such","that","the","their","then",
    "there","these","they","this","to","was","will","with",
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","them","themselves","what","which","who",
    "whom","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","nor","only","own","same","so","than","too","very",
    "s","t","can","just","don","should","now","d","ll","m","o","re","ve","y",
    "ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma",
    "mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn",
}

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:['\-][a-zA-Z0-9]+)*")


def load_stopwords(path):
    """Load newline-delimited stop words as lowercase tokens.

    Args:
        path: Path to a UTF-8 stop-word file.

    Returns:
        A set of stripped, lowercased stop words.

    Raises:
        OSError: If the stop-word file cannot be read.
    """
    with open(path, encoding="utf-8") as f:
        return {line.strip().lower() for line in f}

STOPWORDS = load_stopwords(
    "../data/stoplist_en_ranksnl_large.txt"
)
STOPWORDS.update(LUCENE_STOPS)


def normalize_ascii(text):
    """Return an ASCII-only representation of `text`.

    Non-ASCII marks are stripped after NFKD normalization so downstream token
    matching remains aligned with Lucene-style ASCII analysis.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def clean_query(text: str) -> str:
    """Normalize social-text markers before building the expanded query.

    Args:
        text: Raw query text.

    Returns:
        Query text with mentions/hashtags converted to bare terms, emoji
        removed, and whitespace collapsed.
    """
    text = re.sub(r"\B@(\w+)", r"\1", text)
    text = re.sub(r"\B#(\w+)", r"\1", text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def raw_tokenize(text, stopwords=None):
    """Tokenize corpus text for expansion-term counting.

    Args:
        text: Source document text.
        stopwords: Optional stop-word set. When omitted, the module-level
            `STOPWORDS` set is used.

    Returns:
        Lowercased ASCII tokens with mentions, stop words, short terms, and
        possessive suffixes removed.
    """
    if stopwords is None:
        stopwords = STOPWORDS

    text = text.strip().lower()
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\B#(\w+)", r"\1", text)
    text = normalize_ascii(text)

    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)

    result = []
    for t in tokens:
        if t.endswith("'s"):
            t = t[:-2]
        if t in stopwords:
            continue
        if len(t) < 3:
            continue
        result.append(t)

    return result

# ──────────────────────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────────────────────
print("Loading spaCy (tagger only)...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

print(f"Loading {MODEL_NAME} on {DEVICE}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

if DEVICE == "cuda":
    try:
        model.half()
        print("fp16 enabled")
        print(f"exstimated VRAM for corpus encode: ~{(1024 * 4 * CORPUS_BATCH_SIZE) / 1024**2:.0f} MB/batch")
    except Exception as e:
        print(f"fp16 not available ({e}), using fp32")

# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────
print("Loading data...")
with open(f"{DATA_BASE}/collection_data.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
with open(f"{DATA_BASE}/fr_DEV_en.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

corpus_texts: list[str] = [
    (p.get("title", "") + " " + p.get("abstract", "")).strip()
    for p in papers
]
query_texts: list[str] = [q.get("text", "") for q in queries]

# ──────────────────────────────────────────────────────────────
# Pre-tokenize corpus
# ──────────────────────────────────────────────────────────────
print("Pre-tokenizing corpus (raw, no stemming)...")
corpus_raw_tokens: list[list[str]] = [
    raw_tokenize(t) for t in tqdm(corpus_texts, desc="Tokenizing corpus")
]

# ──────────────────────────────────────────────────────────────
# Encode corpus without the BGE query instruction prefix.
# ──────────────────────────────────────────────────────────────
print("Encoding corpus (GPU, no prefix)...")
corpus_embeddings = model.encode(
    corpus_texts,
    batch_size=CORPUS_BATCH_SIZE,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=DEVICE,
    normalize_embeddings=True,
)

# ──────────────────────────────────────────────────────────────
# Encode queries with the BGE retrieval instruction prefix.
# ──────────────────────────────────────────────────────────────
print(f"Encoding queries (GPU, with prefix: '{BGE_QUERY_PREFIX[:40]}...')...")
query_texts_prefixed = [BGE_QUERY_PREFIX + t for t in query_texts]
query_embeddings = model.encode(
    query_texts_prefixed,
    batch_size=QUERY_BATCH_SIZE,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=DEVICE,
    normalize_embeddings=True,
)

# ──────────────────────────────────────────────────────────────
# Semantic search
# ──────────────────────────────────────────────────────────────
print("Semantic search (GPU)...")
all_hits = util.semantic_search(
    query_embeddings,
    corpus_embeddings,
    top_k=TOP_K_DOCS,
    score_function=util.dot_score,
)

# ──────────────────────────────────────────────────────────────
# Keyword extraction via spaCy (batch)
# ──────────────────────────────────────────────────────────────
print("Extracting keywords from queries (spaCy batch)...")
query_docs = list(nlp.pipe(query_texts, batch_size=SPACY_BATCH_SIZE, n_process=1))

def extract_query_keywords(doc: spacy.tokens.Doc) -> str:
    """Extract stable noun-like query keywords from a spaCy document.

    Args:
        doc: spaCy document produced with POS tagging enabled.

    Returns:
        A space-separated string of unique noun/proper-noun lemmas, preserving
        first occurrence order.
    """
    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and len(token.text) > 2
    ]
    return " ".join(dict.fromkeys(keywords))

# ──────────────────────────────────────────────────────────────
# Assemble expanded queries
# ──────────────────────────────────────────────────────────────
print("Assembling expanded queries...")
expanded_queries: list[dict] = []

for i, q in enumerate(tqdm(queries, desc="Expanding")):
    original  = query_texts[i]
    query_doc = query_docs[i]

    keywords = extract_query_keywords(query_doc)

    expansion_pool: list[str] = []
    for hit in all_hits[i]:
        cid    = hit["corpus_id"]
        score  = hit["score"]
        weight = max(1, round(score * 5))
        expansion_pool.extend(corpus_raw_tokens[cid] * weight)

    counter = Counter(expansion_pool)
    expansion_terms_raw = [w for w, _ in counter.most_common(TOP_K_TERMS * 2)]

    keywords_set = set(keywords.lower().split())
    expansion_filtered = [
        t for t in expansion_terms_raw
        if t not in keywords_set
    ][:TOP_K_TERMS]

    expansion_str = " ".join(expansion_filtered)
    original_clean = clean_query(original)
    expanded = " ".join(filter(None, [original_clean, keywords, expansion_str])).strip()

    expanded_queries.append({
        "index":    q["index"],
        "original": original,
        "keywords": keywords,
        "expanded": expanded,
        "pubkey":   q.get("pubkey", ""),
    })

# ──────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────
output_path = f"{DATA_BASE}/expanded_queries_bge_large_frDEV_en.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(expanded_queries, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(expanded_queries)} expanded queries → {output_path}")
