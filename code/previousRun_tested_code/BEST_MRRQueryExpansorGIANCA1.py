import json
import re
import spacy
import emoji
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
 
# -----------------------------
# Load models 
# -----------------------------
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading BERTweet tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

# -----------------------------
# Slang / abbreviation dictionary
# -----------------------------
SLANG_DICT = {
    "gr8": "great",
    "lol": "laugh",
    "omg": "oh my god",
    "idk": "i do not know",
    "btw": "by the way",
    "tbh": "to be honest",
    "smh": "shaking my head",
    "afaik": "as far as i know",
    "imo": "in my opinion",
    "fyi": "for your information",
    # Add more as needed
}

# -----------------------------
# Stopwords
# -----------------------------
STOPWORDS = nlp.Defaults.stop_words

# -----------------------------
# Helper functions
# -----------------------------
def split_hashtag(token: str) -> list[str]:
    """
    Split hashtags into words using simple camel-case or underscores.
    """
    token = token.lstrip("#")
    # Split camel case
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', token)
    # Split underscores
    split_words = []
    for w in words:
        split_words.extend(w.split("_"))
    return [w.lower() for w in split_words if w]

def normalize_emoji(text: str) -> str:
    """
    Convert emojis to textual descriptions using emoji.demojize.
    """
    return emoji.demojize(text, delimiters=(" ", " "))

def normalize_slang(token: str) -> str:
    """
    Replace common social media abbreviations/slang.
    """
    return SLANG_DICT.get(token.lower(), token)

# -----------------------------
# 1. Normalization
# -----------------------------
def normalize_query(text: str) -> str:
    """
    Full social media query normalization:
    - BERTweet tokenization
    - Hashtag splitting
    - Emoji normalization
    - Slang/abbreviation normalization
    - Stopword removal
    """
    if not text:
        return ""

    # Step 1: Emoji normalization
    text = normalize_emoji(text)

    # Step 2: BERTweet tokenization
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in ["[CLS]", "[SEP]"]]

    normalized_tokens = []
    for token in tokens:
        # Step 3: Hashtag splitting
        if token.startswith("#"):
            split_tokens = split_hashtag(token)
            normalized_tokens.extend(split_tokens)
        else:
            normalized_tokens.append(token)

    # Step 4: Slang normalization
    normalized_tokens = [normalize_slang(t) for t in normalized_tokens]

    # Step 5: Stopword removal and remove very short tokens
    normalized_tokens = [
        t for t in normalized_tokens
        if t.lower() not in STOPWORDS and len(t) > 1
    ]

    return " ".join(normalized_tokens)


# -----------------------------
# 2. Dependency keyword extraction
# -----------------------------
def extract_keywords(text: str) -> str:
    if not text:
        return ""

    doc = nlp(text)

    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop
    ]

    return " ".join(keywords)

# -----------------------------
# 3. Load data
# -----------------------------
with open("data/collection_data.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

with open("data/en_train.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

# Build corpus
corpus_texts = [
    (p.get("title", "") + " " + p.get("abstract", "")).strip()
    for p in papers
]

# -----------------------------
# 4. SBERT model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding corpus...")
corpus_embeddings = model.encode(
    corpus_texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

# -----------------------------
# 5. Process queries
# -----------------------------
expanded_queries = []

for q in tqdm(queries, desc="Processing queries"):
    original = q.get("text", "")

    # Step 1: Normalize
    #normalized = normalize_query(original)
    normalized = original

    # Step 2: Extract keywords (syntax-based)
    keywords = extract_keywords(normalized)

    # Step 3: SBERT embedding expansion
    query_emb = model.encode(normalized, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, corpus_embeddings, top_k=5)[0]

    expansion_terms = []
    for hit in hits:
        doc_text = corpus_texts[hit["corpus_id"]]
        expansion_terms.extend(doc_text.split())

    # Deduplicate + limit
    expansion_terms = list(dict.fromkeys(expansion_terms))[:15]

    # Final query
    expanded = " ".join([
        #normalized,
        keywords,
        " ".join(expansion_terms)
    ]).strip()

    expanded_queries.append({
        "index": q["index"],
        "original": original,
        "keywords": keywords,
        "expanded": expanded,
        "pubkey": q.get("pubkey", "")
    })


# -----------------------------
# 6. Save output
# -----------------------------
output_path = "data/expanded_queries_3.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(expanded_queries, f, indent=2, ensure_ascii=False)

print(f"\nSaved expanded queries to: {output_path}")