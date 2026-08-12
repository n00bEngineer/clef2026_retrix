# Pseudo Relevance Feedback (PRF) Implementation

## Overview

PRF (Pseudo Relevance Feedback) is a classic Information Retrieval technique that improves search **recall** by automatically expanding queries with related terms from top-ranked results.

**How it works:**
1. **Pass 1**: Execute initial search with the original query
2. **Extract**: Analyze the top 3-5 results and count term frequencies
3. **Expand**: Add the most frequent non-stopword terms to the query
4. **Pass 2**: Re-search with the expanded query to find related documents

This is particularly useful in scientific/medical domains where terminology variations are common (e.g., "vaccine" vs "immunization", "COVID-19" vs "SARS-CoV-2").

## Architecture

The implementation consists of:

### `PseudoRelevanceFeedback.java`
Main class with core PRF functionality:

- **`PRFResult`** (inner class): Encapsulates PRF output
  - `topDocIds`: Top documents from initial search
  - `expandedTerms`: Extracted high-frequency terms
  - `expandedQuery`: Original query + expansion terms

- **`performPRF(...)`**: Single PRF iteration
  - Executes initial search
  - Extracts terms from top-K results
  - Selects top-N most frequent terms
  - Builds expanded query
  
- **`twoPassSearchWithPRF(...)`**: Complete two-pass search
  - Performs PRF expansion
  - Returns both Pass 1 and Pass 2 results
  - Includes metadata (expanded terms, expanded query)

### `PRFExample.java`
Demonstration class showing how to use PRF in practice

## Usage

### Basic Single Pass (Expansion Only)

```java
import unipd.se.*;
import org.apache.lucene.store.Directory;
import java.util.HashMap;
import java.util.Map;

// Setup
List<Paper> papers = DataLoader.loadPapers("code/data/collection_data.json");
Directory index = IndexerV1.buildIndex(papers);

Map<String, Float> fields = new HashMap<>();
fields.put("title", 2.0f);      // Title boost
fields.put("abstract", 1.0f);   // Abstract weight

// Perform PRF
PseudoRelevanceFeedback.PRFResult result = PseudoRelevanceFeedback.performPRF(
    index,
    "vaccine effectiveness",  // original query
    fields,
    5,      // topKRelevant: analyze top 5 initial results
    10,     // topNTerms: extract top 10 expansion terms
    100     // searchTopK: retrieve top 100 in initial search
);

// Access results
System.out.println("Top 5 results: " + result.topDocIds);
System.out.println("Expansion terms: " + result.expandedTerms);
System.out.println("Expanded query: " + result.expandedQuery);
```

### Two-Pass Search with PRF

```java
// Perform complete two-pass search with PRF
Map<String, Object> twoPassResult = PseudoRelevanceFeedback.twoPassSearchWithPRF(
    index,
    "vaccine effectiveness",
    fields,
    5,      // topKRelevant
    10,     // topNTerms
    100     // finalTopK: retrieve top 100 in final results
);

// Access results
List<String> pass1Docs = (List<String>) twoPassResult.get("pass1_docs");
List<String> pass2Docs = (List<String>) twoPassResult.get("pass2_docs");
List<String> expandedTerms = (List<String>) twoPassResult.get("expanded_terms");
String expandedQuery = (String) twoPassResult.get("expanded_query");
```

### Command Line Demo

```bash
# Compile
mvn clean compile

# Run PRF example
java -cp target/classes unipd.se.PRFExample code/data/collection_data.json "vaccine effectiveness"
```

Output includes:
- Initial top 5 results
- Extracted expansion terms (sorted by frequency)
- Expanded query string
- Final top 100 results from Pass 2

## Parameters

### Key Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topKRelevant` | int | 5 | Number of top results from Pass 1 to analyze for term extraction (typically 3-5) |
| `topNTerms` | int | 10 | Number of expansion terms to extract (typically 10-20) |
| `searchTopK` / `finalTopK` | int | 100 | Number of documents to retrieve (100-1000 depending on needs) |

### Tuning Recommendations

- **High Recall, Lower Precision**: Increase `topNTerms` (e.g., 20-30) and `topKRelevant` (e.g., 5-10)
- **Balanced**: Use `topKRelevant=5, topNTerms=10` (default)
- **High Precision**: Decrease both parameters

## How It Works Internally

### 1. Term Extraction
- Uses the same `MyEnglishAnalyzer_V2` as indexing
- Ensures consistency: same tokenization, stemming, stopword removal
- Tokenizes title and abstract from top-K documents

### 2. Term Ranking
Terms are sorted by:
1. **Frequency** (descending) - most common terms first
2. **Alphabetically** (ascending) - tie-breaking for determinism

### 3. Query Expansion
Format: `"original_query term1 OR term2 OR term3 ..."`

The `SimpleQueryParser` interprets this as:
- Must contain original query terms (boosted 2x if original was title search)
- Should contain any of the expansion terms (SHOULD clauses)
- Documents with more expansion terms rank higher

## Performance Characteristics

- **Time Complexity**: O(N log N) for term extraction and sorting (N = unique terms in top-K docs)
- **Space**: O(M) where M = unique terms across top-K documents (typically < 1000)
- **Index Access**: Two full index traversals (Pass 1 + Pass 2)

## Integration with SearcherV2

To integrate PRF into the main search pipeline:

```java
// Option 1: Use PRF results directly
Map<String, Object> prfResults = PseudoRelevanceFeedback.twoPassSearchWithPRF(
    index, query, fields, 5, 10, 100
);
List<String> finalDocs = (List<String>) prfResults.get("pass2_docs");

// Option 2: Create ExpandedQueryDoc and use with SearcherV2
ExpandedQueryDoc expandedQuery = new ExpandedQueryDoc(
    index,
    originalQuery,
    prfResult.expandedQuery
);
Map<String, List<String>> results = SearcherV2.search(
    index,
    Collections.singletonList(expandedQuery),
    2.0f,  // title boost
    100
);
```

## Limitations & Considerations

1. **Initial Query Quality**: PRF assumes top results are relevant. Bad initial queries may degrade results.
2. **Vocabulary Mismatch**: Works best when collection contains relevant vocabulary variations.
3. **Computational Cost**: Two passes through index; consider caching for repeated queries.
4. **Stopword Handling**: Relies on stoplist; ensure `stoplist_en_ranksnl_large.txt` is accurate.

## Evaluation Metrics

PRF primarily improves:
- **Recall**: Higher chance of finding relevant documents
- **MAP (Mean Average Precision)**: Often improves at lower cutoffs

May slightly reduce:
- **Precision**: More documents retrieved, some non-relevant

## Example: Medical Query Expansion

```
Original Query: "vaccine adverse effects"

Top 5 Results contain terms:
- "immunization" (freq: 8)
- "side effects" (freq: 7)
- "inoculation" (freq: 5)
- "safety" (freq: 4)
- "complications" (freq: 3)

Expanded Query: "vaccine adverse effects immunization OR side OR effects OR inoculation OR safety OR complications"

Pass 2 finds additional documents discussing:
- "immunization safety"
- "vaccine side effects"
- "inoculation adverse reactions"
```

---