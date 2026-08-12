# PRF Implementation - Summary

## How It Works

**The PRF Process:**

```
Query: "vaccine effectiveness"
    ↓
[Pass 1: Initial Search]
    ↓ Retrieve top 5 results
    ↓
[Term Extraction]
    Analyze titles + abstracts
    Count term frequencies
    Remove stopwords (using MyEnglishAnalyzer_V2)
    Extract top 10 terms: immunization, safety, efficacy, trials, ...
    ↓
[Query Expansion]
    "vaccine effectiveness OR immunization OR safety OR efficacy OR trials OR ..."
    ↓
[Pass 2: Expanded Search]
    ↓ Retrieve top 100 results with expanded query
    ↓
Final Results: More documents with related terminology found!
```

## Quick Start

### Compile
```bash
mvn clean compile
```

### Run Example
```bash
java -cp target/classes unipd.se.PRFExample \
  code/data/collection_data.json \
  "vaccine effectiveness"
```

### Basic Usage
```java
import unipd.se.*;

// Setup
List<Paper> papers = DataLoader.loadPapers("code/data/collection_data.json");
Directory index = IndexerV1.buildIndex(papers);

Map<String, Float> fields = new HashMap<>();
fields.put("title", 2.0f);
fields.put("abstract", 1.0f);

// Perform PRF
PseudoRelevanceFeedback.PRFResult result =
   PseudoRelevanceFeedback.performPRF(
      index,
      "vaccine effectiveness",
      fields,
      5,      // analyze top 5
      10,     // extract 10 expansion terms
      100     // retrieve top 100
   );

// Results
System.out.println(result.topDocIds);      // Top 5 initial docs
System.out.println(result.expandedTerms);  // Extracted terms
System.out.println(result.expandedQuery);  // Full expanded query
```

## Key Features

- [x] **Automatic Query Expansion** - Extract relevant terms from top results
- [x] **Two-Pass Search** - Initial search + expanded search
- [x] **Consistent Tokenization** - Uses same analyzer as indexing
- [x] **Stopword Handling** - Automatically filters stop words
- [x] **Configurable Parameters** - Tune for recall vs precision
- [x] **Multiple APIs**
  - Single pass expansion with `performPRF()`
  - Complete two-pass search with `twoPassSearchWithPRF()`
- [x] **Well Documented** - Complete javadoc, examples, and guides

## Configuration

### Balanced (Default)
```java
topKRelevant = 5;    // Analyze top 5 results
topNTerms = 10;      // Extract top 10 terms
finalTopK = 100;     // Retrieve 100 final results
```

### High Recall (Find More)
```java
topKRelevant = 10;   // Analyze more documents
topNTerms = 20;      // Extract more terms
finalTopK = 1000;    // Retrieve more results
```

### High Precision (Better Quality)
```java
topKRelevant = 3;    // Analyze less documents
topNTerms = 5;       // Extract less terms
finalTopK = 100;     // Standard retrieval
```

## Expected Improvements

- **Recall**: +10-30% (more relevant documents found)
- **MAP**: Often improves especially at lower cutoffs
- **Precision**: Generally maintained or slightly decreased

**Best for:**
- Scientific/medical queries (many synonyms)
- Multi-word queries (related concepts)
- Technical domains with terminology variations

## Implementation Details

### Term Extraction
- Tokenizes title + abstract using `MyEnglishAnalyzer_V2`
- Applies KStem stemming for consistent term matching
- Filters stopwords using `stoplist_en_ranksnl_large.txt`
- Accumulates term frequencies across all top-K documents

### Term Selection
- Sorts by frequency (descending)
- Applies alphabetical tie-breaking for determinism
- Selects top-N most frequent terms

### Query Expansion
- Format: `"original_query term1 OR term2 OR term3 ..."`
- SimpleQueryParser interprets with SHOULD clauses
- Results ranked by term matching and BM25 relevance

## How to Use

### Option 1: Direct Integration
```java
// Get PRF results
Map<String, Object> prfResult = PseudoRelevanceFeedback.twoPassSearchWithPRF(
   index, query, fields, 5, 10, 100
);
List<String> results = (List<String>) prfResult.get("pass2_docs");
```

### Option 2: With SearcherV2
```java
// Create expanded query document
ExpandedQueryDoc expandedQuery = new ExpandedQueryDoc(
   index,
   originalQuery,
   prfResult.expandedQuery
);

// Use with existing SearcherV2
Map<String, List<String>> results = SearcherV2.search(
   index,
   Collections.singletonList(expandedQuery),
   2.0f,  // title boost
   100
);
```

---