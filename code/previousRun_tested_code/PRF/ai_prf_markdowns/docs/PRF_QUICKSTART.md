# Pseudo-Relevance Feedback (PRF) Quick Start

## Quick Example

```java
import unipd.se.*;
import org.apache.lucene.store.Directory;
import java.util.HashMap;
import java.util.Map;

// Setup
List<Paper> papers = DataLoader.loadPapers("code/data/collection_data.json");
Directory index = IndexerV1.buildIndex(papers);

Map<String, Float> fields = new HashMap<>();
fields.put("title", 2.0f);
fields.put("abstract", 1.0f);

// Run PRF
PseudoRelevanceFeedback.PRFResult result =
    PseudoRelevanceFeedback.performPRF(
        index,
        "vaccine effectiveness",
        fields,
        5,      // top 5 results to analyze
        10,     // extract 10 expansion terms
        100     // retrieve top 100
    );

// Results
System.out.println("Expansion terms: " + result.expandedTerms);
System.out.println("Expanded query: " + result.expandedQuery);
```

## Running Examples

### Single Query Demo
```bash
mvn clean compile
java -cp target/classes unipd.se.PRFExample \
  code/data/collection_data.json \
  "vaccine effectiveness"
```

### Batch Processing
```bash
java -cp target/classes unipd.se.PRFBatchExample \
  code/data/collection_data.json \
  code/data/expanded_queries_bge_large.json
```

## Key Methods

### 1. Single PRF Pass
```java
PseudoRelevanceFeedback.PRFResult performPRF(
    Directory dir,
    String originalQuery,
    Map<String, Float> fields,
    int topKRelevant,    // documents to analyze (3-5)
    int topNTerms,       // expansion terms to extract (10-20)
    int searchTopK       // initial retrieval size (100-1000)
)
```

### 2. Complete Two-Pass Search
```java
Map<String, Object> twoPassSearchWithPRF(
    Directory dir,
    String originalQuery,
    Map<String, Float> fields,
    int topKRelevant,
    int topNTerms,
    int finalTopK         // final result size
)
```

## Configuration Examples

### Balanced (Default)
```java
topKRelevant = 5;   // Analyze top 5
topNTerms = 10;     // Extract 10 terms
finalTopK = 100;    // Retrieve top 100
```

### High Recall
```java
topKRelevant = 10;  // Analyze more documents
topNTerms = 20;     // Extract more terms
finalTopK = 1000;   // Retrieve more results
```

### High Precision
```java
topKRelevant = 3;   // Analyze fewer documents
topNTerms = 5;      // Extract fewer terms
finalTopK = 100;    // Standard retrieval
```

## How It Works

```
1. Initial Search
   Query: "vaccine effectiveness"
   ↓ Retrieve top 5 documents

2. Term Extraction
   Analyze: titles + abstracts of top 5
   Count: term frequencies
   Remove: stopwords (handled by MyEnglishAnalyzer_V2)
   Extract top 10 terms: immunization, safety, efficacy, trials, ...

3. Query Expansion
   Build: "vaccine effectiveness OR immunization OR safety OR efficacy OR trials OR ..."

4. Second Search
   Execute: expanded query
   ↓ Retrieve: top 100 final results
```

## Analyzer Integration

PRF uses the same `MyEnglishAnalyzer_V2` as indexing to ensure:
- **Consistent tokenization** - Same word splitting
- **Stemming** - KStem applied consistently
- **Stopword removal** - Custom stoplist applied

This ensures extracted terms match what was indexed.

## Output Files

When running examples, results are saved to:

```
results/
├── prf_demo_results.json          (from PRFExample)
│   ├── original_query
│   ├── expanded_query
│   ├── expanded_terms
│   ├── pass1_results
│   └── pass2_results
│
└── prf_batch_comparison.json      (from PRFBatchExample)
    ├── parameters
    └── query_comparisons[]
        ├── query_text
        ├── expanded_terms
        ├── bm25_results_count
        ├── prf_pass2_results
        └── comparison_metrics
```

---