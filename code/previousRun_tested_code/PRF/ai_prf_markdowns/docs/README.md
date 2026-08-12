# PRF (Pseudo Relevance Feedback) - Complete Implementation

## Using PRF

### Simple One-Liner
```java
// Single PRF pass with default settings
PseudoRelevanceFeedback.PRFResult result = 
   PseudoRelevanceFeedback.performPRF(
      index, "vaccine", fields, 5, 10, 100
   );
```

### Two-Pass Search
```java
// Complete search with expansion and re-ranking
Map<String, Object> results = 
   PseudoRelevanceFeedback.twoPassSearchWithPRF(
      index, "vaccine", fields, 5, 10, 100
   );

List<String> finalDocs = (List<String>) results.get("pass2_docs");
List<String> expandedTerms = (List<String>) results.get("expanded_terms");
```

### Using Configuration Presets
```java
// High recall - find more documents
PRFConfig.Config config = PRFConfig.HIGH_RECALL;

Map<String, Object> results = 
   PseudoRelevanceFeedback.twoPassSearchWithPRF(
      index, query, fields,
      config.topKRelevant,
      config.topNTerms,
      config.finalTopK
   );
```

## Configuration Options

### Quick Presets
```java
PRFConfig.BALANCED              // Default: k=5, n=10, topk=100
PRFConfig.HIGH_RECALL           // k=10, n=20, topk=1000
PRFConfig.HIGH_PRECISION        // k=3, n=5, topk=100
PRFConfig.SCIENTIFIC_MEDICAL    // Optimized for medical queries
PRFConfig.SHORT_QUERY           // For 1-2 word queries
PRFConfig.LONG_QUERY            // For longer multi-word queries
PRFConfig.FAST                  // Speed optimized
```

### Custom Configuration
```java
PRFConfig.Config myConfig = PRFConfig.custom(
   "MY_CONFIG",
   "My custom configuration",
   8,    // topKRelevant: analyze top 8
   15,   // topNTerms: extract 15 terms
   200   // finalTopK: retrieve 200
);
```

## Parameters Explained

| Parameter | Range | Impact | Default |
|-----------|-------|--------|---------|
| `topKRelevant` | 3-20 | How many initial results to analyze | 5 |
| `topNTerms` | 5-30 | How many expansion terms to extract | 10 |
| `finalTopK` | 50-2000 | Final result set size | 100 |

**Tuning Guide:**
- ↑ `topKRelevant` + ↑ `topNTerms` = Higher Recall
- ↓ `topKRelevant` + ↓ `topNTerms` = Higher Precision

## Examples

### Example 1: Single Query with Default Settings
```java
import unipd.se.*;

List<Paper> papers = DataLoader.loadPapers("code/data/collection_data.json");
Directory index = IndexerV1.buildIndex(papers);

Map<String, Float> fields = new HashMap<>();
fields.put("title", 2.0f);
fields.put("abstract", 1.0f);

// Run PRF
PseudoRelevanceFeedback.PRFResult result = 
   PseudoRelevanceFeedback.performPRF(
      index, "vaccine", fields, 5, 10, 100
   );

System.out.println("Top docs: " + result.topDocIds);
System.out.println("Expansion: " + result.expandedTerms);
```

### Example 2: Batch Processing with Comparison
```bash
java -cp target/classes unipd.se.PRFBatchExample \
  code/data/collection_data.json \
  code/data/expanded_queries_bge_large.json
```

### Example 3: Parameter Sensitivity Testing
```bash
java -cp target/classes unipd.se.PRFValidator
```

**Best performance with:**
- Medical/scientific queries
- Queries with technical synonyms
- Multi-word queries
- Collections with terminology variations

## Integration with the Pipeline

### Option 1: Standalone
```java
Map<String, Object> prfResults = 
   PseudoRelevanceFeedback.twoPassSearchWithPRF(
      index, query, fields, 5, 10, 100
   );
```

### Option 2: With SearcherV2
```java
// Create expanded query
ExpandedQueryDoc expandedQuery = new ExpandedQueryDoc(
   index,
   originalQuery,
   prfResult.expandedQuery
);

// Use existing SearcherV2
SearcherV2.search(index, 
   Collections.singletonList(expandedQuery), 
   2.0f, 100);
```

## Key Implementation Details

- **Tokenization**: Uses same `MyEnglishAnalyzer_V2` as indexing
- **Stemming**: KStem applied consistently
- **Stopword Removal**: Custom stoplist from `stoplist_en_ranksnl_large.txt`
- **Term Frequency**: Simple accumulation across top-K documents
- **Deterministic**: Alphabetical tie-breaking for reproducibility

## Common Issues & Solutions

**Issue**: No expansion terms extracted
- → Increase `topKRelevant` to analyze more documents
- → Check if query has relevant results

**Issue**: Results are worse with PRF
- → Reduce `topNTerms` to extract fewer, better terms
- → Check if initial query quality is high
- → Consider using `PRFConfig.HIGH_PRECISION` preset (unica cosa che non ho provato, potrebbe aiutare ma non è la svolta)

**Issue**: Too slow
- → Use `PRFConfig.FAST` preset (fatto apposta zioca si chiama FAST per un motivo)
- → (eventualmente) Reduce `topKRelevant` and `topNTerms`

---