# PRF Examples

This directory contains example programs demonstrating how to use Pseudo Relevance Feedback (PRF).

## Example Programs

### 1. PRFExample
Single query demonstration showing:
- How to use PRF with one query
- Term extraction and expansion
- Results saved to JSON

**Usage:**
```bash
java -cp ../../target/classes prf.PRFExample \
  ../../code/data/collection_data.json \
  "vaccine effectiveness"
```

### 2. PRFBatchExample
Batch processing example showing:
- Processing multiple queries with PRF
- Comparing BM25 vs PRF results
- Detailed comparison statistics

**Usage:**
```bash
java -cp ../../target/classes prf.PRFBatchExample \
  ../../code/data/collection_data.json \
  ../../code/data/expanded_queries_bge_large.json
```

### 3. PRFValidator
Testing and validation utilities:
- Parameter sensitivity testing
- Consistency verification (determinism checks)
- Edge case handling

**Usage:**
```bash
java -cp ../../target/classes prf.PRFValidator
```

## How to Run

1. **Navigate to project root**
   ```bash
   cd /path/to/seupd2526-retrix
   ```

2. **Compile all classes**
   ```bash
   mvn clean compile
   ```

3. **Run an example**
   ```bash
   java -cp target/classes prf.PRFExample code/data/collection_data.json "vaccine"
   ```

## Output

Examples produce:
- Console output showing PRF process
- JSON files in `results/` directory
- Detailed statistics and comparisons

## Configuration Parameters

All examples use configurable parameters:
- `topKRelevant` - Top-K docs to analyze (default: 5)
- `topNTerms` - Top-N expansion terms (default: 10)
- `finalTopK` - Final result set size (default: 100)

## Notes

- All examples use the same `MyEnglishAnalyzer_V2` as the main indexer
- Examples are deterministic - same query produces same results
- Examples automatically create `results/` directory if needed

## For More Information

See documentation files in `../docs/`:
- `PRF_README.md` - Complete guide
- `PRF_QUICKSTART.md` - Quick reference
- `PRF_DOCUMENTATION.md` - Technical details

**For configuration examples, run:**
```bash
java -cp ../../target/classes prf.PRFConfig
```
