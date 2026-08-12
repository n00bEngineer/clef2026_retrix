# PRF Organization Complete

All Pseudo Relevance Feedback (PRF) files have been organized into a dedicated directory structure for better maintainability and navigation.
The goal:
> After the first search, we take the top 3-5 results and we extract their most frequent non-stopword terms, and add them to the query for a second search pass.

## Directory Structure

```
prf/
├── README.md                           Main PRF directory guide
├── src/unipd/se/                       Core source files
│   ├── PseudoRelevanceFeedback.java
│   ├── PRFConfig.java
│   └── PRFIntegration.java
├── examples/                           Example programs
│   ├── PRFExample.java
│   ├── PRFBatchExample.java
│   ├── PRFValidator.java
│   ├── PRFConfig.java (copy)
│   └── README.md
└── docs/                               Complete documentation
    ├── README.md
    ├── PRF_README.md
    ├── PRF_QUICKSTART.md
    ├── PRF_DOCUMENTATION.md
    ├── PRF_IMPLEMENTATION_SUMMARY.md
    ├── PRF_IMPLEMENTATION_COMPLETE.md
    ├── PRF_CHECKLIST.md
    └── FILE_REFERENCE.md
```

## Quick Navigation

| What | Where |
|------|-------|
| **Overview** | `prf/README.md` |
| **Main Guide** | `prf/docs/PRF_README.md` |
| **Quick Ref** | `prf/docs/PRF_QUICKSTART.md` |
| **Examples** | `prf/examples/` |
| **Core Code** | `prf/src/unipd/se/` |
| **All Docs** | `prf/docs/` |

## Getting Started

### Run an Example
```bash
# From project root
mvn clean compile
java -cp target/classes unipd.se.PRFExample \
  code/data/collection_data.json "vaccine"
```

### Check Results
```bash
cat results/prf_demo_results.json
```

## Documentation Files

All 7 documentation files are in `prf/docs/`:

1. **PRF_README.md** - Complete guide
2. **PRF_QUICKSTART.md** - Quick reference
3. **PRF_DOCUMENTATION.md** - Technical details
4. **PRF_IMPLEMENTATION_SUMMARY.md** - Overview
5. **PRF_CHECKLIST.md** - Verification
6. **FILE_REFERENCE.md** - File organization
7. **README.md** - Documentation index

## Core Components

### prf/src/unipd/se/ (3 classes)
- **PseudoRelevanceFeedback.java** - Core algorithm
- **PRFConfig.java** - 9 configuration presets
- **PRFIntegration.java** - 4 integration patterns

### prf/examples/ (3 examples + guide)
- **PRFExample.java** - Single query demo
- **PRFBatchExample.java** - Batch processing
- **PRFValidator.java** - Testing utilities
- **README.md** - How to run examples

### prf/docs/ (7 comprehensive guides)
- Complete API reference
- Usage examples
- Integration patterns
- Troubleshooting
- Architecture overview

## File Locations

**To compile:**
```bash
mvn clean compile
# Automatically compiles prf/src/ files
```

**To run examples:**
```bash
java -cp target/classes unipd.se.PRFExample ...
# Uses compiled classes from prf/src/
```

**To read docs:**
```bash
cat prf/docs/README.md
# All documentation in prf/docs/
```

**(eventualmente) To modify code:**
```bash
# Edit files in:
# - prf/src/unipd/se/PseudoRelevanceFeedback.java
# - prf/src/unipd/se/PRFConfig.java
# - prf/src/unipd/se/PRFIntegration.java
```

---