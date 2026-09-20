# Retrix - Search Engines 2025/2026
This repository contains the Retrix system that obtained the 5th place at CLEF 2026 CheckThat!
Task 1, Source Retrieval for Scientific Web Claims. **This result has been achieved with just a pair of laptops and a RTX 3090, so it is not only easily achievable but also it beats more expensive setups of other competitors (some have multiple NVIDIA H100 GPUs).**

![Results of all the groups partecipating in the challenge](results/screenCheckThat.png).

To have access to our fine-tuned models send an email to me or visit the Huggin face repo https://huggingface.co/fabiuz332.

## Group Participants
- Baldan Fabio (correspondant author)
- Donati Davide 
- Garberino Alvise
- Padoan Giancarlo
- Tessari Marco

## Repository Structure

- `code/src/main/java`: Java evaluation code.
- `code/py`: Python scripts for translation, query expansion, bi-encoder retrieval, reranking, and format conversion.
- `code/data`: train, development, test, collection, and generated query/result data.
- `code/environment`: pinned Python requirement files for the two virtual environments used in the project.
- `finetuned_models`: fine-tuned reranker checkpoints and training scripts.
- `runs`: intermediate ranking JSON files.
- `results`: evaluation summaries and CLEF/CodaBench submission helpers.
- `homework-1`: first report source and PDF.
- `homework-2`: final CLEF paper source and PDF.
- `slides`: presentation material.
- `Analisi_Run.ods`: spreadsheet used to compare and annotate run results.

## Requirements

- Python 3.12.13 for the Python retrieval and reranking pipeline.
- JDK 25 for the Java evaluator.
- Apache Maven 3.9 or newer for compiling and running the Java evaluator.
- A CUDA-capable GPU is strongly recommended for model inference. The largest
  steps use Hugging Face Transformers, BGE-M3/FlagEmbedding, FAISS, and the
  Nemotron reranker.
- A Hugging Face token may be required to download gated or rate-limited models:

```bash
export HF_TOKEN="hf_your_token_here"
```

**Note that this README has the only purpose of showing the right order of use of our retrieval system, paths can vary.**

## Python Environments

Create the general Transformers environment:

```bash
python3.12 -m venv code/environment/venv312
source code/environment/venv312/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r code/environment/venv312/requirements_venv312.txt
deactivate
```

Create the FlagEmbedding/BGE-M3 environment:

```bash
python3.12 -m venv code/environment/venv312FlagEmb
source code/environment/venv312FlagEmb/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r code/environment/venv312FlagEmb/requirements_venv312FlagEmb.txt
deactivate
```

Run all commands below from the repository root.

## Reproducibility Pipeline

### 1. Translate French and German Test Queries

Use the general Transformers environment:

```bash
source code/environment/venv312/bin/activate

python3 code/py/translate_queries.py \
  --input code/data/Test_set/final_fr_test.json \
  --lang fr \
  --output code/data/Test_set/final_fr_TRADOTTOen_test.json

python3 code/py/translate_queries.py \
  --input code/data/Test_set/final_de_test.json \
  --lang de \
  --output code/data/Test_set/final_de_TRADOTTOen_test.json
```

Expected outputs:

- `code/data/Test_set/final_fr_TRADOTTOen_test.json`
- `code/data/Test_set/final_de_TRADOTTOen_test.json`

### 2. Query Expansion

The final expanded test-query files used by the retrieval pipeline are:

- `code/data/Test_set/expanded_queries_bge_large_enFINAL.json`
- `code/data/Test_set/expanded_queries_bge_large_frFINAL_en.json`
- `code/data/Test_set/expanded_queries_bge_large_deFINAL_en.json`

The expansion script is:

```bash
python3 code/py/QueryExpansorBGE-large.py
```

Before rerunning it for a different split, verify the input and output paths in
`code/py/QueryExpansorBGE-large.py`, because this script currently stores those
paths as constants rather than command-line arguments.

### 3. First-stage BGE-M3 Hybrid Retrieval

Use the FlagEmbedding environment:

```bash
source code/environment/venv312FlagEmb/bin/activate

python3 code/py/Bi_encoder_m3_hybrid.py \
  --queries code/data/Test_set/expanded_queries_bge_large_enFINAL.json \
  --corpus code/data/collection_data.json \
  --output runs/res_bge_m3_hybrid_enFINAL_topk3000.json \
  --model BAAI/bge-m3 \
  --retrieval-mode hybrid \
  --top-k 3000 \
  --precompute-corpus-hybrid

python3 code/py/Bi_encoder_m3_hybrid.py \
  --queries code/data/Test_set/expanded_queries_bge_large_frFINAL_en.json \
  --corpus code/data/collection_data.json \
  --output runs/res_bge_m3_hybrid_frFINAL_topk3000.json \
  --model BAAI/bge-m3 \
  --retrieval-mode hybrid \
  --top-k 3000 \
  --precompute-corpus-hybrid

python3 code/py/Bi_encoder_m3_hybrid.py \
  --queries code/data/Test_set/expanded_queries_bge_large_deFINAL_en.json \
  --corpus code/data/collection_data.json \
  --output runs/res_bge_m3_hybrid_deFINAL_topk3000.json \
  --model BAAI/bge-m3 \
  --retrieval-mode hybrid \
  --top-k 3000 \
  --precompute-corpus-hybrid
```

Expected outputs:

- `runs/res_bge_m3_hybrid_enFINAL_topk3000.json`
- `runs/res_bge_m3_hybrid_frFINAL_topk3000.json`
- `runs/res_bge_m3_hybrid_deFINAL_topk3000.json`

### 4. Nemotron Reranking

Use the general Transformers environment:

```bash
deactivate 2>/dev/null || true
source code/environment/venv312/bin/activate

python3 code/py/evaluate_nemotronLora.py \
  --model_dir finetuned_models/fineTune_nemotron/nemotronFT_Train2026-All2025 \
  --base_model nvidia/llama-nemotron-rerank-1b-v2 \
  --topics code/data/Test_set/final_en_test.json \
  --corpus code/data/collection_data.json \
  --bm25_results runs/res_bge_m3_hybrid_enFINAL_topk3000.json \
  --output runs/reranked_results_nemotron_enFINAL.json \
  --top_k 2000 \
  --rerank_top 100

python3 code/py/evaluate_nemotronLora.py \
  --model_dir finetuned_models/fineTune_nemotron/nemotronFT_Train2026-All2025 \
  --base_model nvidia/llama-nemotron-rerank-1b-v2 \
  --topics code/data/Test_set/final_fr_TRADOTTOen_test.json \
  --corpus code/data/collection_data.json \
  --bm25_results runs/res_bge_m3_hybrid_frFINAL_topk3000.json \
  --output runs/reranked_results_nemotron_frFINAL.json \
  --top_k 2000 \
  --rerank_top 100

python3 code/py/evaluate_nemotronLora.py \
  --model_dir finetuned_models/fineTune_nemotron/nemotronFT_Train2026-All2025 \
  --base_model nvidia/llama-nemotron-rerank-1b-v2 \
  --topics code/data/Test_set/final_de_TRADOTTOen_test.json \
  --corpus code/data/collection_data.json \
  --bm25_results runs/res_bge_m3_hybrid_deFINAL_topk3000.json \
  --output runs/reranked_results_nemotron_deFINAL.json \
  --top_k 2000 \
  --rerank_top 100
```

Expected outputs:

- `runs/reranked_results_nemotron_enFINAL.json`
- `runs/reranked_results_nemotron_frFINAL.json`
- `runs/reranked_results_nemotron_deFINAL.json`

### 5. Java Evaluation on Development Runs

The Java evaluator is intended for development data that includes gold `pubkey`
fields. Build and run it with Maven:

```bash
mvn clean package

mvn exec:java \
  -Dexec.args="_ code/data/Dev_set/ENexpanded_queries_bge_largeDEV.json code/data/Dev_set/reranked_results_nemotronFTAarsen20252026_Lora_enDEVtopk1000_onbiEncoderTopk1000.json"
```

The evaluator writes metrics to `results/evaluation_results_reranked*.json`.

### 6. Final CLEF/CodaBench Outputs

Convert each final reranked JSON file to a two-column TSV with header:

```text
index	preds
```

Expected final files:

- `results/clef_submission/predictions_en.tsv`
- `results/clef_submission/predictions_fr.tsv`
- `results/clef_submission/predictions_de.tsv`

Expected data-row counts, excluding the header:

- English: 6076 rows
- French: 1220 rows
- German: 876 rows

After verifying the row counts, package the prediction files:

```bash
cd results/clef_submission
zip predictions.zip predictions_en.tsv predictions_fr.tsv predictions_de.tsv
```

## License

All contents of this repository are shared using the
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

![CC logo](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)
