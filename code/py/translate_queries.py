#!/usr/bin/env python3
"""
Translate multilingual CLEF queries from French or German into English.

The script uses EuroLLM through the Hugging Face text-generation pipeline so the
model's chat template is applied consistently. Generated text is post-processed
to remove prompt echoes and common assistant-label artifacts before it is saved
as a JSON query file.

Model license: Apache 2.0.

Usage:
  python translate_queries.py --input fr_train.json --lang fr --output fr_train_en.json
  python translate_queries.py --input de_train.json --lang de --output de_train_en.json

Options:
  --input          Source query JSON path
  --lang           Source language: 'fr' or 'de'
  --output         Output JSON path with English translations
  --model          Hugging Face model ID
  --batch_size     Queries per generation batch
  --max_new_tokens Maximum generated tokens per translation
  --cache_dir      Hugging Face cache directory
  --keep_original  Preserve the source text in `text_original`
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL = "utter-project/EuroLLM-9B-Instruct-2512"

LANG_NAMES = {
    "fr": "French",
    "de": "German",
}

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Translate the user's text to English. "
    "Output ONLY the English translation. "
    "Do not include the original text, explanations, labels, or any other content."
)

# Cleanup patterns are applied in order to the raw model output. They cover
# common EuroLLM artifacts observed during batched chat-template generation:
#   - "English: \n assistant\n..."
#   - "assistant\n..."
#   - "English:\n..."
#   - repeated prompt text before the translation
CLEANUP_PATTERNS = [
    # Drop any echoed conversation prefix up to the assistant marker.
    (r"(?s)^.*?\bassistant\b\s*", ""),
    # Drop language and translation labels that should not reach Lucene.
    (r"(?i)^(english|en|translation|traduction|übersetzung)\s*:\s*", ""),
    # Drop the instruction phrase when the model repeats the prompt verbatim.
    (r"(?s)^.*?nothing else\.\s*", ""),
    # Source-language line filters were intentionally left out because they were
    # too aggressive for short multilingual queries.
]


def clean_translation(raw: str) -> str:
    """Normalize a generated translation by removing known prompt artifacts.

    Args:
        raw: Raw assistant output returned by the generation pipeline.

    Returns:
        The cleaned English translation with empty lines removed.
    """
    text = raw.strip()
    for pattern, replacement in CLEANUP_PATTERNS:
        text = re.sub(pattern, replacement, text)
        text = text.strip()
    # Empty generated lines can otherwise become whitespace-only Lucene terms.
    text = "\n".join(line for line in text.splitlines() if line.strip())
    return text.strip()


def build_messages(text: str, src_lang: str) -> list[dict]:
    """Build the chat-template messages sent to EuroLLM.

    Args:
        text: Source-language query text.
        src_lang: Two-letter source language code present in `LANG_NAMES`.

    Returns:
        A system/user message list compatible with the Hugging Face pipeline.

    Raises:
        KeyError: If `src_lang` is not configured in `LANG_NAMES`.
    """
    lang_name = LANG_NAMES[src_lang]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Translate this {lang_name} text to English. "
                f"Output only the English translation, nothing else.\n\n"
                f"{text}"
            ),
        },
    ]


def load_pipeline(model_name: str, cache_dir: str | None, device: str):
    """Load the EuroLLM generation pipeline and configure tokenizer padding.

    Args:
        model_name: Hugging Face model identifier or local model path.
        cache_dir: Optional Hugging Face cache directory.
        device: Requested execution device; actual placement is delegated to
            `device_map="auto"` for large-model loading.

    Returns:
        A configured text-generation pipeline.
    """
    log.info(f"Loading pipeline: {model_name}")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "cache_dir": cache_dir,
        },
        device_map="auto",
    )
    # Decoder-only models require left padding for stable batched generation.
    pipe.tokenizer.padding_side = "left"
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {torch.cuda.get_device_name(0)} ({mem:.0f}GB VRAM)")
    log.info("Pipeline loaded | bfloat16 | device_map=auto")
    return pipe


def translate_all(
        queries: list[dict],
        src_lang: str,
        pipe,
        batch_size: int,
        max_new_tokens: int,
        keep_original: bool,
) -> list[dict]:
    """Translate every query record and preserve the input JSON schema.

    Args:
        queries: Query dictionaries containing at least `index` and `text`.
        src_lang: Two-letter source language code supported by `LANG_NAMES`.
        pipe: Hugging Face text-generation pipeline.
        batch_size: Number of queries sent to the pipeline per batch.
        max_new_tokens: Maximum generated tokens per query.
        keep_original: Whether to add `text_original` to each output record.

    Returns:
        Query dictionaries with translated `text` values and optional metadata.

    Raises:
        KeyError: If an input query lacks required keys or `src_lang` is invalid.
    """
    total = len(queries)
    log.info(
        f"Beginning translation: {total} query | batch_size={batch_size} | "
        f"{LANG_NAMES[src_lang]} -> English"
    )

    # Build messages up front so batching does not interleave prompt creation
    # with model execution.
    all_messages = [build_messages(q["text"], src_lang) for q in queries]

    translated_texts = []
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch_messages = all_messages[i : i + batch_size]

        outputs = pipe(
            batch_messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=pipe.tokenizer.pad_token_id,
            eos_token_id=pipe.tokenizer.eos_token_id,
            batch_size=batch_size,
        )

        for out in outputs:
            # The pipeline returns the chat history; the last message is the
            # assistant answer when chat-template output is structured.
            raw = out[0]["generated_text"]
            if isinstance(raw, list):
                assistant_content = raw[-1]["content"]
            else:
                # String output is cleaned by the regex layer above.
                assistant_content = raw

            cleaned = clean_translation(assistant_content)
            translated_texts.append(cleaned)

        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        speed = done / elapsed
        eta = (total - done) / speed if speed > 0 else 0
        log.info(f"  {done}/{total} | {speed:.1f} q/s | ETA {eta:.0f}s")

    elapsed_total = time.time() - t0
    log.info(
        f"Completed: {total} queries in {elapsed_total:.1f}s "
        f"({total / elapsed_total:.1f} q/s)"
    )

    output = []
    for q, translated in zip(queries, translated_texts):
        entry = {"index": q["index"]}
        if keep_original:
            entry["text_original"] = q["text"]
        entry["text"] = translated
        if "pubkey" in q:
            entry["pubkey"] = q["pubkey"]
        output.append(entry)

    return output


def main():
    """Parse CLI arguments, translate the input queries, and write JSON output.

    Side effects:
        Loads a large Hugging Face model, reads the input JSON file, creates the
        output directory if needed, writes the translated JSON file, and logs a
        small qualitative sample.
    """
    parser = argparse.ArgumentParser(
        description="Translates query FR/DE -> EN with EuroLLM-9B-Instruct (Apache 2.0)"
    )
    parser.add_argument("--input",    required=True,  help="Source path of the queries in JSON format")
    parser.add_argument("--lang",     required=True,  choices=["fr", "de"], help="Source language")
    parser.add_argument("--output",   required=True,  help="Output path of the queries in JSON format (EN)")
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Queries per batch (default: 8 — safe for 9B bfloat16 on 24GB VRAM)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Max generated tokens per translation (default: 256)",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--keep_original", action="store_true",
        help="Adds 'text_original' for quality check",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        log.warning("CUDA not available — CPU inference will be very slow")

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"File not found: {input_path}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        queries = json.load(f)
    log.info(f"Loaded {len(queries)} queries from {input_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(args.model, args.cache_dir, device)

    translated = translate_all(
        queries, args.lang, pipe,
        args.batch_size, args.max_new_tokens, args.keep_original,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    log.info(f"Output saved: {output_path}")

    # Log a small sample for manual quality checks without changing output data.
    log.info("--- Translation sample (first 5) ---")
    for i in range(min(5, len(queries))):
        log.info(f"[idx={queries[i]['index']}]")
        if args.keep_original:
            log.info(f"  SRC: {queries[i]['text'][:120]}")
        log.info(f"  TGT: {translated[i]['text'][:120]}")


if __name__ == "__main__":
    main()
