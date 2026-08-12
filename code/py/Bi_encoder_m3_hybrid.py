"""Run dense or BGE-M3 hybrid retrieval over the Retrix corpus.

This script loads query and corpus JSON files, encodes text with either
SentenceTransformers or FlagEmbedding's BGE-M3 implementation, builds a FAISS
inner-product index over corpus dense vectors, and writes `{qid: [pubkey, ...]}`
rankings for the Java evaluator. Hybrid mode combines dense, sparse lexical,
and ColBERT-style multi-vector scores for BGE-M3 candidates.

The implementation is designed for large local runs: it detects CPU,
single-GPU, and multi-GPU environments, adapts batch size after CUDA OOM errors,
and can optionally precompute BGE-M3 sparse/ColBERT corpus features when enough
system memory is available.
"""

import argparse
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import faiss
import numpy as np
import torch
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 2000
DEFAULT_BATCH_SIZE = 1024
DEFAULT_BGE_M3_MAX_LENGTH = 1024
DEFAULT_RETRIEVAL_MODE = "hybrid"
DEFAULT_CANDIDATE_MULTIPLIER = 3
DEFAULT_MULTIVECTOR_MAX_LENGTH = 384
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_SPARSE_WEIGHT = 0.15
DEFAULT_COLBERT_WEIGHT = 0.35
DEFAULT_COLBERT_SCORE_BATCH_SIZE = 512
DEFAULT_CORPUS_COLBERT_DTYPE = "float16"
DEFAULT_LOG_EVERY_QUERIES = 100
DEFAULT_CPU_THREADS = 6

SCRIPT_PATH = Path(__file__).resolve()
# These relative parents match the repository layout used by the coursework.
CODE_ROOT = SCRIPT_PATH.parents[1] 
REPO_ROOT = SCRIPT_PATH.parents[2]

DEFAULT_QUERIES = CODE_ROOT / "data" / "expanded_queries_bge_large_frDEV_en.json"
DEFAULT_CORPUS = CODE_ROOT / "data" / "collection_data.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "bi_encoder_results_bge_large_frDEVen_topk1000.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for retrieval and resource configuration."""
    parser = argparse.ArgumentParser(
        description="Encode queries and corpus with a bi-encoder and run dense retrieval."
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--retrieval-mode",
        choices=("auto", "dense", "hybrid", "multivector"),
        default=DEFAULT_RETRIEVAL_MODE,
        help=(
            "Retrieval strategy. "
            '"auto" uses "hybrid" for BAAI/bge-m3 and "dense" otherwise. '
            '"multivector" is accepted as alias of "hybrid".'
        ),
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=DEFAULT_CANDIDATE_MULTIPLIER,
        help=(
            "For hybrid mode: retrieve candidate_multiplier * top_k with dense FAISS "
            "before combining dense+sparse+colbert scores."
        ),
    )
    parser.add_argument(
        "--multivector-max-length",
        "--hybrid-max-length",
        dest="multivector_max_length",
        type=int,
        default=DEFAULT_MULTIVECTOR_MAX_LENGTH,
        help="Max token length used for sparse+colbert scoring in hybrid mode.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
        help="Dense score weight in hybrid mode.",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=DEFAULT_SPARSE_WEIGHT,
        help="Sparse lexical score weight in hybrid mode.",
    )
    parser.add_argument(
        "--colbert-weight",
        type=float,
        default=DEFAULT_COLBERT_WEIGHT,
        help="Multi-vector ColBERT score weight in hybrid mode.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Optional maximum sequence length for the encoder. "
            "If not provided and model is BAAI/bge-m3, defaults to 1024."
        ),
    )
    parser.add_argument("--query-field", default="auto")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--colbert-score-batch-size",
        type=int,
        default=DEFAULT_COLBERT_SCORE_BATCH_SIZE,
        help=(
            "How many candidate documents to score together during ColBERT reranking. "
            "Higher values increase GPU utilization but also VRAM use."
        ),
    )
    parser.add_argument(
        "--precompute-corpus-hybrid",
        action="store_true",
        help=(
            "Precompute sparse and ColBERT corpus features once and reuse them for all queries. "
            "This is much faster in hybrid mode but requires substantially more system RAM."
        ),
    )
    parser.add_argument(
        "--corpus-colbert-dtype",
        choices=("float16", "float32"),
        default=DEFAULT_CORPUS_COLBERT_DTYPE,
        help=(
            "Storage dtype for precomputed corpus ColBERT vectors. "
            "`float16` uses less RAM and is usually faster; `float32` preserves full precision."
        ),
    )
    parser.add_argument(
        "--log-every-queries",
        type=int,
        default=DEFAULT_LOG_EVERY_QUERIES,
        help=(
            "How often to print hybrid retrieval progress. "
            "For example, 100 logs every 100 queries."
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help=(
            "Number of CPU threads to use for FAISS CPU and PyTorch CPU backends. "
            "Leave unset to keep library defaults."
        ),
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional debug limit. If set, only the first N queries are processed.",
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help="Optional prefix added to every query before encoding.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file.

    Args:
        path: JSON file path.

    Returns:
        The decoded JSON object.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the content is not valid JSON.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_installed_version(package_name: str) -> str | None:
    """Return the installed package version, or `None` when absent."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def patch_transformers_flash_attn_compat() -> None:
    """Patch a missing transformers helper expected by some FlagEmbedding builds.

    Side effects:
        Adds `is_flash_attn_greater_or_equal_2_10` to `transformers.utils` when
        an older compatible helper is present. No patch is applied if
        transformers is unavailable or already exposes the helper.
    """
    try:
        import transformers.utils as transformers_utils
    except ImportError:
        return

    if hasattr(transformers_utils, "is_flash_attn_greater_or_equal_2_10"):
        return

    legacy_checker = getattr(transformers_utils, "is_flash_attn_greater_or_equal", None)
    if legacy_checker is None:
        return

    def _is_flash_attn_greater_or_equal_2_10() -> bool:
        """Compatibility shim for FlagEmbedding imports."""
        try:
            return bool(legacy_checker("2.1.0"))
        except Exception:
            return False

    transformers_utils.is_flash_attn_greater_or_equal_2_10 = _is_flash_attn_greater_or_equal_2_10


def resolve_retrieval_mode(model_name: str, requested_mode: str) -> str:
    """Resolve CLI retrieval mode aliases and model-specific defaults."""
    if requested_mode == "multivector":
        return "hybrid"
    if requested_mode != "auto":
        return requested_mode
    if model_name.lower().startswith("baai/bge-m3"):
        return "hybrid"
    return "dense"


def resolve_execution_mode() -> tuple[str, list[str]]:
    """Select CPU, single-GPU, or multi-GPU execution from visible CUDA devices.

    Returns:
        A tuple of execution mode and target device strings. Multi-GPU mode
        returns one `cuda:N` entry per visible GPU; CPU and single-GPU modes
        return an empty device list because downstream libraries use defaults.
    """
    if not torch.cuda.is_available():
        return "cpu", []

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return "cuda", []

    devices = [f"cuda:{gpu_idx}" for gpu_idx in range(gpu_count)]
    return "multi-gpu", devices


def build_bge_m3_model(model_name: str, execution_mode: str, target_devices: list[str]) -> Any:
    """Instantiate a BGE-M3 FlagEmbedding model for the current hardware.

    Args:
        model_name: Hugging Face model ID or local path.
        execution_mode: One of `cpu`, `cuda`, or `multi-gpu`.
        target_devices: Device IDs used only for multi-GPU initialization.

    Returns:
        A `BGEM3FlagModel` instance.

    Raises:
        ImportError: If FlagEmbedding is missing or incompatible.
        TypeError: If none of the supported initialization keyword variants
            match the installed FlagEmbedding version.
    """
    patch_transformers_flash_attn_compat()
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as error:
        flagembedding_version = get_installed_version("FlagEmbedding")
        transformers_version = get_installed_version("transformers")
        if flagembedding_version is not None:
            version_bits = [f"FlagEmbedding=={flagembedding_version}"]
            if transformers_version is not None:
                version_bits.append(f"transformers=={transformers_version}")
            version_summary = ", ".join(version_bits)
            hint = (
                "FlagEmbedding is installed but could not be imported. "
                "This is usually caused by an incompatible `transformers` version. "
                f"Detected {version_summary}. "
                "Install compatible versions in the same virtual environment, for example: "
                "python -m pip install -U 'transformers>=4.44.2,<6' FlagEmbedding"
            )
            if "is_flash_attn_greater_or_equal_2_10" in str(error):
                hint += (
                    " The missing `is_flash_attn_greater_or_equal_2_10` symbol means "
                    "your installed `transformers` is older than what this FlagEmbedding "
                    "build expects."
                )
            raise ImportError(hint) from error
        raise ImportError(
            "FlagEmbedding is required for bge-m3 retrieval. "
            "Install it with: pip install -U FlagEmbedding"
        ) from error

    base_kwargs: dict[str, Any] = {"use_fp16": execution_mode != "cpu"}
    init_attempts: list[tuple[dict[str, Any], str | None]] = []

    if execution_mode == "cpu":
        init_attempts.append(({**base_kwargs, "devices": "cpu"}, None))
        init_attempts.append(({**base_kwargs, "device": "cpu"}, None))
    elif execution_mode == "cuda":
        init_attempts.append(({**base_kwargs, "devices": "cuda:0"}, None))
        init_attempts.append(({**base_kwargs, "device": "cuda:0"}, None))
    else:
        fallback_device = target_devices[0] if target_devices else "cuda:0"
        init_attempts.append(({**base_kwargs, "devices": target_devices}, None))
        init_attempts.append(
            (
                {**base_kwargs, "devices": fallback_device},
                "Installed FlagEmbedding does not support multi-device inference; "
                f"falling back to single device {fallback_device}.",
            )
        )
        init_attempts.append(
            (
                {**base_kwargs, "device": fallback_device},
                "Installed FlagEmbedding does not support multi-device inference; "
                f"falling back to single device {fallback_device}.",
            )
        )

    last_type_error: TypeError | None = None
    for init_kwargs, message in init_attempts:
        try:
            model = BGEM3FlagModel(model_name, **init_kwargs)
            if message is not None:
                print(message)
            return model
        except TypeError as error:
            last_type_error = error

    if last_type_error is not None:
        raise last_type_error

    return BGEM3FlagModel(model_name, **base_kwargs)


def infer_query_prefix(model_name: str, provided_prefix: str | None) -> str:
    """Infer the query encoding prefix for BGE-family dense models.

    Args:
        model_name: Model ID used for encoding.
        provided_prefix: Explicit CLI prefix, if any.

    Returns:
        The prefix that should be prepended to each query before dense encoding.
    """
    if provided_prefix is not None:
        return provided_prefix
    if model_name.lower().startswith("baai/bge-m3"):
        return ""
    if model_name.lower().startswith("baai/bge"):
        return "Represent this sentence for searching relevant passages: "
    return ""


def resolve_effective_max_length(model_name: str, provided_max_length: int | None) -> int | None:
    """Resolve the encoder max length from CLI input and model defaults.

    Args:
        model_name: Model ID used for encoding.
        provided_max_length: Optional CLI max-length value.

    Returns:
        A positive max length, or `None` to keep the model/library default.

    Raises:
        ValueError: If an explicit max length is not positive.
    """
    if provided_max_length is not None:
        if provided_max_length <= 0:
            raise ValueError("--max-length must be greater than 0 when provided.")
        return provided_max_length

    if model_name.lower().startswith("baai/bge-m3"):
        return DEFAULT_BGE_M3_MAX_LENGTH

    return None


def is_cuda_oom(error: Exception) -> bool:
    """Return whether an exception represents a CUDA out-of-memory failure."""
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(error).lower()


def score_to_float(score: Any) -> float:
    """Convert tensor-like or scalar scores to plain Python floats."""
    if hasattr(score, "item"):
        return float(score.item())
    return float(score)


def format_bytes(num_bytes: int) -> str:
    """Format a byte count using binary units for progress messages."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as a compact human-readable duration."""
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def configure_cpu_threads(cpu_threads: int | None) -> None:
    """Configure FAISS and PyTorch CPU thread counts.

    Args:
        cpu_threads: Desired CPU thread count. `None` leaves library defaults.

    Raises:
        ValueError: If `cpu_threads` is provided but not positive.

    Side effects:
        Updates FAISS OpenMP threads and PyTorch intra/inter-op thread settings
        when supported, then prints the active configuration.
    """
    if cpu_threads is None:
        detected = os.cpu_count()
        print(
            "CPU threading: using library defaults"
            + (f" (system reports {detected} logical cores)." if detected is not None else ".")
        )
        return

    if cpu_threads <= 0:
        raise ValueError("--cpu-threads must be greater than 0 when provided.")

    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(cpu_threads)

    torch.set_num_threads(cpu_threads)
    interop_threads = max(1, min(cpu_threads, 4))
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    print(
        "CPU threading configured: "
        f"faiss_threads={cpu_threads}, "
        f"torch_threads={torch.get_num_threads()}, "
        f"torch_interop_threads={interop_threads}"
    )


def get_colbert_score_device() -> torch.device:
    """Return the device used for local ColBERT score reductions."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def compute_colbert_scores_batched(
    query_colbert: np.ndarray,
    candidate_colbert: list[np.ndarray],
    batch_size: int,
) -> np.ndarray:
    """Compute ColBERT-style late-interaction scores for candidate documents.

    Args:
        query_colbert: Query token representations with shape `(query_tokens, dim)`.
        candidate_colbert: Candidate document token representations.
        batch_size: Number of candidate documents scored per tensor batch.

    Returns:
        A float32 score array aligned with `candidate_colbert`.

    Raises:
        ValueError: If `batch_size` is not positive.
        Exception: Re-raises non-OOM tensor failures.
    """
    current_batch_size = batch_size
    if current_batch_size <= 0:
        raise ValueError("--colbert-score-batch-size must be greater than 0.")
    if not candidate_colbert:
        return np.empty((0,), dtype=np.float32)

    device = get_colbert_score_device()
    q_reps = torch.as_tensor(query_colbert, dtype=torch.float32, device=device)
    if q_reps.ndim != 2 or q_reps.shape[0] == 0:
        return np.zeros(len(candidate_colbert), dtype=np.float32)

    while True:
        try:
            score_chunks: list[np.ndarray] = []
            for start in range(0, len(candidate_colbert), current_batch_size):
                batch = candidate_colbert[start : start + current_batch_size]
                non_empty_indices: list[int] = []
                non_empty_docs: list[np.ndarray] = []
                batch_scores = np.zeros(len(batch), dtype=np.float32)

                for local_idx, doc_colbert in enumerate(batch):
                    if isinstance(doc_colbert, np.ndarray) and doc_colbert.ndim == 2 and doc_colbert.shape[0] > 0:
                        non_empty_indices.append(local_idx)
                        non_empty_docs.append(doc_colbert)

                if not non_empty_docs:
                    score_chunks.append(batch_scores)
                    continue

                # Pack variable-length document token vectors into a padded
                # tensor so the late-interaction reduction can run in one pass.
                max_doc_tokens = max(doc.shape[0] for doc in non_empty_docs)
                embedding_dim = q_reps.shape[1]
                doc_tensor = torch.zeros(
                    (len(non_empty_docs), max_doc_tokens, embedding_dim),
                    dtype=torch.float32,
                    device=device,
                )
                doc_mask = torch.zeros(
                    (len(non_empty_docs), max_doc_tokens),
                    dtype=torch.bool,
                    device=device,
                )

                for tensor_row, doc_colbert in enumerate(non_empty_docs):
                    doc_reps = torch.as_tensor(doc_colbert, dtype=torch.float32, device=device)
                    doc_length = doc_reps.shape[0]
                    doc_tensor[tensor_row, :doc_length] = doc_reps
                    doc_mask[tensor_row, :doc_length] = True

                # ColBERT scoring keeps the best document-token match for each
                # query token, then averages the summed matches by query length.
                token_scores = torch.einsum("qd,bkd->bqk", q_reps, doc_tensor)
                token_scores = token_scores.masked_fill(~doc_mask.unsqueeze(1), float("-inf"))
                max_scores = token_scores.max(dim=-1).values
                reduced_scores = max_scores.sum(dim=-1) / q_reps.shape[0]
                reduced_scores_np = reduced_scores.detach().cpu().numpy().astype(np.float32, copy=False)

                for output_idx, score in zip(non_empty_indices, reduced_scores_np):
                    batch_scores[output_idx] = float(score)

                score_chunks.append(batch_scores)

            return np.concatenate(score_chunks, axis=0)
        except Exception as error:
            if not (torch.cuda.is_available() and is_cuda_oom(error) and current_batch_size > 1):
                raise

            next_batch_size = max(1, current_batch_size // 2)
            print(
                "CUDA OOM while ColBERT reranking "
                f"with colbert_score_batch_size={current_batch_size}. "
                f"Retrying with colbert_score_batch_size={next_batch_size}."
            )
            torch.cuda.empty_cache()
            current_batch_size = next_batch_size


def precompute_corpus_hybrid_features(
    model: Any,
    corpus_texts: list[str],
    batch_size: int,
    max_length: int,
    colbert_storage_dtype: str,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    """Precompute BGE-M3 sparse and ColBERT corpus features.

    Args:
        model: BGE-M3 model exposing `encode`.
        corpus_texts: Corpus documents to encode.
        batch_size: Encoding batch size.
        max_length: Maximum token length for sparse/ColBERT features.
        colbert_storage_dtype: Storage dtype name for ColBERT vectors.

    Returns:
        Sparse lexical weights and ColBERT vectors aligned with `corpus_texts`.

    Raises:
        KeyError: If an unsupported dtype name is supplied.

    Side effects:
        Encodes the full corpus and prints memory-use information.
    """
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
    }
    storage_dtype = dtype_map[colbert_storage_dtype]

    print(
        "Precomputing corpus sparse and ColBERT features for hybrid retrieval. "
        "This uses more RAM but avoids re-encoding candidate documents for every query."
    )
    corpus_hybrid_output = encode_bge_m3(
        model=model,
        texts=corpus_texts,
        batch_size=batch_size,
        label="corpus (hybrid cache)",
        max_length=max_length,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    corpus_sparse = corpus_hybrid_output["lexical_weights"]
    corpus_colbert = [
        np.asarray(doc_colbert, dtype=storage_dtype)
        for doc_colbert in corpus_hybrid_output["colbert_vecs"]
    ]
    total_colbert_bytes = sum(doc_colbert.nbytes for doc_colbert in corpus_colbert)
    print(
        "Stored precomputed corpus ColBERT vectors as "
        f"{colbert_storage_dtype} ({format_bytes(total_colbert_bytes)} total)."
    )
    return corpus_sparse, corpus_colbert


def encode_bge_m3(
    model: Any,
    texts: list[str],
    batch_size: int,
    label: str,
    max_length: int | None,
    return_dense: bool,
    return_sparse: bool,
    return_colbert_vecs: bool,
) -> dict[str, Any]:
    """Encode text batches with BGE-M3, adapting after CUDA OOM.

    Args:
        model: BGE-M3 model exposing FlagEmbedding's `encode` API.
        texts: Text inputs to encode.
        batch_size: Preferred encode batch size.
        label: Human-readable label used in progress messages.
        max_length: Optional maximum token length.
        return_dense: Whether to include dense vectors.
        return_sparse: Whether to include lexical weights.
        return_colbert_vecs: Whether to include ColBERT token vectors.

    Returns:
        A dictionary containing the requested BGE-M3 outputs.

    Raises:
        Exception: Re-raises non-OOM encoding failures.
    """
    current_batch_size = batch_size
    while True:
        try:
            total_items = len(texts)
            if total_items == 0:
                output: dict[str, Any] = {}
                if return_dense:
                    output["dense_vecs"] = np.empty((0, 0), dtype=np.float32)
                if return_sparse:
                    output["lexical_weights"] = []
                if return_colbert_vecs:
                    output["colbert_vecs"] = []
                return output

            dense_chunks: list[np.ndarray] = []
            sparse_chunks: list[Any] = []
            colbert_chunks: list[Any] = []
            total_chunks = (total_items + current_batch_size - 1) // current_batch_size

            for chunk_idx, start in enumerate(range(0, total_items, current_batch_size), start=1):
                end = min(total_items, start + current_batch_size)
                print(
                    f"Encoding {label} with bge-m3: chunk {chunk_idx}/{total_chunks} "
                    f"({start + 1}-{end} of {total_items}), batch_size={current_batch_size}..."
                )
                chunk_output = model.encode(
                    texts[start:end],
                    batch_size=current_batch_size,
                    max_length=max_length,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert_vecs=return_colbert_vecs,
                )

                if return_dense:
                    dense = np.asarray(chunk_output["dense_vecs"], dtype=np.float32)
                    faiss.normalize_L2(dense)
                    dense_chunks.append(dense)
                if return_sparse:
                    sparse_chunks.extend(chunk_output["lexical_weights"])
                if return_colbert_vecs:
                    colbert_chunks.extend(chunk_output["colbert_vecs"])

            output = {}
            if return_dense:
                output["dense_vecs"] = np.concatenate(dense_chunks, axis=0)
            if return_sparse:
                output["lexical_weights"] = sparse_chunks
            if return_colbert_vecs:
                output["colbert_vecs"] = colbert_chunks
            return output
        except Exception as error:
            if not (torch.cuda.is_available() and is_cuda_oom(error) and current_batch_size > 1):
                raise

            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"CUDA OOM while bge-m3 encoding {label} with batch_size={current_batch_size}. "
                f"Retrying with batch_size={next_batch_size}."
            )
            torch.cuda.empty_cache()
            current_batch_size = next_batch_size


def pick_query_text(item: dict[str, Any], query_field: str) -> str:
    """Select the query text field from a query record.

    Args:
        item: Query record dictionary.
        query_field: Explicit field name or `auto`.

    Returns:
        The selected stripped query text, or an empty string if none is usable.
    """
    if query_field != "auto":
        value = item.get(query_field, "")
        return value.strip() if isinstance(value, str) else ""

    for field_name in ("expanded", "original", "text", "query"):
        value = item.get(field_name, "")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def iter_query_records(raw_queries: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield normalized `(qid, item)` pairs from supported query JSON shapes.

    Args:
        raw_queries: Either a list of query dictionaries or a mapping from qid
            to query dictionaries/strings.

    Yields:
        Query IDs and dictionary records. String values are wrapped as
        `{"text": value}`.
    """
    if isinstance(raw_queries, list):
        for position, item in enumerate(raw_queries):
            if isinstance(item, dict):
                qid = str(item.get("index", item.get("qid", item.get("id", position))))
                yield qid, item
    elif isinstance(raw_queries, dict):
        for qid, item in raw_queries.items():
            if isinstance(item, dict):
                yield str(qid), item
            elif isinstance(item, str):
                yield str(qid), {"text": item}


def load_queries(
    path: Path,
    query_field: str,
    max_queries: int | None,
    query_prefix: str,
) -> tuple[list[str], list[str]]:
    """Load, normalize, and optionally limit query records.

    Args:
        path: Query JSON path.
        query_field: Explicit field to read, or `auto`.
        max_queries: Optional maximum number of valid queries to load.
        query_prefix: Optional prefix prepended before encoding.

    Returns:
        Query IDs and query texts aligned by index.

    Raises:
        ValueError: If no valid query text is loaded.
    """
    raw_queries = load_json(path)

    query_ids: list[str] = []
    query_texts: list[str] = []
    skipped = 0

    for qid, item in iter_query_records(raw_queries):
        text = pick_query_text(item, query_field)
        if not text:
            skipped += 1
            continue

        query_ids.append(qid)
        query_texts.append(f"{query_prefix}{text}" if query_prefix else text)

        if max_queries is not None and len(query_ids) >= max_queries:
            break

    print(f"Loaded {len(query_ids)} queries from {path}")
    if skipped:
        print(f"Skipped {skipped} queries with empty text")

    if not query_ids:
        raise ValueError("No valid queries were loaded.")

    return query_ids, query_texts


def build_document_text(item: dict[str, Any]) -> str:
    """Build the retrievable document text from title and abstract fields."""
    parts = [
        str(item.get("title", "")).strip(),
        str(item.get("abstract", "")).strip(),
    ]
    return " ".join(part for part in parts if part)


@dataclass
class FaissIndexHandle:
    """Container for a FAISS index and optional GPU resource owner."""

    index: Any
    device: str
    gpu_resources: Any | None = None


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    """Load corpus documents and normalize IDs/text for retrieval.

    Args:
        path: Corpus JSON path. The file must contain a list of document
            dictionaries.

    Returns:
        Pubkeys and document texts aligned by index.

    Raises:
        ValueError: If the corpus shape is invalid or no usable documents load.
    """
    raw_corpus = load_json(path)
    if not isinstance(raw_corpus, list):
        raise ValueError("Corpus JSON must be a list of documents.")

    pubkeys: list[str] = []
    documents: list[str] = []
    seen_pubkeys: set[str] = set()
    skipped = 0

    for item in raw_corpus:
        if not isinstance(item, dict):
            skipped += 1
            continue

        pubkey = item.get("pubkey", item.get("id"))
        if pubkey is None:
            skipped += 1
            continue

        pubkey_str = str(pubkey)
        if pubkey_str in seen_pubkeys:
            continue

        text = build_document_text(item)
        if not text:
            skipped += 1
            continue

        seen_pubkeys.add(pubkey_str)
        pubkeys.append(pubkey_str)
        documents.append(text)

    print(f"Loaded {len(pubkeys)} documents from {path}")
    if skipped:
        print(f"Skipped {skipped} corpus entries without usable text or id")

    if not pubkeys:
        raise ValueError("No valid corpus documents were loaded.")

    return pubkeys, documents


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    label: str,
) -> np.ndarray:
    """Encode texts with SentenceTransformers on CPU or one GPU.

    Args:
        model: SentenceTransformer model instance.
        texts: Texts to encode.
        batch_size: Preferred batch size.
        label: Human-readable label used in progress messages.

    Returns:
        L2-normalized float32 embeddings.

    Raises:
        Exception: Re-raises non-OOM encoding failures.
    """
    current_batch_size = batch_size
    while True:
        print(f"Encoding {label} ({len(texts)} items), batch_size={current_batch_size}...")
        try:
            embeddings = model.encode(
                texts,
                batch_size=current_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as error:
            if not (torch.cuda.is_available() and is_cuda_oom(error) and current_batch_size > 1):
                raise

            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"CUDA OOM while encoding {label} with batch_size={current_batch_size}. "
                f"Retrying with batch_size={next_batch_size}."
            )
            torch.cuda.empty_cache()
            current_batch_size = next_batch_size


def encode_texts_multi_gpu(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    label: str,
    pool: Any,
) -> np.ndarray:
    """Encode texts with SentenceTransformers' multi-process GPU pool.

    Args:
        model: SentenceTransformer model that owns the process pool.
        texts: Texts to encode.
        batch_size: Preferred batch size.
        label: Human-readable label used in progress messages.
        pool: Pool returned by `start_multi_process_pool`.

    Returns:
        L2-normalized float32 embeddings.

    Raises:
        Exception: Re-raises non-OOM encoding failures.
    """
    current_batch_size = batch_size
    while True:
        print(
            f"Encoding {label} ({len(texts)} items) with multi-GPU, "
            f"batch_size={current_batch_size}..."
        )
        try:
            embeddings = model.encode_multi_process(
                texts,
                pool=pool,
                batch_size=current_batch_size,
            )
            normalized = np.asarray(embeddings, dtype=np.float32)
            faiss.normalize_L2(normalized)
            return normalized
        except Exception as error:
            if not (is_cuda_oom(error) and current_batch_size > 1):
                raise

            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"CUDA OOM while multi-GPU encoding {label} with batch_size={current_batch_size}. "
                f"Retrying with batch_size={next_batch_size}."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current_batch_size = next_batch_size


def build_faiss_index(corpus_embeddings: np.ndarray) -> FaissIndexHandle:
    """Build an inner-product FAISS index over corpus embeddings.

    Args:
        corpus_embeddings: Float32 L2-normalized corpus embedding matrix.

    Returns:
        A FAISS index handle using GPU resources when available and supported,
        otherwise a CPU index.
    """
    dimension = corpus_embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(dimension)
    cpu_index.add(corpus_embeddings)

    if not torch.cuda.is_available():
        print("Using FAISS CPU index (no CUDA GPU detected for FAISS).")
        return FaissIndexHandle(index=cpu_index, device="cpu")

    has_gpu_faiss = hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")
    if not has_gpu_faiss:
        print("Using FAISS CPU index (installed FAISS build does not include GPU support).")
        return FaissIndexHandle(index=cpu_index, device="cpu")

    try:
        gpu_resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
        print("Using FAISS GPU index on cuda:0.")
        return FaissIndexHandle(
            index=gpu_index,
            device="gpu",
            gpu_resources=gpu_resources,
        )
    except Exception as error:
        print(
            "FAISS GPU index initialization failed; falling back to CPU index. "
            f"Reason: {error}"
        )
        return FaissIndexHandle(index=cpu_index, device="cpu")


def dense_search(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    corpus_pubkeys: list[str],
    index_handle: FaissIndexHandle,
    top_k: int,
) -> dict[str, list[str]]:
    """Run dense nearest-neighbor search for all queries.

    Args:
        query_ids: Query IDs aligned with `query_embeddings`.
        query_embeddings: Query embedding matrix.
        corpus_pubkeys: Corpus IDs aligned with the FAISS index rows.
        index_handle: FAISS index wrapper.
        top_k: Maximum number of documents returned per query.

    Returns:
        Query-to-ranked-pubkeys mapping.
    """
    effective_top_k = min(top_k, len(corpus_pubkeys))
    _, indices = index_handle.index.search(query_embeddings, effective_top_k)

    results: dict[str, list[str]] = {}
    for row, qid in enumerate(query_ids):
        ranked_pubkeys = [
            corpus_pubkeys[col]
            for col in indices[row]
            if 0 <= col < len(corpus_pubkeys)
        ]
        results[qid] = ranked_pubkeys

    print(f"Completed dense search for {len(query_ids)} queries")
    return results


def hybrid_search(
    model: Any,
    query_ids: list[str],
    query_texts: list[str],
    corpus_pubkeys: list[str],
    corpus_texts: list[str],
    index_handle: FaissIndexHandle,
    top_k: int,
    candidate_multiplier: int,
    batch_size: int,
    hybrid_max_length: int,
    colbert_score_batch_size: int,
    dense_weight: float,
    sparse_weight: float,
    colbert_weight: float,
    corpus_sparse_cache: list[dict[str, float]] | None = None,
    corpus_colbert_cache: list[np.ndarray] | None = None,
    query_log_interval: int = DEFAULT_LOG_EVERY_QUERIES,
) -> dict[str, list[str]]:
    """Run BGE-M3 hybrid retrieval over dense candidates.

    Dense FAISS search first selects an expanded candidate pool. Each candidate
    is then rescored with weighted dense, sparse lexical, and ColBERT-style
    late-interaction scores.

    Args:
        model: BGE-M3 model exposing dense, lexical, and ColBERT outputs.
        query_ids: Query IDs aligned with `query_texts`.
        query_texts: Query texts to encode for hybrid scoring.
        corpus_pubkeys: Corpus IDs aligned with `corpus_texts` and the FAISS index.
        corpus_texts: Corpus document texts.
        index_handle: Dense FAISS candidate index.
        top_k: Number of final documents returned per query.
        candidate_multiplier: Multiplier used to size the dense candidate pool.
        batch_size: Preferred BGE-M3 encoding batch size.
        hybrid_max_length: Token length for sparse and ColBERT representations.
        colbert_score_batch_size: Candidate batch size for tensor scoring.
        dense_weight: Dense score contribution.
        sparse_weight: Sparse lexical score contribution.
        colbert_weight: ColBERT score contribution.
        corpus_sparse_cache: Optional precomputed sparse corpus features.
        corpus_colbert_cache: Optional precomputed ColBERT corpus features.
        query_log_interval: Number of queries between progress logs.

    Returns:
        Query-to-ranked-pubkeys mapping.

    Raises:
        ValueError: If any required numeric parameter is invalid.
    """
    if candidate_multiplier <= 0:
        raise ValueError("--candidate-multiplier must be greater than 0.")
    if hybrid_max_length <= 0:
        raise ValueError("--multivector-max-length/--hybrid-max-length must be greater than 0.")
    if colbert_score_batch_size <= 0:
        raise ValueError("--colbert-score-batch-size must be greater than 0.")
    if query_log_interval <= 0:
        raise ValueError("--log-every-queries must be greater than 0.")
    if dense_weight == 0.0 and sparse_weight == 0.0 and colbert_weight == 0.0:
        raise ValueError(
            "At least one hybrid weight must be non-zero "
            "(--dense-weight, --sparse-weight, --colbert-weight)."
        )

    candidate_k = min(
        len(corpus_pubkeys),
        max(top_k, top_k * candidate_multiplier),
    )

    query_output = encode_bge_m3(
        model=model,
        texts=query_texts,
        batch_size=batch_size,
        label="queries (hybrid)",
        max_length=hybrid_max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    query_embeddings = np.asarray(query_output["dense_vecs"], dtype=np.float32)
    query_sparse = query_output["lexical_weights"]
    query_colbert = query_output["colbert_vecs"]
    dense_scores, dense_indices = index_handle.index.search(query_embeddings, candidate_k)

    results: dict[str, list[str]] = {}
    total_queries = len(query_ids)
    started_at = time.perf_counter()
    for row, qid in enumerate(query_ids):
        candidate_doc_indices = [
            int(doc_idx)
            for doc_idx in dense_indices[row]
            if 0 <= int(doc_idx) < len(corpus_pubkeys)
        ]

        if not candidate_doc_indices:
            results[qid] = []
            continue

        if corpus_sparse_cache is not None and corpus_colbert_cache is not None:
            candidate_sparse = [corpus_sparse_cache[doc_idx] for doc_idx in candidate_doc_indices]
            candidate_colbert = [corpus_colbert_cache[doc_idx] for doc_idx in candidate_doc_indices]
        else:
            candidate_texts = [corpus_texts[doc_idx] for doc_idx in candidate_doc_indices]
            candidate_output = encode_bge_m3(
                model=model,
                texts=candidate_texts,
                batch_size=batch_size,
                label=f"candidates for query {qid} (hybrid)",
                max_length=hybrid_max_length,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=True,
            )
            candidate_sparse = candidate_output["lexical_weights"]
            candidate_colbert = candidate_output["colbert_vecs"]

        q_sparse = query_sparse[row]
        q_colbert = query_colbert[row]
        sparse_scores = np.asarray(
            model.compute_lexical_matching_score([q_sparse], candidate_sparse),
            dtype=np.float32,
        ).reshape(-1)
        colbert_scores = compute_colbert_scores_batched(
            query_colbert=q_colbert,
            candidate_colbert=candidate_colbert,
            batch_size=colbert_score_batch_size,
        )
        dense_scores_row = np.asarray(dense_scores[row][: len(candidate_doc_indices)], dtype=np.float32)
        # All score families are already aligned with `candidate_doc_indices`.
        final_scores = (
            dense_weight * dense_scores_row
            + sparse_weight * sparse_scores
            + colbert_weight * colbert_scores
        )
        ranked_positions = np.argsort(-final_scores)[:top_k]
        results[qid] = [corpus_pubkeys[candidate_doc_indices[int(pos)]] for pos in ranked_positions]

        completed_queries = row + 1
        if completed_queries == 1 or completed_queries % query_log_interval == 0 or completed_queries == total_queries:
            elapsed = time.perf_counter() - started_at
            queries_per_second = completed_queries / elapsed if elapsed > 0 else 0.0
            remaining_queries = total_queries - completed_queries
            eta_seconds = remaining_queries / queries_per_second if queries_per_second > 0 else float("inf")
            print(
                "Hybrid progress: "
                f"{completed_queries}/{total_queries} queries "
                f"({completed_queries / total_queries:.1%}), "
                f"elapsed={format_duration(elapsed)}, "
                f"eta={format_duration(eta_seconds)}, "
                f"throughput={queries_per_second:.2f} q/s"
            )

    print(f"Completed hybrid search for {len(query_ids)} queries")
    return results


def save_results(path: Path, results: dict[str, list[str]]) -> None:
    """Write retrieval results as Evaluator.java-compatible JSON.

    Args:
        path: Output JSON path.
        results: Query-to-ranked-pubkeys mapping.

    Side effects:
        Creates the parent directory when needed and writes the JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} query rankings to {path}")


def main() -> None:
    """Execute the configured retrieval pipeline.

    Side effects:
        Reads query and corpus files, loads neural retrieval models, may allocate
        substantial GPU/CPU memory, builds a FAISS index, and writes rankings to
        the configured output path.

    Raises:
        ValueError: If CLI options are inconsistent or input files contain no
            usable records.
        ImportError: If the requested retrieval backend is unavailable.
    """
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.colbert_score_batch_size <= 0:
        raise ValueError("--colbert-score-batch-size must be greater than 0.")
    if args.log_every_queries <= 0:
        raise ValueError("--log-every-queries must be greater than 0.")
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be greater than 0 when provided.")
    if args.candidate_multiplier <= 0:
        raise ValueError("--candidate-multiplier must be greater than 0.")
    if args.multivector_max_length <= 0:
        raise ValueError("--multivector-max-length/--hybrid-max-length must be greater than 0.")
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("--max-queries must be greater than 0 when provided.")

    query_prefix = infer_query_prefix(args.model, args.query_prefix)
    effective_max_length = resolve_effective_max_length(args.model, args.max_length)
    retrieval_mode = resolve_retrieval_mode(args.model, args.retrieval_mode)
    configure_cpu_threads(args.cpu_threads)
    execution_mode, target_devices = resolve_execution_mode()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"Detected GPUs: {gpu_count}")
    if execution_mode == "cpu":
        print("Execution mode: CPU (no CUDA GPU detected)")
    elif execution_mode == "cuda":
        print("Execution mode: single-GPU")
        print(f"Primary GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Execution mode: multi-GPU ({', '.join(target_devices)})")

    print(f"Bi-encoder model: {args.model}")
    print(f"Retrieval mode: {retrieval_mode}")
    if query_prefix:
        print(f"Using query prefix: {query_prefix}")
    if effective_max_length is not None:
        print(f"Encoder max length: {effective_max_length}")

    query_ids, query_texts = load_queries(
        path=args.queries,
        query_field=args.query_field,
        max_queries=args.max_queries,
        query_prefix=query_prefix,
    )
    corpus_pubkeys, corpus_texts = load_corpus(args.corpus)

    if retrieval_mode == "hybrid" and not args.model.lower().startswith("baai/bge-m3"):
        raise ValueError(
            '--retrieval-mode "hybrid" requires BGE-M3 (e.g., --model BAAI/bge-m3).'
        )

    if args.model.lower().startswith("baai/bge-m3"):
        model = build_bge_m3_model(
            model_name=args.model,
            execution_mode=execution_mode,
            target_devices=target_devices,
        )
        corpus_dense_output = encode_bge_m3(
            model=model,
            texts=corpus_texts,
            batch_size=args.batch_size,
            label="corpus (dense index)",
            max_length=effective_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        corpus_embeddings = np.asarray(corpus_dense_output["dense_vecs"], dtype=np.float32)

        if retrieval_mode == "hybrid":
            corpus_sparse_cache: list[dict[str, float]] | None = None
            corpus_colbert_cache: list[np.ndarray] | None = None
            if args.precompute_corpus_hybrid:
                corpus_sparse_cache, corpus_colbert_cache = precompute_corpus_hybrid_features(
                    model=model,
                    corpus_texts=corpus_texts,
                    batch_size=args.batch_size,
                    max_length=args.multivector_max_length,
                    colbert_storage_dtype=args.corpus_colbert_dtype,
                )
            print("Building FAISS index...")
            index_handle = build_faiss_index(corpus_embeddings)
            results = hybrid_search(
                model=model,
                query_ids=query_ids,
                query_texts=query_texts,
                corpus_pubkeys=corpus_pubkeys,
                corpus_texts=corpus_texts,
                index_handle=index_handle,
                top_k=args.top_k,
                candidate_multiplier=args.candidate_multiplier,
                batch_size=args.batch_size,
                hybrid_max_length=args.multivector_max_length,
                colbert_score_batch_size=args.colbert_score_batch_size,
                dense_weight=args.dense_weight,
                sparse_weight=args.sparse_weight,
                colbert_weight=args.colbert_weight,
                corpus_sparse_cache=corpus_sparse_cache,
                corpus_colbert_cache=corpus_colbert_cache,
                query_log_interval=args.log_every_queries,
            )
            save_results(args.output, results)
            return

        query_dense_output = encode_bge_m3(
            model=model,
            texts=query_texts,
            batch_size=args.batch_size,
            label="queries (dense)",
            max_length=effective_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        query_embeddings = np.asarray(query_dense_output["dense_vecs"], dtype=np.float32)
    else:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for dense retrieval mode. "
                "Install it with: pip install -U sentence-transformers"
            )
        if execution_mode == "multi-gpu":
            model = SentenceTransformer(args.model)
            if effective_max_length is not None:
                model.max_seq_length = effective_max_length
            pool = model.start_multi_process_pool(target_devices=target_devices)
            try:
                corpus_embeddings = encode_texts_multi_gpu(
                    model=model,
                    texts=corpus_texts,
                    batch_size=args.batch_size,
                    label="corpus",
                    pool=pool,
                )
                query_embeddings = encode_texts_multi_gpu(
                    model=model,
                    texts=query_texts,
                    batch_size=args.batch_size,
                    label="queries",
                    pool=pool,
                )
            finally:
                model.stop_multi_process_pool(pool)
        else:
            model = SentenceTransformer(args.model, device=execution_mode)
            if effective_max_length is not None:
                model.max_seq_length = effective_max_length
            corpus_embeddings = encode_texts(
                model=model,
                texts=corpus_texts,
                batch_size=args.batch_size,
                label="corpus",
            )
            query_embeddings = encode_texts(
                model=model,
                texts=query_texts,
                batch_size=args.batch_size,
                label="queries",
            )

    print("Building FAISS index...")
    index_handle = build_faiss_index(corpus_embeddings)

    results = dense_search(
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        corpus_pubkeys=corpus_pubkeys,
        index_handle=index_handle,
        top_k=args.top_k,
    )
    save_results(args.output, results)


if __name__ == "__main__":
    main()
