import json
from pathlib import Path


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def chunked(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Back-translation cache must be a JSON object: {path}")

    cache: dict[str, dict[str, str]] = {}
    for direction, entries in payload.items():
        if not isinstance(direction, str) or not isinstance(entries, dict):
            continue
        normalized_entries = {}
        for source_text, translated_text in entries.items():
            source = normalize_text(str(source_text))
            translated = normalize_text(str(translated_text))
            if source and translated:
                normalized_entries[source] = translated
        if normalized_entries:
            cache[direction] = normalized_entries

    return cache


def save_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, ensure_ascii=False)


def resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for back translation, but no CUDA device is available.")
    return device


class Seq2SeqTranslator:
    def __init__(
        self,
        model_name_or_path: str,
        device_name: str,
        max_input_length: int,
        num_beams: int,
    ) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Back translation requires `transformers`. Install it with: "
                "pip install -U transformers sentencepiece"
            ) from error

        self.device = resolve_device(device_name)
        self.max_input_length = max_input_length
        self.num_beams = num_beams
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        import torch

        self.torch = torch

    def translate_batch(self, texts: list[str]) -> list[str]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max(32, min(256, self.max_input_length)),
                num_beams=self.num_beams,
            )

        return [
            normalize_text(text)
            for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        ]


def build_model_name(model_template: str, source_lang: str, target_lang: str) -> str:
    return model_template.format(src=source_lang, tgt=target_lang)


def translate_with_cache(
    texts: list[str],
    *,
    source_lang: str,
    target_lang: str,
    model_template: str,
    batch_size: int,
    device_name: str,
    max_input_length: int,
    num_beams: int,
    cache: dict[str, dict[str, str]],
) -> tuple[dict[str, str], bool]:
    direction_key = f"{source_lang}->{target_lang}"
    direction_cache = cache.setdefault(direction_key, {})

    unique_texts = list(dict.fromkeys(normalize_text(text) for text in texts if normalize_text(text)))
    missing_texts = [text for text in unique_texts if text not in direction_cache]
    if not missing_texts:
        return {text: direction_cache[text] for text in unique_texts if text in direction_cache}, False

    model_name = build_model_name(model_template, source_lang, target_lang)
    print(
        f"Back translation: loading model {model_name} for {len(missing_texts)} "
        f"missing texts ({direction_key})."
    )
    translator = Seq2SeqTranslator(
        model_name_or_path=model_name,
        device_name=device_name,
        max_input_length=max_input_length,
        num_beams=num_beams,
    )

    translated_count = 0
    for batch in chunked(missing_texts, batch_size):
        translated_batch = translator.translate_batch(batch)
        for source_text, translated_text in zip(batch, translated_batch):
            translated = normalize_text(translated_text)
            if translated:
                direction_cache[source_text] = translated
        translated_count += len(batch)
        print(
            f"Back translation: cached {translated_count}/{len(missing_texts)} "
            f"texts for {direction_key}."
        )

    return {text: direction_cache[text] for text in unique_texts if text in direction_cache}, True


def build_back_translation_map(
    texts: list[str],
    *,
    pivot_langs: list[str],
    source_lang: str = "en",
    model_template: str = "Helsinki-NLP/opus-mt-{src}-{tgt}",
    batch_size: int = 8,
    device_name: str = "auto",
    max_input_length: int = 192,
    num_beams: int = 4,
    cache_path: Path | None = None,
) -> dict[str, list[dict[str, str]]]:
    unique_texts = list(dict.fromkeys(normalize_text(text) for text in texts if normalize_text(text)))
    back_translation_map: dict[str, list[dict[str, str]]] = {text: [] for text in unique_texts}
    if not unique_texts:
        return back_translation_map

    cache = load_cache(cache_path) if cache_path is not None else {}
    normalized_pivots = [lang.strip() for lang in pivot_langs if lang.strip()]
    if not normalized_pivots:
        raise ValueError("At least one pivot language is required for back translation.")

    print(
        f"Back translation: preparing {len(unique_texts)} unique training texts "
        f"across {len(normalized_pivots)} pivot languages."
    )

    cache_changed = False

    for pivot_lang in normalized_pivots:
        if pivot_lang == source_lang:
            continue

        forward_map, forward_changed = translate_with_cache(
            unique_texts,
            source_lang=source_lang,
            target_lang=pivot_lang,
            model_template=model_template,
            batch_size=batch_size,
            device_name=device_name,
            max_input_length=max_input_length,
            num_beams=num_beams,
            cache=cache,
        )
        pivot_texts = [forward_map[text] for text in unique_texts if text in forward_map]
        backward_map, backward_changed = translate_with_cache(
            pivot_texts,
            source_lang=pivot_lang,
            target_lang=source_lang,
            model_template=model_template,
            batch_size=batch_size,
            device_name=device_name,
            max_input_length=max_input_length,
            num_beams=num_beams,
            cache=cache,
        )
        cache_changed = cache_changed or forward_changed or backward_changed

        for original_text in unique_texts:
            pivot_text = forward_map.get(original_text)
            if not pivot_text:
                continue

            back_translated = normalize_text(backward_map.get(pivot_text, ""))
            if not back_translated or back_translated == original_text:
                continue

            variants_for_text = back_translation_map[original_text]
            seen_texts = {variant["text"] for variant in variants_for_text}
            if back_translated in seen_texts:
                continue

            variants_for_text.append(
                {
                    "pivot_lang": pivot_lang,
                    "text": back_translated,
                }
            )

    if cache_path is not None and cache_changed:
        save_cache(cache_path, cache)
        print(f"Back translation: saved cache to {cache_path}")

    generated_variants = sum(len(variants) for variants in back_translation_map.values())
    print(f"Back translation: generated {generated_variants} unique augmented texts.")
    return back_translation_map
