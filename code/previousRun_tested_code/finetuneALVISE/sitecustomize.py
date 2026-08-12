"""Compatibility patches auto-loaded by Python for local FlagEmbedding tooling."""


def _patch_transformers_flagembedding_compat() -> None:
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_import_utils
        from transformers import Trainer
        import inspect
    except ImportError:
        return

    if not hasattr(transformers_utils, "is_flash_attn_greater_or_equal_2_10"):
        legacy_checker = getattr(transformers_utils, "is_flash_attn_greater_or_equal", None)
        if legacy_checker is not None:

            def _is_flash_attn_greater_or_equal_2_10() -> bool:
                try:
                    return bool(legacy_checker("2.1.0"))
                except Exception:
                    return False

            transformers_utils.is_flash_attn_greater_or_equal_2_10 = (
                _is_flash_attn_greater_or_equal_2_10
            )

    if not hasattr(transformers_import_utils, "is_torch_fx_available"):

        def _is_torch_fx_available() -> bool:
            try:
                return bool(transformers_import_utils.is_torch_available())
            except Exception:
                return False

        transformers_import_utils.is_torch_fx_available = _is_torch_fx_available

    trainer_init_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" not in trainer_init_signature.parameters and not getattr(
        Trainer.__init__, "_flagembedding_tokenizer_compat", False
    ):
        original_trainer_init = Trainer.__init__

        def _trainer_init_with_tokenizer_compat(self, *args, tokenizer=None, processing_class=None, **kwargs):
            if tokenizer is not None and processing_class is None:
                processing_class = tokenizer
            result = original_trainer_init(
                self,
                *args,
                processing_class=processing_class,
                **kwargs,
            )
            if not hasattr(self, "tokenizer"):
                self.tokenizer = processing_class
            return result

        _trainer_init_with_tokenizer_compat._flagembedding_tokenizer_compat = True
        Trainer.__init__ = _trainer_init_with_tokenizer_compat


_patch_transformers_flagembedding_compat()
