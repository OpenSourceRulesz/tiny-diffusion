"""tiny-diffusion: a toy text diffusion LM with a frozen-block tool loop."""

_LAZY = {
    "WordTokenizer": ".tokenizer",
    "SPECIALS": ".tokenizer",
    "TinyDiffusionLM": ".model",
    "diffusion_loss": ".diffusion",
    "generate": ".diffusion",
    "TOOLS": ".tools",
    "run_call": ".tools",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name], __name__)
        return getattr(mod, name)
    raise AttributeError(name)
