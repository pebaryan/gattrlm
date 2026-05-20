"""Backward-compatible shim — re-exports from the canonical `attractor` package.

New code should import from `attractor` directly:

    from attractor.models.config import Config
    from attractor import create_model
"""


def create_config(name: str, **kwargs):
    """Create a model config by name. Delegates to attractor."""
    from attractor import create_config as _create_config
    return _create_config(name, **kwargs)


def create_model(name: str, **kwargs):
    """Create a model by name. Delegates to attractor."""
    from attractor import create_model as _create_model
    return _create_model(name, **kwargs)


def from_pretrained(repo_id: str, device="cpu", dtype=None, **kwargs):
    """Load a pretrained model from a HuggingFace repository. Delegates to attractor."""
    from attractor import from_pretrained as _from_pretrained
    return _from_pretrained(repo_id, device=device, dtype=dtype, **kwargs)


def save_pretrained(model, path: str):
    """Save a model's config and weights. Delegates to attractor."""
    from attractor import save_pretrained as _save_pretrained
    return _save_pretrained(model, path)


def __getattr__(name):
    if name == "GPT":
        from attractor.models.gpt.gpt import GPT
        return GPT
    if name == "GPTConfig":
        from attractor.models.gpt.config import GPTConfig
        return GPTConfig
    if name == "Parcae":
        from attractor.models.parcae.parcae import Parcae
        return Parcae
    if name == "ParcaeConfig":
        from attractor.models.parcae.config import ParcaeConfig
        return ParcaeConfig
    if name == "EQLM":
        from attractor.models.eqlm.eqlm import EQLM
        return EQLM
    if name == "EQLMConfig":
        from attractor.models.eqlm.config import EQLMConfig
        return EQLMConfig
    raise AttributeError(f"module 'parcae_lm' has no attribute {name!r}")
