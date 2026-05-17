from parcae_lm.models.eqlm.config import EQLMConfig
from parcae_lm.models.eqlm.init import EQLMInit


def __getattr__(name):
    if name == "EQLM":
        from parcae_lm.models.eqlm.eqlm import EQLM
        return EQLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["EQLMConfig", "EQLMInit", "EQLM"]