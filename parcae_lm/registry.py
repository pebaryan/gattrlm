# Config files in parcae_lm/configs/ import DEFAULT_HF_ORG from here.
# We define it directly to avoid a circular import:
#   parcae_lm.registry → attractor.registry → _rescan_model_configs() → parcae_lm configs → parcae_lm.registry
DEFAULT_HF_ORG = "SandyResearch"


def __getattr__(name):
    """Lazily re-export symbols from attractor.registry (avoids circular imports)."""
    if name in ("add_model_config_path", "name_to_config", "configs", "_natural_key", "_rescan_model_configs"):
        from attractor.registry import _rescan_model_configs as _  # ensure attractor.registry is loaded
        import sys
        mod = sys.modules[__name__]
        from attractor.registry import add_model_config_path, name_to_config, configs, _natural_key, _rescan_model_configs
        mod.add_model_config_path = add_model_config_path
        mod.name_to_config = name_to_config
        mod.configs = configs
        mod._natural_key = _natural_key
        mod._rescan_model_configs = _rescan_model_configs
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
