from attractor.registry import DEFAULT_HF_ORG


# Native Clifford LM: standard transformer prelude + FP head with both
# Clifford attention and Clifford MLP. Sizing mirrors clifford-small-140m.
configs = [
    dict(
        name="native-clifford-small-140m",
        hf_config=dict(org=DEFAULT_HF_ORG, name="native-clifford-small-140m"),
        block_size=2048,
        vocab_size=32768,
        padding_multiple=64,
        n_embd=1024,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=4096,
        bias=False,
        tie_embeddings=True,
        architecture_class_name="CliffordLM",  # dispatches to CliffordLMConfig
        block_class_name="TransformerPreNormBlock",
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
        mlp_class_name="BaseMLP",  # used by prelude/coda only
        nonlin_name="ReLU2",
        qk_norm=True,
        logit_softcap=None,
        n_backbone_layers=7,
        n_fp_blocks=1,
        solver="anderson",
        max_iter=64,
        min_iter=6,
        tol=3e-4,
        anderson_m=5,
        anderson_beta=1.0,
        backward_type="onestep",
        backward_max_iter=64,
        backward_min_iter=6,
        backward_tol=3e-4,
        adjoint_grad_clip=None,
        layer_scale_init=0.75,
        gamma_max=0.75,
        fp_lr_scale=0.5,
        fp_wd=0.1,
        init_strategy="scaled-zero",
        init_orthogonal=True,
        # Clifford signature.
        clifford_p=3,
        clifford_q=0,
        clifford_r=0,
        # Clifford MLP sublayer (same as clifford-small-140m).
        n_clifford_channels=512,
        n_clifford_hidden=512,
        # Native Clifford attention.
        native_attention=True,
        n_clifford_attn_heads=8,
        n_clifford_attn_channels_per_head=8,
    ),
]
