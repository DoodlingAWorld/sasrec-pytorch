"""Configuration for SASRec.

A single dataclass holds every knob so that training runs are reproducible and
self-documenting. Defaults match the paper's MovieLens-1M setup (Section IV-C).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class SASRecConfig:
    # ---- data ----
    dataset: str = "ml-1m"
    max_len: int = 200          # n: max sequence length (paper uses 200 for ML-1M)
    min_count: int = 5          # k-core: drop users/items with < k interactions

    # ---- model ----
    hidden_dim: int = 50        # d: latent dimensionality
    num_blocks: int = 2         # b: number of stacked self-attention blocks
    num_heads: int = 1          # paper uses single-head attention for small d
    dropout: float = 0.2        # ML-1M uses 0.2; sparser datasets use 0.5

    # ---- training ----
    lr: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 201       # paper trains up to a few hundred epochs w/ early stop
    patience: int = 20          # early stop if val NDCG@10 doesn't improve for this many evals
    eval_every: int = 20        # run validation every N epochs
    l2_emb: float = 0.0         # optional L2 on embeddings (paper relies on dropout)

    # ---- evaluation ----
    num_neg_eval: int = 100     # negatives sampled per user at eval time
    topk: int = 10              # @10 metrics

    # ---- misc ----
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def fast_dev_config(**overrides) -> SASRecConfig:
    """Tiny config to validate the pipeline end-to-end in minutes (not for real metrics)."""
    cfg = SASRecConfig(
        max_len=50,
        hidden_dim=32,
        num_blocks=1,
        num_epochs=5,
        eval_every=1,
        patience=5,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
