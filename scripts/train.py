#!/usr/bin/env python3
"""Train SASRec and save the model + metrics.

Examples
--------
    python scripts/train.py                 # full ML-1M repro (n=200, d=50, b=2)
    python scripts/train.py --fast-dev      # quick smoke test (minutes)
    python scripts/train.py --epochs 150 --seed 1
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sasrec.config import SASRecConfig, fast_dev_config  # noqa: E402
from sasrec.train import train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fast-dev", action="store_true", help="tiny config for a smoke test")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="experiments/runs")
    args = p.parse_args()

    cfg = fast_dev_config(seed=args.seed) if args.fast_dev else SASRecConfig(seed=args.seed)
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.max_len is not None:
        cfg.max_len = args.max_len
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.num_blocks is not None:
        cfg.num_blocks = args.num_blocks

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "fastdev" if args.fast_dev else "ml1m"
    run_name = f"{tag}_seed{cfg.seed}_d{cfg.hidden_dim}_n{cfg.max_len}_b{cfg.num_blocks}"

    out = train(cfg, verbose=True)

    metrics = {
        "config": cfg.to_dict(),
        "best_val_ndcg": out["best_val_ndcg"],
        "test_ndcg": out["test_ndcg"],
        "test_hit": out["test_hit"],
        "num_items": out["num_items"],
        "num_users": out["num_users"],
    }
    with open(os.path.join(args.out_dir, f"{run_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(out["model"].state_dict(), os.path.join(args.out_dir, f"{run_name}.pt"))
    print(f"\nSaved metrics + model to {args.out_dir}/{run_name}.*")
    print(f"TEST NDCG@10={out['test_ndcg']:.4f}  Hit@10={out['test_hit']:.4f}")


if __name__ == "__main__":
    main()
