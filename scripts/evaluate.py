#!/usr/bin/env python3
"""Load a trained SASRec checkpoint and report test Hit@10 / NDCG@10."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sasrec.config import SASRecConfig  # noqa: E402
from sasrec.eval import evaluate  # noqa: E402
from sasrec.model import SASRec  # noqa: E402
from sasrec.train import load_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="path to a .pt state_dict saved by scripts/train.py")
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=50)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = SASRecConfig(
        max_len=args.max_len, hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks, seed=args.seed,
    )
    train_d, valid_d, test_d, _, num_items = load_dataset(cfg)

    model = SASRec(num_items, cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    ndcg, hit = evaluate(
        model, train_d, test_d, num_items, cfg.max_len,
        extra_context=valid_d, num_neg=cfg.num_neg_eval, topk=cfg.topk, seed=cfg.seed,
    )
    print(f"TEST  NDCG@{cfg.topk}={ndcg:.4f}  Hit@{cfg.topk}={hit:.4f}")


if __name__ == "__main__":
    main()
