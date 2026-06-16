#!/usr/bin/env python3
"""Download + preprocess a dataset and print its statistics (Table II in the paper)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sasrec.data import build_user_sequences, download_ml1m, leave_one_out_split, parse_ml1m  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--min-count", type=int, default=5)
    args = p.parse_args()

    path = download_ml1m(args.raw_dir)
    rows = parse_ml1m(path)
    seqs, num_users, num_items = build_user_sequences(rows, args.min_count)
    train, valid, test = leave_one_out_split(seqs)

    n_actions = sum(len(s) for s in seqs.values())
    print("MovieLens-1M (after preprocessing)")
    print(f"  users         : {num_users}")
    print(f"  items         : {num_items}")
    print(f"  actions       : {n_actions:,}")
    print(f"  actions/user  : {n_actions / num_users:.1f}")
    print(f"  actions/item  : {n_actions / num_items:.1f}")
    print(f"  train users   : {sum(1 for s in train.values() if len(s) >= 2)}")
    print(f"  eval users    : {sum(1 for s in test.values() if len(s) >= 1)}")


if __name__ == "__main__":
    main()
