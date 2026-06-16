"""Evaluation: leave-one-out ranking against sampled negatives (Hit@K, NDCG@K)."""

from __future__ import annotations

import math

import numpy as np
import torch

from .data import left_pad_sequence


def hit_at_k(rank: int, k: int) -> float:
    """1.0 if the single relevant item lands in the top-k (0-based rank < k)."""
    return 1.0 if rank < k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    """NDCG@k for a single relevant item at 0-based position ``rank``.

    With one relevant item the ideal DCG is 1, so NDCG = DCG = 1 / log2(rank + 2)
    if the item is within the top-k, else 0.
    """
    if rank >= k:
        return 0.0

    return 1.0 / math.log2(rank + 2)


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #
def _sample_negatives(
    exclude: set[int], num_items: int, n: int, rng: np.random.Generator
) -> list[int]:
    negs: list[int] = []
    seen = set(exclude)
    while len(negs) < n:
        c = int(rng.integers(1, num_items + 1))
        if c not in seen:
            seen.add(c)
            negs.append(c)
    return negs


@torch.no_grad()
def evaluate(
    model,
    user_train: dict[int, list[int]],
    user_target: dict[int, list[int]],
    num_items: int,
    max_len: int,
    *,
    extra_context: dict[int, list[int]] | None = None,
    num_neg: int = 100,
    topk: int = 10,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[float, float]:
    """Compute (NDCG@k, Hit@k) over all users that have a target.

    Parameters
    ----------
    user_train     : history used as model input.
    user_target    : the held-out item per user (valid or test).
    extra_context  : optional items appended to the input *after* train (e.g. the
                     validation item is part of the input when evaluating on test).
    """
    model.eval()
    rng = np.random.default_rng(seed)

    users = [u for u, t in user_target.items() if len(t) > 0 and len(user_train.get(u, [])) > 0]

    seqs, cands, exclude_sizes = [], [], []
    for u in users:
        hist = list(user_train[u])
        if extra_context is not None:
            hist = hist + list(extra_context.get(u, []))
        seqs.append(left_pad_sequence(hist, max_len))

        target = user_target[u][0]
        exclude = set(hist) | {target}
        negs = _sample_negatives(exclude, num_items, num_neg, rng)
        cands.append([target] + negs)  # index 0 is the ground-truth item

    seqs_t = torch.from_numpy(np.stack(seqs)).to(device)
    cands_t = torch.tensor(cands, dtype=torch.long, device=device)

    ndcg_sum = 0.0
    hit_sum = 0.0
    n = len(users)
    for start in range(0, n, batch_size):
        sb = seqs_t[start:start + batch_size]
        cb = cands_t[start:start + batch_size]
        scores = model.predict(sb, cb)                 # [b, 1+num_neg]
        # rank of the true item (column 0): how many candidates score strictly higher.
        ranks = (scores > scores[:, :1]).sum(dim=1).cpu().numpy()
        for r in ranks:
            ndcg_sum += ndcg_at_k(int(r), topk)
            hit_sum += hit_at_k(int(r), topk)

    return ndcg_sum / n, hit_sum / n
