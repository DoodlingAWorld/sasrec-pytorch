"""Tests for ranking metrics. test_ndcg_* also specs the EXERCISE function ndcg_at_k."""

import math

from sasrec.eval import hit_at_k, ndcg_at_k


def test_hit_at_k():
    assert hit_at_k(rank=0, k=10) == 1.0
    assert hit_at_k(rank=9, k=10) == 1.0
    assert hit_at_k(rank=10, k=10) == 0.0   # 0-based rank 10 is the 11th item
    assert hit_at_k(rank=100, k=10) == 0.0


def test_ndcg_at_k_known_values():
    # rank 0 (top of the list): perfect -> 1 / log2(2) = 1.0
    assert math.isclose(ndcg_at_k(rank=0, k=10), 1.0)
    # rank 1 (2nd position): 1 / log2(3)
    assert math.isclose(ndcg_at_k(rank=1, k=10), 1.0 / math.log2(3))
    # rank 2 (3rd position): 1 / log2(4) = 0.5
    assert math.isclose(ndcg_at_k(rank=2, k=10), 0.5)


def test_ndcg_at_k_outside_topk_is_zero():
    assert ndcg_at_k(rank=10, k=10) == 0.0
    assert ndcg_at_k(rank=50, k=10) == 0.0


def test_ndcg_monotonically_decreases_with_rank():
    vals = [ndcg_at_k(r, k=10) for r in range(10)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
