# Hands-on exercises

Three core functions in this repo are left **unimplemented on purpose** (they raise
`NotImplementedError`). Implementing them yourself is the fastest way to actually learn the
paper. They are ordered easy to harder. For each one, the test file is the spec: make it pass.

Run a single test file in a tight loop while you iterate:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_metrics.py -q   # etc.
```

The whole pipeline (training, evaluation) only works once all three are implemented.

## Exercise 1: `ndcg_at_k` (easiest)

* File: `src/sasrec/eval.py`  ·  Test: `tests/test_metrics.py`

Normalized Discounted Cumulative Gain at k, for the leave-one-out setting where there is
exactly one relevant item. With a single relevant item the ideal DCG is 1, so:

> NDCG@k = 1 / log2(rank + 2) if the item is within the top-k (0-based `rank < k`), else 0.

Why `rank + 2`? The standard DCG discount is `log2(position + 1)` with 1-based positions.
`log2(1) = 0` would divide by zero for the top spot, so for a 0-based rank it becomes `log2(rank + 2)`.

Goal: understand why NDCG rewards ranking the right item higher, and how it differs from Hit@k
(which only checks whether the item is in the top-k at all).

## Exercise 2: `leave_one_out_split` (easy)

* File: `src/sasrec/data.py`  ·  Test: `tests/test_data.py`

Split each user's chronological sequence into train, valid, and test: `test = seq[-1]`,
`valid = seq[-2]`, `train = seq[:-2]`. Users with fewer than 3 interactions cannot form all
three splits, so put the whole sequence in train and leave valid and test empty.

Goal: understand the leave-one-out evaluation protocol used across sequential recommendation,
and why the split must respect chronological order (no peeking at the future). This is the most
common place where sloppy evaluation quietly inflates results.

## Exercise 3: `PointWiseFFN` (medium)

* File: `src/sasrec/modules.py`  ·  Test: `tests/test_model.py`

A position-wise two-layer feed-forward network: `Linear(d, d)`, ReLU, dropout, `Linear(d, d)`,
dropout, applied identically at every time step. It runs after the self-attention sublayer in
each block.

Goal: understand why a Transformer needs the FFN at all. Self-attention is a weighted average
of value vectors, which is a linear operation. The FFN injects the non-linearity and lets the
model mix latent dimensions. `test_model.py` exercises it through the full block, so a wrong FFN
breaks `test_forward_shapes` and `test_causality_no_future_leak`.

## Stretch (in a separate repo, later)

The trap-heavy module is `CausalSelfAttention`, where most Transformer bugs live (the causal and
padding masks). After the three exercises above, the extensions listed in the README (longer
sequences, BERT4Rec, a new dataset) are where the real portfolio differentiation happens.
