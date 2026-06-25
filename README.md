# SASRec-PyTorch: Self-Attentive Sequential Recommendation, reproduced

A clean, from-scratch **PyTorch** reproduction of
[**SASRec** (Kang & McAuley, ICDM 2018)](https://arxiv.org/abs/1808.09781). SASRec was the
first model to bring the causal self-attention Transformer to sequential recommendation, and
it is the ancestor of today's industrial sequence models (Pinterest's PinnerFormer, Google's
sequential recommenders, Meta's generative recommenders).

This repo has two goals. First, faithfully reproduce the paper's MovieLens-1M results. Second,
serve as a foundation for extensions, which live in their own repos (see [Related repos](#related-repos)).

## What problem does SASRec solve?

Given a user's chronological sequence of interactions `(s1, s2, ..., st)`, predict the next
item `st+1`. Earlier approaches sat at two extremes. Markov Chains look only at the last item:
strong on sparse data, but blind to long-range context. RNNs summarize all history in a hidden
state: expressive, but data-hungry and slow to train. SASRec gets the best of both. Self-attention
draws context from the entire history (like an RNN) while adaptively focusing on the few items
that actually matter for the next prediction (like a Markov Chain), and it trains an order of
magnitude faster because attention parallelizes.

## Architecture

```
items ->  Embedding (item + learned positional)  ->  dropout
      ->  N x [ causal self-attention  +  point-wise FFN ]   (residual + LayerNorm + dropout)
      ->  final LayerNorm
score ->  dot product of the last position's representation with the (tied) item embedding
```

* **Causal masking:** position `i` attends only to positions `<= i`, so it never sees the future.
* **Tied embeddings:** one item-embedding table both encodes the input and scores candidates.
* **Loss:** binary cross-entropy with one sampled negative per step (Adam optimizer).

## Results (MovieLens-1M, 100 sampled negatives)

| Source | NDCG@10 | Hit@10 |
|---|---|---|
| Paper (Table III) | 0.5905 | 0.8245 |
| **This repo** (seed 42) | **0.5925** | **0.8222** |

The reproduction lands right on the paper's numbers (NDCG@10 a touch higher, Hit@10 a touch
lower). The preprocessed dataset statistics (6,040 users, 3,416 items, ~1.0M actions) match the
paper's Table II exactly, and the model trains on CPU in about an hour at roughly 14 seconds per
epoch (300 epochs, default config n=200, d=50, b=2).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel
pip install -r requirements.txt

python scripts/prepare_data.py          # download + preprocess, print Table II stats
python scripts/train.py --fast-dev      # smoke test in minutes
python scripts/train.py                 # full reproduction (n=200, d=50, b=2)
python scripts/evaluate.py experiments/runs/ml1m_seed42.pt
pytest                                   # correctness tests
```

On a Meta devvm, external hosts route through the forward proxy
(`pip install --proxy http://fwdproxy:8080 ...`); the dataset downloader uses it automatically.

## Repository layout

```
src/sasrec/
  data.py      download + k-core filter + chronological order + leave-one-out + Dataset
  modules.py   CausalSelfAttention, PointWiseFFN, SASRecBlock
  model.py     SASRec: embeddings + stacked blocks + tied-embedding scoring
  losses.py    BCE with per-step negative sampling
  eval.py      Hit@K / NDCG@K against sampled negatives
  train.py     training loop with early stopping on validation NDCG
  config.py    one dataclass for every hyperparameter (plus a fast-dev preset)
scripts/       prepare_data.py, train.py, evaluate.py
tests/         data, model (causality, tied embedding), metrics
```

## Reproduction notes

These follow the authors' released code, which differs slightly from the paper text. The
differences are what let a reproduction actually hit the numbers:

* The residual adds the LayerNorm'd input, not the raw input (`y = LN(x); out = y + sublayer(y)`).
* Item embeddings are scaled by `sqrt(d)` before adding positional embeddings.
* Attention is single-head (multi-head was slightly worse for the small `d` used here).
* Pad positions are zeroed after every block, and pad keys are masked in attention.
* Negatives are sampled per-position, fresh each epoch, excluding the user's own items.

## Related repos

This repo is the faithful reproduction of SASRec. Extensions live in their own standalone repos:

* **`sasrec-longseq`** (planned): efficiency and longer-sequence scaling (Table V), vectorized
  negative sampling, sampled-vs-full-softmax loss, throughput profiling.
* **`bert4rec-pytorch`** (planned): a bidirectional, masked ("cloze") variant, compared to SASRec.
* **`sasrec-newdomain`** (planned): applying the model to a dataset outside the paper.

## Citation

```bibtex
@inproceedings{kang2018sasrec,
  title={Self-Attentive Sequential Recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={ICDM},
  year={2018}
}
```

Reference implementations consulted for correctness (not copied): the authors'
[kang205/SASRec](https://github.com/kang205/SASRec) (TensorFlow) and
[pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch).
