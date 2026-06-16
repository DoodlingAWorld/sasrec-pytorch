"""Tests for model correctness: shapes, causality (no future leak), tied embedding, masking."""

import torch

from sasrec.config import SASRecConfig
from sasrec.model import SASRec


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = SASRecConfig(max_len=10, hidden_dim=16, num_blocks=2, dropout=0.0)
    model = SASRec(num_items=50, cfg=cfg).eval()  # eval() -> dropout off, deterministic
    return model, cfg


def test_forward_shapes():
    model, cfg = _tiny_model()
    B, L = 4, cfg.max_len
    seq = torch.randint(1, 51, (B, L))
    pos = torch.randint(1, 51, (B, L))
    neg = torch.randint(1, 51, (B, L))
    pos_logits, neg_logits = model(seq, pos, neg)
    assert pos_logits.shape == (B, L)
    assert neg_logits.shape == (B, L)


def test_predict_shapes():
    model, cfg = _tiny_model()
    B, L, C = 3, cfg.max_len, 7
    seq = torch.randint(1, 51, (B, L))
    cands = torch.randint(1, 51, (B, C))
    scores = model.predict(seq, cands)
    assert scores.shape == (B, C)


def test_causality_no_future_leak():
    """Changing item at position t must NOT change the representation at positions < t."""
    model, cfg = _tiny_model()
    L = cfg.max_len
    seq = torch.randint(1, 51, (1, L))
    feats1 = model.seq_repr(seq)

    seq2 = seq.clone()
    # change the LAST item to something different
    seq2[0, -1] = (seq2[0, -1] % 50) + 1
    if seq2[0, -1] == seq[0, -1]:
        seq2[0, -1] = (seq2[0, -1] % 50) + 1
    feats2 = model.seq_repr(seq2)

    # all positions except the last must be identical (no information from the future)
    assert torch.allclose(feats1[:, :-1, :], feats2[:, :-1, :], atol=1e-6)
    # the last position is allowed to (and should) change
    assert not torch.allclose(feats1[:, -1, :], feats2[:, -1, :], atol=1e-6)


def test_padding_row_is_zero():
    model, _ = _tiny_model()
    assert torch.allclose(model.item_emb.weight[0], torch.zeros_like(model.item_emb.weight[0]))


def test_prediction_uses_tied_item_embedding():
    """predict() must score candidates with the SAME item embedding used for input."""
    model, cfg = _tiny_model()
    L = cfg.max_len
    seq = torch.randint(1, 51, (1, L))
    cands = torch.tensor([[3, 7, 11]])
    scores = model.predict(seq, cands)

    # reproduce the score manually from item_emb -> confirms weight tying
    feats = model.seq_repr(seq)
    last = feats[:, -1, :]
    manual = (model.item_emb(cands) * last.unsqueeze(1)).sum(-1)
    assert torch.allclose(scores, manual, atol=1e-6)
