"""Core building blocks of SASRec: causal self-attention, point-wise FFN, and the block.

Reproduction subtleties (these differ from a naive reading of the paper, and match the
authors' released code — they matter for hitting the reported numbers):

  * Residual adds the **LayerNorm'd** input, not the raw input. The paper text writes
    ``g(x) = x + Dropout(g(LayerNorm(x)))``, but the released code computes
    ``y = LayerNorm(x); out = y + sublayer(y)``. We follow the code.
  * Dropout lives *inside* the sublayers (on attention weights and inside the FFN),
    plus a dropout on the input embedding (in model.py).
  * Pad positions are explicitly zeroed after every block.
  * Single-head attention (no multi-head split) — the paper found multi-head slightly
    worse for the small d used in recommendation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LN_EPS = 1e-8  # matches the authors' layer-norm epsilon


class CausalSelfAttention(nn.Module):
    """Single-head, causal, pad-aware scaled dot-product self-attention.

    Position i may attend to positions j <= i (causal) and only to non-pad keys.
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = dim ** -0.5  # 1/sqrt(d): keeps logits well-scaled for softmax

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] ; pad_mask: [B, L] bool (True = real token). Returns [B, L, D]."""
        B, L, D = x.shape
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        logits = (Q @ K.transpose(1, 2)) * self.scale            # [B, L, L]

        # causal: lower-triangular allows j <= i
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        # key padding: cannot attend to pad keys
        allowed = causal.unsqueeze(0) & pad_mask.unsqueeze(1)    # [B, L, L]
        logits = logits.masked_fill(~allowed, float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        # A query that is itself a pad (or has no valid keys) yields an all -inf row ->
        # softmax = nan. Those positions get zeroed by the block's pad mask anyway, but
        # we scrub the nans here so they don't poison gradients.
        attn = torch.nan_to_num(attn)
        attn = self.dropout(attn)
        return attn @ V                                          # [B, L, D]


class PointWiseFFN(nn.Module):
    """Position-wise two-layer feed-forward network (ReLU), applied identically per step.

    Endows the otherwise-linear attention output with non-linearity and lets it mix
    information across latent dimensions. ``d -> d`` (no expansion), matching the paper.
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.fc2(self.dropout1(self.relu(self.fc1(x)))))


class SASRecBlock(nn.Module):
    """One self-attention block: pre-LN attention + pre-LN FFN, each with a residual."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.attn = CausalSelfAttention(dim, dropout)
        self.ffn_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.ffn = PointWiseFFN(dim, dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        y = self.attn_norm(x)
        x = y + self.attn(y, pad_mask)          # residual on the normalized input
        z = self.ffn_norm(x)
        x = z + self.ffn(z)                      # residual on the normalized input
        x = x * pad_mask.unsqueeze(-1)           # zero out pad positions
        return x
