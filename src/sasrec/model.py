"""The SASRec model: embeddings + stacked self-attention blocks + tied-embedding scoring."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import SASRecConfig
from .modules import SASRecBlock, LN_EPS


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

    The same item embedding table is used both to encode the input sequence and to
    score candidate items (weight tying) — the paper's ablation shows this is important.
    """

    def __init__(self, num_items: int, cfg: SASRecConfig):
        super().__init__()
        self.num_items = num_items
        self.max_len = cfg.max_len
        d = cfg.hidden_dim

        # +1 row for the padding id (0); padding_idx keeps its embedding fixed at zero.
        self.item_emb = nn.Embedding(num_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, d)  # learned absolute positions
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [SASRecBlock(d, cfg.dropout) for _ in range(cfg.num_blocks)]
        )
        self.last_norm = nn.LayerNorm(d, eps=LN_EPS)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        # keep the padding row at exactly zero
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)

    # ------------------------------------------------------------------ #
    def seq_repr(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode a batch of (left-padded) item-id sequences -> [B, L, D] features."""
        B, L = seq.shape
        pad_mask = seq != 0  # [B, L] bool

        x = self.item_emb(seq) * (self.item_emb.embedding_dim ** 0.5)  # scale (ref. detail)
        positions = torch.arange(L, device=seq.device)
        x = x + self.pos_emb(positions).unsqueeze(0)
        x = self.emb_dropout(x)
        x = x * pad_mask.unsqueeze(-1)

        for block in self.blocks:
            x = block(x, pad_mask)
        return self.last_norm(x)  # [B, L, D]

    # ------------------------------------------------------------------ #
    def forward(
        self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward. Returns (pos_logits, neg_logits), each [B, L]."""
        feats = self.seq_repr(seq)              # [B, L, D]
        pos_e = self.item_emb(pos)              # [B, L, D]
        neg_e = self.item_emb(neg)              # [B, L, D]
        pos_logits = (feats * pos_e).sum(dim=-1)
        neg_logits = (feats * neg_e).sum(dim=-1)
        return pos_logits, neg_logits

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Score candidate items for the *next* step. seq:[B,L], candidates:[B,C] -> [B,C]."""
        feats = self.seq_repr(seq)             # [B, L, D]
        last = feats[:, -1, :]                 # rep after the most recent (rightmost) item
        cand_e = self.item_emb(candidates)     # [B, C, D]
        return (cand_e * last.unsqueeze(1)).sum(dim=-1)  # [B, C]
