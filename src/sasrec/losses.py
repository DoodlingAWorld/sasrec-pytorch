"""Training objective: binary cross-entropy over (positive, negative) at each valid step."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sasrec_bce_loss(
    pos_logits: torch.Tensor, neg_logits: torch.Tensor, pos_targets: torch.Tensor
) -> torch.Tensor:
    """Binary cross-entropy on valid (non-pad) positions.

    Each valid step contributes ``-log σ(pos) - log(1 - σ(neg))``. Positions whose
    target is the padding id (0) are ignored.

    Parameters
    ----------
    pos_logits, neg_logits : [B, L]  scores for the true next item and a sampled negative.
    pos_targets : [B, L]  the positive target ids (used only to find valid positions).
    """
    valid = pos_targets != 0  # [B, L] bool
    pos_l = pos_logits[valid]
    neg_l = neg_logits[valid]
    loss = F.binary_cross_entropy_with_logits(
        pos_l, torch.ones_like(pos_l)
    ) + F.binary_cross_entropy_with_logits(neg_l, torch.zeros_like(neg_l))
    return loss
