"""Training orchestration: data -> model -> train loop with early stopping on val NDCG."""

from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import SASRecConfig
from .data import (
    SASRecTrainDataset,
    build_user_sequences,
    download_ml1m,
    leave_one_out_split,
    parse_ml1m,
)
from .eval import evaluate
from .losses import sasrec_bce_loss
from .model import SASRec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(cfg: SASRecConfig, raw_dir: str = "data/raw"):
    """Download + preprocess + split. Returns (train, valid, test, num_users, num_items)."""
    if cfg.dataset != "ml-1m":
        raise ValueError(f"Only ml-1m wired up so far; got {cfg.dataset!r}")
    ratings_path = download_ml1m(raw_dir)
    rows = parse_ml1m(ratings_path)
    user_seqs, num_users, num_items = build_user_sequences(rows, cfg.min_count)
    user_train, user_valid, user_test = leave_one_out_split(user_seqs)
    return user_train, user_valid, user_test, num_users, num_items


def train(cfg: SASRecConfig, raw_dir: str = "data/raw", verbose: bool = True):
    """Full training run. Returns dict with best val/test metrics and the trained model."""
    set_seed(cfg.seed)
    device = cfg.device

    user_train, user_valid, user_test, num_users, num_items = load_dataset(cfg, raw_dir)
    if verbose:
        n_actions = sum(len(s) for s in user_train.values())
        print(f"users={num_users} items={num_items} train_actions={n_actions}")

    ds = SASRecTrainDataset(user_train, num_items, cfg.max_len, seed=cfg.seed)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    model = SASRec(num_items, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))

    best_val_ndcg = -1.0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        for seq, pos, neg in dl:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            pos_logits, neg_logits = model(seq, pos, neg)
            loss = sasrec_bce_loss(pos_logits, neg_logits, pos)
            if cfg.l2_emb > 0:
                loss = loss + cfg.l2_emb * model.item_emb.weight.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        epoch_time = time.time() - t0

        if epoch % cfg.eval_every == 0 or epoch == cfg.num_epochs:
            val_ndcg, val_hit = evaluate(
                model, user_train, user_valid, num_items, cfg.max_len,
                num_neg=cfg.num_neg_eval, topk=cfg.topk, seed=cfg.seed, device=device,
            )
            if verbose:
                print(
                    f"epoch {epoch:3d} | loss {running / max(nb,1):.4f} "
                    f"| {epoch_time:.1f}s | val NDCG@{cfg.topk} {val_ndcg:.4f} "
                    f"Hit@{cfg.topk} {val_hit:.4f}"
                )
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= cfg.patience:
                    if verbose:
                        print(f"early stop at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test: input includes train + validation action (paper protocol).
    test_ndcg, test_hit = evaluate(
        model, user_train, user_test, num_items, cfg.max_len,
        extra_context=user_valid, num_neg=cfg.num_neg_eval, topk=cfg.topk,
        seed=cfg.seed, device=device,
    )
    if verbose:
        print(f"TEST NDCG@{cfg.topk} {test_ndcg:.4f} | Hit@{cfg.topk} {test_hit:.4f}")

    return {
        "model": model,
        "best_val_ndcg": best_val_ndcg,
        "test_ndcg": test_ndcg,
        "test_hit": test_hit,
        "num_items": num_items,
        "num_users": num_users,
    }
