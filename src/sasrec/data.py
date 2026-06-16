"""Data pipeline for sequential recommendation.

Responsibilities:
  1. Download a raw dataset (MovieLens-1M by default).
  2. Preprocess: k-core filtering, contiguous re-indexing, chronological ordering.
  3. Leave-one-out split (train / valid / test) per the SASRec evaluation protocol.
  4. A torch Dataset that yields fixed-length, left-padded (seq, pos, neg) training triples.

Design notes
------------
* Item id 0 is RESERVED for padding everywhere. Real items are 1..num_items.
* We deliberately avoid pandas: the parsing is trivial and a lighter dependency
  footprint makes the repo easier to run.
* "k-core" filtering iteratively drops users and items with fewer than `min_count`
  interactions until the set is stable (dropping a rare item can make a user fall
  below the threshold, and vice-versa).
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

# Primary: the canonical GroupLens release (works on any machine with internet).
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
# Fallback: a Hugging Face mirror of the original ratings.dat (same `user::item::rating::ts`
# format). Used when the primary host is unreachable (e.g. a sandbox where only
# huggingface.co is allowlisted). md5/size match the original 1,000,209 ratings.
ML1M_HF_RATINGS = (
    "https://huggingface.co/datasets/nasserCha/movielens_rating_1m/resolve/main/ratings.dat"
)


# --------------------------------------------------------------------------- #
# Download + parse
# --------------------------------------------------------------------------- #
def _make_opener(proxy: str | None):
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener()


def download_ml1m(raw_dir: str, proxy: str | None = "http://fwdproxy:8080") -> str:
    """Download MovieLens-1M and return the path to ratings.dat.

    Tries the canonical GroupLens zip first; on failure falls back to a Hugging Face
    mirror of ratings.dat. `proxy` is needed on Meta devvms (external egress goes
    through fwdproxy); pass proxy=None on a machine with direct internet access.
    """
    os.makedirs(raw_dir, exist_ok=True)
    ratings_path = os.path.join(raw_dir, "ml-1m", "ratings.dat")
    if os.path.exists(ratings_path):
        return ratings_path

    opener = _make_opener(proxy)
    urllib.request.install_opener(opener)

    zip_path = os.path.join(raw_dir, "ml-1m.zip")
    try:
        print(f"Downloading {ML1M_URL} ...")
        urllib.request.urlretrieve(ML1M_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        return ratings_path
    except Exception as e:  # noqa: BLE001 - any network/zip failure -> try the mirror
        print(f"Primary source failed ({e}); falling back to Hugging Face mirror ...")
        os.makedirs(os.path.dirname(ratings_path), exist_ok=True)
        urllib.request.urlretrieve(ML1M_HF_RATINGS, ratings_path)
        return ratings_path


def parse_ml1m(ratings_path: str) -> list[tuple[int, int, int]]:
    """Parse ratings.dat -> list of (user, item, timestamp).

    Format: UserID::MovieID::Rating::Timestamp  (rating ignored = implicit feedback).
    """
    rows = []
    with open(ratings_path, "r", encoding="latin-1") as f:
        for line in f:
            u, i, _rating, ts = line.strip().split("::")
            rows.append((int(u), int(i), int(ts)))
    return rows


# --------------------------------------------------------------------------- #
# Preprocess
# --------------------------------------------------------------------------- #
def kcore_filter(
    rows: list[tuple[int, int, int]], min_count: int
) -> list[tuple[int, int, int]]:
    """Iteratively drop users/items with < min_count interactions until stable."""
    rows = list(rows)
    while True:
        user_cnt: dict[int, int] = defaultdict(int)
        item_cnt: dict[int, int] = defaultdict(int)
        for u, i, _ in rows:
            user_cnt[u] += 1
            item_cnt[i] += 1
        keep = [
            (u, i, ts)
            for (u, i, ts) in rows
            if user_cnt[u] >= min_count and item_cnt[i] >= min_count
        ]
        if len(keep) == len(rows):
            return keep
        rows = keep


def build_user_sequences(
    rows: list[tuple[int, int, int]], min_count: int = 5
) -> tuple[dict[int, list[int]], int, int]:
    """Full preprocessing.

    Returns
    -------
    user_seqs : dict[user_id -> [item_id, ...]]  chronologically ordered, ids re-indexed.
    num_users : int
    num_items : int   (real items are 1..num_items; 0 is padding)
    """
    rows = kcore_filter(rows, min_count)

    # Re-index users and items to contiguous ids starting at 1 (0 = padding).
    user_map: dict[int, int] = {}
    item_map: dict[int, int] = {}
    by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)  # u -> [(ts, item)]
    for u, i, ts in rows:
        if u not in user_map:
            user_map[u] = len(user_map) + 1
        if i not in item_map:
            item_map[i] = len(item_map) + 1
        by_user[user_map[u]].append((ts, item_map[i]))

    user_seqs: dict[int, list[int]] = {}
    for u, pairs in by_user.items():
        pairs.sort(key=lambda p: p[0])  # sort by timestamp (stable for ties)
        user_seqs[u] = [item for _ts, item in pairs]

    return user_seqs, len(user_map), len(item_map)


# --------------------------------------------------------------------------- #
# Leave-one-out split
# --------------------------------------------------------------------------- #
def leave_one_out_split(
    user_seqs: dict[int, list[int]],
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]]]:
    """Split each user's chronological sequence into train / valid / test.

    Protocol (SASRec, Section IV-A):
      * test[u]  = the single most recent action      -> seq[-1]
      * valid[u] = the second most recent action      -> seq[-2]
      * train[u] = everything before that             -> seq[:-2]

    Users with fewer than 3 interactions cannot form all three splits; for those,
    put the whole sequence in train and leave valid/test empty (they are skipped
    at eval time). After 5-core filtering on ML-1M this case is rare/absent.

    Returns
    -------
    (user_train, user_valid, user_test) : three dicts keyed by user id.
    """
    user_train: dict[int, list[int]] = {}
    user_valid: dict[int, list[int]] = {}
    user_test: dict[int, list[int]] = {}

    for u, seq in user_seqs.items():
        if len(seq) < 3:
            user_train[u] = seq
            user_valid[u] = []
            user_test[u] = []
        else:
            user_train[u] = seq[:-2]
            user_valid[u] = [seq[-2]]
            user_test[u] = [seq[-1]]

    return user_train, user_valid, user_test


# --------------------------------------------------------------------------- #
# Training Dataset
# --------------------------------------------------------------------------- #
def _random_neq(low: int, high: int, exclude: set[int], rng: np.random.Generator) -> int:
    """Sample an int in [low, high) not in `exclude`."""
    t = int(rng.integers(low, high))
    while t in exclude:
        t = int(rng.integers(low, high))
    return t


class SASRecTrainDataset(Dataset):
    """Yields fixed-length, left-padded (seq, pos, neg) triples for one user.

    For a user's training items ``ts``:
      * ``seq`` is the input  = ts[:-1], left-padded to ``max_len``.
      * ``pos`` is the target = ts[1:]  (the next item at each step), left-padded.
      * ``neg`` is one negative per valid (non-pad) position, sampled uniformly
        from items NOT in the user's full item set (fresh every epoch / __getitem__).

    Left-padding means short sequences sit at the right end of the window, so the
    most recent action is always at index ``max_len - 1``.
    """

    def __init__(
        self,
        user_train: dict[int, list[int]],
        num_items: int,
        max_len: int,
        seed: int = 0,
    ):
        # Keep only users with >= 2 training items (need at least one (input,target) pair).
        self.users = [u for u, s in user_train.items() if len(s) >= 2]
        self.user_train = user_train
        self.num_items = num_items
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        u = self.users[idx]
        ts = self.user_train[u]
        max_len = self.max_len

        seq = np.zeros(max_len, dtype=np.int64)
        pos = np.zeros(max_len, dtype=np.int64)
        neg = np.zeros(max_len, dtype=np.int64)
        interacted = set(ts)

        nxt = ts[-1]
        i = max_len - 1
        for item in reversed(ts[:-1]):
            seq[i] = item
            pos[i] = nxt
            if nxt != 0:
                neg[i] = _random_neq(1, self.num_items + 1, interacted, self.rng)
            nxt = item
            i -= 1
            if i == -1:
                break

        return (
            torch.from_numpy(seq),
            torch.from_numpy(pos),
            torch.from_numpy(neg),
        )


def left_pad_sequence(items: list[int], max_len: int) -> np.ndarray:
    """Left-pad/truncate a list of item ids to a fixed-length int64 array (for eval)."""
    seq = np.zeros(max_len, dtype=np.int64)
    if len(items) == 0:
        return seq
    items = items[-max_len:]
    seq[max_len - len(items):] = items
    return seq
