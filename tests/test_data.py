"""Tests for the data pipeline. test_leave_one_out_* also specs the EXERCISE function."""

import numpy as np

from sasrec.data import (
    build_user_sequences,
    kcore_filter,
    leave_one_out_split,
    left_pad_sequence,
)


def test_kcore_filter_drops_rare_users_and_items():
    # items 10,11,12 each appear 2x (survive). item 99 appears once -> dropped.
    # user 3 has a single action -> dropped (dropping it leaves item 10 at 2, still ok).
    rows = [
        (1, 10, 1), (1, 11, 2), (1, 12, 3), (1, 99, 4),
        (2, 10, 5), (2, 11, 6), (2, 12, 7),
        (3, 10, 8),
    ]
    kept = kcore_filter(rows, min_count=2)
    items = {i for _u, i, _ts in kept}
    users = {u for u, _i, _ts in kept}
    assert 99 not in items          # rare item removed
    assert 3 not in users           # rare user removed
    assert users == {1, 2}
    assert items == {10, 11, 12}


def test_build_user_sequences_reindexes_and_orders_by_time():
    rows = [
        (7, 100, 30), (7, 200, 10), (7, 300, 20),  # out of time order
        (8, 100, 5), (8, 300, 6),
    ]
    seqs, num_users, num_items = build_user_sequences(rows, min_count=1)  # no filtering here
    # ids are contiguous starting at 1 (0 reserved for padding)
    assert num_users == 2
    assert num_items == 3
    all_items = {i for s in seqs.values() for i in s}
    assert all_items == {1, 2, 3}
    # each user's sequence is chronologically ordered (item ids reflect first-seen order:
    # 100->1, 200->2, 300->3); user 7 by time = [200,300,100] = [2,3,1]
    u7 = [u for u in seqs if seqs[u] == [2, 3, 1]]
    assert len(u7) == 1


def test_leave_one_out_split_basic():
    user_seqs = {1: [5, 6, 7, 8], 2: [9, 10, 11]}
    train, valid, test = leave_one_out_split(user_seqs)
    assert train[1] == [5, 6] and valid[1] == [7] and test[1] == [8]
    assert train[2] == [9] and valid[2] == [10] and test[2] == [11]


def test_leave_one_out_split_short_user_goes_to_train():
    user_seqs = {1: [5, 6]}  # < 3 interactions: can't form valid+test
    train, valid, test = leave_one_out_split(user_seqs)
    assert train[1] == [5, 6]
    assert valid[1] == [] and test[1] == []


def test_left_pad_sequence_left_pads_and_truncates():
    assert list(left_pad_sequence([1, 2, 3], 5)) == [0, 0, 1, 2, 3]
    assert list(left_pad_sequence([1, 2, 3, 4, 5, 6], 3)) == [4, 5, 6]  # keep most recent
    assert list(left_pad_sequence([], 3)) == [0, 0, 0]
