"""Smoke check: the image-level train/val split is disjoint, deterministic,
and its union recovers the full row range — a leak here would silently
undermine the val-selection protocol in notebooks/03."""

import torch

from symbolic.dataset import (
    _image_level_split_row_indices,
    _selected_indices_from_metadata,
)


def _fake_records(num_images: int = 10) -> list[dict]:
    records = []
    row = 0
    for i in range(num_images):
        num_rois = 50 + i * 7
        records.append({"row_start": row, "row_stop": row + num_rois})
        row += num_rois
    return records


def test_train_val_rows_are_disjoint():
    records = _fake_records()
    train_rows = _image_level_split_row_indices(records, "train", val_fraction=0.2)
    val_rows = _image_level_split_row_indices(records, "val", val_fraction=0.2)
    assert set(train_rows.tolist()).isdisjoint(set(val_rows.tolist()))


def test_train_val_union_is_full_row_range():
    records = _fake_records()
    total_rows = records[-1]["row_stop"]
    train_rows = _image_level_split_row_indices(records, "train", val_fraction=0.2)
    val_rows = _image_level_split_row_indices(records, "val", val_fraction=0.2)
    union = sorted(train_rows.tolist() + val_rows.tolist())
    assert union == list(range(total_rows))


def test_split_is_deterministic_for_a_fixed_seed():
    records = _fake_records()
    first = _image_level_split_row_indices(records, "train", split_seed=7)
    second = _image_level_split_row_indices(records, "train", split_seed=7)
    assert torch.equal(first, second)


def test_different_seeds_can_change_the_partition():
    records = _fake_records(num_images=20)
    a = _image_level_split_row_indices(records, "val", split_seed=0)
    b = _image_level_split_row_indices(records, "val", split_seed=1)
    assert a.tolist() != b.tolist()


def test_invalid_split_name_raises():
    records = _fake_records()
    try:
        _image_level_split_row_indices(records, "test")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown split name")


def test_restrict_to_never_leaks_rows_outside_the_split():
    records = _fake_records()
    val_rows = _image_level_split_row_indices(records, "val", val_fraction=0.2)
    total_rows = records[-1]["row_stop"]
    metadata = {"teacher_labels": torch.zeros(total_rows, dtype=torch.int64)}

    # No neg_ratio: restrict_to alone must reproduce exactly the split rows.
    kept_all = _selected_indices_from_metadata(
        metadata, random_state=42, neg_ratio=None, restrict_to=val_rows
    )
    assert torch.equal(torch.sort(kept_all).values, torch.sort(val_rows).values)

    # With neg_ratio, the selection must still be a subset of the split rows.
    kept_subset = _selected_indices_from_metadata(
        metadata, random_state=42, neg_ratio=1.0, restrict_to=val_rows
    )
    assert set(kept_subset.tolist()).issubset(set(val_rows.tolist()))


if __name__ == "__main__":
    test_train_val_rows_are_disjoint()
    test_train_val_union_is_full_row_range()
    test_split_is_deterministic_for_a_fixed_seed()
    test_different_seeds_can_change_the_partition()
    test_invalid_split_name_raises()
    test_restrict_to_never_leaks_rows_outside_the_split()
    print("OK")
