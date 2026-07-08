from pathlib import Path

import numpy as np
import pytest

from channel_estimation.dataset import (
    EXPECTED_SPLIT_KEYS,
    generate_synthetic_dataset,
    load_npz_dataset,
    split_dataset,
)


def test_load_dataset_missing_path():
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_npz_dataset(Path("does-not-exist.npz"))


def test_load_dataset_round_trip(tmp_path):
    path = tmp_path / "sample.npz"
    arrays = {key: np.zeros((2, 3, 2), dtype=np.float32) for key in EXPECTED_SPLIT_KEYS}
    np.savez(path, **arrays)
    loaded = load_npz_dataset(path)
    assert set(loaded) == set(EXPECTED_SPLIT_KEYS)
    assert loaded["x_train"].shape == (2, 3, 2)


def test_generate_synthetic_dataset_shapes():
    dataset = generate_synthetic_dataset(num_samples=20, num_pilots=8)
    assert dataset["x_train"].shape == (14, 8, 2)
    assert dataset["x_val"].shape == (3, 8, 2)
    assert dataset["x_test"].shape == (3, 8, 2)
    assert dataset["x_train"].dtype == np.float32


def test_load_dataset_accepts_ofdm_grid_shape(tmp_path):
    path = tmp_path / "grid.npz"
    arrays = {
        key: np.zeros((2, 14, 16, 2), dtype=np.float32) for key in EXPECTED_SPLIT_KEYS
    }
    np.savez(path, **arrays)
    loaded = load_npz_dataset(path)
    assert loaded["x_train"].shape == (2, 14, 16, 2)


def test_load_dataset_rejects_mismatched_split_shapes(tmp_path):
    path = tmp_path / "malformed.npz"
    arrays = {key: np.zeros((2, 3, 2), dtype=np.float32) for key in EXPECTED_SPLIT_KEYS}
    arrays["y_val"] = np.zeros((2, 4, 2), dtype=np.float32)
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="validation split"):
        load_npz_dataset(path)


def test_load_dataset_rejects_inconsistent_pilot_dimensions(tmp_path):
    path = tmp_path / "inconsistent.npz"
    arrays = {key: np.zeros((2, 3, 2), dtype=np.float32) for key in EXPECTED_SPLIT_KEYS}
    arrays["x_test"] = np.zeros((2, 4, 2), dtype=np.float32)
    arrays["y_test"] = np.zeros((2, 4, 2), dtype=np.float32)
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="same pilot dimensions"):
        load_npz_dataset(path)


def test_split_dataset_rejects_empty_partition():
    values = np.zeros((3, 4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="non-empty"):
        split_dataset(values, values)
