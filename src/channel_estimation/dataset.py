"""Dataset loading, inspection, splitting, and synthetic generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .sionna_ofdm import GridSpec

from .ofdm import (
    generate_pilot_symbols,
    generate_rayleigh_channel,
    simulate_pilot_observations,
)

EXPECTED_SPLIT_KEYS = (
    "x_train",
    "y_train",
    "x_val",
    "y_val",
    "x_test",
    "y_test",
)


def load_npz_dataset(
    path: str | Path,
    *,
    required_keys: tuple[str, ...] | None = EXPECTED_SPLIT_KEYS,
) -> dict[str, NDArray[np.generic]]:
    """Load an NPZ dataset into memory and optionally validate required keys."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as archive:
        if required_keys:
            missing = sorted(set(required_keys) - set(archive.files))
            if missing:
                raise KeyError(f"Dataset is missing required arrays: {missing}")
        dataset = {key: archive[key] for key in archive.files}

    if required_keys == EXPECTED_SPLIT_KEYS:
        validate_split_dataset(dataset)
    return dataset


def validate_split_dataset(dataset: Mapping[str, ArrayLike]) -> None:
    """Validate shapes for the standard train, validation, and test arrays.

    both the original flat pilot representation ``(samples, pilots, 2)`` and the
    ofdm resource-grid representation ``(samples, ofdm-symbols, subcarriers, 2)``
    are accepted; the only fixed requirement is a trailing real/imaginary axis of
    length two and consistent feature dimensions across all splits.
    """
    expected_feature_shape: tuple[int, ...] | None = None
    for split, label in (
        ("train", "training"),
        ("val", "validation"),
        ("test", "test"),
    ):
        inputs = np.asarray(dataset[f"x_{split}"])
        targets = np.asarray(dataset[f"y_{split}"])
        if inputs.shape != targets.shape:
            raise ValueError(f"Input and target shapes differ for the {label} split.")
        if inputs.ndim not in (3, 4) or inputs.shape[-1] != 2:
            raise ValueError(
                f"The {label} split must end in a real/imaginary axis of length 2 "
                "and be a flat pilot or ofdm-grid representation."
            )
        if inputs.shape[0] == 0:
            raise ValueError(f"The {label} split must not be empty.")
        if not np.issubdtype(inputs.dtype, np.number) or not np.issubdtype(
            targets.dtype, np.number
        ):
            raise ValueError(f"The {label} split must contain numeric arrays.")

        feature_shape = inputs.shape[1:]
        if expected_feature_shape is None:
            expected_feature_shape = feature_shape
        elif feature_shape != expected_feature_shape:
            raise ValueError("All dataset splits must use the same pilot dimensions.")


def inspect_dataset(
    dataset: Mapping[str, ArrayLike],
) -> dict[str, dict[str, object]]:
    """Return serializable shape and dtype metadata for dataset arrays."""
    return {
        key: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
        }
        for key, value in dataset.items()
    }


def complex_to_features(values: ArrayLike) -> NDArray[np.floating]:
    """Represent complex values with a final real/imaginary feature axis."""
    array = np.asarray(values)
    if not np.iscomplexobj(array):
        raise ValueError("complex_to_features expects a complex-valued array.")
    return np.stack((array.real, array.imag), axis=-1)


def features_to_complex(values: ArrayLike) -> NDArray[np.complexfloating]:
    """Convert a final ``[..., 2]`` real/imaginary axis to complex values."""
    array = np.asarray(values)
    if array.shape[-1:] != (2,):
        raise ValueError("Expected a final feature axis of length 2.")
    return array[..., 0] + 1j * array[..., 1]


def split_dataset(
    inputs: ArrayLike,
    targets: ArrayLike,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> dict[str, NDArray[np.generic]]:
    """Split aligned arrays deterministically without shuffling."""
    x = np.asarray(inputs)
    y = np.asarray(targets)
    if len(x) != len(y):
        raise ValueError("inputs and targets must contain the same sample count.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test split.")

    train_end = int(len(x) * train_fraction)
    validation_end = train_end + int(len(x) * validation_fraction)
    if train_end == 0 or validation_end == train_end or validation_end == len(x):
        raise ValueError(
            "Split fractions must produce non-empty train, validation, and test sets."
        )
    return {
        "x_train": x[:train_end],
        "y_train": y[:train_end],
        "x_val": x[train_end:validation_end],
        "y_val": y[train_end:validation_end],
        "x_test": x[validation_end:],
        "y_test": y[validation_end:],
    }


def generate_synthetic_dataset(
    *,
    num_samples: int = 5000,
    num_pilots: int = 64,
    snr_db: float = 10.0,
    random_seed: int = 42,
    dtype: np.dtype = np.float32,
) -> dict[str, NDArray[np.generic]]:
    """Regenerate the notebook's Rayleigh/AWGN channel-estimation dataset."""
    if num_samples <= 0 or num_pilots <= 0:
        raise ValueError("num_samples and num_pilots must be positive.")

    rng = np.random.default_rng(random_seed)
    channels = generate_rayleigh_channel((num_samples, num_pilots), rng)
    pilots = generate_pilot_symbols(num_pilots, batch_size=num_samples)
    observations = simulate_pilot_observations(channels, pilots, snr_db, rng)
    inputs = complex_to_features(observations).astype(dtype)
    targets = complex_to_features(channels).astype(dtype)
    return split_dataset(inputs, targets)


def generate_grid_dataset(
    spec: "GridSpec",
    *,
    num_samples: int = 2000,
    snr_db: float = 10.0,
    random_seed: int = 42,
    batch_size: int = 256,
    input_source: str = "received",
    dtype: np.dtype = np.float32,
) -> dict[str, NDArray[np.generic]]:
    """Generate a split dataset of ofdm resource grids using Sionna.

    each example is paired with the true frequency-domain channel over the full
    grid (target), both represented with a trailing real/imaginary axis.
    ``input_source`` selects the neural input: ``"received"`` uses the raw
    received resource grid, while ``"ls-nn"`` and ``"ls-lin"`` use pilot-aware
    least-squares interpolation as a denoising/refinement input. produces arrays shaped
    ``(samples, num-ofdm-symbols, num-subcarriers, 2)``.

    this requires the ml stack (Sionna 2.x, which uses PyTorch); ``sionna_ofdm``
    is imported lazily so the rest of the dataset utilities stay usable without
    it. it has not been executed in this environment.
    """
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive.")
    if input_source not in ("received", "ls-nn", "ls-lin"):
        raise ValueError("input_source must be 'received', 'ls-nn', or 'ls-lin'.")

    from .sionna_ofdm import simulate_grid, simulate_grid_tensors, to_numpy

    received_batches: list[NDArray[np.complexfloating]] = []
    channel_batches: list[NDArray[np.complexfloating]] = []
    collected = 0
    batch_index = 0
    while collected < num_samples:
        this_batch = min(batch_size, num_samples - collected)
        if input_source == "received":
            received, channels, _no = simulate_grid(
                spec,
                snr_db,
                batch_size=this_batch,
                seed=random_seed + batch_index,
            )
        else:
            from .baselines import grid_ls_estimate

            y, h_freq, no, resource_grid = simulate_grid_tensors(
                spec,
                snr_db,
                batch_size=this_batch,
                seed=random_seed + batch_index,
            )
            interpolation = input_source.removeprefix("ls-")
            received = grid_ls_estimate(
                y, no, resource_grid, interpolation_type=interpolation
            ).reshape((this_batch, spec.num_ofdm_symbols, spec.num_subcarriers))
            channels = to_numpy(h_freq).reshape(
                (this_batch, spec.num_ofdm_symbols, spec.num_subcarriers)
            )
        received_batches.append(received)
        channel_batches.append(channels)
        collected += this_batch
        batch_index += 1

    observations = np.concatenate(received_batches, axis=0)
    channels = np.concatenate(channel_batches, axis=0)
    inputs = complex_to_features(observations).astype(dtype)
    targets = complex_to_features(channels).astype(dtype)
    return split_dataset(inputs, targets)


def save_npz_dataset(
    path: str | Path,
    dataset: Mapping[str, ArrayLike],
) -> Path:
    """Save dataset arrays to a compressed NPZ archive."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, **{key: np.asarray(value) for key, value in dataset.items()}
    )
    return output_path
