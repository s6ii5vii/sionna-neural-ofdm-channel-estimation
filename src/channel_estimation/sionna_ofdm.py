"""sionna-based ofdm resource-grid simulation for channel estimation.

this module builds a realistic ofdm link using nvidia sionna's phy layer: an
ofdm resource grid with embedded pilot symbols, a 3gpp tdl (or rayleigh block
fading) channel, and awgn. it produces supervised ``(received-grid,
true-channel)`` pairs used to train and evaluate channel estimators over a full
resource grid rather than at isolated pilot locations.

design notes
------------
* sionna and tensorflow are imported lazily inside each function so the
  numpy-only baseline, metrics, dataset-split, and config code remain importable
  and testable without the heavy ml stack installed.
* sionna 1.x exposes its link-level (physical-layer) api under ``sionna.phy``.
  the symbol names used here target that layout; pin an exact sionna version and
  verify against it, because module paths shifted across the 0.x -> 1.x change.
* all public helpers return plain numpy arrays with the single-antenna
  transmit/receive dimensions squeezed out, so downstream code sees channel and
  observation grids shaped ``(batch, num-ofdm-symbols, num-subcarriers)``.

this file has not been executed in this environment (no tensorflow/sionna/gpu
available here); treat it as implemented-but-unverified until run on a machine
with the ml stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .ofdm import noise_power_from_snr

_VALID_CHANNEL_KINDS = ("tdl-a", "tdl-b", "tdl-c", "tdl-d", "tdl-e", "rayleigh")


@dataclass(frozen=True)
class GridSpec:
    """Immutable description of the ofdm resource grid and channel.

    the fields mirror the ``experiment`` block of an experiment config so a
    single spec can be built from yaml and threaded through simulation,
    estimation, and evaluation without re-reading the config each time.
    """

    num_subcarriers: int
    num_ofdm_symbols: int
    pilot_ofdm_symbol_indices: tuple[int, ...]
    channel_kind: str = "tdl-a"
    delay_spread_ns: float = 100.0
    max_doppler_hz: float = 0.0
    carrier_frequency_hz: float = 3.5e9
    subcarrier_spacing_hz: float = 30e3
    cyclic_prefix_length: int = 6
    num_bits_per_symbol: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.num_subcarriers, int) or self.num_subcarriers <= 0:
            raise ValueError("num_subcarriers must be a positive integer.")
        if not isinstance(self.num_ofdm_symbols, int) or self.num_ofdm_symbols <= 0:
            raise ValueError("num_ofdm_symbols must be a positive integer.")
        if not self.pilot_ofdm_symbol_indices:
            raise ValueError("pilot_ofdm_symbol_indices must not be empty.")
        if any(
            not isinstance(index, int) or not 0 <= index < self.num_ofdm_symbols
            for index in self.pilot_ofdm_symbol_indices
        ):
            raise ValueError(
                "every pilot symbol index must be an integer within the grid."
            )
        if self.channel_kind not in _VALID_CHANNEL_KINDS:
            raise ValueError(f"channel_kind must be one of {_VALID_CHANNEL_KINDS}.")
        if not np.isfinite(self.delay_spread_ns) or self.delay_spread_ns <= 0:
            raise ValueError("delay_spread_ns must be a positive finite value.")
        if not np.isfinite(self.max_doppler_hz) or self.max_doppler_hz < 0:
            raise ValueError("max_doppler_hz must be a non-negative finite value.")

    @classmethod
    def from_experiment(cls, experiment: Mapping[str, Any]) -> "GridSpec":
        """Build a spec from a validated grid ``experiment`` config mapping."""
        return cls(
            num_subcarriers=int(experiment["num-subcarriers"]),
            num_ofdm_symbols=int(experiment["num-ofdm-symbols"]),
            pilot_ofdm_symbol_indices=tuple(experiment["pilot-ofdm-symbol-indices"]),
            channel_kind=str(experiment.get("channel-model", "tdl-a")),
            delay_spread_ns=float(experiment.get("delay-spread-ns", 100.0)),
            max_doppler_hz=float(experiment.get("max-doppler-hz", 0.0)),
        )


def build_resource_grid(spec: GridSpec) -> Any:
    """Build a sionna ``ResourceGrid`` with a kronecker pilot pattern."""
    try:
        from sionna.phy.ofdm import ResourceGrid
    except ImportError as exc:  # pragma: no cover - requires ml stack
        raise ImportError(
            "sionna is required for resource-grid simulation. install the project "
            "with the 'ml' extra or use requirements.txt."
        ) from exc

    return ResourceGrid(
        num_ofdm_symbols=spec.num_ofdm_symbols,
        fft_size=spec.num_subcarriers,
        subcarrier_spacing=spec.subcarrier_spacing_hz,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=spec.cyclic_prefix_length,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=list(spec.pilot_ofdm_symbol_indices),
    )


def build_channel_model(spec: GridSpec) -> Any:
    """Build a 3gpp tdl or rayleigh block-fading channel model."""
    if spec.channel_kind == "rayleigh":
        try:
            from sionna.phy.channel import RayleighBlockFading
        except ImportError as exc:  # pragma: no cover - requires ml stack
            raise ImportError("sionna is required for channel modelling.") from exc
        return RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1)

    try:
        from sionna.phy.channel.tr38901 import TDL
    except ImportError as exc:  # pragma: no cover - requires ml stack
        raise ImportError("sionna is required for channel modelling.") from exc

    model_letter = spec.channel_kind.split("-")[-1].upper()
    return TDL(
        model=model_letter,
        delay_spread=spec.delay_spread_ns * 1e-9,
        carrier_frequency=spec.carrier_frequency_hz,
        max_speed=_doppler_to_speed(spec.max_doppler_hz, spec.carrier_frequency_hz),
        num_rx_ant=1,
        num_tx_ant=1,
    )


def _doppler_to_speed(max_doppler_hz: float, carrier_frequency_hz: float) -> float:
    """Convert a maximum doppler shift in hz to a user speed in m/s."""
    speed_of_light = 299_792_458.0
    if max_doppler_hz <= 0:
        return 0.0
    return float(max_doppler_hz * speed_of_light / carrier_frequency_hz)


def simulate_grid_tensors(
    spec: GridSpec,
    snr_db: float,
    *,
    batch_size: int,
    seed: int,
) -> tuple[Any, Any, float, Any]:
    """Simulate one batch and return the raw sionna tensors and resource grid.

    returns ``(y, h_freq, no, resource_grid)`` where ``y`` and ``h_freq`` keep
    their native sionna shapes so they can be fed to ``LSChannelEstimator``.
    random qam data occupies the non-pilot resource elements so the observation
    is a realistic mixed data/pilot grid.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    try:
        import tensorflow as tf
        from sionna.phy.channel import ApplyOFDMChannel, GenerateOFDMChannel
        from sionna.phy.mapping import QAMSource
        from sionna.phy.ofdm import ResourceGridMapper
    except ImportError as exc:  # pragma: no cover - requires ml stack
        raise ImportError(
            "tensorflow and sionna are required for simulation. install the 'ml' "
            "extra or use requirements.txt."
        ) from exc

    tf.random.set_seed(seed)

    resource_grid = build_resource_grid(spec)
    channel_model = build_channel_model(spec)

    qam_source = QAMSource(spec.num_bits_per_symbol)
    rg_mapper = ResourceGridMapper(resource_grid)
    generate_channel = GenerateOFDMChannel(channel_model, resource_grid)
    apply_channel = ApplyOFDMChannel(add_awgn=True)

    # random payload symbols on the data-carrying resource elements.
    data = qam_source([batch_size, 1, 1, resource_grid.num_data_symbols])
    x_rg = rg_mapper(data)

    h_freq = generate_channel(batch_size)
    no = noise_power_from_snr(float(snr_db))
    y = apply_channel(x_rg, h_freq, no)

    return y, h_freq, float(no), resource_grid


def simulate_grid(
    spec: GridSpec,
    snr_db: float,
    *,
    batch_size: int,
    seed: int,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating], float]:
    """Simulate one batch of received grids and matching true channels.

    returns ``(y, h_true, no)`` where ``y`` and ``h_true`` are complex arrays of
    shape ``(batch, num-ofdm-symbols, num-subcarriers)`` and ``no`` is the linear
    noise power used for the awgn.
    """
    y, h_freq, no, _resource_grid = simulate_grid_tensors(
        spec, snr_db, batch_size=batch_size, seed=seed
    )
    return _squeeze_grid(y), _squeeze_grid(h_freq), no


def _squeeze_grid(tensor: Any) -> NDArray[np.complexfloating]:
    """Reduce a single-antenna sionna grid tensor to ``(batch, symbols, sc)``.

    sionna returns received grids shaped
    ``(batch, num-rx, num-rx-ant, num-ofdm-symbols, fft-size)`` and channels
    shaped ``(batch, num-rx, num-rx-ant, num-tx, num-tx-ant, num-ofdm-symbols,
    fft-size)``. with a single antenna on each side every leading pair collapses
    to one, leaving the batch and the two grid axes.
    """
    array = np.asarray(tensor)
    if array.ndim < 3:
        raise ValueError("expected a sionna grid tensor with at least three axes.")
    batch = array.shape[0]
    grid = array.shape[-2:]
    return array.reshape((batch, *grid))
