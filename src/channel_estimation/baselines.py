"""Classical channel-estimation baselines."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def least_squares_estimate(
    received_pilots: ArrayLike,
    pilot_symbols: ArrayLike,
) -> NDArray[np.complex128]:
    """Estimate channel coefficients by dividing observations by known pilots.

    For the scalar pilot model ``y = h*x + n``, the LS solution is ``h_hat = y/x``.
    Pilot symbols must be non-zero and broadcast-compatible with the observations.
    """
    received = np.asarray(received_pilots)
    pilots = np.asarray(pilot_symbols)
    try:
        pilots_are_finite = bool(np.all(np.isfinite(pilots)))
    except TypeError as exc:
        raise ValueError("Pilot symbols must be numeric and finite.") from exc
    if not pilots_are_finite:
        raise ValueError("Pilot symbols must be finite.")
    if np.any(pilots == 0):
        raise ValueError("Least-squares estimation requires non-zero pilot symbols.")
    try:
        return np.asarray(received / pilots)
    except ValueError as exc:
        raise ValueError(
            "received_pilots and pilot_symbols must be broadcast-compatible."
        ) from exc


def lmmse_estimate(*_args: object, **_kwargs: object) -> None:
    """Planned LMMSE baseline.

    A correct implementation requires explicit channel and noise covariance
    assumptions. It remains intentionally unimplemented until those assumptions
    are defined and validated.
    """
    raise NotImplementedError(
        "LMMSE estimation is planned but requires validated covariance models."
    )
