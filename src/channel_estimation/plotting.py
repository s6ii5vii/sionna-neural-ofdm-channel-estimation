"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _save_curve(
    x_values: Iterable[float],
    y_values: Iterable[float],
    *,
    ylabel: str,
    title: str,
    output_path: str | Path,
    logarithmic_y: bool,
) -> Path:
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    plot = axis.semilogy if logarithmic_y else axis.plot
    plot(list(x_values), list(y_values), marker="o")
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.35)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_nmse_vs_snr(
    snr_values: Iterable[float],
    nmse_values: Iterable[float],
    output_path: str | Path,
) -> Path:
    """Save an NMSE-versus-SNR curve."""
    return _save_curve(
        snr_values,
        nmse_values,
        ylabel="NMSE",
        title="Least-Squares Channel Estimation",
        output_path=output_path,
        logarithmic_y=True,
    )


def save_ber_vs_snr(
    snr_values: Iterable[float],
    ber_values: Iterable[float],
    output_path: str | Path,
) -> Path:
    """Save a BER-versus-SNR curve."""
    return _save_curve(
        snr_values,
        ber_values,
        ylabel="BER",
        title="Bit Error Rate",
        output_path=output_path,
        logarithmic_y=True,
    )
