"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Iterable, Mapping, Sequence, Tuple


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


def save_nmse_comparison(
    series: Mapping[str, Tuple[Sequence[float], Sequence[float]]],
    output_path: str | Path,
    *,
    title: str = "Channel Estimation NMSE vs SNR",
) -> Path:
    """Save one NMSE-versus-SNR curve per estimator on shared axes.

    ``series`` maps an estimator label to a ``(snr_values, nmse_values)`` pair.
    """
    import matplotlib.pyplot as plt

    if not series:
        raise ValueError("series must contain at least one estimator.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    for label, (snr_values, nmse_values) in series.items():
        axis.semilogy(list(snr_values), list(nmse_values), marker="o", label=label)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("NMSE")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_nmse_sweep_comparison(
    summary_rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    title: str = "Channel Estimation Sweep NMSE vs SNR",
) -> Path:
    """Save mean NMSE curves with one-standard-deviation bands.

    ``summary_rows`` is the output of the grid sweep summarizer, with one row
    per channel model, SNR, and estimator.
    """
    import matplotlib.pyplot as plt

    if not summary_rows:
        raise ValueError("summary_rows must contain at least one row.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, dict[float, Mapping[str, object]]]] = {}
    for row in summary_rows:
        channel_model = str(row["channel-model"])
        estimator = str(row["estimator"])
        snr_db = float(row["snr-db"])
        grouped.setdefault(channel_model, {}).setdefault(estimator, {})[snr_db] = row

    channel_models = sorted(grouped)
    columns = min(3, len(channel_models))
    rows = math.ceil(len(channel_models) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 4 * rows), squeeze=False, sharey=True
    )
    for axis, channel_model in zip(axes.flat, channel_models, strict=False):
        for estimator, by_snr in sorted(grouped[channel_model].items()):
            snr_values = sorted(by_snr)
            values = [float(by_snr[snr]["nmse-mean"]) for snr in snr_values]
            spreads = [float(by_snr[snr]["nmse-std"]) for snr in snr_values]
            lower = [max(value - spread, 1e-12) for value, spread in zip(values, spreads)]
            upper = [value + spread for value, spread in zip(values, spreads)]
            axis.semilogy(snr_values, values, marker="o", label=estimator)
            axis.fill_between(snr_values, lower, upper, alpha=0.12)
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("NMSE")
        axis.set_title(channel_model.upper())
        axis.grid(True, which="both", alpha=0.35)
        axis.legend()
    for axis in list(axes.flat)[len(channel_models):]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


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
