# Current Project Status

Last updated: 2026-07-14

## Summary

This project now has a working, reproducible pipeline for comparing classical
and lightweight neural OFDM channel estimators on a Sionna resource-grid
experiment. The latest validated notebook run shows a useful and honest result:
the small CNN (convolutional neural network) improves substantially over LS
(least-squares) interpolation, while covariance-informed LMMSE (linear minimum
mean squared error) remains the strongest estimator because it uses the
configured TDL (tapped delay line) channel covariance structure.

The current result should be framed as a validated single-condition result, not
yet as a broad robustness claim. The next evidence target is the repeated TDL
(tapped delay line) channel and seed sweep.

## What Works Now

- NumPy Rayleigh/AWGN (additive white Gaussian noise) pilot-observation
  simulation.
- LS (least-squares) channel estimation for the simple pilot model.
- NMSE (normalized mean squared error) and BER (bit error rate) metric
  utilities.
- NPZ (NumPy compressed archive) dataset loading, inspection, splitting, and
  regeneration.
- Configuration-driven experiment runners.
- Sionna OFDM (orthogonal frequency-division multiplexing) resource-grid
  simulation for TDL (tapped delay line) and Rayleigh channels.
- Sionna LS (least-squares) channel estimation with nearest-neighbor and linear
  interpolation.
- Covariance-informed Sionna LMMSE (linear minimum mean squared error)
  interpolation for TDL profiles.
- Small PyTorch neural estimators, including the convolutional full-grid model.
- Deterministic training with separate dataset, model, and evaluation seeds.
- Repeated-seed sweep infrastructure across TDL (tapped delay line) channel
  models.
- Focused tests for configuration, datasets, metrics, OFDM (orthogonal
  frequency-division multiplexing) helpers, baselines, sweep summaries, and
  model construction where dependencies are available.

## Latest Validated Result

The latest notebook validation used the grid neural comparison experiment over a
TDL-A (tapped delay line A) channel with 1,000 evaluation samples per SNR
(signal-to-noise ratio).

Observed NMSE (normalized mean squared error) pattern:

| SNR (signal-to-noise ratio, dB) | LS-nn (least-squares with nearest-neighbor interpolation) | LS-lin (least-squares with linear interpolation) | LMMSE (linear minimum mean squared error) | Neural CNN (convolutional neural network) |
| ---: | ---: | ---: | ---: | ---: |
| -5 | 3.169561 | 2.856816 | 0.044503 | 0.418829 |
| 0 | 1.002303 | 0.903405 | 0.017139 | 0.122998 |
| 5 | 0.316956 | 0.285682 | 0.006320 | 0.037841 |
| 10 | 0.100230 | 0.090340 | 0.002263 | 0.013734 |
| 15 | 0.031696 | 0.028568 | 0.000819 | 0.006667 |

Interpretation:

- LS (least-squares) interpolation improves as SNR (signal-to-noise ratio)
  increases, but it remains the weakest estimator family in this run.
- The CNN (convolutional neural network) consistently beats LS-nn
  (least-squares with nearest-neighbor interpolation) and LS-lin
  (least-squares with linear interpolation) across all tested SNR values.
- LMMSE (linear minimum mean squared error) is best overall because it is
  model-informed and uses TDL (tapped delay line) covariance assumptions.
- The strongest defensible claim is that the CNN is useful versus LS
  interpolation in this validated TDL-A (tapped delay line A) setting, not that
  it beats the strongest classical benchmark.

## Recent Work Completed

- Fixed the Sionna LMMSE (linear minimum mean squared error) precision mismatch
  by using single-precision TDL (tapped delay line) covariance matrices.
- Added robustness-style unit coverage for LMMSE across multiple TDL profiles,
  delay spreads, Doppler values, and interpolation orders.
- Validated the local non-ML (machine learning) test suite after the LMMSE
  changes.
- Updated the README and living report with the current validated takeaway.
- Fixed the grid neural sweep wrapper so it accepts the package CLI (command
  line interface) flags for channel models and seeds.
- Pushed the latest updates to `origin/main`.

Recent commits:

- `c54a60a test lmmse robustness across tdl profiles`
- `0e496a7 document validated grid comparison`

## Verification Evidence

The latest local test run completed successfully:

```text
python -m pytest
59 passed, 3 skipped
```

The skipped tests are expected in the current local Windows environment because
`torch` and `sionna` are not installed there. The real grid neural experiment
and robustness sweep should be run in the notebook or ML (machine learning)
environment where those dependencies are available.

The sweep CLI (command line interface) is now available:

```text
python experiments/grid-neural-comparison-v1/run-sweep.py --help
```

It exposes:

- `--channel-models`
- `--seeds`

## What Is Not Proven Yet

- The CNN (convolutional neural network) has not yet been proven robust across
  multiple TDL (tapped delay line) profiles and seeds.
- The CNN has not yet been evaluated as an out-of-distribution estimator without
  retraining.
- The current CNN trains at one dataset-generation SNR (signal-to-noise ratio).
- BER (bit error rate) is implemented as a metric utility but is not connected
  to a documented end-to-end OFDM (orthogonal frequency-division multiplexing)
  decision path.
- Runtime, model-size, and deployment-cost comparisons still need to be
  integrated into the main result tables.
- LMMSE (linear minimum mean squared error) should continue to be described as
  model-informed, not resource equivalent to the CNN.

## Next Step

Run the repeated-channel robustness sweep in the ML (machine learning)
environment:

```bash
python experiments/grid-neural-comparison-v1/run-sweep.py \
  experiments/grid-neural-comparison-v1/config.yaml \
  --channel-models tdl-a,tdl-b,tdl-c \
  --seeds 42,43,44
```

That command retrains the CNN (convolutional neural network) for each
channel/seed configuration, evaluates LS-nn (least-squares with
nearest-neighbor interpolation), LS-lin (least-squares with linear
interpolation), LMMSE (linear minimum mean squared error), and CNN at every
configured SNR (signal-to-noise ratio), and writes raw, summary, margin, and
figure artifacts under `results/`.

After the sweep completes, the next documentation update should summarize:

- CNN (convolutional neural network) versus LS (least-squares) improvement by
  channel model and SNR (signal-to-noise ratio).
- CNN versus LMMSE (linear minimum mean squared error) gap by channel model and
  SNR.
- Variation across seeds.
- Any channel profiles where the CNN gain is weak or unstable.
- Whether the result supports a broader robustness claim or only a narrower
  TDL-A (tapped delay line A)-style claim.
