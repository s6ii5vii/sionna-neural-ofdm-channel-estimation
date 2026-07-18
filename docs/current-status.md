# Current Project Status

Last updated: 2026-07-18

## Summary

This project now has a working, reproducible pipeline for comparing classical
and lightweight neural OFDM channel estimators on a Sionna resource-grid
experiment. The latest validated notebook run shows a useful and honest result:
the small CNN (convolutional neural network) improves substantially over LS
(least-squares) interpolation, while covariance-informed LMMSE (linear minimum
mean squared error) remains the strongest estimator because it uses the
configured TDL (tapped delay line) channel covariance structure.

The current result should be framed as a validated robustness result across the
tested TDL (tapped delay line) profiles, not yet as a universal wireless-channel
claim. The next evidence target is out-of-distribution evaluation without
retraining.

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

The latest notebook validation used the grid neural comparison experiment over
TDL-A (tapped delay line A), TDL-B (tapped delay line B), and TDL-C (tapped
delay line C), with three random seeds per channel profile and 1,000 evaluation
samples per SNR (signal-to-noise ratio). The run used commit
`df071f5cf948d01113b21b7f991bc59b68fbd99d` in a Colab GPU (graphics processing
unit) environment with Sionna 2.0.1 and PyTorch 2.11.0.

Mean NMSE (normalized mean squared error) pattern across three seeds:

| Channel | SNR (signal-to-noise ratio, dB) | LS-lin (least-squares with linear interpolation) | LMMSE (linear minimum mean squared error) | Neural CNN (convolutional neural network) |
| --- | ---: | ---: | ---: | ---: |
| TDL-A | -5 | 2.929670 | 0.046973 | 0.429709 |
| TDL-A | 0 | 0.926443 | 0.018089 | 0.124818 |
| TDL-A | 5 | 0.292967 | 0.006671 | 0.037888 |
| TDL-A | 10 | 0.092644 | 0.002380 | 0.013392 |
| TDL-A | 15 | 0.029297 | 0.000857 | 0.006219 |
| TDL-B | -5 | 2.898596 | 0.047160 | 0.451389 |
| TDL-B | 0 | 0.916617 | 0.017956 | 0.129753 |
| TDL-B | 5 | 0.289860 | 0.006427 | 0.038592 |
| TDL-B | 10 | 0.091662 | 0.002189 | 0.013147 |
| TDL-B | 15 | 0.028986 | 0.000754 | 0.005771 |
| TDL-C | -5 | 2.887714 | 0.047799 | 0.457173 |
| TDL-C | 0 | 0.913175 | 0.019403 | 0.129366 |
| TDL-C | 5 | 0.288771 | 0.007192 | 0.037892 |
| TDL-C | 10 | 0.091318 | 0.002584 | 0.012898 |
| TDL-C | 15 | 0.028877 | 0.000907 | 0.005757 |

Interpretation:

- LS (least-squares) interpolation improves as SNR (signal-to-noise ratio)
  increases, but it remains the weakest estimator family in this run.
- The CNN (convolutional neural network) consistently beats LS-nn
  (least-squares with nearest-neighbor interpolation) and LS-lin
  (least-squares with linear interpolation) across all tested SNR values.
- LMMSE (linear minimum mean squared error) is best overall because it is
  model-informed and uses TDL (tapped delay line) covariance assumptions.
- The CNN has a 100% win rate against LS-lin across the 15 channel/SNR
  combinations, with mean improvement ranging from 78.75% to 87.07%.
- The strongest defensible claim is now that the CNN is robustly useful versus
  LS interpolation across the tested TDL-A/B/C profiles, not that it beats the
  strongest model-informed classical benchmark.

## Recent Work Completed

- Fixed the Sionna LMMSE (linear minimum mean squared error) precision mismatch
  by using single-precision TDL (tapped delay line) covariance matrices.
- Added robustness-style unit coverage for LMMSE across multiple TDL profiles,
  delay spreads, Doppler values, and interpolation orders.
- Validated the local non-ML (machine learning) test suite after the LMMSE
  changes.
- Updated the README and living report with the current validated takeaway.
- Ran the repeated-channel robustness sweep across TDL-A, TDL-B, and TDL-C with
  seeds 42, 43, and 44.
- Fixed the grid neural sweep wrapper so it accepts the package CLI (command
  line interface) flags for channel models and seeds.
- Pushed the latest updates to `origin/main`.

Recent commits:

- `c54a60a test lmmse robustness across tdl profiles`
- `0e496a7 document validated grid comparison`
- `df071f5 fix grid sweep colab handoff`

## Verification Evidence

The latest local test run completed successfully:

```text
python -m pytest
59 passed, 3 skipped
```

The skipped tests are expected in the current local Windows environment because
`torch` and `sionna` are not installed there. The real grid neural experiment
and robustness sweep were run in the notebook ML (machine learning) environment
where those dependencies are available.

The Colab run also completed the full test suite:

```text
python -m pytest -q
62 passed in 7.91s
```

The sweep CLI (command line interface) is now available:

```text
python experiments/grid-neural-comparison-v1/run-sweep.py --help
```

It exposes:

- `--channel-models`
- `--seeds`

## What Is Not Proven Yet

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

Evaluate out-of-distribution behavior without retraining. Reason: the completed
robustness sweep retrained a CNN (convolutional neural network) for each TDL
(tapped delay line) channel profile, so the next harder question is whether a
CNN trained on one profile or SNR (signal-to-noise ratio) range still helps when
the channel model, mobility, delay spread, or SNR distribution shifts.
