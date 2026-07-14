# Current Project Status

Last updated: 2026-07-14

## Summary

This project now has a working, reproducible pipeline for comparing classical
and lightweight neural OFDM channel estimators on a Sionna resource-grid
experiment. The latest validated notebook run shows a useful and honest result:
the small CNN improves substantially over LS interpolation, while
covariance-informed LMMSE remains the strongest estimator because it uses the
configured TDL channel covariance structure.

The current result should be framed as a validated single-condition result, not
yet as a broad robustness claim. The next evidence target is the repeated TDL
channel and seed sweep.

## What Works Now

- NumPy Rayleigh/AWGN pilot-observation simulation.
- Least-squares channel estimation for the simple pilot model.
- NMSE and BER metric utilities.
- NPZ dataset loading, inspection, splitting, and regeneration.
- Configuration-driven experiment runners.
- Sionna OFDM resource-grid simulation for TDL and Rayleigh channels.
- Sionna LS channel estimation with nearest-neighbor and linear interpolation.
- Covariance-informed Sionna LMMSE interpolation for TDL profiles.
- Small PyTorch neural estimators, including the convolutional full-grid model.
- Deterministic training with separate dataset, model, and evaluation seeds.
- Repeated-seed sweep infrastructure across TDL channel models.
- Focused tests for configuration, datasets, metrics, OFDM helpers, baselines,
  sweep summaries, and model construction where dependencies are available.

## Latest Validated Result

The latest notebook validation used the grid neural comparison experiment over a
TDL-A channel with 1,000 evaluation samples per SNR.

Observed NMSE pattern:

| SNR (dB) | LS-nn | LS-lin | LMMSE | Neural CNN |
| ---: | ---: | ---: | ---: | ---: |
| -5 | 3.169561 | 2.856816 | 0.044503 | 0.418829 |
| 0 | 1.002303 | 0.903405 | 0.017139 | 0.122998 |
| 5 | 0.316956 | 0.285682 | 0.006320 | 0.037841 |
| 10 | 0.100230 | 0.090340 | 0.002263 | 0.013734 |
| 15 | 0.031696 | 0.028568 | 0.000819 | 0.006667 |

Interpretation:

- LS interpolation improves as SNR increases, but it remains the weakest
  estimator family in this run.
- The CNN consistently beats LS-nn and LS-lin across all tested SNR values.
- LMMSE is best overall because it is model-informed and uses TDL covariance
  assumptions.
- The strongest defensible claim is that the CNN is useful versus LS
  interpolation in this validated TDL-A setting, not that it beats the strongest
  classical benchmark.

## Recent Work Completed

- Fixed the Sionna LMMSE precision mismatch by using single-precision TDL
  covariance matrices.
- Added robustness-style unit coverage for LMMSE across multiple TDL profiles,
  delay spreads, Doppler values, and interpolation orders.
- Validated the local non-ML test suite after the LMMSE changes.
- Updated the README and living report with the current validated takeaway.
- Fixed the grid neural sweep wrapper so it accepts the package CLI flags for
  channel models and seeds.
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
and robustness sweep should be run in the notebook or ML environment where those
dependencies are available.

The sweep CLI is now available:

```text
python experiments/grid-neural-comparison-v1/run-sweep.py --help
```

It exposes:

- `--channel-models`
- `--seeds`

## What Is Not Proven Yet

- The CNN has not yet been proven robust across multiple TDL profiles and seeds.
- The CNN has not yet been evaluated as an out-of-distribution estimator without
  retraining.
- The current CNN trains at one dataset-generation SNR.
- BER is implemented as a metric utility but is not connected to a documented
  end-to-end OFDM decision path.
- Runtime, model-size, and deployment-cost comparisons still need to be
  integrated into the main result tables.
- LMMSE should continue to be described as model-informed, not resource
  equivalent to the CNN.

## Next Step

Run the repeated-channel robustness sweep in the ML environment:

```bash
python experiments/grid-neural-comparison-v1/run-sweep.py \
  experiments/grid-neural-comparison-v1/config.yaml \
  --channel-models tdl-a,tdl-b,tdl-c \
  --seeds 42,43,44
```

That command retrains the CNN for each channel/seed configuration, evaluates
LS-nn, LS-lin, LMMSE, and CNN at every configured SNR, and writes raw, summary,
margin, and figure artifacts under `results/`.

After the sweep completes, the next documentation update should summarize:

- CNN versus LS improvement by channel model and SNR.
- CNN versus LMMSE gap by channel model and SNR.
- Variation across seeds.
- Any channel profiles where the CNN gain is weak or unstable.
- Whether the result supports a broader robustness claim or only a narrower
  TDL-A-style claim.
