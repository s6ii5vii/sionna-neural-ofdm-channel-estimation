# Experiments

Each directory contains a versioned configuration and a small runner. Install the
package in editable mode before running an experiment:

```bash
python -m pip install -e .
python experiments/baseline-ls/run-experiment.py
```

Experiment scripts write generated tables and figures under `results/`. They do
not contain precomputed or invented scientific results.

The grid neural comparison also includes a heavier validation sweep:

```bash
python experiments/grid-neural-comparison-v1/run-sweep.py
```

By default, the sweep trains and evaluates the CNN (convolutional neural
network) across three TDL (tapped delay line) channel models and three random
seeds. Each run uses separate training-data, model, and evaluation seed streams.
It writes raw NMSE (normalized mean squared error) rows, summary statistics,
neural margin statistics against LS-lin (least-squares estimator with linear
interpolation) and covariance-informed LMMSE (linear minimum mean squared
error), and a channel-specific comparison figure under `results/`.

`grid-neural-comparison-v2/` is a harder version of the same comparison: a
wider, frequency-selective, time-varying channel, pilots on every 4th
subcarrier, a CNN trained across -5 to 15 dB with SNR (signal-to-noise ratio)
input, dilated convolutions, and a residual output, plus an LMMSE (linear
minimum mean squared error) baseline built from mismatched channel statistics.
It uses the same `run-experiment.py` and `run-sweep.py` entry points. It has not
been validated on a GPU yet.
