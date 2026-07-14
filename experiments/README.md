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

By default, the sweep trains and evaluates the CNN (convolutional neural network)
across three TDL (tapped delay line) channel models and three random seeds. It
writes raw NMSE (normalized mean squared error) rows, summary statistics, neural
margin statistics against LS-lin (least-squares estimator with linear
interpolation), and a comparison figure under `results/`.
