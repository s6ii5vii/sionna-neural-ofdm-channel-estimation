# Experiments

Each directory contains a versioned configuration and a small runner. Install the
package in editable mode before running an experiment:

```bash
python -m pip install -e .
python experiments/baseline-ls/run-experiment.py
```

Experiment scripts write generated tables and figures under `results/`. They do
not contain precomputed or invented scientific results.
