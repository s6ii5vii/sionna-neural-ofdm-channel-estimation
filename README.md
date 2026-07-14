# Lightweight Neural OFDM Channel Estimation for Low-Resource 6G-Like Environments

An early-stage reproducible research software project for evaluating classical
and lightweight neural channel estimators under constrained wireless
conditions.

## Research question

Can a lightweight neural estimator improve pilot-based OFDM channel estimation
under low SNR, sparse pilots, limited training data, and limited inference
compute while remaining practical for constrained deployment scenarios?

The repository establishes the simulation, dataset, classical baselines,
lightweight neural estimator, metrics, and repeated-seed experiment structure
needed to investigate that question. It does not yet make a reviewed neural
performance claim.

## Why this project matters

Reliable channel-state estimates are required for coherent OFDM reception.
Neural estimators can be useful when analytical assumptions are incomplete, but
their value must be tested against simple baselines and under explicit resource
constraints. This project emphasizes:

- reproducible configurations and seeded simulations;
- honest comparison with classical estimators;
- metrics such as NMSE, BER where symbol decisions are available, model size,
  and runtime;
- small, inspectable implementations rather than an oversized model;
- documented assumptions, limitations, and generated artifacts.

## Current status

Implemented:

- NumPy Rayleigh/AWGN pilot-observation simulation matching the original
  notebook experiments;
- LS channel estimation;
- NMSE and BER utilities;
- split NPZ dataset loading, inspection, and regeneration;
- configuration-driven LS sweeps across SNR values;
- LMMSE estimation with an explicit, estimated channel covariance (NumPy,
  unit-tested);
- a Sionna OFDM resource-grid simulation module (TDL/Rayleigh fading, pilot
  grid, AWGN) and a Sionna LS-with-interpolation baseline wrapper;
- a Sionna-backed grid dataset generator producing full resource-grid
  observation/channel pairs;
- a small PyTorch MLP estimator and a deliberately small convolutional full-grid
  estimator;
- deterministic neural training with independent data, model, and evaluation
  seeds, best-validation checkpoint selection, and dataset provenance checks;
- repeated-seed validation sweeps across TDL channel models;
- tests for datasets, metrics, LS, and LMMSE baselines.

A July 2026 interactive notebook run of the grid neural comparison validated the
current TDL-A experiment path with 1,000 evaluation samples per SNR. In that run,
the CNN substantially improved over LS interpolation across the tested SNR
range, while model-informed LMMSE remained the strongest estimator because it
uses the configured TDL covariance structure. The Sionna simulation, grid
baselines, and neural comparison are not executed in CI (they require Sionna
2.x, PyTorch, and ideally a GPU), so broader robustness claims still require
the repeated-channel sweep below.

Planned:

- out-of-distribution robustness sweeps without retraining;
- controlled neural architecture and dataset-size ablations;
- BER evaluation through an end-to-end link;
- runtime and parameter-count reporting.

## Repository structure

```text
src/channel_estimation/   Python import package (underscore required by Python)
experiments/              versioned experiment configurations and runners
notebooks/                explanation, visualization, and guided demos
data/                     committed sample data and dataset documentation
results/                  generated figures and tables
docs/                     research framing, design, alignment, and report scaffold
tests/                    focused correctness tests
archive/learning-log/     original dated development notes
```

## Quickstart

Python 3.11 or newer is required by the current Sionna 2.x stack.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the ML stack when working on Sionna or the neural estimator:

```bash
python -m pip install -e ".[ml,test]"
```

`requirements.txt` contains the complete expected environment. Sionna 2.x (which
uses PyTorch) can be heavy and platform-sensitive, so a supported Colab or GPU
environment may be preferable.

## Run the LS baseline

```bash
python experiments/baseline-ls/run-experiment.py
```

The runner reads `experiments/baseline-ls/config.yaml` and writes:

- `results/tables/baseline-ls.csv`
- `results/figures/baseline-ls-nmse.png`

Run the implemented classical portion of the first constrained profile with:

```bash
python experiments/low-resource-v1/run-experiment.py
```

The sparse-pilot profile currently evaluates only selected pilot locations. It
does not yet reconstruct unobserved subcarriers.

## Run the Sionna grid baseline

The grid experiment simulates a full OFDM resource grid over a 3GPP TDL-A
channel and compares least-squares estimation with nearest and linear
interpolation plus covariance-informed LMMSE interpolation across the whole
grid. It requires the ml stack and, realistically, a GPU:

```bash
python -m pip install -e ".[ml]"
python experiments/grid-tdl-v1/run-experiment.py
```

It writes `results/tables/grid-tdl-v1.csv` and a per-estimator NMSE-vs-SNR
figure. These Sionna paths are implemented but have not been executed in CI;
validate them on a suitable environment before interpreting the outputs.

Neural training (grid CNN or flat MLP, selected automatically from the dataset
rank) runs through:

```bash
python -m channel_estimation.train experiments/low-resource-v1/config.yaml
```

Training additionally writes a `*.report.json` next to the checkpoint recording
test NMSE, parameter count, serialized model size, and per-example latency.

## Compare the CNN against least squares (LS)

`experiments/grid-neural-comparison-v1/` is a single-configuration
head-to-head comparison of LS with nearest-neighbor interpolation (LS-nn), LS
with linear interpolation (LS-lin), covariance-informed linear minimum mean
squared error (LMMSE), and a small convolutional neural network (CNN). The config has three
sections: `experiment` (the shared resource-grid and channel), `training` (the
CNN training loop and an optional `dataset-generation` block that lets training
auto-create the grid dataset if it does not already exist), and `neural` (the
checkpoint path and architecture the evaluator loads for scoring).

```bash
# 1. Train (auto-generates the grid dataset if missing):
python -m channel_estimation.train experiments/grid-neural-comparison-v1/config.yaml

# 2. Score all three estimators at every SNR and emit one CSV / one plot:
python experiments/grid-neural-comparison-v1/run-experiment.py
```

Requires the ml stack. An experimental Colab helper notebook is available under
`notebooks/`, but validate its outputs before interpreting results.

The latest validated single-run pattern is:

- LS interpolation error decreases with SNR but remains the weakest family;
- the CNN is consistently better than LS-nn and LS-lin in the tested TDL-A run;
- LMMSE is best overall and should be framed as a model-informed upper classical
  benchmark, not as a resource-equivalent learned baseline.

Run the broader robustness sweep across TDL profiles and seeds with:

```bash
python experiments/grid-neural-comparison-v1/run-sweep.py \
  experiments/grid-neural-comparison-v1/config.yaml \
  --channel-models tdl-a,tdl-b,tdl-c \
  --seeds 42,43,44
```

The sweep retrains each CNN for its channel/seed configuration, evaluates LS,
LMMSE, and CNN at every configured SNR, and writes raw, summary, margin, and
figure artifacts under `results/`.

## Inspect or regenerate the dataset

```python
from channel_estimation.dataset import inspect_dataset, load_npz_dataset

dataset = load_npz_dataset("data/channel-estimation-dataset.npz")
print(inspect_dataset(dataset))
```

The committed sample contains 5,000 examples represented by real and imaginary
features, split into train, validation, and test arrays. See
[`data/README.md`](data/README.md) for its schema and regeneration policy.

## Low-resource framing

The first study profiles cover:

- low SNR and noisy observations;
- reduced pilot density;
- smaller training datasets;
- a target model-size budget;
- limited inference compute;
- robustness across later, explicitly defined channel conditions.

The exact definitions, assumptions, and exclusions are documented in
[`docs/low-resource-framing.md`](docs/low-resource-framing.md).

## Proposal relevance

This codebase is preparation evidence for scientific ML, reproducible
experiment and evaluation, lightweight ML, and network-simulation work. It
demonstrates a progression from a technical question to baselines, datasets,
tests, configurations, and a report structure. It is not presented as a GSoC
project, accepted research, or evidence of a 6G breakthrough.

See [`docs/proposal-alignment.md`](docs/proposal-alignment.md) for the detailed
mapping and boundaries.

## Limitations

- The current NumPy model is a simplified independent Rayleigh/AWGN pilot model,
  not a complete standards-compliant link simulation.
- The committed dataset was generated at one SNR with unit pilots.
- The current validated single TDL-A notebook run supports only a narrow
  performance statement: the CNN beats LS interpolation on that run, while
  covariance-informed LMMSE remains stronger.
- LMMSE now uses a covariance matrix estimated from channel realizations; it is
  only as good as that covariance assumption and the data it is estimated from.
- BER is implemented as a metric utility but is not yet connected to an
  end-to-end coded OFDM receiver.
- Generated results require validation before scientific interpretation.

## Report and citation

[`docs/report.md`](docs/report.md) is a mini-paper-style living report. There is
no formal publication or archival citation for this project yet. Until a
versioned release exists, cite the repository URL and commit hash when referring
to a specific experiment.

## License

This project is available under the terms in [`LICENSE`](LICENSE).
