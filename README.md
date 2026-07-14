# Lightweight Neural OFDM (orthogonal frequency-division multiplexing) Channel Estimation for Low-Resource 6G-Like Environments

An early-stage reproducible research software project for evaluating classical
and lightweight neural channel estimators under constrained wireless
conditions.

## Research question

Can a lightweight neural estimator improve pilot-based OFDM (orthogonal
frequency-division multiplexing) channel estimation under low SNR
(signal-to-noise ratio), sparse pilots, limited training data, and limited inference
compute while remaining practical for constrained deployment scenarios?

The repository establishes the simulation, dataset, classical baselines,
lightweight neural estimator, metrics, and repeated-seed experiment structure
needed to investigate that question. It does not yet make a reviewed neural
performance claim.

## Why this project matters

Reliable channel-state estimates are required for coherent OFDM (orthogonal
frequency-division multiplexing) reception.
Neural estimators can be useful when analytical assumptions are incomplete, but
their value must be tested against simple baselines and under explicit resource
constraints. This project emphasizes:

- reproducible configurations and seeded simulations;
- honest comparison with classical estimators;
- metrics such as NMSE (normalized mean squared error), BER (bit error rate)
  where symbol decisions are available, model size, and runtime;
- small, inspectable implementations rather than an oversized model;
- documented assumptions, limitations, and generated artifacts.

## Current status

Implemented:

- NumPy Rayleigh/AWGN (additive white Gaussian noise) pilot-observation
  simulation matching the original notebook experiments;
- LS (least-squares) channel estimation;
- NMSE (normalized mean squared error) and BER (bit error rate) utilities;
- split NPZ (NumPy compressed archive) dataset loading, inspection, and
  regeneration;
- configuration-driven LS (least-squares) sweeps across SNR (signal-to-noise
  ratio) values;
- LMMSE (linear minimum mean squared error) estimation with an explicit,
  estimated channel covariance (NumPy, unit-tested);
- a Sionna OFDM (orthogonal frequency-division multiplexing) resource-grid
  simulation module (TDL (tapped delay line)/Rayleigh fading, pilot grid, AWGN
  (additive white Gaussian noise)) and a Sionna LS (least-squares)-with-interpolation
  baseline wrapper;
- a Sionna-backed grid dataset generator producing full resource-grid
  observation/channel pairs;
- a small PyTorch MLP (multilayer perceptron) estimator and a deliberately
  small convolutional full-grid estimator;
- deterministic neural training with independent data, model, and evaluation
  seeds, best-validation checkpoint selection, and dataset provenance checks;
- repeated-seed validation sweeps across TDL (tapped delay line) channel
  models;
- tests for datasets, metrics, LS (least-squares), and LMMSE (linear minimum
  mean squared error) baselines.

A July 2026 interactive notebook run of the grid neural comparison validated the
current TDL-A (tapped delay line A) experiment path with 1,000 evaluation
samples per SNR (signal-to-noise ratio). In that run, the CNN (convolutional
neural network) substantially improved over LS (least-squares) interpolation
across the tested SNR range, while model-informed LMMSE (linear minimum mean
squared error) remained the strongest estimator because it uses the configured
TDL (tapped delay line) covariance structure. The Sionna simulation, grid
baselines, and neural comparison are not executed in CI (continuous
integration) because they require Sionna 2.x, PyTorch, and ideally a GPU
(graphics processing unit), so broader robustness claims still require the
repeated-channel sweep below.

Planned:

- out-of-distribution robustness sweeps without retraining;
- controlled neural architecture and dataset-size ablations;
- BER (bit error rate) evaluation through an end-to-end link;
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

Install the ML (machine learning) stack when working on Sionna or the neural
estimator:

```bash
python -m pip install -e ".[ml,test]"
```

`requirements.txt` contains the complete expected environment. Sionna 2.x (which
uses PyTorch) can be heavy and platform-sensitive, so a supported Colab or GPU
(graphics processing unit) environment may be preferable.

## Run the LS (least-squares) baseline

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

The grid experiment simulates a full OFDM (orthogonal frequency-division
multiplexing) resource grid over a 3GPP (3rd Generation Partnership Project)
TDL-A (tapped delay line A) channel and compares LS (least-squares) estimation
with nearest and linear interpolation plus covariance-informed LMMSE (linear
minimum mean squared error) interpolation across the whole grid. It requires
the ML (machine learning) stack and, realistically, a GPU (graphics processing
unit):

```bash
python -m pip install -e ".[ml]"
python experiments/grid-tdl-v1/run-experiment.py
```

It writes `results/tables/grid-tdl-v1.csv` and a per-estimator NMSE
(normalized mean squared error)-vs-SNR (signal-to-noise ratio) figure. These
Sionna paths are implemented but have not been executed in CI (continuous
integration); validate them on a suitable environment before interpreting the
outputs.

Neural training (grid CNN (convolutional neural network) or flat MLP
(multilayer perceptron), selected automatically from the dataset rank) runs
through:

```bash
python -m channel_estimation.train experiments/low-resource-v1/config.yaml
```

Training additionally writes a `*.report.json` next to the checkpoint recording
test NMSE (normalized mean squared error), parameter count, serialized model
size, and per-example latency.

## Compare the CNN (convolutional neural network) against least squares (LS)

`experiments/grid-neural-comparison-v1/` is a single-configuration
head-to-head comparison of LS (least-squares) with nearest-neighbor
interpolation (LS-nn), LS with linear interpolation (LS-lin),
covariance-informed LMMSE (linear minimum mean squared error), and a small CNN
(convolutional neural network). The config has three sections: `experiment`
(the shared resource-grid and channel), `training` (the CNN training loop and
an optional `dataset-generation` block that lets training auto-create the grid
dataset if it does not already exist), and `neural` (the checkpoint path and
architecture the evaluator loads for scoring).

```bash
# 1. Train (auto-generates the grid dataset if missing):
python -m channel_estimation.train experiments/grid-neural-comparison-v1/config.yaml

# 2. Score all three estimators at every SNR (signal-to-noise ratio) and emit one CSV (comma-separated values file) / one plot:
python experiments/grid-neural-comparison-v1/run-experiment.py
```

Requires the ML (machine learning) stack. An experimental Colab helper notebook
is available under `notebooks/`, but validate its outputs before interpreting
results.

The latest validated single-run pattern is:

- LS (least-squares) interpolation error decreases with SNR (signal-to-noise
  ratio) but remains the weakest family;
- the CNN (convolutional neural network) is consistently better than LS-nn
  (least-squares with nearest-neighbor interpolation) and LS-lin
  (least-squares with linear interpolation) in the tested TDL-A (tapped delay
  line A) run;
- LMMSE (linear minimum mean squared error) is best overall and should be
  framed as a model-informed upper classical benchmark, not as a
  resource-equivalent learned baseline.

Run the broader robustness sweep across TDL (tapped delay line) profiles and
seeds with:

```bash
python experiments/grid-neural-comparison-v1/run-sweep.py \
  experiments/grid-neural-comparison-v1/config.yaml \
  --channel-models tdl-a,tdl-b,tdl-c \
  --seeds 42,43,44
```

The sweep retrains each CNN (convolutional neural network) for its channel/seed
configuration, evaluates LS (least-squares), LMMSE (linear minimum mean squared
error), and CNN at every configured SNR (signal-to-noise ratio), and writes raw,
summary, margin, and figure artifacts under `results/`.

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

- low SNR (signal-to-noise ratio) and noisy observations;
- reduced pilot density;
- smaller training datasets;
- a target model-size budget;
- limited inference compute;
- robustness across later, explicitly defined channel conditions.

The exact definitions, assumptions, and exclusions are documented in
[`docs/low-resource-framing.md`](docs/low-resource-framing.md).

## Proposal relevance

This codebase is preparation evidence for scientific ML (machine learning),
reproducible experiment and evaluation, lightweight ML, and network-simulation
work. It
demonstrates a progression from a technical question to baselines, datasets,
tests, configurations, and a report structure. It is not presented as a GSoC
(Google Summer of Code) project, accepted research, or evidence of a 6G
breakthrough.

See [`docs/proposal-alignment.md`](docs/proposal-alignment.md) for the detailed
mapping and boundaries.

## Limitations

- The current NumPy model is a simplified independent Rayleigh/AWGN (additive
  white Gaussian noise) pilot model, not a complete standards-compliant link
  simulation.
- The committed dataset was generated at one SNR (signal-to-noise ratio) with
  unit pilots.
- The current validated single TDL-A (tapped delay line A) notebook run
  supports only a narrow performance statement: the CNN (convolutional neural
  network) beats LS (least-squares) interpolation on that run, while
  covariance-informed LMMSE (linear minimum mean squared error) remains stronger.
- LMMSE now uses a covariance matrix estimated from channel realizations; it is
  only as good as that covariance assumption and the data it is estimated from.
- BER (bit error rate) is implemented as a metric utility but is not yet
  connected to an end-to-end coded OFDM (orthogonal frequency-division
  multiplexing) receiver.
- Generated results require validation before scientific interpretation.

## Report and citation

[`docs/report.md`](docs/report.md) is a mini-paper-style living report. There is
no formal publication or archival citation for this project yet. Until a
versioned release exists, cite the repository URL and commit hash when referring
to a specific experiment.

## License

This project is available under the terms in [`LICENSE`](LICENSE).
