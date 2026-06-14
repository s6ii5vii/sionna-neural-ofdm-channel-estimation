# Lightweight Neural OFDM Channel Estimation for Low-Resource 6G-Like Environments

An early-stage reproducible research software project for evaluating classical
and lightweight neural channel estimators under constrained wireless
conditions.

## Research question

Can a lightweight neural estimator improve pilot-based OFDM channel estimation
under low SNR, sparse pilots, limited training data, and limited inference
compute while remaining practical for constrained deployment scenarios?

The repository currently establishes the simulation, dataset, least-squares
(LS) baseline, metrics, and experiment structure needed to investigate that
question. It does not yet establish that a neural estimator outperforms the
classical baseline.

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
- a small TensorFlow/Keras estimator skeleton;
- tests for datasets, metrics, and the LS baseline.

Planned:

- validated full OFDM resource-grid experiments in Sionna;
- interpolation or estimation across unobserved subcarriers for sparse pilots;
- LMMSE with explicit covariance assumptions;
- neural training and controlled ablations;
- BER evaluation through an end-to-end link;
- runtime and parameter-count reporting.

## Repository structure

```text
src/channel_estimation/   reusable simulation, baseline, metric, model, and I/O code
experiments/              versioned experiment configurations and runners
notebooks/                explanation, visualization, and guided demos
data/                     committed sample data and dataset documentation
results/                  generated figures and tables
docs/                     research framing, design, alignment, and report scaffold
tests/                    focused correctness tests
archive/learning_log/     original dated development notes
```

## Quickstart

Python 3.10-3.12 is recommended for the broadest TensorFlow and Sionna
compatibility.

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

`requirements.txt` contains the complete expected environment. TensorFlow and
Sionna can be heavy and platform-sensitive, so a supported Colab or GPU
environment may be preferable.

## Run the LS baseline

```bash
python experiments/baseline_ls/run_experiment.py
```

The runner reads `experiments/baseline_ls/config.yaml` and writes:

- `results/tables/baseline_ls.csv`
- `results/figures/baseline_ls_nmse.png`

Run the implemented classical portion of the first constrained profile with:

```bash
python experiments/low_resource_v1/run_experiment.py
```

The sparse-pilot profile currently evaluates only selected pilot locations. It
does not yet reconstruct unobserved subcarriers.

## Inspect or regenerate the dataset

```python
from channel_estimation.dataset import inspect_dataset, load_npz_dataset

dataset = load_npz_dataset("data/channel_estimation_dataset.npz")
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
[`docs/low_resource_framing.md`](docs/low_resource_framing.md).

## Proposal relevance

This codebase is preparation evidence for scientific ML, reproducible
experiment and evaluation, lightweight ML, and network-simulation work. It
demonstrates a progression from a technical question to baselines, datasets,
tests, configurations, and a report structure. It is not presented as a GSoC
project, accepted research, or evidence of a 6G breakthrough.

See [`docs/proposal_alignment.md`](docs/proposal_alignment.md) for the detailed
mapping and boundaries.

## Limitations

- The current NumPy model is a simplified independent Rayleigh/AWGN pilot model,
  not a complete standards-compliant link simulation.
- The committed dataset was generated at one SNR with unit pilots.
- No neural performance results are reported yet.
- LMMSE is intentionally unimplemented until covariance assumptions are
  specified.
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
