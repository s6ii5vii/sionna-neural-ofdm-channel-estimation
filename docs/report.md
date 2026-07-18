# Lightweight Neural OFDM (orthogonal frequency-division multiplexing) Channel Estimation for Low-Resource 6G-Like Environments

## Abstract

This living report describes an early-stage reproducible study of pilot-based
OFDM (orthogonal frequency-division multiplexing) channel estimation under
constrained conditions. The current codebase implements a simplified
Rayleigh/AWGN (additive white Gaussian noise) simulation, Sionna resource-grid
simulation, classical estimators, dataset tooling, and an intentionally small
neural estimator. A July 2026 validation notebook shows the CNN (convolutional
neural network) improving over LS (least-squares) interpolation across TDL-A
(tapped delay line A), TDL-B (tapped delay line B), and TDL-C (tapped delay line
C), while covariance-informed LMMSE (linear minimum mean squared error) remains
the strongest estimator. Experiments evaluate whether a lightweight model offers
useful error or robustness tradeoffs under low SNR (signal-to-noise ratio),
sparse pilots, limited data, and compute constraints.

## Introduction

Accurate channel estimates are required for coherent wireless reception.
Classical estimators provide strong, interpretable baselines, while learned
estimators may help when model assumptions are incomplete. Any learned method
must justify its additional data and compute cost through controlled comparison.

## Background

Document OFDM (orthogonal frequency-division multiplexing) resource grids,
pilots, fading, AWGN (additive white Gaussian noise), LS (least-squares)
estimation, LMMSE (linear minimum mean squared error) assumptions, and relevant
neural channel-estimation literature here. References must be added before this
section is treated as a literature review.

## Problem setup

The current implemented model is `y = h*x + n`, with independent unit-power
Rayleigh coefficients, known non-zero pilots, and circular complex Gaussian
noise. Complex values are represented as real/imaginary feature pairs for ML
(machine learning).

## Low-resource constraints

The study will vary SNR (signal-to-noise ratio), pilot density,
training-dataset size, model capacity, and measured inference cost. The initial
sparse-pilot experiment evaluates only observed positions; full-grid
reconstruction is future work.

## Methods

Experiments are configured in YAML (YAML Ain't Markup Language) and run through
reusable package modules. Seeds, sample counts, SNR (signal-to-noise ratio)
values, pilot density, and output paths are explicit. Figures and CSV
(comma-separated values) tables are generated rather than manually transcribed.

## Baselines

LS (least-squares) estimation is implemented as element-wise division by known
non-zero pilots. LMMSE (linear minimum mean squared error) is implemented both
for the simplified NumPy model and as a covariance-informed Sionna grid
interpolator using the configured TDL (tapped delay line) profile.

## Neural estimator

The PyTorch MLP (multilayer perceptron) flattens real/imaginary pilot features,
applies one small hidden layer, and predicts channel features; a small CNN
(convolutional neural network) estimates the channel across a full resource
grid. Architecture selection, parameter counts, training protocol,
deterministic seed policy, and best-validation checkpoint selection are
implemented; broader ablations remain to be completed.

## Experiments

The implemented baseline profile sweeps LS (least-squares) NMSE (normalized
mean squared error) over SNR (signal-to-noise ratio). The first low-resource
profile reduces pilot density and evaluation sample count while declaring data
and model-size targets. The grid neural comparison evaluates LS-nn
(least-squares with nearest-neighbor interpolation), LS-lin (least-squares with
linear interpolation), model-informed LMMSE (linear minimum mean squared error),
and a small CNN (convolutional neural network) on a Sionna TDL (tapped delay
line) resource grid. Repeated-seed validation across standardized Sionna TDL
channel profiles is implemented.

## Results

The latest validated notebook run used the grid neural comparison over TDL-A
(tapped delay line A), TDL-B (tapped delay line B), and TDL-C (tapped delay line
C), with three random seeds per channel profile and 1,000 evaluation samples per
SNR (signal-to-noise ratio). It produced the expected monotonic NMSE (normalized
mean squared error) improvement with SNR for every estimator. The CNN
(convolutional neural network) substantially reduced NMSE relative to LS-lin
(least-squares with linear interpolation), with 100% win rate across the 15
channel/SNR combinations and mean improvement ranging from 78.75% to 87.07%.
LMMSE (linear minimum mean squared error) still produced the lowest NMSE in
every tested case. This supports a stronger but still bounded conclusion: the
lightweight CNN is robustly useful versus LS interpolation across the tested
TDL-A/B/C settings, but it does not beat the model-informed LMMSE benchmark.

The next evidence target is out-of-distribution evaluation without retraining.
Future tables should include pilot density, training-data volume, parameter
count, model size, and measured latency. BER (bit error rate) should be reported
only after an end-to-end decision path exists.

## Limitations

The committed sample dataset uses one SNR (signal-to-noise ratio), each grid
CNN (convolutional neural network) in the sweep trains at one dataset-generation
SNR, the grid LMMSE (linear minimum mean squared error) benchmark assumes
knowledge of the configured TDL (tapped delay line) profile, and no end-to-end
BER (bit error rate) receiver is connected yet.

## Future work

1. Evaluate out-of-distribution channel conditions without retraining.
2. Add cross-profile training/evaluation matrices.
3. Add pilot-pattern and mixed-SNR ablations.
4. Measure memory use and operation count alongside corrected latency metrics.
5. Connect BER to a documented end-to-end receiver.

## References

References will be added as the background and methods are developed. No
placeholder citation should be interpreted as a completed literature review.
