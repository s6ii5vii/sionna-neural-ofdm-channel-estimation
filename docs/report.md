# Lightweight Neural OFDM Channel Estimation for Low-Resource 6G-Like Environments

## Abstract

This living report describes an early-stage reproducible study of pilot-based
OFDM channel estimation under constrained conditions. The current codebase
implements a simplified Rayleigh/AWGN simulation, a least-squares baseline,
dataset tooling, and an intentionally small neural-estimator skeleton. No neural
performance claim is made. Future experiments will evaluate whether a
lightweight model offers useful error or robustness tradeoffs under low SNR,
sparse pilots, limited data, and compute constraints.

## Introduction

Accurate channel estimates are required for coherent wireless reception.
Classical estimators provide strong, interpretable baselines, while learned
estimators may help when model assumptions are incomplete. Any learned method
must justify its additional data and compute cost through controlled comparison.

## Background

Document OFDM resource grids, pilots, fading, AWGN, LS estimation, LMMSE
assumptions, and relevant neural channel-estimation literature here. References
must be added before this section is treated as a literature review.

## Problem setup

The current implemented model is `y = h*x + n`, with independent unit-power
Rayleigh coefficients, known non-zero pilots, and circular complex Gaussian
noise. Complex values are represented as real/imaginary feature pairs for ML.

## Low-resource constraints

The study will vary SNR, pilot density, training-dataset size, model capacity,
and measured inference cost. The initial sparse-pilot experiment evaluates only
observed positions; full-grid reconstruction is future work.

## Methods

Experiments are configured in YAML and run through reusable package modules.
Seeds, sample counts, SNR values, pilot density, and output paths are explicit.
Figures and CSV tables are generated rather than manually transcribed.

## Baselines

LS estimation is implemented as element-wise division by known non-zero pilots.
LMMSE remains planned until channel and noise covariance assumptions are
documented and validated.

## Neural estimator

The initial Keras model flattens real/imaginary pilot features, applies one small
hidden dense layer, and predicts channel features. Architecture selection,
parameter counts, training protocol, and ablations remain to be completed.

## Experiments

The implemented baseline profile sweeps LS NMSE over SNR. The first
low-resource profile reduces pilot density and evaluation sample count while
declaring data and model-size targets. Repeated seeds and standardized Sionna
channel profiles are planned.

## Results

No reviewed neural-estimator results are reported yet.

Future tables should include LS and neural NMSE by SNR, repeated-seed variation,
pilot density, training-data volume, parameter count, model size, and measured
latency. BER should be reported only after an end-to-end decision path exists.

## Limitations

The current simulator is simplified, the committed dataset uses one SNR, sparse
pilots lack full-grid reconstruction, LMMSE is absent, and the neural baseline
has not been trained or evaluated in a controlled study.

## Future work

1. Validate a complete Sionna OFDM pipeline.
2. Implement sparse-pilot interpolation or full-grid learned reconstruction.
3. Add a covariance-defined LMMSE baseline.
4. Train and evaluate the lightweight estimator.
5. Add repeated seeds and out-of-distribution channel conditions.
6. Measure model size, latency, and memory use.
7. Connect BER to a documented end-to-end receiver.

## References

References will be added as the background and methods are developed. No
placeholder citation should be interpreted as a completed literature review.
