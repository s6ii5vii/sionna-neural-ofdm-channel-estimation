# Lightweight Neural OFDM Channel Estimation for Low-Resource 6G-Like Environments

## Abstract

This living report describes an early-stage reproducible study of pilot-based
OFDM channel estimation under constrained conditions. The current codebase
implements a simplified Rayleigh/AWGN simulation, Sionna resource-grid
simulation, classical estimators, dataset tooling, and an intentionally small
neural estimator. No reviewed neural performance claim is made. Experiments evaluate whether a
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
LMMSE is implemented both for the simplified NumPy model and as a
covariance-informed Sionna grid interpolator using the configured TDL profile.

## Neural estimator

The PyTorch MLP flattens real/imaginary pilot features, applies one small hidden
layer, and predicts channel features; a small convolutional network estimates the
channel across a full resource grid. Architecture selection, parameter counts,
training protocol, deterministic seed policy, and best-validation checkpoint
selection are implemented; broader ablations remain to be completed.

## Experiments

The implemented baseline profile sweeps LS NMSE over SNR. The first
low-resource profile reduces pilot density and evaluation sample count while
declaring data and model-size targets. Repeated-seed validation across
standardized Sionna TDL channel profiles is implemented.

## Results

No reviewed neural-estimator results are reported yet.

Future tables should include LS and neural NMSE by SNR, repeated-seed variation,
pilot density, training-data volume, parameter count, model size, and measured
latency. BER should be reported only after an end-to-end decision path exists.

## Limitations

The committed sample dataset uses one SNR, the grid CNN trains at one SNR, the
grid LMMSE benchmark assumes knowledge of the configured TDL profile, and no
end-to-end bit-error-rate receiver is connected yet.

## Future work

1. Review repeated-seed validation results.
2. Evaluate out-of-distribution channel conditions without retraining.
3. Add pilot-pattern and mixed-SNR ablations.
4. Measure memory use and operation count alongside corrected latency metrics.
5. Connect BER to a documented end-to-end receiver.

## References

References will be added as the background and methods are developed. No
placeholder citation should be interpreted as a completed literature review.
