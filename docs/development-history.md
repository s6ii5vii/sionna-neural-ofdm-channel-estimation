# Development history

The project began as a sequence of exploratory notebooks and was later
restructured into a reproducible research codebase. The original dated notes
are retained under `archive/learning-log/`.

## Environment and Sionna validation

The initial work established a working TensorFlow and NVIDIA Sionna environment,
verified imports and accelerator visibility, and introduced the relationship
between OFDM (orthogonal frequency-division multiplexing) resource grids,
fading channels, and channel estimation.

## Basic OFDM (orthogonal frequency-division multiplexing) simulation and pilot structure

The next stage created a simple OFDM (orthogonal frequency-division
multiplexing) resource grid, generated Rayleigh block fading examples,
inspected channel tensor shapes, and visualized channel magnitudes. This
provided the first concrete model of pilots, subcarriers, and wireless channel
variation.

## Least-squares estimation and SNR experiments

A scalar pilot model, `y = h*x + n`, was used to implement manual LS
(least-squares) channel estimation. The estimator was evaluated across SNR
(signal-to-noise ratio) values using NMSE (normalized mean squared error).
These experiments now form the basis of the reusable LS baseline and metric
modules.

## Dataset generation

The exploratory pipeline generated 5,000 independent Rayleigh/AWGN (additive
white Gaussian noise) examples, represented complex observations and targets as
real/imaginary feature pairs, created train, validation, and test splits, and
saved them to NPZ (NumPy compressed archive). That artifact is retained as a
sample dataset while generation logic now lives in the package.

## Reproducible research restructure

The codebase was reorganized around a `src` package, versioned experiments,
tests, research-facing documentation, generated result directories, and
explanatory notebooks. Future development should extend these shared modules
rather than creating isolated notebook implementations.
