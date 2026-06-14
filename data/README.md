# Data

## Committed sample dataset

`channel_estimation_dataset.npz` is a small reference dataset generated for the
first neural-estimator experiments. It contains 5,000 simulated examples with
64 pilot observations per example at 10 dB SNR.

The current archive contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `x_train` | `(3500, 64, 2)` | noisy pilot observations |
| `y_train` | `(3500, 64, 2)` | true channel coefficients |
| `x_val` | `(750, 64, 2)` | validation observations |
| `y_val` | `(750, 64, 2)` | validation targets |
| `x_test` | `(750, 64, 2)` | test observations |
| `y_test` | `(750, 64, 2)` | test targets |

The final axis stores real and imaginary components. The committed arrays use
`float64` because they came from the original notebook pipeline. New generated
datasets default to `float32`.

## Regeneration

Use `channel_estimation.dataset.generate_synthetic_dataset` and
`save_npz_dataset` for the current independent Rayleigh/AWGN model. New dataset
versions should record:

- random seed;
- SNR or SNR sampling policy;
- pilot count and density;
- channel model;
- train, validation, and test split policy;
- package and configuration version.

The committed file is a sample and compatibility fixture, not a canonical
benchmark. Do not commit large generated datasets directly. Store them in
`data/generated/` or external artifact storage and document how to reproduce or
retrieve them.
