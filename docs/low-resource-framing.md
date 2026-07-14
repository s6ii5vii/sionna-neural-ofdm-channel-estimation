# Low-resource framing

## Motivation

Neural channel estimators are often evaluated under broad model-capacity and
data assumptions. This project instead asks whether a deliberately small model
can provide useful estimation behavior when wireless observations, training
data, and deployment compute are constrained.

"Low-resource" is a collection of explicit experiment dimensions, not a claim
that one configuration represents every constrained wireless system.

## Low SNR (signal-to-noise ratio) and noisy channels

Low-SNR (low signal-to-noise ratio) profiles test estimation when pilot
observations are heavily corrupted by noise. Experiments must report the SNR
(signal-to-noise ratio) definition, signal-power convention, random seed policy,
and performance across a range rather than at one favorable operating point.

## Sparse pilots and reduced overhead

Reducing pilot density can improve spectral efficiency but leaves fewer direct
channel observations. The current sparse profile selects a subset of pilot
locations and evaluates LS (least-squares) at those observed locations only.
Full-grid evaluation requires a documented interpolation or model-based
reconstruction method and is not yet implemented.

## Limited training data

Dataset-size ablations should compare fixed train, validation, and test policies
across multiple sample budgets. Test data must remain isolated from model and
hyperparameter selection.

## Limited inference compute

Practical evaluation should include latency on named hardware, parameter count,
serialized model size, and, where feasible, estimated operation count or memory
use. A model is not lightweight merely because it has a short implementation.

## Lightweight model size

The initial PyTorch estimators (a one-hidden-layer MLP (multilayer perceptron)
and a small convolutional grid network) expose their width as a configuration
value. The first profile sets a parameter-count target rather than claiming the
current architecture already meets a deployment requirement.

## Changing channel conditions

Robustness must eventually be tested across channel distributions that differ
from training conditions. Candidate studies include SNR (signal-to-noise ratio)
shifts, changed fading profiles, mobility assumptions, delay spreads, and pilot
patterns. These require validated Sionna configurations before results can be
interpreted.

## Relevance

The constraints reflect settings where data collection, pilot overhead, device
compute, energy, or reliable connectivity may be limited. They provide a useful
experimental lens for edge and constrained wireless systems without asserting
that the simplified current simulator models a deployed 6G network.

## What this project does not claim

- It is not a 6G breakthrough or standards proposal.
- It does not yet demonstrate superiority over classical estimators.
- It is not publication-ready research.
- It does not yet model a complete coded OFDM (orthogonal frequency-division
  multiplexing) link or standardized deployment.
- "Low-resource" does not imply universal efficiency without measured compute,
  memory, and accuracy tradeoffs.
