# Day 3 Notes

## What I learned today
- Least Squares estimates the channel using known pilot symbols.
- The received signal can be modeled as y = h x + n.
- Higher SNR gives cleaner received pilots and better channel estimation.
- NMSE measures how far the estimated channel is from the true channel.

## What I built today
- Created known pilot symbols.
- Simulated a complex wireless channel.
- Added complex noise to the received pilot signal.
- Implemented a manual Least Squares channel estimator.
- Plotted true channel magnitude against LS-estimated channel magnitude.
- Evaluated NMSE across multiple SNR values.

## What confused me today
- How complex numbers represent wireless signals.
- Why NMSE is better than plain MSE.
- How SNR affects estimation quality.

## What I will do next
- Generate more structured training data.
- Prepare pilot observations and true channel labels for the neural network.
- Start building the dataset generation pipeline.