# Day 2 Notes

## What I learned today
- OFDM systems organize wireless data across time and frequency using a resource grid.
- Rayleigh fading models random wireless channel changes caused by multipath propagation.
- Block fading means the channel stays constant over a short block of time.
- Wireless channel coefficients can be stored as tensors on the GPU.
- GPU tensors must be moved to the CPU before converting them to NumPy for plotting.

## What I built today
- Created a basic OFDM resource grid visualization.
- Initialized a Rayleigh block fading channel in Sionna.
- Generated channel coefficients for a simple SISO wireless setup.
- Plotted channel magnitude over time.
- Simulated 100 independent Rayleigh fading channel examples.
- Plotted the distribution of fading channel magnitudes.

## What confused me today
- Why the first channel magnitude plot was flat.
- Why the channel tensor had many dimensions.
- Why GPU tensors could not be converted directly to NumPy.

## What I will do next
- Move from simple fading visualization to actual pilot-based channel estimation.
- Learn how transmitted pilot symbols help estimate the wireless channel.
- Begin building the classical Least Squares baseline.