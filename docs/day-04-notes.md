# Day 4 Notes

## What I learned today
- Neural networks need many input-output examples to learn.
- For channel estimation, the input is the noisy received pilot signal.
- The target is the true wireless channel.
- Complex wireless signals can be represented using real and imaginary parts.
- Training, validation, and test splits help evaluate models fairly.

## What I built today
- Generated thousands of simulated wireless channel examples.
- Created known pilot symbols for each example.
- Added complex noise to received pilot signals.
- Converted complex input and target arrays into real-imaginary ML format.
- Split the dataset into training, validation, and test sets.
- Saved the dataset as a NumPy `.npz` file.

## What confused me today
- Why neural networks cannot directly use complex numbers easily.
- Why the dataset shape becomes `(samples, pilots, 2)`.
- Whether generated datasets should be committed to GitHub.

## What I will do next
- Load the generated dataset.
- Build a simple neural network estimator.
- Train the model to predict the true channel from noisy pilot observations.
- Compare the neural estimator against Least Squares.