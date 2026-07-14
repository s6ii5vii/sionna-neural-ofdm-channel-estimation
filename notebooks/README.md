# Notebooks

These notebooks explain and visualize the project. They are not the main
experiment engine.

1. `01-ofdm-system-overview.ipynb` introduces the simplified OFDM (orthogonal
   frequency-division multiplexing) pilot model.
2. `02-ls-baseline-demo.ipynb` demonstrates LS (least-squares) estimation and
   NMSE (normalized mean squared error).
3. `03-dataset-exploration.ipynb` inspects the committed sample dataset.
4. `04-low-resource-experiment-demo.ipynb` examines the first constrained profile.
5. `05-colab-grid-baseline.ipynb` runs the Sionna resource-grid baseline on a
   Colab GPU (graphics processing unit) (clone, install, verify the
   `sionna.phy` API (application programming interface), test, and sweep).

Run `python -m pip install -e .` from the repository root before opening them.
Reusable logic belongs in `src/channel_estimation/`; reproducible sweeps belong
in `experiments/`.
