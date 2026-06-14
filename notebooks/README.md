# Notebooks

These notebooks explain and visualize the project. They are not the main
experiment engine.

1. `01_ofdm_system_overview.ipynb` introduces the simplified pilot model.
2. `02_ls_baseline_demo.ipynb` demonstrates LS estimation and NMSE.
3. `03_dataset_exploration.ipynb` inspects the committed sample dataset.
4. `04_low_resource_experiment_demo.ipynb` examines the first constrained profile.

Run `python -m pip install -e .` from the repository root before opening them.
Reusable logic belongs in `src/channel_estimation/`; reproducible sweeps belong
in `experiments/`.
