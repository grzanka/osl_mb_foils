# Environment Setup

This project uses a small Conda environment tailored for the depth-dose validation notebook (notebooks/pub_something_2026/mc_ccb/0.1_ccb_mc_depth_validation.ipynb). It includes:

- python 3.11
- pandas (data handling, HDF5 I/O)
- matplotlib (plots)
- scipy (interpolation)
- pytables (pandas HDF5 backend)
- ipykernel (needed for running notebooks in VS Code/Jupyter)

## Create and activate the environment

You can create the environment from the provided YAML (recommended) or via an inline command. Both work on Windows PowerShell and Linux/macOS shells when Conda is on your PATH.

### Option A: Using environment.yml (recommended)
```bash
conda env create -f environment.yml && conda activate osl-mb-foils
```

### Option B: One-liner without the YAML
```bash
conda create -n osl-mb-foils -y -c conda-forge python=3.11 pandas matplotlib scipy pytables ipykernel && conda activate osl-mb-foils
```

After activation, you can run the notebook in Jupyter/VS Code.
