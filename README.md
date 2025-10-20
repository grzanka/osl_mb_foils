Directory summary

- `src/` — main Python package containing data processing and analysis modules (e.g., `analysis.py`, `detector.py`, `paths.py`, `visualise/`, `plot.py`, `settings.py`).
- `notebooks/` — Jupyter notebooks for exploration and published materials, including a `playground/` and `pub_materials_2024/` with example analyses.

# osl_mb_foils
Data analysis of 2D OSL magnesium-borate foils

## Quick setup (Linux)

Copy-paste these commands on a fresh Linux machine. This project uses Poetry (see `pyproject.toml`). The instructions create an in-project virtual environment, install dependencies, and run a quick check.

Prerequisites: git, Python 3.11 (or compatible), and curl or wget.

1) Clone the repo

```bash
git clone https://github.com/grzanka/osl_mb_foils.git
```

```bash
cd osl_mb_foils
```

2) Install Poetry (if you don't have it)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

```bash
export PATH="$HOME/.local/bin:$PATH"
```

3) Create and use the project virtual environment, then install deps

```bash
poetry install
```

```bash
poetry shell
```

Poetry virtual environment location and removal

By default this project configures Poetry to create an in-project virtual environment (a `.venv` folder inside the repo). To remove and rebuild the environment:

```bash
rm -rf .venv
```

```bash
poetry install
```

4) Quick checks

```bash
python -c "import numpy, pandas, matplotlib, scipy, cv2, tables; print('ok')"
```

```bash
python -c "from src import analysis; print('analysis module loaded:', hasattr(analysis, '__file__'))"
```

- `data/` and `raw/` — measured data and generated HDF5 files (large files may be stored externally).

1) Clone the repo

```bash
git clone https://github.com/grzanka/osl_mb_foils.git
```

```bash
cd osl_mb_foils
```

2) Install Poetry (if you don't have it)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

```bash
export PATH="$HOME/.local/bin:$PATH"
```

3) Create and use the project virtual environment, then install deps

```bash
poetry install
```

```bash
poetry shell
```

4) Quick checks

```bash
python -c "import numpy, pandas, matplotlib, scipy, cv2, tables; print('ok')"
```

```bash
python -c "from src import analysis; print('analysis module loaded:', hasattr(analysis, '__file__'))"
```

Notes
- Python version is specified in `pyproject.toml` (>= 3.11). Use pyenv or system package manager to install it if needed.
- Developer tools (pre-commit, yapf) are listed under the `dev` group in Poetry; use `poetry install --with dev` if needed.
- Data is stored in `data/` and `raw/` directories; large files may not be included in the repository.

That's it — you should be ready to run notebooks under `notebooks/` or scripts under `src/`.

Notes
- Python version is specified in `pyproject.toml` (>= 3.11). Use pyenv or system package manager to install it if needed.
- Developer tools (pre-commit, yapf) are listed under the `dev` group in Poetry; use `poetry install --with dev` if needed.
- Data is stored in `data/` and `raw/` directories; large files may not be included in the repository.

That's it — you should be ready to run notebooks under `notebooks/` or scripts under `src/`.
