<!-- Agent instructions for the carm-paraver workspace -->
# CARM-Paraver — Agent instructions

Purpose
- Short guide so an AI assistant can quickly be productive in this repository.

Quick summary
- Entry point: Paraver_CARM.py (GUI application using Dash).
- Main dependencies: listed in `requirements.txt` and `pyproject.toml` (Python >= 3.10).
- Typical workflow: install deps, prepare PATH (Paraver and repo), launch via Paraver or run script directly.

## How to run locally

1. Create a virtual environment and install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure the repository root and Paraver's `bin` are on `PATH` or run `./setup.sh /path/to/Paraver/bin`.

3. Typical invocation (when running outside Paraver):

```
python Paraver_CARM.py --csv /path/to/mask.csv /path/to/trace.prv
```

## Notes and conventions
- The app is launched from Paraver via a context menu; when launched this way Paraver prints a link to open the Dash GUI.
- The code expects CARM results CSV files in the user data directory (platformdirs app name "carm") under `roofline/` named `<machine>_roofline.csv`.
- Config templates live under `paraver_carm_configs/` (Intel/IntelV2 subfolders).
- The CLI uses argparse; `--csv` (mask) and the `trace_path` (positional) are required.
- Time units and scaling factors are set inside `Paraver_CARM.py` (see `scaling_factors`).

## Files to inspect first
- `Paraver_CARM.py` — main application (entry point).
- `README.md` — usage notes and counters mapping.

## If something's missing
- Open `README.md` and `Paraver_CARM.py` for repository-specific details (counter names, config mappings).
- When in doubt about run arguments, search for `parser.add_argument` inside `Paraver_CARM.py`.

## Linting, formatting and testing
After code changes, run:

```bash
ruff check --fix
ruff format
pytest
```