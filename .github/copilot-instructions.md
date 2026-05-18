<!-- Copilot instructions for the carm-paraver workspace -->
# CARM-Paraver — Copilot Instructions

Purpose
- Short guide so an AI assistant can quickly be productive in this repository.

Quick summary
- Entry point: Paraver_CARM.py (GUI application using Dash).
- Main dependencies: listed in `requirements.txt` and `pyproject.toml` (Python >= 3.10).
- Typical workflow: install deps, prepare PATH (Paraver and repo), launch via Paraver or run script directly.

How to run locally

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

Notes and conventions
- The app is launched from Paraver via a context menu; when launched this way Paraver prints a link to open the Dash GUI.
- The code expects CARM results CSV files in `carm_results/roofline` named `<machine>_roofline.csv`.
- Config templates live under `paraver_carm_configs/` (Intel/IntelV2 subfolders).
- The CLI uses argparse; `--csv` (mask) and the `trace_path` (positional) are required.
- Time units and scaling factors are set inside `Paraver_CARM.py` (see `scaling_factors`).
- Large traces can be slow; prefer analyzing a focused time window (README recommends ~50ms sections).

Files to inspect first
- `Paraver_CARM.py` — main application (entry point).
- `requirements.txt` and `pyproject.toml` — dependency and metadata.
- `README.md` — usage notes and counters mapping.

Common tasks you can ask the assistant
- "Open the GUI locally with example traces" — installs deps and runs the app.
- "Add a unit test for X" — propose test harness and add test files.
- "Refactor the large function that processes counters" — identify hotspots, propose and apply changes.
- "Add CI that runs ruff and basic smoke test" — create a minimal workflow.

Suggested next agent customizations
- `create-infra-agent` — automates install/run smoke tests and checks `pyproject.toml`.
- `create-refactor-skill` — scans for large functions (e.g., long loops in `Paraver_CARM.py`) and proposes modularization.

If something's missing
- Open `README.md` and `Paraver_CARM.py` for repository-specific details (counter names, config mappings).
- When in doubt about run arguments, search for `parser.add_argument` inside `Paraver_CARM.py`.

Contact
- Leave an issue in the repo or ask the maintainer for sample traces and environment specifics.
