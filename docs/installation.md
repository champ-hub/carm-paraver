---
title: Installation
nav_order: 2
parent: Home
---

# Installation

## Requirements

- **Python** 3.9 or later
- **[Paraver](https://tools.bsc.es/downloads)** (version 4.12 and later) and **[Extrae](https://github.com/bsc-performance-tools/extrae)** — trace visualization and generation tools from BSC.

## Installing via pip

The recommended way to install CARM-Paraver is from PyPI:

```bash
pip install carm-paraver
```

### Using a Virtual Environment

If you encounter dependency conflicts, use a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install carm-paraver
```

When using a virtual environment, you must run Paraver from the same environment so it can find the CARM-Paraver executable:

```bash
source .venv/bin/activate
wxparaver
```

## Installing from Source

Alternatively, you can install from the repository's source. Clone the repository and install:

```bash
git clone https://github.com/champ-hub/carm-paraver.git
cd carm-paraver
pip install .
```

## First-time Setup: Adding Paraver to PATH

CARM-Paraver needs `paramedir` (part of the Paraver installation) to be available on your system `PATH`. Add Paraver's `bin` directory to your shell configuration:

```bash
# Add this to your .bashrc or .bash_profile (adjust the path as needed)
export PATH=/path/to/paraver/bin:$PATH
```

After updating the file, reload it or start a new terminal session:

```bash
source ~/.bashrc
```
