---
title: Benchmarking
nav_order: 5
parent: Home
---

# CARM Benchmarking

To display your architecture's roofline in the CARM GUI, you need to benchmark it using the **[CARM Tool](https://github.com/champ-hub/carm-roofline)**.

**Important:** For compatibility, use the [latest version of the CARM Tool](https://pypi.org/project/carm-roofline/) from PyPI, which is also installed via `pip`.

## How It Works

1. Run the CARM Tool on the target architecture to measure peak performance (flops/cycle) and peak memory bandwidth for each memory level.
2. The tool generates a roofline CSV file for your machine.
3. CARM-Paraver reads this file to display the roofline in the GUI.

The data typically lives in `~/.local/share/carm/roofline/<machine>_roofline.csv`. If not, check where [platformdirs](https://pypi.org/project/platformdirs/4.1.0/) stores the data on your system.

## Sample Rooflines

The CARM Tool ships with a set of sample rooflines from a **MareNostrum 5 GPP node**, which can be used for testing and development without benchmarking your own system.
