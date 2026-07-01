---
title: Home
nav_order: 1
description: CARM-Paraver — Cache-Aware Roofline Model analysis for Paraver traces
---

# CARM-Paraver

**CARM-Paraver** is a Dash-based GUI that enables [Cache-Aware Roofline Model (CARM)](https://jp.zirlab.es/carm/) analysis of [Paraver](https://tools.bsc.es/paraver) traces for floating-point operations. It can be launched from the Paraver interface and send labeled events back to Paraver for visualization.

The CARM allows for roofline analysis of your application, displaying its computational bursts as points on the roofline. This can be used to identify bottlenecks and optimization opportunities for the respective code section:

- **Memory-bound points** benefit from optimization strategies that improve data locality and reduce memory traffic.
- **Compute-bound points** benefit from optimization strategies that increase computational intensity.
- The **distance of your points to the roofline** indicates how much performance can be gained by optimizing your code.

## Quick Start

```bash
pip install carm-paraver
```

Ensure Paraver's `bin` directory is on your `PATH`, then use [Extrae](https://github.com/bsc-performance-tools/extrae) to generate a trace with the required hardware counters (see [Trace Requirements](trace-requirements)), load it in Paraver, and launch the CARM GUI via the context menu.

See the [Usage](usage) page for a step-by-step guide.

## Table of Contents

| Page | Description |
|------|-------------|
| [Installation](installation) | Requirements, installation via pip or source, virtual environment setup |
| [Usage](usage) | Step-by-step guide: generating traces, launching the GUI, basic workflow |
| [Trace Requirements](trace-requirements) | Required Extrae hardware counters for Intel and AMD CPUs |
| [Benchmarking](benchmarking) | How to benchmark your architecture using the CARM Tool |
| [GUI Features](features) | Launch configuration, left and right sidebar options, normalization |
| [Performance](performance) | Tips for improving GUI responsiveness with large traces |

## About

CARM-Paraver is developed and maintained by the [CARM Contributors](https://github.com/champ-hub/carm-paraver). It is part of the CHARM (Compression and Hybrid Architectures for Modern computing) ecosystem.
