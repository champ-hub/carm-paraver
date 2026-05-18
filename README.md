# CARM-Paraver GUI

This Graphical User Interface was developed to allow for the analysis of [Paraver](https://tools.bsc.es/paraver) traces in the scope of the Cache-Aware Roofline Model (CARM) for floating-point operations. This GUI relies on CARM results obtained via the CARM Tool which can be found in its ([Github repository](https://github.com/champ-hub/carm-roofline)), for instructions on running the CARM Tool please consult the README and other documentation available in its repository. For instruction on running Paraver and obtaining Paraver/Extrae traces please consult the Paraver/Extrae documentation.

# Requirements
- python (tested with 3.9.25, 3.10.12, 3.12.3)
- [Paraver](https://tools.bsc.es/downloads)

# How to use

## Installation
You can install the CARM Paraver GUI through pip:
```bash
pip install carm-paraver
```
You can also install it from source by cloning this repository and running:
```bash
pip install .
```
If you have version conflicts with the dependencies, you can use a Python virtual environment to install the package and its dependencies in an isolated environment. To do this, you can run:
```bash
python -m venv .venv
source .venv/bin/activate
pip install carm-paraver
```
If you install in a virtual environment, make sure to run Paraver from the same environment:
```bash
source .venv/bin/activate
wxparaver
```

## Running
The GUI is launched via the Paraver interface like so:
1. Load a Paraver trace with the required counters, and zoom into a section of interest.
    - **GUI performance is heavily dependent on the time range selected. It is recommended the analysis be focused on a region with a manageable number of events.**
2. Right click the timeline and select the option to launch the CARM GUI.
3. Configure the options within the Paraver interface to your liking (these can also be changed within the CARM GUI), and click "Run".
4. Click the link printed in the Paraver console to open the GUI in your browser.

It is also possible to launch the GUI from outside a Paraver timeline, using the *Run Application* option in the Paraver interface.

If you get any errors, be sure to consult the Setup instructions below.

## Setup

### Setting up your PATH
carm-paraver needs `paramedir` to be in your PATH in order to run. To add it, add paraver's bin directory to your PATH. You can make this permanent by appending it to your .bashrc or .bash_profile (change the path accordingly):

```bash
export PATH=/path/to/paraver/bin:$PATH
```

### Paraver Trace Requirements
Avoid labeling regions **with MPI calls inside them**. Focus on labeling regions of pure computation, as MPI calls will prevent region and hardware counter timestamps from matching, which is required for the CARM analysis.

Keep in mind the CARM GUI needs CARM results from the [CARM Tool](https://github.com/champ-hub/carm-roofline) in order to plot Paraver timestamps, this repository includes some example CARM results sourced from the [MareNostrum 5](https://www.bsc.es/supportkc/docs/MareNostrum5/overview/) supercomputer in the carm_results folder. The ability to add additional CARM results will be added soon.

To use the CARM interface, a Paraver/Extrae trace is needed which was instrumented with Intel FP and memory counters such as:

| FP/Mem Operation       | Intel Counter                              |
| ---------------------- | ------------------------------------------ |
| Intel FP Scalar DP     | `FP_ARITH_INST_RETIRED:SCALAR_DOUBLE`      |
| Intel FP Scalar SP     | `FP_ARITH_INST_RETIRED:SCALAR_SINGLE`      |
| Intel FP SSE DP        | `FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE` |
| Intel FP SSE SP        | `FP_ARITH_INST_RETIRED:128B_PACKED_SINGLE` |
| Intel FP AVX2 DP       | `FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE` |
| Intel FP AVX2 SP       | `FP_ARITH_INST_RETIRED:256B_PACKED_SINGLE` |
| Intel FP AVX512 DP     | `FP_ARITH_INST_RETIRED:512B_PACKED_DOUBLE` |
| Intel FP AVX512 SP     | `FP_ARITH_INST_RETIRED:512B_PACKED_SINGLE` |
| Intel Loads            | `MEM_INST_RETIRED:ALL_LOADS`               |
| Intel Stores           | `MEM_INST_RETIRED:ALL_STORES`              |
| Intel Loads and Stores | `MEM_INST_RETIRED:ALL`                     |

In AMD CPUs, the following counters can be used:

| FP/Mem Operation       | AMD Counter                                    |
| ---------------------- | ---------------------------------------------- |
| Mul/Add Flops          | `retired_sse_avx_operations:dp_mult_add_flops` |
| Add/Sub Flops          | `retired_sse_avx_operations:dp_add_sub_flops`  |
| Mul Flops              | `retired_sse_avx_operations:dp_mult_flops`     |
| Div Flops              | `retired_sse_avx_operations:dp_div_flops`      |
| Mul/Add Flops (SP)     | `retired_sse_avx_operations:sp_mult_add_flops` |
| Add/Sub Flops (SP)     | `retired_sse_avx_operations:sp_add_sub_flops`  |
| Mul Flops (SP)         | `retired_sse_avx_operations:sp_mult_flops`     |
| Div Flops (SP)         | `retired_sse_avx_operations:sp_div_flops`      |
| Loads                  | `ls_dispatch:ld_dispatch`                      |
| Stores                 | `ls_dispatch:store_dispatch`                   |

At least one FP and one memory counter (separate load and store counters are recommended for a more detailed analysis) must be available in the trace to be analyzed, otherwise the CARM analysis is not possible. It is also recommended to keep all counters in a single counter set (when obtaining the trace via Extrae), this usually allows for all FP counters of a given precision (DP or SP) and the load and store counters. Precisions can also be mixed but the amount of counters used must fit in a single counter set.

## Features

### Left Sidebar

**Use Paraver/CARM Colors:**
Controls which coloring scheme is used in the CARM GUI: the same colors as the Paraver timeline (if enabled) or the selected CARM GUI coloring scheme (see right sidebar options).

**Use Semantic Window / All Timestamps:**
Controls whether the Paraver semantic window is used: if enabled, displays only the timestamps that are within the semantic window of the Paraver timeline. If disabled, all timestamps in the trace are displayed in the CARM GUI.

**Plot Raw/Accumulated Values:**
Controls whether timestamps (with the same underlying Paraver value) are averaged. Allows for similar timestamps to be grouped into a single, per-thread point, or to plot all timestamps individually.

**Re-Sync Timeline With Paraver:**
Re-syncs the plotted timestamps in the CARM GUI with the timestamps being viewed in the Paraver timeline from which the CARM GUI was launched. This first requires the **Time Sync** button to be clicked on the Paraver side, the CARM GUI will usually keep itself synced to the Paraver timeline whenever the **Time Sync** button is clicked in the Paraver interface. In case the user changes the displayed timestamps in the CARM GUI and wishes to return to the same interval that they have in the Paraver timeline, they can use the **Re-Sync Timeline With Paraver** button.

**Send Timestamps Roof Labels:**
Labels the timestamps based on which roof they are under, for viewing in Paraver. The path of the generated trace will be printed in the Paraver console, and can be clicked to open the trace in Paraver. You can then select the trace and click *New single timeline window* to view the timestamps with the new labels.

**Send Timestamps LD/ST Percentage Colors:**
Same as above, but labels the timestamps based on the percentage of loads to stores.

**Send Timestamps SP/DP Percentage Colors:**
Same as above, but labels the timestamps based on the percentage of single to double precision operations.

### Right Sidebar
The right sidebar controls the CARM GUI specific features, which include various filtering and coloring options as well as graphical annotations.

Useful options include:
- **Filter points** by vector ISA or precision
- **Color points** based on thread ID, precision, vector ISA or load/store ratio
    - Note that this requires the left sidebar option to be set to "Use CARM GUI Colors".

The plot can be configured to normalize the performance roof to the number of threads. The normalized roofs represent the performance per thread, which matches the Paraver timestamps (also per thread). This mode is recommended when relating application performance to the underlying hardware. The non-normalized roofs represent the overall performance of the architecture, and is best for understanding the hardware capabilities.