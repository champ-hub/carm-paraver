# CARM-Paraver GUI

This GUI allows the analysis of [Paraver](https://tools.bsc.es/paraver) traces on the Cache-Aware Roofline Model (CARM) for floating-point operations. It can be launched from the Paraver interface and send labeled events back to Paraver for visualization.

# Requirements
- Python (tested with 3.9.25, 3.10.12, 3.12.3)
- [Paraver, Extrae](https://tools.bsc.es/downloads)

# How to use

## Installation
**The recommended way to install the package** is via `pip`:
```bash
pip install carm-paraver
```
Alternatively, you can install it from source by cloning this repository and running:
```bash
pip install .
```
If the install fails due to dependency conflicts, you can use a Python virtual environment to install the package and its dependencies in an isolated environment. To do this, you can run:
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

## First-time Setup
CARM-Paraver needs `paramedir` to be in your PATH in order to run. To add it, add Paraver's bin directory to your PATH. You can make this permanent by appending it to your `.bashrc` or `.bash_profile` (change the path accordingly):

```bash
export PATH=/path/to/paraver/bin:$PATH
```

## Running
The GUI is launched via the Paraver interface like so:
1. Use [Extrae](https://github.com/bsc-performance-tools/extrae) to generate a trace with the required counters ([see how to configure Extrae below](#paraver-trace-requirements)).
2. Load the trace in Paraver, and zoom into a section of interest.
3. Right click the timeline and select the option to launch the CARM GUI.
4. Configure the options in Paraver to your liking (see [Launch Configuration](#launch-configuration)), and click "Run".
5. Click the link printed in the Paraver console to open the GUI in your browser.

You will now have the CARM GUI open, showing the architecture's roofline, and the events from the Paraver trace represented as points on the plot. Their position on the roofline, which is determined by their performance and arithmetic intensity, can be used to identify bottlenecks and optimization opportunities for the respective code section. Check the [CARM GUI Features](#carm-gui-features) section for more details about the GUI, and how you can label events and send them back to Paraver for visualization.

If you get any errors, be sure to consult the [First-time Setup](#first-time-setup) and [Paraver Trace Requirements](#paraver-trace-requirements) sections.

## Paraver Trace Requirements

To enable CARM analysis, your Paraver trace needs to include information on the floating-point and memory operations performed by the application. To do this, [configure Extrae](https://tools.bsc.es/doc/html/extrae/xml.html#xml-section-performance-counters) to include the counters in the tables below.

#### Which counters to include?
Include only the necessary counters for your analysis, so they fit in a single counter set. If too many counters are active, accuracy may be reduced.

Take the application examples below. For each case, the tables below indicate which counters you should include in your Extrae configuration:
- **App 1**: The application only uses double precision, but you don't know which vector ISAs it uses.
- **App 2**: The application is vectorized with AVX2, using both precisions.

If you are unsure, include all counters and prune them later as you learn more about the application. Using separate load and store counters is recommended, as it allows for a more detailed analysis.

#### Intel CPUs
| FP/Mem Operation | Intel Counter                              | App 1   | App 2   |
| ---------------- | ------------------------------------------ | ------- | ------- |
| Scalar DP Insts  | `FP_ARITH_INST_RETIRED:SCALAR_DOUBLE`      | &check; | &check; |
| Scalar SP Insts  | `FP_ARITH_INST_RETIRED:SCALAR_SINGLE`      |         | &check; |
| SSE DP Insts     | `FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE` | &check; |         |
| SSE SP Insts     | `FP_ARITH_INST_RETIRED:128B_PACKED_SINGLE` |         |         |
| AVX2 DP Insts    | `FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE` | &check; | &check; |
| AVX2 SP Insts    | `FP_ARITH_INST_RETIRED:256B_PACKED_SINGLE` |         | &check; |
| AVX512 DP Insts  | `FP_ARITH_INST_RETIRED:512B_PACKED_DOUBLE` | &check; |         |
| AVX512 SP Insts  | `FP_ARITH_INST_RETIRED:512B_PACKED_SINGLE` |         |         |
| Loads            | `MEM_INST_RETIRED:ALL_LOADS`               | &check; | &check; |
| Stores           | `MEM_INST_RETIRED:ALL_STORES`              | &check; | &check; |
| Loads and Stores | `MEM_INST_RETIRED:ALL`                     |         |         |

#### AMD CPUs
| FP/Mem Operation | AMD Counter                                    | App 1   | App 2   |
| ---------------- | ---------------------------------------------- | ------- | ------- |
| Mul/Add DP Flops | `retired_sse_avx_operations:dp_mult_add_flops` | &check; | &check; |
| Mul/Add SP Flops | `retired_sse_avx_operations:sp_mult_add_flops` |         | &check; |
| Add/Sub DP Flops | `retired_sse_avx_operations:dp_add_sub_flops`  | &check; | &check; |
| Add/Sub SP Flops | `retired_sse_avx_operations:sp_add_sub_flops`  |         | &check; |
| Mul DP Flops     | `retired_sse_avx_operations:dp_mult_flops`     | &check; | &check; |
| Mul SP Flops     | `retired_sse_avx_operations:sp_mult_flops`     |         | &check; |
| Div DP Flops     | `retired_sse_avx_operations:dp_div_flops`      | &check; | &check; |
| Div SP Flops     | `retired_sse_avx_operations:sp_div_flops`      |         | &check; |
| Loads            | `ls_dispatch:ld_dispatch`                      | &check; | &check; |
| Stores           | `ls_dispatch:store_dispatch`                   | &check; | &check; |

#### Additional recommendations
For best results, when labeling your code with [Extrae events](https://tools.bsc.es/doc/html/extrae/api.html), e.g. with `Extrae_eventandcounters` calls, **avoid labeling regions that include MPI calls**. Focus on labeling regions of pure computation, as MPI calls will cause the region and hardware counter timestamps to not match, preventing them from being shown on the CARM GUI.

## CARM Benchmarking

To benchmark your architecture and display its roofline in the CARM GUI, use the [CARM Tool](https://github.com/champ-hub/carm-roofline). **NOTE: Compatibility between the two tools is in development, more details to come**.

This tool ships a series of sample rooflines from a MareNostrum 5 GPP node.

## CARM GUI Features

### Launch Configuration
**Use window colors:**
Controls which coloring scheme is used in the CARM GUI: the same colors as the Paraver timeline (if enabled) or the selected CARM GUI coloring scheme (see right sidebar options).

**Use Semantic Window:**
Controls whether the Paraver semantic window is used: if enabled, the GUI displays only the timestamps that are within the semantic window of the Paraver timeline. If disabled, all timestamps in the trace are displayed.

**Accumulate values:**
Controls whether timestamps (with the same underlying Paraver value) are averaged. Allows for similar timestamps to be grouped into a single, per-thread point, or to plot all timestamps individually.

### Left Sidebar

**Use Paraver/CARM Colors:**
Same as above's "Use window colors"

**Use Semantic Window / All Timestamps:**
Same as above's "Use Semantic Window"

**Plot Raw/Accumulated Values:**
Same as above's "Accumulate values"

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

## GUI Performance
The GUI may become slow when plotting a very large number of events. To improve performance, you can:
- Enable the "Accumulate values" option to group similar events into a single point.
- Enable the "Use Semantic Window" option to only plot events visible in Paraver.
- Focus your analysis on a smaller time window in the Paraver timeline.