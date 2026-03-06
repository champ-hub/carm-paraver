# carm-paraver

This Graphical User Interface was developed to allow for the analysis of [Paraver](https://tools.bsc.es/paraver) traces in the scope of the Cache-Aware Roofline Model (CARM) for floating-point operations. This GUI relies on CARM results obtained via the CARM Tool which can be found in its ([Github repository](https://github.com/champ-hub/carm-roofline)), for instructions on running the CARM Tool please consult the README and other documentation available in its repository. For instruction on running Paraver and obtaining Paraver/Extrae traces please consult the Paraver/Extrae documentation.

# Requirements
- python (tested with python 3.10.12, 3.12.3)
    - dash
    - dash-bootstrap-components
    - dash-daq
    - numpy
    - pandas
    - plotly

- [Paraver](https://tools.bsc.es/downloads)

# How to use

## Setup
The GUI is launched via the Paraver interface, the option to do so can be found by right clicking any Paraver timeline, and then expanding the "Run" dropdown were the CARM option can be selected. This will launch a window within Paraver where you can configure and launch the CARM GUI. These configurations can later be adjusted within the GUI as well.

### Paraver Trace Requirements
Avoid labeling regions **with MPI calls inside them**. Focus on labeling regions of pure computation, as MPI calls will prevent region and hardware counter timestamps from matching, which is required for the CARM analysis. 

### Python Dependencies
The CARM GUI requires some Python packages to be installed, they can be installed using the requirements.txt file:

```
pip install -r requirements.txt
```
In some cases you might need the flag --break-system-packages or a Python virtual environment (recommended) to install the packages (this is likely the case if you get the error: externally-managed-environment PEP 668).

### Other Dependencies
Add the path to the root directory of this repository, and the path to Paraver's bin directory to their PATH like so:

```
export PATH="$PATH:/path/to/repository/carm-paraver"
export PATH="$PATH:/path/to/Paraver/bin"
```
In case you want to keep these folders added to your PATH permanently you can run setup.sh like so:
```sh
./setup.sh /path/to/Paraver/bin # relative or absolute paths work
```
After these steps Paraver can be launched, and the option to launch CARM from a Paraver timeline should be available.

Keep in mind the CARM GUI needs CARM results from the [CARM Tool](https://github.com/champ-hub/carm-roofline) in order to plot Paraver timestamps, this repository includes some example CARM results sourced from the [MareNostrum 5](https://www.bsc.es/supportkc/docs/MareNostrum5/overview/) supercomputer in the carm_results folder. To add more CARM results simply add the output `<machine>_roofline.csv` files from the CARM Tool to the carm_results folder.

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


At least one FP and one memory counter (separate load and store counters are recommended for a more detailed analysis) must be available in the trace to be analyzed, otherwise the CARM analysis is not possible. It is also recommended to keep all counters in a single counter set (when obtaining the trace via Extrae), this usually allows for all FP counters of a given precision (DP or SP) and the load and store counters. Precisions can also be mixed but the amount of counters used must fit in a single counter set.

## Steps

After performing the setup above, you can:

1. Load a Paraver trace with the required counters, and zoom into a section of interest.
    - **Processing time is heavily dependent on the time range selected. It is recommended the analysis be focused on a ~50ms section to avoid a long wait.**
2. Right click the timeline and select the option to launch the CARM GUI.
3. Configure the options within the Paraver interface to your liking, and click "Run".
4. Click the link printed in the Paraver console to open the GUI in your browser.

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

### Note
The CARM GUI can also be launched from outside a Paraver timeline, for this click the "Run Application" option (Gear Icon) in the top bar of Paraver.