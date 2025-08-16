# carm-paraver

This Graphical User Interface was developed to allow for the analysis of [Paraver](https://tools.bsc.es/paraver) traces in the scope of the Cache-Aware Roofline Model (CARM) for floating-point operations. This GUI relies on CARM results obtained via the CARM Tool which can be found in its ([Github repositoy](https://github.com/champ-hub/carm-roofline)), for instructions on running the CARM Tool please consult the README and other documentation available in it's repository. For instruction on running Paraver and obtaining Paraver/Extrae traces please consult the Paraver/Extrae documentation.

# Requirements
- python (only tested with python 3.12.3)
    - dash
    - dash-bootstrap-components
    - dash-daq
    - numpy
    - pandas
    - plotly

- [Paraver](https://tools.bsc.es/downloads)

# How to use

## Setup
The GUI is launched via the Paraver interface, the option to do so can be found by right clicking any Paraver timeline, and then expanding the "Run" dropdown were the CARM option can be selected to launch the Paraver windown that will configure and launch the CARM GUI.

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
```
./setup.sh /path/to/Paraver/bin"
```
After these steps Paraver can be launched, and the option to launch CARM from a Paraver timeline should be available.

Keep in mind the CARM GUI needs CARM results from the CARM Tool in order to plot Paraver timestamps, this repository includes some example CARM results sourced from the [MareNostrum 5](https://www.bsc.es/supportkc/docs/MareNostrum5/overview/) supercomputer in the carm_results folder. To add more CARM results simply add the output xxx_roofline.csv files from the CARM Tool to the carm_results folder.

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


## Features

### Left Sidebar

**Use Paraver/CARM Colors:**  
Controls wether to use the same coloring as the Paraver timeline for timestamps in the CARM GUI.

**Use Semantic Window / All Timestamps:**  
Enables the Paraver semantic window, which makes it so only timestamps that are displayed in the Paraver timeline window used to launch the GUI are shown in the CARM GUI.

**Plot Raw/Accumulated Values:**  
Enables the accumulation and averaging of values across multiple timestamps that share the same underlying Paraver value (the value displayed for the timestamps on the Paraver timeline) uninterrupted into a single timestamp.

**Re-Sync Timeline With Paraver:**  
Re-syncs the plotted timestamps in the CARM GUI with the timestamps being viewed in the Paraver timeline from which the CARM GUI was launched. This first requires the **Time Sync** button to be clicked on the Paraver side, the CARM GUI will usually keep itself synced to the Paraver timeline whenever the **Time Sync** button is clicked in the Paraver interface. In case the user changes the displayed timestamps in the CARM GUI and wishes to return to the same interval that they have in the Paraver timeline, they can use the **Re-Sync Timeline With Paraver** button.

**Send Timestamps Roof Labels:**  
Sends a trace which can be viewed in Paraver, where each timestamp is identified by the corresponding roof above them. The trace path will be printed in the terminal of the Paraver interface and can be clicked to open in Paraver.

**Send Timestamps LD/ST|SP/DP Colors:**  
Sends a trace which can be viewed in Paraver, where each timestamp is identified and colored in the same way as the corresponding CARM GUI timestamp coloring options. The trace path will be printed in the terminal of the Paraver interface and can be clicked to open in Paraver.

### Right Sidebar  
The right sidebar controls the CARM GUI specific features, which include various filtering and coloring options as well as graphical annotations.

### Note
The CARM GUI can also be launched from outside a Paraver timeline, for this click the "Run Application" option (Gear Icon) in the top bar of Paraver.