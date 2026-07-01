---
title: GUI Features
nav_order: 6
parent: Home
---

# CARM GUI Features

## Launch Configuration

When launching the CARM GUI from Paraver, a dialog presents the following options:

### Use Window Colors
Controls the coloring scheme used in the CARM GUI:
- **Enabled:** Uses the same colors as the Paraver timeline.
- **Disabled:** Uses the selected CARM GUI coloring scheme (configured via the right sidebar).

This setting is also available in the left sidebar as **Use Paraver/CARM Colors**.

### Use Semantic Window
Controls whether the Paraver semantic window is used:
- **Enabled:** The GUI displays only the timestamps that fall within the semantic window of the Paraver timeline.
- **Disabled:** All timestamps in the trace are displayed.

This setting is also available in the left sidebar as **Use Semantic Window / All Timestamps**.

### Accumulate Values
Controls how timestamps with the same underlying Paraver value are handled:
- **Enabled:** Similar timestamps are averaged into a single, per-thread point.
- **Disabled:** Each timestamp is plotted individually.

This setting is also available in the left sidebar as **Plot Raw/Accumulated Values**.

## Left Sidebar

The left sidebar provides controls that mirror the launch configuration and additional Paraver-specific actions.

### Controls

| Control | Description |
|---------|-------------|
| **Use Paraver/CARM Colors** | Same as the *Use Window Colors* launch option. |
| **Use Semantic Window / All Timestamps** | Same as the *Use Semantic Window* launch option. |
| **Plot Raw/Accumulated Values** | Same as the *Accumulate Values* launch option. |
| **Re-Sync Timeline With Paraver** | Re-syncs the plotted timestamps in the CARM GUI with the timestamps being viewed in the Paraver timeline. This first requires the **Time Sync** button to be clicked on the Paraver side (the CARM GUI usually keeps itself synced automatically). Use this button if you changed the displayed timestamps in the CARM GUI and want to return to the Paraver timeline interval. |

### Sending Labels Back to Paraver

The left sidebar includes buttons to label timestamps and send them back to Paraver for visualization:

- **Send Roof Labels:** Labels each timestamp based on which roof of the roofline it falls under. The path of the generated trace is printed in the Paraver console; click it to open the trace in Paraver, then select the trace and click *New single timeline window*.
- **Send LD/ST Percentage Colors:** Labels timestamps based on the percentage of loads vs. stores.
- **Send SP/DP Percentage Colors:** Labels timestamps based on the percentage of single-precision vs. double-precision operations.
- **Send Performance:** Labels timestamps based on their performance (GFLOP/s).
- **Send Arithmetic Intensity:** Labels timestamps based on their arithmetic intensity (FLOP/byte).
- **Send Roof Proximity:** Sends 4 sets of labels, each based on the proximity of timestamps to the 4 roofs (L1 through DRAM). A value of 1 means the timestamp is on/above the roof, while a value of e.g. 0.5 means the timestamp has half of the attainable performance of that roof.

## Right Sidebar

The right sidebar controls CARM-specific visualization features, including filtering, coloring, and annotations.

### Useful Options

- **Filter points** by vector ISA (e.g. SSE, AVX2, AVX512) or precision (single/double).
- **Color points** based on thread ID, precision, vector ISA, or load/store ratio.
  - Note: requires the left sidebar option to be set to **Use CARM GUI Colors**.

### Roofline Normalization

By default, the plot is configured to normalize the performance roof to the number of threads:

- **Normalized roofs** represent the performance **per thread**, matching the Paraver timestamps, which have per-thread metrics. This mode is **recommended when relating application performance to the underlying hardware**.
- **Non-normalized roofs** represent the overall performance of the architecture, best for **understanding the raw hardware capabilities**.
