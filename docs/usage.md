---
title: Usage
nav_order: 3
parent: Home
---

# Usage

## Step-by-Step: Launching the CARM GUI

1. **Generate a trace with Extrae**

   Use [Extrae](https://github.com/bsc-performance-tools/extrae) to instrument your application and generate a Paraver trace. Your Extrae configuration must include the required hardware counters — see [Trace Requirements](trace-requirements) for the necessary counters and configuration advice.

2. **Load the trace in Paraver**

   Open the generated trace in Paraver and zoom into a section of interest.

3. **Launch the CARM GUI**

   Right-click the timeline and select the option to launch the CARM GUI. Configure the options in the dialog that appears (see [Launch Configuration](features#launch-configuration) for details), then click **Run**.

4. **Open the GUI in your browser**

   Click the link printed in the Paraver console to open the CARM GUI in your browser.

## What You'll See

The CARM GUI displays:

- The **architecture's roofline** (peak performance bounds for compute and memory)
- **Your application's events** as points on the roofline plot
- Each point's position is determined by its **performance** (flops/s) and **arithmetic intensity** (flops/byte)

The position of your points on the roofline helps identify bottlenecks and optimization opportunities. See [GUI Features](features) for details on filtering, coloring, and annotation options.

## Troubleshooting

If you encounter errors:

- Verify that `paramedir` is on your `PATH` (see [Installation](installation#first-time-setup-adding-paraver-to-path)).
- Check that your trace includes the required hardware counters (see [Trace Requirements](trace-requirements)).
- Ensure CARM-Paraver is installed in the same Python environment from which you launch Paraver.
