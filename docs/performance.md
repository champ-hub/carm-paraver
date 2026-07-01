---
title: Performance
nav_order: 7
parent: Home
---

# GUI Performance

The CARM GUI may become slow when plotting a very large number of events. To improve performance, try the following strategies:

## Accumulate Values

Enable the **Accumulate values** option (available in the launch dialog or left sidebar). This groups similar events into a single point, reducing the number of data points the GUI needs to render.

## Use Semantic Window

Enable the **Use Semantic Window** option (available in the launch dialog or left sidebar). This restricts the GUI to only plot events visible in the current Paraver timeline view, discarding data outside the window of interest.

## Narrow Your Analysis Window

In the Paraver timeline, focus your analysis on a smaller time window. A narrower view means fewer events to plot and a more responsive GUI.
