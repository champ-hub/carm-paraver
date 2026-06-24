#!/usr/bin/env python3

import argparse
import atexit
import copy
import ctypes
import datetime
import errno
import logging
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from importlib import resources
from typing import Any

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import numpy as np

# Third Party Libraries
# Run: pip install dash dash-bootstrap-components dash-daq numpy pandas plotly
# To get all of the Libraries in case requirements.txt method fails
import pandas as pd
import platformdirs
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from pandas import DataFrame

# Local Python Scripts
from . import GUI_utils as ut
from .analysis_helpers import (
    TimestampColorContext,
    WindowMode,
    build_timestamp_scatter_trace,
    build_timestamp_tooltip_args,
    calculate_roofline_profile,
    fallback_roofline_df_by_isa,
    filter_base_and_intel_data,
    filter_roofline_df_by_query,
    get_timestamp_point,
    infer_effective_isa_from_timestamp_columns,
    iter_timestamp_points,
    prepare_timestamp_series,
    resolve_analysis_paraver_toggles,
    resolve_interval_point_index_and_legend,
    resolve_roofline_angle_bounds,
    resolve_roofline_x_bounds,
    resolve_timestamp_legend_state,
    resolve_timestamp_slice_bounds,
    resolve_toggle_enabled,
    select_timestamp_color,
    should_plot_timestamp_point,
    should_reset_annotations_for_lines,
)


def set_process_death_signal():
    """Set the process to receive a SIGTERM signal when its parent process dies."""
    libc = ctypes.CDLL("libc.so.6")
    PR_PDEATHSIG = 1
    result = libc.prctl(PR_PDEATHSIG, signal.SIGTERM)
    if result != 0:
        raise OSError("prctl failed")


class ProgressBar:
    """Prints a terminal progress bar that updates in-place, ensuring 100% is printed exactly once."""

    def __init__(self, total: int, bar_width: int = 30):
        self.total = total
        self.bar_width = bar_width
        self._done = False

    def update(self, processed: int) -> None:
        """Print the progress bar for the given number of processed items.

        When ``processed >= total`` the bar is shown at 100% followed by a
        newline; subsequent calls are no-ops.
        """
        if self._done:
            return

        if self.total <= 0:
            return

        if processed >= self.total:
            self._print(1.0)
            print()
            self._done = True
        else:
            progress = min(processed / self.total, 0.99)
            self._print(progress)

    def _print(self, progress: float) -> None:
        if progress >= 1.0:
            segments = self.bar_width
        else:
            segments = min(math.ceil(self.bar_width * progress), self.bar_width - 1)
        print(
            f"[{'#' * segments}{' ' * (self.bar_width - segments)}] {progress * 100:.1f}%",
            end="\r",
            flush=True,
        )


def _carm_btn(label: str, button_id: str, tooltip: str | None = None):
    """Create a sidebar button with optional Tooltip."""
    btn = dbc.Button(
        label,
        id=button_id,
        className="mb-2",
        style={"width": "100%"},
        n_clicks=0,
    )
    if tooltip:
        return html.Div([btn, dbc.Tooltip(tooltip, target=button_id)])
    return btn


set_process_death_signal()

VERSION = "1.0.0"

# determine a usable port before performing expensive setup
base_port = int(os.environ.get("CARM_PORT", "8050"))
max_attempts = 5
SELECTED_PORT = None
for _attempt in range(max_attempts):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", base_port))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"WARNING: port {base_port} already in use; trying next port",
                file=sys.stderr,
                flush=True,
            )
            sock.close()
            base_port += 1
            continue
        else:
            sock.close()
            raise
    sock.close()
    SELECTED_PORT = base_port
    break

if SELECTED_PORT is None:
    print(
        "ERROR: could not find an open port after several attempts; exiting.",
        file=sys.stderr,
    )
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")


def _resolve_roofline_data_dir() -> str:
    data_dir = platformdirs.user_data_dir("carm", appauthor=False)
    roofline_dir = os.path.join(data_dir, "roofline")
    os.makedirs(roofline_dir, exist_ok=True)
    return roofline_dir


def _seed_roofline_data(roofline_dir: str) -> None:
    if any(name.endswith(".csv") for name in os.listdir(roofline_dir)):
        return

    sample_ref = resources.files("carm_paraver").joinpath(
        "sample_data",
        "roofline",
        "MN5_roofline.csv",
    )
    try:
        with resources.as_file(sample_ref) as sample_path:
            shutil.copy2(sample_path, os.path.join(roofline_dir, sample_path.name))
    except FileNotFoundError:
        print(
            "ERROR: bundled MN5 roofline sample is missing; unable to seed data directory.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    except OSError as exc:
        print(
            f"ERROR: unable to seed roofline data in {roofline_dir}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


carm_results_path = _resolve_roofline_data_dir()
_seed_roofline_data(carm_results_path)

# Global Variables
n_segments = 0
total_FP_inst = 0
total_FP_ops = 0
total_GFLOPS = 0
total_mem_inst = 0
total_time = 0
total_ai = 0
total_threads = 0
totals = {}
unique_threadIDs = []
appname = ""
lines_origin = {}
lines_origin2 = {}
dropdown_custom = 0
MAX_RANGE = 5000
no_sync = True
current_file_timestamps = []
first_load = 0
max_dots_auto = 1000
data_points = 0

# Define the start and end colors for the age coloring
start_color = (135, 206, 250)  # Light Blue
end_color = (0, 0, 139)  # Dark Blue


# CONSTANTS
TIME_SCALE_FACTORS = {
    "seconds": 1000000,
    "milliseconds": 1000,
    "microseconds": 1,
    "nanoseconds": 0.001,
}

prev_range = [0, 1]


intel_performance_counters = {
    "Intel_FP_Scalar_DP": 1,
    "Intel_FP_Scalar_SP": 1,
    "Intel_FP_SSE_DP": 2,
    "Intel_FP_SSE_SP": 4,
    "Intel_FP_AVX2_DP": 4,
    "Intel_FP_AVX2_SP": 8,
    "Intel_FP_AVX512_DP": 8,
    "Intel_FP_AVX512_SP": 16,
    "Intel_Loads": 1,
    "Intel_Stores": 1,
    "Intel_Loads_Stores": 1,
}

intel_performance_counters_mapping = {
    "Intel_FP_Scalar_DP": "FP_ARITH_INST_RETIRED:SCALAR_DOUBLE",
    "Intel_FP_Scalar_SP": "FP_ARITH_INST_RETIRED:SCALAR_SINGLE",
    "Intel_FP_SSE_DP": "FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE",
    "Intel_FP_SSE_SP": "FP_ARITH_INST_RETIRED:128B_PACKED_SINGLE",
    "Intel_FP_AVX2_DP": "FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE",
    "Intel_FP_AVX2_SP": "FP_ARITH_INST_RETIRED:256B_PACKED_SINGLE",
    "Intel_FP_AVX512_DP": "FP_ARITH_INST_RETIRED:512B_PACKED_DOUBLE",
    "Intel_FP_AVX512_SP": "FP_ARITH_INST_RETIRED:512B_PACKED_SINGLE",
    "Intel_Loads": "MEM_INST_RETIRED:ALL_LOADS",
    "Intel_Stores": "MEM_INST_RETIRED:ALL_STORES",
    "Intel_Loads_Stores": "MEM_INST_RETIRED:ALL",
}

memory_counters = {"Intel_Loads", "Intel_Stores", "Intel_Loads_Stores"}
fp_counters = {key for key in intel_performance_counters_mapping if key.startswith("Intel_FP_")}

intel_configs_partial = (
    "FP_Scalar_DP",
    "FP_SSE_DP",
    "FP_AVX2_DP",
    "FP_AVX512_DP",
    "FP_Scalar_SP",
    "FP_SSE_SP",
    "FP_AVX2_SP",
    "FP_AVX512_SP",
    "Loads",
    "Stores",
)

intel_configs = [
    os.path.join(script_dir, "paraver_carm_configs", version, f"Intel_{config}.cfg")
    for version in ["Intel", "IntelV2"]
    for config in intel_configs_partial
]

amd_performance_counters = {
    "retired_sse_avx_operations:dp_mult_add_flops": 1,
    "retired_sse_avx_operations:dp_add_sub_flops": 1,
    "retired_sse_avx_operations:dp_mult_flops": 1,
    "retired_sse_avx_operations:dp_div_flops": 1,
    "retired_sse_avx_operations:sp_mult_add_flops": 1,
    "retired_sse_avx_operations:sp_add_sub_flops": 1,
    "retired_sse_avx_operations:sp_mult_flops": 1,
    "retired_sse_avx_operations:sp_div_flops": 1,
    "ls_dispatch:ld_dispatch": 1,
    "ls_dispatch:store_dispatch": 1,
}

base_statistics = {
    "ThreadID": [],
    "Timestamp": [],
    "Duration": [],
    "Duration_Percent": [],
    "FP_Percent": [],
    "Memory_Percent": [],
    "GFLOPS": [],
    "FLOP": [],
    "Bandwidth": [],
    "Bytes": [],
    "Arithmetic_Intensity": [],
    "R": [],
    "G": [],
    "B": [],
    "Paraver_Value": [],
    "Paraver_Label": [],
}
full_base_statistics = {
    "ThreadID": [],
    "Timestamp": [],
    "Duration": [],
    "GFLOPS": [],
    "Arithmetic_Intensity": [],
    "Paraver_Label": [],
}

intel_statistics2 = {
    "ThreadID": [],
    "Timestamp": [],
    "Paraver_Label": [],
    "Intel_FP_Scalar_SP": [],
    "Intel_FP_Scalar_DP": [],
    "Intel_FP_SSE_SP": [],
    "Intel_FP_SSE_DP": [],
    "Intel_FP_AVX2_SP": [],
    "Intel_FP_AVX2_DP": [],
    "Intel_FP_AVX512_SP": [],
    "Intel_FP_AVX512_DP": [],
    "Intel_FP_SP": [],
    "Intel_FP_DP": [],
    "Intel_FP_Total": [],
    "Intel_FP_DP_Percent": [],
    "Intel_Load": [],
    "Intel_Store": [],
    "Intel_Load_Percent": [],
}


# Extract counter data
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("-v", "--version", action="store_true")
pre_args, remaining_args = pre_parser.parse_known_args()

if pre_args.version:
    print(f"Paraver_CARM version {VERSION}")
    sys.exit(0)
parser = argparse.ArgumentParser(description="Paraver CARM Dash App")

parser.add_argument("--min_dur", type=float, default=1, help="Minimum duration filter")
parser.add_argument(
    "--color_csv",
    action="store_true",
    help="Use color CSV (.legend.csv) corresponding to the mask CSV",
)
parser.add_argument("--mask_csv", action="store_true", help="Use mask CSV")
parser.add_argument("-ac", action="store_true", help="Optional flag for accumulate values mode")
parser.add_argument("--csv", type=str, required=True, help="Path to the mask CSV")
parser.add_argument("trace_path", type=str, help="Path to the .prv file")
parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")

args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)

logging.debug(f"Parsed arguments: {args}")

min_dur = args.min_dur
use_paraver_coloring = args.color_csv
use_mask_csv = args.mask_csv
ac_mode = args.ac
mask_csv_path = args.csv
prv_trace_path = args.trace_path

data_source_directory = os.path.dirname(prv_trace_path)
color_csv_path = ""

if mask_csv_path != "":
    if mask_csv_path.endswith(".csv"):
        color_csv_path = mask_csv_path.replace(".csv", ".legend.csv")
        sync_csv_path = mask_csv_path.replace(".csv", ".paraver_sync.csv")
        if not os.path.isfile(sync_csv_path):
            no_sync = False

if color_csv_path != "":
    if not color_csv_path.endswith(".legend.csv"):
        print(f"ERROR: Expected a legend file ending with '.legend.csv', got: {color_csv_path}")
        sys.exit(1)

if mask_csv_path != "" and mask_csv_path.endswith(".csv"):
    if use_mask_csv:
        mask_button_text = "Use All Timestamps"
        mask_button_offset = 0
    else:
        mask_button_text = "Use Semantic Window"
        mask_button_offset = 1

    if ac_mode:
        ac_button_text = "Plot Raw Values"
        ac_button_offset = 0
    else:
        ac_button_text = "Plot Accumulated Values"
        ac_button_offset = 1

    legend_filename = os.path.basename(color_csv_path)

    with open(mask_csv_path) as f:
        first_line = f.readline().strip()

    parts = first_line.split(":")
    prv_filename = os.path.basename(parts[3])
    prv_stem = os.path.splitext(prv_filename)[0]

    window_mode_str = parts[5] if len(parts) > 5 else "window_in_code_mode"
    if window_mode_str == WindowMode.GRADIENT.value:
        window_mode = WindowMode.GRADIENT
    else:
        window_mode = WindowMode.CODE
    time_unit = parts[4] if len(parts) > 4 else "Unknown"

    if legend_filename.endswith(".legend.csv"):
        legend_stem = legend_filename.replace(".legend.csv", "")
    else:
        legend_stem = os.path.splitext(legend_filename)[0]

    if legend_stem.endswith(prv_stem):
        window_name = legend_stem[: -len(prv_stem)].rstrip("_")
    else:
        window_name = legend_stem

    color_sync_button_style = {"width": "100%"}
    timeline_warning_style = {"display": "none"}
    timeline_warning_card_style = {"display": "none"}
else:
    window_name = ""
    coloring_button_text = ""
    mask_button_text = ""
    ac_button_text = ""
    mask_button_offset = -1
    color_button_offset = -1
    ac_button_offset = -1

    color_sync_button_style = {"width": "100%", "display": "none"}
    timeline_warning_style = {
        "color": "black",
        "textAlign": "center",
        "fontSize": "16px",
    }
    timeline_warning_card_style = {
        "margin": "0px auto 15px auto",
        "padding": "0px",
        "text-align": "center",
    }

    time_unit = "Microseconds"
    use_mask_csv = False
    use_paraver_coloring = False
    window_mode = WindowMode.CODE

scaling_unit = TIME_SCALE_FACTORS.get(time_unit.lower(), 1)

if not os.path.exists(prv_trace_path):
    print(f"ERROR: The path '{prv_trace_path}' does not exist")
    sys.exit(1)

ok = ut.find_and_run("paramedir")
if not ok:
    print(
        "Paramedir not found! Add the Paraver bin/ directory to your PATH.\n"
        "Add the following to your .bashrc or .bash_profile (change the path accordingly):\n"
        "  export PATH=/path/to/paraver/bin:$PATH"
    )
    sys.exit(1)


def _write_temp_cfgs_with_timeunit(cfg_paths, time_unit_value, output_dir):
    """Write copies of cfg files to `output_dir` with `window_units` set to `time_unit_value`.

    Returns list of written file paths. If a source file is missing it is skipped.
    """
    unit = time_unit_value if time_unit_value and time_unit_value != "Unknown" else "Microseconds"
    written = []
    for src in cfg_paths:
        try:
            with open(src) as fh:
                content = fh.read()
        except Exception:
            continue

        if re.search(r"(?m)^window_units\s+\S+", content):
            new_content = re.sub(r"(?m)^window_units\s+\S+", f"window_units {unit}", content)
        else:
            new_content = content + f"\nwindow_units {unit}\n"

        dst = os.path.join(output_dir, os.path.basename(src))
        try:
            with open(dst, "w") as fh:
                fh.write(new_content)
            written.append(dst)
        except Exception:
            # on failure, try to continue with other files
            continue

    return written


paramedir_tempdir = None
counter_csv_dir = os.getcwd()

if prv_trace_path.endswith(".prv") or prv_trace_path.endswith(".gz"):
    print(f"Running Paramedir to parse the trace in {prv_trace_path}", flush=True)
    paramedir_tempdir = tempfile.TemporaryDirectory(prefix="carm-paramedir-")
    atexit.register(paramedir_tempdir.cleanup)
    counter_csv_dir = paramedir_tempdir.name
    temp_cfgs = _write_temp_cfgs_with_timeunit(intel_configs, time_unit, counter_csv_dir)
    subprocess.run(
        ["paramedir", prv_trace_path, *temp_cfgs],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        cwd=counter_csv_dir,
    )

    print("Paramedir execution finished, calculating CARM metrics.", flush=True)

# Get CARM results
csv_files = sorted(f for f in os.listdir(carm_results_path) if f.endswith("_roofline.csv"))
if not csv_files:
    print(
        f"ERROR: No CARM roofline results found in {carm_results_path}. Add files named *_roofline.csv.",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)

# Extract machine names from filenames
machine_names = [file.replace("_roofline.csv", "") for file in csv_files]

counter_data_df = None
missing_files = set()
found_files = []
sp_counters_available = False
dp_counters_available = False

# Loop through each counter
for counter_name in intel_performance_counters.keys():
    filename = f"{counter_name}.csv"
    counter_path = os.path.join(counter_csv_dir, filename)
    if os.path.exists(counter_path):
        found_files.append(filename[:-4])
        df = pd.read_csv(
            counter_path,
            sep="\t",
            header=None,
            skiprows=1,
            names=["ThreadID", "Timestamp", "Duration", counter_name],
        )
        if counter_data_df is None:
            counter_data_df = df
        else:
            # Merge with the existing DataFrame on ThreadID, Timestamp, and Duration
            counter_data_df = pd.merge(
                counter_data_df,
                df,
                on=["ThreadID", "Timestamp", "Duration"],
                how="outer",
            )
        try:
            os.remove(counter_path)
        except OSError as e:
            print(
                f"Warning: Could not delete {counter_path} ({e}), future results may be affected",
                flush=True,
            )

    else:
        missing_files.add(filename[:-4])
        # If the file is missing, create a DataFrame with zeros for this counter
        if counter_data_df is None:
            counter_data_df = pd.DataFrame(columns=["ThreadID", "Timestamp", "Duration", counter_name])
            counter_data_df[counter_name] = 0
        else:
            counter_data_df[counter_name] = 0

counter_data_df.sort_values(by="Timestamp", ascending=True, inplace=True)

no_mem = False

# Check if all memory counters are missing
if memory_counters <= missing_files:
    print(
        "ERROR: No memory counters found. Please add at least one of the following Intel memory counters "
        "to your XML file (separate Loads and Stores recommended):",
        flush=True,
    )
    for key in memory_counters:
        print(
            f"{key.replace('Intel_', '').replace('_', ' ')} -> {intel_performance_counters_mapping[key]}",
            flush=True,
        )
    no_mem = True
    sys.exit(1)

# Check if all FP (floating point) counters are missing
if fp_counters <= missing_files:
    print(
        "ERROR: No floating-point counters found. Please add at least one of the following Intel FP counters "
        "to your XML file:",
        flush=True,
    )
    for key in sorted(fp_counters):
        print(
            f"{key.replace('_', ' ')} -> {intel_performance_counters_mapping[key]}",
            flush=True,
        )
    sys.exit(1)


if "Intel_Loads" not in missing_files and "Intel_Stores" not in missing_files and "Intel_Loads_Stores" in missing_files:
    missing_files.remove("Intel_Loads_Stores")

if any("SP" in s for s in found_files):
    sp_counters_available = True
if any("DP" in s for s in found_files):
    dp_counters_available = True

missing_msg = (
    "Counters for some events are missing from the trace. If any of the following operations\nare relevant to your "
    "application, you should add the corresponding counters:\n\n"
    + "\n  ".join(
        [
            f"{f.replace('_', ' ')} -> {intel_performance_counters_mapping.get(f, 'No mapping found')}"
            for f in missing_files
        ]
    )
    + "\n\nThe analysis will proceed with the available counters.\nSee the documentation for more details."
)

is_modal_open = len(missing_files) > 0

assert isinstance(counter_data_df, DataFrame)
biggest_timestamp = counter_data_df["Timestamp"].max()

total_time = (biggest_timestamp - counter_data_df["Timestamp"].min()) * scaling_unit

total_threads = counter_data_df["ThreadID"].nunique()
unique_threadIDs = counter_data_df["ThreadID"].unique().tolist()
unique_threadIDs_checkbox = [{"label": thread_id, "value": thread_id} for thread_id in unique_threadIDs]

filename_with_ext = os.path.basename(prv_trace_path)
appname = os.path.splitext(filename_with_ext)[0]

if counter_data_df is not None:
    # Calculate totals for each counter column
    for column in counter_data_df.columns:
        # Exclude non-counter columns
        if column not in ["ThreadID", "Timestamp", "Duration"]:
            totals[column] = counter_data_df[column].sum()

    for counter, total in totals.items():
        # Check if the counter name contains "FP"
        if "FP" in counter:
            total_FP_inst += total
            total_FP_ops += total * intel_performance_counters[counter]
        else:
            total_mem_inst += total

else:
    print("No data to calculate totals.")

# TODO: This works globally, but should probably be done per-timestamp for best accuracy
# Calculate approximate size of memory instructions based on the FP instructions present
bytes_modifier = (
    4 * (totals["Intel_FP_Scalar_SP"] / total_FP_inst)
    + 8 * (totals["Intel_FP_Scalar_DP"] / total_FP_inst)
    + 16 * ((totals["Intel_FP_SSE_SP"] + totals["Intel_FP_SSE_DP"]) / total_FP_inst)
    + 32 * ((totals["Intel_FP_AVX2_SP"] + totals["Intel_FP_AVX2_DP"]) / total_FP_inst)
    + 64 * ((totals["Intel_FP_AVX512_SP"] + totals["Intel_FP_AVX512_DP"]) / total_FP_inst)
)
# Calculate totals for the trace
total_ai = total_FP_ops / (total_mem_inst * bytes_modifier)
total_GFLOPS = total_FP_ops / (total_time * 1e3)

if color_csv_path != "":
    legend = []
    with open(color_csv_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            match = re.match(r'^([\d\.]+)(?:-([\d\.]+))?\s+"([^"]+)"\s+(\d+),(\d+),(\d+)$', stripped)
            if match:
                start_str = match.group(1)
                end_str = match.group(2)
                label = match.group(3)
                r, g, b = int(match.group(4)), int(match.group(5)), int(match.group(6))

                value_start = float(start_str)
                value_end = float(end_str) if end_str else value_start

                legend.append(
                    {
                        "value_start": value_start,
                        "value_end": value_end,
                        "value_label": label,
                        "R": r,
                        "G": g,
                        "B": b,
                    }
                )
            else:
                print(f"Skipping malformed line: {stripped}")

            if use_paraver_coloring:
                coloring_button_text = "Use CARM GUI Colors"
                color_button_offset = 0
            else:
                coloring_button_text = "Use Paraver Timeline Colors"
                color_button_offset = 1

    legend_df = pd.DataFrame(legend)
    legend_df["value_start"] = legend_df["value_start"].astype(float)
    legend_df["value_end"] = legend_df["value_end"].astype(float)

    with open(mask_csv_path) as f:
        lines = f.readlines()

        data_lines = [line for line in lines if not line.startswith("#")]

        trace_df = pd.DataFrame(
            [line.strip().split("\t") for line in data_lines],
            columns=["ThreadID", "Timestamp", "Duration", "LegendValue"],
        )

        trace_df["Timestamp"] = trace_df["Timestamp"].astype(float)
        trace_df["LegendValue"] = trace_df["LegendValue"].astype(float)

    trace_df = trace_df.sort_values("LegendValue").reset_index(drop=True)
    legend_df = legend_df.sort_values("value_start").reset_index(drop=True)
    assert isinstance(trace_df, DataFrame)
    assert isinstance(legend_df, DataFrame)

    color_df = pd.merge_asof(
        trace_df,
        legend_df,
        left_on="LegendValue",
        right_on="value_start",
        direction="backward",
    )

    del trace_df

    in_range_mask = (color_df["LegendValue"] >= color_df["value_start"]) & (
        color_df["LegendValue"] <= color_df["value_end"]
    )
    color_df = color_df[in_range_mask].copy()

    color_df = color_df[["ThreadID", "Timestamp", "R", "G", "B", "LegendValue", "value_label"]]

    a = "1"

# Calculate metrics for each trace timestamp
no_flops = 0
match = 0
columns_to_check = [
    "Intel_FP_Scalar_SP",
    "Intel_FP_Scalar_DP",
    "Intel_FP_SSE_SP",
    "Intel_FP_SSE_DP",
    "Intel_FP_AVX2_SP",
    "Intel_FP_AVX2_DP",
    "Intel_FP_AVX512_SP",
    "Intel_FP_AVX512_DP",
]
counter_data_df = counter_data_df.fillna(0)
assert isinstance(counter_data_df, DataFrame)
# Report progress during processing of rows
total_rows = len(counter_data_df)
rows_chars = len(str(total_rows))
step = max(1, total_rows // 100) if total_rows > 0 else 1
prog_bar_width = 30  # Total width of the progress bar
processed = 0
bar = ProgressBar(total_rows, prog_bar_width)
if total_rows > 50_000:
    print(
        f"WARNING: Displaying a large number of rows ({total_rows}) may slow down the UI. Consider zooming "
        f"into a smaller time range in Paraver before launching CARM for a better experience.",
        flush=True,
    )
else:
    print(f"Processing {total_rows} rows for CARM metrics...", flush=True)  # Initial message


_time_start = time.time()

# Pre-merge color information into the counter dataframe for O(1) lookups
# This avoids expensive per-row filtering of `color_df` inside the loop.
if color_csv_path != "":
    counter_data_df = pd.merge(
        counter_data_df,
        color_df[["ThreadID", "Timestamp", "R", "G", "B", "LegendValue", "value_label"]],
        on=["ThreadID", "Timestamp"],
        how="left",
    )
    del color_df
else:
    counter_data_df = counter_data_df.copy()

for row in counter_data_df.itertuples(index=False):
    processed += 1
    if processed % step == 0 or processed == total_rows:
        bar.update(processed)
    duration = row.Duration * scaling_unit
    timestamp = row.Timestamp
    # if FLOP counters are all zero or NaN, skip calculations and set metrics to zero/defaults
    if all(pd.isnull(getattr(row, col)) or getattr(row, col) == 0 for col in columns_to_check):
        no_flops += 1
        full_base_statistics["ThreadID"].append(row.ThreadID)
        full_base_statistics["Timestamp"].append(timestamp)
        full_base_statistics["Duration"].append(duration)
        full_base_statistics["GFLOPS"].append(0)
        full_base_statistics["Arithmetic_Intensity"].append(0)
        full_base_statistics["Paraver_Label"].append("")
        continue

    fp_inst = (
        row.Intel_FP_Scalar_SP
        + row.Intel_FP_Scalar_DP
        + row.Intel_FP_SSE_SP
        + row.Intel_FP_SSE_DP
        + row.Intel_FP_AVX2_SP
        + row.Intel_FP_AVX2_DP
        + row.Intel_FP_AVX512_SP
        + row.Intel_FP_AVX512_DP
    )

    sp_ops = (
        row.Intel_FP_Scalar_SP * 1 + row.Intel_FP_SSE_SP * 4 + row.Intel_FP_AVX2_SP * 8 + row.Intel_FP_AVX512_SP * 16
    )
    dp_ops = (
        row.Intel_FP_Scalar_DP * 1 + row.Intel_FP_SSE_DP * 2 + row.Intel_FP_AVX2_DP * 4 + row.Intel_FP_AVX512_DP * 8
    )
    fp_ops = sp_ops + dp_ops

    mem_ops = row.Intel_Loads + row.Intel_Stores
    # Calculate approximate size of memory instructions based on the FP instructions present
    bytes_modifier = (
        4 * (row.Intel_FP_Scalar_SP / fp_inst)
        + 8 * (row.Intel_FP_Scalar_DP / fp_inst)
        + 16 * ((row.Intel_FP_SSE_SP + row.Intel_FP_SSE_DP) / fp_inst)
        + 32 * ((row.Intel_FP_AVX2_SP + row.Intel_FP_AVX2_DP) / fp_inst)
        + 64 * ((row.Intel_FP_AVX512_SP + row.Intel_FP_AVX512_DP) / fp_inst)
    )
    memory_bytes = mem_ops * bytes_modifier

    # Calculate General Statistics
    duration_percent = (duration / total_time) * 100

    if pd.isna(duration) or duration <= 0:
        gflops = 0.0
    else:
        gflops = fp_ops / (duration * 1e3)
    # gflops = fp_ops / (duration * 1e3) if duration > 0 else 0
    fP_percent = fp_ops / total_FP_ops

    bandwidth = memory_bytes / (duration * 1e3) if duration > 0 else 0
    memory_percent = mem_ops / total_mem_inst

    if mem_ops > 0:
        load_percentage = ut.custom_round((row.Intel_Loads / mem_ops) * 100, 1)
        if load_percentage < 0.1:
            load_percentage = 0.1
    else:
        load_percentage = 0

    dp_percentage = ut.custom_round((dp_ops / fp_ops) * 100, 1)
    if dp_percentage < 0.1:
        dp_percentage = 0.1

    arithmethic_intensity = fp_ops / memory_bytes

    base_statistics["ThreadID"].append(row.ThreadID)
    base_statistics["Timestamp"].append(timestamp)
    base_statistics["Duration"].append(duration)
    base_statistics["Duration_Percent"].append(duration_percent)
    base_statistics["FP_Percent"].append(fP_percent)
    base_statistics["Memory_Percent"].append(memory_percent)
    base_statistics["GFLOPS"].append(float(gflops))
    base_statistics["FLOP"].append(float(fp_ops))
    base_statistics["Bandwidth"].append(bandwidth)
    base_statistics["Bytes"].append(float(memory_bytes))
    base_statistics["Arithmetic_Intensity"].append(float(arithmethic_intensity))

    if color_csv_path != "":
        # merged_df contains color columns (may be NaN if no match)
        if pd.notna(getattr(row, "R", None)) and (
            (getattr(row, "R", 0) != 0) or (getattr(row, "G", 0) != 0) or (getattr(row, "B", 0) != 0)
        ):
            match += 1
            base_statistics["Paraver_Value"].append(row.LegendValue)
            base_statistics["Paraver_Label"].append(row.value_label)
            base_statistics["R"].append(int(row.R))
            base_statistics["G"].append(int(row.G))
            base_statistics["B"].append(int(row.B))

            intel_statistics2["Paraver_Label"].append(row.value_label)
            full_base_statistics["Paraver_Label"].append(row.value_label)
        else:
            base_statistics["Paraver_Value"].append("")
            base_statistics["Paraver_Label"].append("No Label")
            base_statistics["R"].append(0)
            base_statistics["G"].append(0)
            base_statistics["B"].append(0)

            intel_statistics2["Paraver_Label"].append("")
            full_base_statistics["Paraver_Label"].append("")
    else:
        base_statistics["Paraver_Value"].append("")
        base_statistics["Paraver_Label"].append("No Label")
        base_statistics["R"].append(0)
        base_statistics["G"].append(0)
        base_statistics["B"].append(0)

        intel_statistics2["Paraver_Label"].append("")
        full_base_statistics["Paraver_Label"].append("")

    full_base_statistics["ThreadID"].append(row.ThreadID)
    full_base_statistics["Timestamp"].append(timestamp)
    full_base_statistics["Duration"].append(duration)
    full_base_statistics["GFLOPS"].append(float(gflops))
    full_base_statistics["Arithmetic_Intensity"].append(float(arithmethic_intensity))

    intel_statistics2["ThreadID"].append(row.ThreadID)
    intel_statistics2["Timestamp"].append(timestamp)
    intel_statistics2["Intel_FP_Scalar_SP"].append(row.Intel_FP_Scalar_SP)
    intel_statistics2["Intel_FP_Scalar_DP"].append(row.Intel_FP_Scalar_DP)
    intel_statistics2["Intel_FP_SSE_SP"].append(row.Intel_FP_SSE_SP * 4)
    intel_statistics2["Intel_FP_SSE_DP"].append(row.Intel_FP_SSE_DP * 2)
    intel_statistics2["Intel_FP_AVX2_SP"].append(row.Intel_FP_AVX2_SP * 8)
    intel_statistics2["Intel_FP_AVX2_DP"].append(row.Intel_FP_AVX2_DP * 4)
    intel_statistics2["Intel_FP_AVX512_SP"].append(row.Intel_FP_AVX512_SP * 16)
    intel_statistics2["Intel_FP_AVX512_DP"].append(row.Intel_FP_AVX512_DP * 8)
    intel_statistics2["Intel_FP_SP"].append(sp_ops)
    intel_statistics2["Intel_FP_DP"].append(dp_ops)
    intel_statistics2["Intel_FP_Total"].append(fp_ops)
    intel_statistics2["Intel_FP_DP_Percent"].append(dp_percentage)
    intel_statistics2["Intel_Load"].append(row.Intel_Loads)
    intel_statistics2["Intel_Store"].append(row.Intel_Stores)
    intel_statistics2["Intel_Load_Percent"].append(load_percentage)

del counter_data_df

_runtime = time.time() - _time_start
print(f"Finished processing {total_rows} rows in {_runtime:.2f} seconds. ")


base_statistics_df = pd.DataFrame(base_statistics)
full_base_statistics_df = pd.DataFrame(full_base_statistics)
intel_statistics_df2 = pd.DataFrame(intel_statistics2)
n_segments = base_statistics_df.shape[0]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder=assets_dir,
    suppress_callback_exceptions=True,
)

# Sidebar Layout Definition
sidebar = dbc.Offcanvas(
    html.Div(
        [
            html.P(
                "Data Filtering",
                className="mb-2",
                style={"color": "white", "textAlign": "center", "fontSize": "20px"},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.P(
                            "Filter by Vector Extension:",
                            className="mb-1",
                            style={"color": "black"},
                        ),
                        dbc.Checklist(
                            id="isa-checklist",
                            options=[
                                {"label": " Scalar", "value": "Scalar"},
                                {"label": " SSE", "value": "SSE"},
                                {"label": " AVX2", "value": "AVX2"},
                                {"label": " AVX512", "value": "AVX512"},
                            ],
                            value=["Scalar", "SSE", "AVX2", "AVX512"],
                            inline=True,
                            className="mb-2",
                            style={"color": "black"},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.P(
                                            "Filter by Precision:",
                                            className="mb-1",
                                            style={"color": "black"},
                                        ),
                                        dbc.Checklist(
                                            id="precision-checklist",
                                            options=[
                                                {"label": " SP", "value": "SP"},
                                                {"label": " DP", "value": "DP"},
                                            ],
                                            value=["SP", "DP"],
                                            inline=True,
                                            className="mb-2",
                                            style={"color": "black"},
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.P(
                                            "Toggle Total:",
                                            className="mb-1",
                                            style={
                                                "color": "black",
                                                "margin-right": "10px",
                                                "display": "none",
                                            },
                                        ),
                                        dbc.Checklist(
                                            id="total-checklist",
                                            options=[{"label": "", "value": "Total"}],
                                            inline=True,
                                            className="mb-1",
                                            style={"color": "black", "display": "none"},
                                        ),
                                    ],
                                    width=6,
                                ),
                            ]
                        ),
                        html.P(
                            "Filter by Thread ID:",
                            className="mb-1",
                            style={"color": "black"},
                        ),
                        dbc.Checklist(
                            id="thread-checklist",
                            options=unique_threadIDs_checkbox,
                            value=unique_threadIDs,
                            inline=True,
                            className="mb-2",
                            style={"color": "black"},
                        ),
                        html.Div(
                            [
                                html.P(
                                    "Cut values lower than:",
                                    className="mb-1",
                                    style={
                                        "color": "black",
                                        "margin-right": "10px",
                                        "flex": "none",
                                    },
                                ),
                                dcc.Input(
                                    id="lower-filter",
                                    type="number",
                                    value=0.0000001,
                                    min=0,
                                    placeholder="Enter value",
                                    debounce=True,
                                    style={"flex": "1", "width": "100%"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        html.Div(
                            [
                                html.P(
                                    "Minimum Duration (ns):",
                                    className="mb-1",
                                    style={
                                        "color": "black",
                                        "margin-right": "10px",
                                        "flex": "none",
                                    },
                                ),
                                dcc.Input(
                                    id="duration-filter",
                                    type="number",
                                    value=min_dur,
                                    min=0,
                                    placeholder="Enter value",
                                    debounce=True,
                                    style={"flex": "1", "width": "100%"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                    ]
                ),
                style={"backgroundColor": "white"},
                className="mb-2",
            ),
            html.P(
                "Graph Customization",
                className="mb-2",
                style={"color": "white", "textAlign": "center", "fontSize": "20px"},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.P(
                            "Color timestamps based on:",
                            className="mb-1",
                            style={"color": "black"},
                        ),
                        dbc.RadioItems(
                            id="color-radio",
                            options=[
                                {"label": " Age", "value": "Youngest"},
                                # {'label': ' Oldest', 'value': 'Oldest'},
                                # {'label': ' Duration', 'value': 'Duration'},
                                {"label": " Thread ID", "value": "Thread ID"},
                                {"label": " Precision", "value": "Precision"},
                                {
                                    "label": " LD/ST Percentage",
                                    "value": "LD/ST Percentage",
                                },
                                {"label": " Vector ISA", "value": "ISA"},
                            ],
                            inline=True,
                            value="Youngest",
                            className="mb-2",
                            style={"color": "black"},
                        ),
                        html.Div(
                            [
                                dbc.Label(
                                    "Use Exponent Notation",
                                    html_for="exponent-switch",
                                    style={"marginRight": "40px"},
                                ),
                                dbc.Switch(id="exponent-switch", label="", value=True),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        html.Div(
                            [
                                dbc.Label(
                                    "Show Lines Legend",
                                    html_for="line-legend-switch",
                                    style={"marginRight": "70px"},
                                ),
                                dbc.Switch(id="line-legend-switch", label="", value=True),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        # normalize roofs per-thread switch
                        html.Div(
                            [
                                dbc.Label(
                                    "Normalize Roofs by Threads",
                                    html_for="normalize-switch",
                                    style={"marginRight": "70px"},
                                ),
                                dbc.Switch(id="normalize-switch", label="", value=True),
                                dbc.Tooltip(
                                    "When enabled, the roofline performance will be normalized by the number of "
                                    "threads, showing the performance per thread. Given timestamp performance is "
                                    "also at the level of a thread, enabling this allows to better compare the "
                                    "points to the roofs. Disabling this will give you a better idea of the "
                                    "overall performance of the system.",
                                    target="normalize-switch",
                                    placement="right",
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                    ]
                ),
                style={"backgroundColor": "white"},
                className="mb-2",
            ),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                [
                                    html.P(
                                        "Lines Width:",
                                        className="mb-1",
                                        style={
                                            "color": "black",
                                            "margin-right": "10px",
                                            "flex": "1",
                                        },
                                    ),
                                    html.P(
                                        "Dots Size:",
                                        className="mb-1",
                                        style={
                                            "color": "black",
                                            "margin-right": "10px",
                                            "flex": "1",
                                        },
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dcc.Input(
                                        id="line-size",
                                        type="number",
                                        min=1,
                                        className="mb-2",
                                        max=100,
                                        step=1,
                                        value=3,
                                        style={
                                            "flex": "1",
                                            "margin-right": "50px",
                                            "width": "70px",
                                        },
                                    ),
                                    dcc.Input(
                                        id="dot-size",
                                        type="number",
                                        min=1,
                                        className="mb-2",
                                        max=100,
                                        step=1,
                                        value=10,
                                        style={"flex": "1", "margin-right": "50px"},
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    html.P(
                                        "Title Font:",
                                        className="mb-1",
                                        style={
                                            "color": "black",
                                            "margin-right": "10px",
                                            "flex": "1",
                                        },
                                    ),
                                    html.P(
                                        "Axis Font:",
                                        className="mb-1",
                                        style={
                                            "color": "black",
                                            "margin-right": "10px",
                                            "flex": "1",
                                        },
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dcc.Input(
                                        id="title-size",
                                        type="number",
                                        className="mb-2",
                                        min=1,
                                        max=100,
                                        step=1,
                                        value=20,
                                        style={"flex": "1", "margin-right": "50px"},
                                    ),
                                    dcc.Input(
                                        id="axis-size",
                                        type="number",
                                        className="mb-2",
                                        min=1,
                                        max=100,
                                        step=1,
                                        value=20,
                                        style={"flex": "1", "margin-right": "50px"},
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    html.P(
                                        "Legend Font:",
                                        className="mb-1",
                                        style={
                                            "color": "black",
                                            "margin-right": "10px",
                                            "flex": "1",
                                        },
                                    ),
                                    html.P(
                                        "Ticks Font:",
                                        className="mb-1",
                                        style={
                                            "color": "black",
                                            "margin-right": "10px",
                                            "flex": "1",
                                        },
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dcc.Input(
                                        id="legend-size",
                                        type="number",
                                        className="mb-2",
                                        min=1,
                                        max=100,
                                        step=1,
                                        value=14,
                                        style={"flex": "1", "margin-right": "50px"},
                                    ),
                                    dcc.Input(
                                        id="tick-size",
                                        type="number",
                                        className="mb-2",
                                        min=1,
                                        max=100,
                                        step=1,
                                        value=18,
                                        style={"flex": "1", "margin-right": "50px"},
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    html.P(
                                        "Annotations:",
                                        className="mb-1",
                                        style={
                                            "color": "black",
                                            "margin-right": "10px",
                                            "flex": "1",
                                        },
                                    ),
                                    html.P(
                                        "Tooltip Font:",
                                        className="mb-1",
                                        style={
                                            "color": "black",
                                            "margin-right": "10px",
                                            "flex": "1",
                                        },
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dcc.Input(
                                        id="annotation-size",
                                        type="number",
                                        className="mb-2",
                                        min=1,
                                        max=100,
                                        step=1,
                                        value=10,
                                        style={"flex": "1", "margin-right": "50px"},
                                    ),
                                    dcc.Input(
                                        id="tooltip-size",
                                        type="number",
                                        className="mb-2",
                                        min=1,
                                        max=100,
                                        step=1,
                                        value=14,
                                        style={"flex": "1", "margin-right": "50px"},
                                    ),
                                ]
                            ),
                        ],
                        title="Change Font/Line Sizes",
                    ),
                ],
                id="font-accordion",
                start_collapsed=True,
                always_open=True,
                flush=True,
                style={"backgroundColor": "#1a1a1a"},
                className="mb-2",
            ),
            dbc.Button(
                "Edit Graph Text",
                id="button-CARM",
                className="mb-2",
                style={"width": "100%"},
                n_clicks=1,
            ),
            html.P(
                "Notations Configuration",
                className="mb-2",
                style={"color": "white", "textAlign": "center", "fontSize": "20px"},
            ),
            html.Div(
                [
                    dbc.Accordion(
                        [],
                        id="annotation-accordion",
                        start_collapsed=True,
                        always_open=True,
                        flush=True,
                        style={"backgroundColor": "#1a1a1a"},
                    )
                ],
                id="angle-inputs-container",
                style={"marginBottom": "15px"},
            ),
            dbc.Button(
                "Create Annotation",
                id="create-annotation-button",
                className="mb-2",
                style={"width": "100%"},
            ),
            dbc.Button(
                "Disable Annotations",
                id="disable-annotation-button",
                className="mb-2",
                style={"width": "100%"},
            ),
        ],
        style={"backgroundColor": "#1a1a1a"},
    ),
    id="offcanvas",
    title=html.H5("Graph Options", style={"color": "white", "fontsize": "30px"}),
    is_open=False,
    placement="end",
    style={"backgroundColor": "#1a1a1a"},
)

sidebar2 = dbc.Offcanvas(
    children=[
        html.P(
            "Paraver -> CARM",
            className="mb-2",
            style={"color": "white", "textAlign": "center", "fontSize": "20px"},
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.P(
                        "To use the Paraver -> CARM features please launch CARM from a Paraver timeline window",
                        className="mb-0",
                        style=timeline_warning_style,
                    ),
                ]
            ),
            style=timeline_warning_card_style,
        ),
        dbc.Button(
            coloring_button_text,
            id="button-paraver-colors",
            className="mb-2",
            style=color_sync_button_style,
            n_clicks=0,
        ),
        dbc.Button(
            mask_button_text,
            id="button-paraver-mask",
            className="mb-2",
            style=color_sync_button_style,
            n_clicks=0,
        ),
        dbc.Button(
            ac_button_text,
            id="button-paraver-accumulate",
            className="mb-4",
            style=color_sync_button_style,
            n_clicks=0,
        ),
        dbc.Button(
            "Re-Sync Timeline With Paraver",
            id="button-paraver-sync",
            className="mb-5",
            style=color_sync_button_style,
            n_clicks=0,
        ),
        html.P(
            "CARM -> Paraver",
            className="mb-2",
            style={"color": "white", "textAlign": "center", "fontSize": "20px"},
        ),
        _carm_btn(
            "Send Roof Labels",
            "button-roof-labels",
            "Labels each timestamp based on which roof is above it (L2, DRAM, etc.)",
        ),
        _carm_btn(
            "Send LD/ST Ratio",
            "button-carm-ldst-colors",
            "Labels each timestamp based on the load-store ratio",
        ),
        _carm_btn(
            "Send SP/DP Ratio",
            "button-carm-spdp-colors",
            "Labels each timestamp based on the single-precision/double-precision ratio",
        ),
        _carm_btn("Send Performance", "button-carm-gflops", "Labels each timestamp based on the GFLOPS performance"),
        _carm_btn(
            "Send Arithmetic Intensity", "button-carm-ai", "Labels each timestamp based on the arithmetic intensity"
        ),
        _carm_btn(
            "Send Roof Proximity",
            "button-carm-roof-proximity",
            "Labels each timestamp based on its proximity to each of the roofs. e.g. 0.2 relative to the L1 means a "
            "perfectly optimization could achieve a 5x speedup. A value of 1.0 means the timestamp is at or above the "
            "roof.",
        ),
    ],
    id="offcanvas2",
    title=html.H5("Paraver Functions", style={"color": "white", "fontsize": "30px"}),
    is_open=False,
    placement="start",
    style={"backgroundColor": "#1a1a1a"},
)

# Main app layout
app.layout = dbc.Container(
    [
        dcc.Interval(id="interval-component", interval=1000, n_intervals=0, disabled=True),
        dcc.Interval(id="paraver-sync-check", interval=1000, n_intervals=0, disabled=True),
        dcc.Store(id="paraver-sync-timestamps", data=[]),
        dcc.Download(id="download-csv"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        html.Img(src="/assets/bsc.svg", height="30px"),
                        id="open-offcanvas2",
                        n_clicks=0,
                        className="btn-sm",
                        style={
                            "border": "none",
                            "background": "transparent",
                            "padding": "0",
                            "margin": "0",
                        },
                    ),
                    width="auto",
                    style={"padding-right": "5px", "padding-left": "5px"},
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="filename",
                        options=[
                            {
                                "label": machine_name,
                                "value": os.path.join(carm_results_path, file),
                            }
                            for machine_name, file in zip(machine_names, csv_files, strict=True)
                        ],
                        multi=False,
                        placeholder="Select Machine Results...",
                    ),
                    width=True,
                ),
                dbc.Col(
                    dbc.Button(
                        html.Img(src="/assets/CARM_icon3.svg", height="30px"),
                        id="open-offcanvas",
                        n_clicks=0,
                        className="btn-sm",
                        style={
                            "border": "none",
                            "background": "transparent",
                            "padding": "0",
                            "margin": "0",
                        },
                    ),
                    width="auto",
                    style={"padding-right": "5px", "padding-left": "5px"},
                ),
            ],
            align="center",
            style={"margin-top": "1px"},
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    id="additional-dropdowns",
                                    style={"margin-top": "10px"},
                                ),
                                html.Div(id="additional-dropdowns2"),
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    html.Div(
                                        [
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.P(
                                                            f"Execution Timestamp Range Selection ({time_unit})",
                                                            style={
                                                                "textAlign": "center",
                                                                "fontWeight": "bold",
                                                                "margin-right": "0px",
                                                                "align-self": "center",
                                                                "margin-top": "-6px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            [
                                                                dcc.RangeSlider(
                                                                    id="time-slider",
                                                                    min=0,
                                                                    max=None,
                                                                    step=1,
                                                                    value=[0, 1],
                                                                    marks=None,
                                                                    tooltip=None,
                                                                    allowCross=False,
                                                                    pushable=2,
                                                                ),
                                                            ],
                                                            style={
                                                                "width": "100%",
                                                                "margin": "0 10px",
                                                            },
                                                        ),
                                                    ]
                                                ),
                                                style={
                                                    # "height": "100px",
                                                    "margin": "0px auto 10px auto",
                                                    "padding": "0px",
                                                    "text-align": "center",
                                                    "flex": "1",
                                                },
                                            ),
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.P(
                                                            f"Execution Timestamps To Plot ({time_unit})",
                                                            style={
                                                                "textAlign": "center",
                                                                "fontWeight": "bold",
                                                                "margin-top": "-6px",
                                                            },
                                                        ),
                                                        dcc.Store(id="data-points-store"),
                                                        html.Div(
                                                            [
                                                                html.Button(
                                                                    "▶️",
                                                                    id="play-pause-button",
                                                                    n_clicks=0,
                                                                    style={
                                                                        "border": "none",
                                                                        "outline": "none",
                                                                        "fontSize": "24px",
                                                                        "backgroundColor": "transparent",
                                                                        "cursor": "pointer",
                                                                        "margin-right": "10px",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        dcc.RangeSlider(
                                                                            id="value-slider",
                                                                            min=0,
                                                                            max=None,
                                                                            step=1,
                                                                            value=[
                                                                                0,
                                                                                1,
                                                                            ],
                                                                            marks=None,
                                                                            tooltip=None,
                                                                            allowCross=False,
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "flex": "1",
                                                                        "min-width": "0",
                                                                        "margin": "0px",
                                                                    },
                                                                ),
                                                                html.Span(
                                                                    "Grouping",
                                                                    style={
                                                                        "fontWeight": "bold",
                                                                        "marginLeft": "15px",
                                                                        "whiteSpace": "nowrap",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Button(
                                                                            "⬇️",
                                                                            id="button-divide",
                                                                            n_clicks=0,
                                                                        ),
                                                                        dcc.Input(
                                                                            id="input-number",
                                                                            type="number",
                                                                            min=1,
                                                                            max=n_segments,
                                                                            step=1,
                                                                            value=1,
                                                                            style={
                                                                                "width": "70px",
                                                                                "margin": "0 0px",
                                                                            },
                                                                        ),
                                                                        html.Button(
                                                                            "⬆️",
                                                                            id="button-multiply",
                                                                            n_clicks=0,
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "display": "inline-block",
                                                                        "margin-left": "5px",
                                                                    },
                                                                ),
                                                                dbc.Checkbox(
                                                                    id="average-checkbox",
                                                                    label="",
                                                                    style={
                                                                        "margin-left": "40px",
                                                                        "display": "none",
                                                                    },
                                                                ),
                                                            ],
                                                            style={
                                                                "display": "flex",
                                                                "align-items": "center",
                                                                "justify-content": "center",
                                                            },
                                                        ),
                                                    ]
                                                ),
                                                style={
                                                    # "height": "100px",
                                                    "margin": "0px 10px 10px 10px",
                                                    "padding": "0px",
                                                    "text-align": "center",
                                                    "flex": "1",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "align-items": "center",
                                            "height": "100%",
                                        },
                                    ),
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="graphs",
                                style={"display": "none"},
                                config={
                                    "toImageButtonOptions": {
                                        "format": "svg",
                                        "filename": "CARM_Tool",
                                    },
                                    "editable": False,
                                    "displaylogo": False,
                                    "edits": {
                                        "annotationPosition": True,
                                    },
                                },
                            ),
                            width=11,
                        ),
                    ],
                    className="g-0",
                ),
            ],
            id="slider-components",
            style={"display": "none"},
        ),
        html.Div(id="graph-size-data", style={"whiteSpace": "pre-wrap", "display": "none"}),
        html.Div(id="graph-size-update", style={"whiteSpace": "pre-wrap", "display": "none"}),
        dcc.Store(id="store-dimensions"),
        dcc.Store(id="graph-lines"),
        dcc.Store(id="graph-lines2"),
        dcc.Store(id="graph-values"),
        dcc.Store(id="graph-values2"),
        dcc.Store(id="graph-isa"),
        dcc.Store(id="graph-xrange"),
        dcc.Store(id="graph-yrange"),
        dcc.Store(id="change-annon"),
        dcc.Store(id="clicked-point-index", data=-1),
        dcc.Store(id="clicked-trace-index", data=-1),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "⬆",
                                    style={
                                        "fontSize": "24px",
                                        "color": "#6c757d",
                                        "marginRight": "10px",
                                    },
                                ),
                                html.Span(
                                    "Select a Machine to View CARM Results",
                                    style={
                                        "fontSize": "20px",
                                        "fontWeight": "bold",
                                        "color": "#6c757d",
                                    },
                                ),
                                html.Span(
                                    "⬆",
                                    style={
                                        "fontSize": "24px",
                                        "color": "#6c757d",
                                        "marginLeft": "10px",
                                    },
                                ),
                            ],
                            id="initial-text",
                            style={"textAlign": "center", "marginTop": "10px"},
                        ),
                        html.Img(
                            src="/assets/carm_bsc.svg",
                            id="initial-image",
                            style={
                                "width": "99%",
                                "height": "90%",
                                "background": "transparent",
                                "marginLeft": "40px",
                            },
                        ),
                    ],
                    width=10,
                    style={"backgroundColor": "#e9ecef", "textAlign": "center"},
                )
            ],
            id="initial-content",
            justify="center",
            style={"backgroundColor": "#e9ecef", "textAlign": "center"},
        ),
        dcc.Store(id="machine-selected", data=False),
        sidebar,
        sidebar2,
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle(
                        "Edit Point Style",
                        style={"text-align": "center", "color": "white"},
                    ),
                    style={"backgroundColor": "#6c757d"},
                ),
                dbc.ModalBody(
                    [
                        daq.ColorPicker(
                            label=" ",
                            id="dot-color-picker",
                            value={"hex": "#0000FF"},  # blue
                        ),
                        html.Hr(),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Label("Size:", style={"marginRight": "10px"}),
                                        dcc.Input(
                                            id="dot-size-input",
                                            type="number",
                                            value=10,
                                            min=1,
                                            max=40,
                                            step=1,
                                            style={
                                                "marginRight": "20px",
                                                "width": "45px",
                                            },
                                        ),
                                        html.Label("Shape:", style={"marginRight": "10px"}),
                                        dcc.Dropdown(
                                            id="dot-symbol-dropdown",
                                            options=[
                                                {"label": "Circle", "value": "circle"},
                                                {"label": "Square", "value": "square"},
                                                {
                                                    "label": "Diamond",
                                                    "value": "diamond",
                                                },
                                                {"label": "Cross", "value": "cross"},
                                                {"label": "X", "value": "x"},
                                                {
                                                    "label": "Triangle-Up",
                                                    "value": "triangle-up",
                                                },
                                                {
                                                    "label": "Triangle-Down",
                                                    "value": "triangle-down",
                                                },
                                            ],
                                            value="circle",
                                            style={"width": "170px"},
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center"},
                                )
                            ],
                            width="auto",
                        ),
                    ],
                    style={"backgroundColor": "#e9ecef"},
                    id="modal-body",
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Submit",
                            id="dot-submit-button",
                            className="ms-auto",
                            n_clicks=0,
                            style={"margin-right": "auto"},
                        ),
                        dbc.Button(
                            "Close",
                            id="close-dot-modal",
                            className="me-auto",
                            n_clicks=0,
                            style={"margin-left": "auto"},
                        ),
                    ],
                    className="w-100 d-flex",
                    style={"backgroundColor": "#6c757d"},
                ),
            ],
            id="point-edit-modal",
            is_open=False,
            style={"width": "auto", "centered": "true"},
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle(
                        "Warning - Missing Counter Data",
                        className="text-center w-100",
                        style={"text-align": "center", "color": "white"},
                    ),
                    style={"text-align": "center", "backgroundColor": "#6c757d"},
                ),
                dbc.ModalBody(
                    [html.Pre(missing_msg)],
                    style={"backgroundColor": "#e9ecef", "text-align": "center"},
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Close",
                            id="close-warning-modal",
                            className="me-auto",
                            n_clicks=0,
                            style={"margin-left": "auto"},
                        ),
                    ],
                    className="w-100 d-flex",
                    style={"backgroundColor": "#6c757d"},
                ),
            ],
            size="lg",
            id="warning-modal",
            is_open=is_modal_open,
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Create a New Annotation"),
                dbc.ModalBody(
                    [
                        dbc.Label("Annotation Text"),
                        dbc.Input(
                            type="text",
                            id="annotation-text-input",
                            placeholder="Enter annotation text",
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Submit",
                        id="submit-annotation",
                        className="ms-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="annotation-modal",
            is_open=False,
        ),
        dcc.Store(id="annotations-store", data=[]),
    ],
    fluid=True,
    className="p-3",
    style={"backgroundColor": "#e9ecef"},
)


# App Callbacks
@app.callback(
    Output("button-paraver-colors", "children"),
    Input("button-paraver-colors", "n_clicks"),
)
def toggle_button_paraver_colors(n_clicks):
    if (n_clicks + color_button_offset) % 2 == 1:
        return "Use Paraver Timeline Colors"
    else:
        return "Use CARM GUI Colors"


@app.callback(
    Output("button-paraver-mask", "children"),
    Input("button-paraver-mask", "n_clicks"),
)
def toggle_button_paraver_mask(n_clicks):
    if (n_clicks + mask_button_offset) % 2 == 1:
        return "Use Semantic Window"
    else:
        return "Use All Timestamps"


@app.callback(
    Output("button-paraver-accumulate", "children"),
    Input("button-paraver-accumulate", "n_clicks"),
)
def toggle_button_paraver_accum(n_clicks):
    if (n_clicks + ac_button_offset) % 2 == 1:
        return "Plot Accumulated Values"
    else:
        return "Plot Raw Values"


@app.callback(
    Output("duration-filter", "value"),
    Input("duration-filter", "value"),
    prevent_initial_call=True,
)
def enforce_non_null(value):
    if value is None:
        return 0
    return value


@app.callback(
    Output("time-slider", "value", allow_duplicate=True),
    Output("value-slider", "value", allow_duplicate=True),
    Output("paraver-sync-timestamps", "data"),
    Input("paraver-sync-check", "n_intervals"),
    Input("button-paraver-sync", "n_clicks"),
    Input("lower-filter", "value"),
    Input("duration-filter", "value"),
    Input("button-paraver-mask", "n_clicks"),
    State("paraver-sync-timestamps", "data"),
    State("filename", "value"),
    prevent_initial_call=True,
)
def update_slider_from_csv(
    n_intervals,
    button_clicks,
    lower_filter,
    duration_filter,
    mask_button,
    current_values,
    selected_file,
):
    def prevent_update_for_reason(reason: str):
        logging.debug(f"Preventing update on update_slider_from_csv: {reason}")
        raise PreventUpdate

    global sync_csv_path
    global current_file_timestamps
    if mask_button_offset == -1:
        prevent_update_for_reason("Mask button offset is -1.")
    if not selected_file:
        prevent_update_for_reason("No file selected.")
    else:
        global no_sync
        global first_load
        try:
            csv_df = pd.read_csv(sync_csv_path)
            new_timestamps = [float(csv_df.iloc[0, 0]), float(csv_df.iloc[1, 0])]
        except Exception:
            first_load += 1
            new_timestamps = current_file_timestamps

        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if new_timestamps == current_file_timestamps and trigger_id != "button-paraver-sync":
            prevent_update_for_reason("Timestamps in CSV have not changed and trigger is not sync button.")

        first_load += 1
        current_file_timestamps = new_timestamps

        if first_load <= 1:
            prevent_update_for_reason("First load.")

        try:
            start_index = (full_base_statistics_df["Timestamp"] - new_timestamps[0]).abs().idxmin()
            end_index = (full_base_statistics_df["Timestamp"] - new_timestamps[1]).abs().idxmin()
            use_paraver_mask = resolve_toggle_enabled(mask_button, mask_button_offset)

            adjusted_start_index = ut.find_nearest_positive(
                full_base_statistics_df,
                start_index,
                float(lower_filter),
                float(duration_filter),
                use_paraver_mask,
                min_bound=0,
            )
            adjusted_end_index = ut.find_nearest_positive(
                full_base_statistics_df,
                end_index,
                float(lower_filter),
                float(duration_filter),
                use_paraver_mask,
                min_bound=adjusted_start_index,
            )

            matching_start_timestamp = full_base_statistics_df.loc[adjusted_start_index, "Timestamp"]
            matching_end_timestamp = full_base_statistics_df.loc[adjusted_end_index, "Timestamp"]

            filtered_base, _ = filter_base_and_intel_data(
                base_statistics_df,
                intel_statistics_df2,
                lower_filter,
                duration_filter,
                use_paraver_mask,
                ut.is_valid_paraver_value,
            )

            new_start_index = filtered_base[filtered_base["Timestamp"] == matching_start_timestamp].index[0]
            new_end_index = filtered_base[filtered_base["Timestamp"] == matching_end_timestamp].index[0]

        except Exception as e:
            if no_sync:
                print("ERROR finding indices in main_df:", e, flush=True)
                print(
                    'Check if the "Cut values lower than" option is not too high for the current region of interest',
                    flush=True,
                )
                no_sync = False
            raise PreventUpdate from None

        new_slider_indices = [int(new_start_index), int(new_end_index)]

        def print_separator():
            print("-" * 50, flush=True)

        if trigger_id == "button-paraver-sync":
            print_separator()
            print(
                "Sync Button Clicked, updating slider to timestamp range {} - {}".format(
                    filtered_base.loc[new_start_index, "Timestamp"],
                    filtered_base.loc[new_end_index, "Timestamp"],
                )
            )

            if adjusted_start_index != start_index:
                print(
                    "INFO: Adjusted Start Timestamp to {} from {} to allow for CARM plotting".format(
                        filtered_base.loc[new_start_index, "Timestamp"],
                        (full_base_statistics_df.loc[start_index, "Timestamp"]),
                    ),
                    flush=True,
                )

            if adjusted_end_index != end_index:
                print(
                    "INFO: Adjusted End Timestamp to {} from {} to allow for CARM plotting".format(
                        filtered_base.loc[new_end_index, "Timestamp"],
                        (full_base_statistics_df.loc[end_index, "Timestamp"]),
                    ),
                    flush=True,
                )

            print_separator()
            no_sync = True
            return new_slider_indices, new_slider_indices, new_timestamps

        if new_slider_indices != current_values:
            print_separator()
            print(
                "Sync CSV values changed, updating slider to timestamp range {} - {}".format(
                    filtered_base.loc[new_start_index, "Timestamp"],
                    filtered_base.loc[new_end_index, "Timestamp"],
                )
            )

            if adjusted_start_index != start_index:
                print(
                    "INFO: Adjusted Start Timestamp to {} from {} to allow for CARM plotting".format(
                        filtered_base.loc[new_start_index, "Timestamp"],
                        (full_base_statistics_df.loc[start_index, "Timestamp"]),
                    ),
                    flush=True,
                )

            if adjusted_end_index != end_index:
                print(
                    "INFO: Adjusted End Timestamp to {} from {} to allow for CARM plotting".format(
                        filtered_base.loc[new_end_index, "Timestamp"],
                        (full_base_statistics_df.loc[end_index, "Timestamp"]),
                    ),
                    flush=True,
                )

            print_separator()
            no_sync = True
            return new_slider_indices, new_slider_indices, new_timestamps


@app.callback(
    Input("button-roof-labels", "n_clicks"),
    Input("graph-lines", "data"),
    prevent_initial_call=True,
)
def generate_csv(n_clicks, lines):
    global full_base_statistics_df, prv_trace_path, time_unit
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id != "button-roof-labels":
        raise PreventUpdate

    if lines is None:
        print("Graph lines data is None, cannot generate roof labels CSV.", flush=True)
        return

    df: pd.DataFrame = full_base_statistics_df.copy()
    df["Roof Label"] = df.apply(lambda row: ut.label_cache_level(row, lines), axis=1)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    metadata_line = f"#{timestamp}:CSV:RUNAPP:{prv_trace_path}:{time_unit}:{WindowMode.CODE.value}:1:6"

    csv_df = df[["ThreadID", "Timestamp", "Duration", "Roof Label"]]
    # natural sort on the thread ID column so values like "1.1.10" come after
    # "1.1.2" instead of being ordered lexicographically.
    csv_df = csv_df.sort_values(
        ["ThreadID", "Timestamp"],
        key=lambda col: ut.natural_sort_series(col) if col.name == "ThreadID" else col,
    )

    output_dir = os.path.dirname(prv_trace_path)
    roof_csv_filepath = os.path.join(output_dir, "carm_roofs.csv")
    with open(roof_csv_filepath, "w") as f:
        f.write(metadata_line + "\n")
        csv_df.to_csv(f, index=False, header=False, sep="\t")

    roof_labels_filepath = os.path.join(output_dir, "carm_roofs.legend.csv")
    labels_data = [
        [1, "L1", 0, 255, 0],  # Green
        [2, "L2", 0, 0, 255],  # Blue
        [3, "L3", 255, 165, 0],  # Orange
        [4, "DRAM", 255, 0, 0],  # Red
        [5, "No Floating Point Operations Found", 75, 0, 130],  # Indigo
        [6, "Above L1", 255, 192, 203],  # Pink
    ]
    with open(roof_labels_filepath, "w") as f:
        for row in labels_data:
            label_line = f'{row[0]} "{row[1]}",{row[2]},{row[3]},{row[4]}\n'
            f.write(label_line)
    print("carm_roofs.csv file written.", flush=True)

    return


@app.callback(
    Input("button-carm-ldst-colors", "n_clicks"),
    Input("button-carm-spdp-colors", "n_clicks"),
    Input(component_id="graphs", component_property="figure"),
    prevent_initial_call=True,
)
def generate_color_csv(n_clicks_ldst, n_clicks_spdp, graph):
    global full_base_statistics_df, prv_trace_path, time_unit, intel_statistics_df2
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id not in ["button-carm-ldst-colors", "button-carm-spdp-colors"]:
        raise PreventUpdate

    df = full_base_statistics_df.copy()

    if trigger_id == "button-carm-ldst-colors":
        df = df.merge(
            intel_statistics_df2[["Timestamp", "ThreadID", "Intel_Load_Percent"]],
            on=["Timestamp", "ThreadID"],
            how="left",
        )
        unique_percentages = df["Intel_Load_Percent"].dropna().unique()
        df["Intel_Load_Percent"] = df["Intel_Load_Percent"].fillna(0)
    elif trigger_id == "button-carm-spdp-colors":
        df = df.merge(
            intel_statistics_df2[["Timestamp", "ThreadID", "Intel_FP_DP_Percent"]],
            on=["Timestamp", "ThreadID"],
            how="left",
        )
        unique_percentages = df["Intel_FP_DP_Percent"].dropna().unique()
        df["Intel_FP_DP_Percent"] = df["Intel_FP_DP_Percent"].fillna(0)

    unique_percentages.sort()
    color_map = []

    for percentage in unique_percentages:
        if trigger_id == "button-carm-ldst-colors":
            r, g, b = ut.blend_colors(0, 0, 0, 0, 0, percentage, 0, "LD/ST Percentage", True)
            extra_string = "Loads"

        elif trigger_id == "button-carm-spdp-colors":
            r, g, b = ut.blend_colors(0, 0, 0, 0, percentage, 0, 0, "Precision", True)
            extra_string = "DP"

        color_map.append(
            {
                "percentage": percentage,
                "percentage_string": f"{percentage}% {extra_string}",
                "r": r,
                "g": g,
                "b": b,
            }
        )

    color_map_df = pd.DataFrame(color_map)
    output_dir = os.path.dirname(prv_trace_path)
    roof_labels_filepath = os.path.join(output_dir, "carm_colors.legend.csv")
    ut.format_ld_st_csv(color_map_df, roof_labels_filepath)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    metadata_line = (
        f"#{timestamp}:CSV:RUNAPP:{prv_trace_path}:{time_unit}:{WindowMode.CODE.value}:{color_map_df['percentage'].min()}:"
        f"{color_map_df['percentage'].max()}"
    )

    if trigger_id == "button-carm-ldst-colors":
        csv_df = df[["ThreadID", "Timestamp", "Duration", "Intel_Load_Percent"]]
    elif trigger_id == "button-carm-spdp-colors":
        csv_df = df[["ThreadID", "Timestamp", "Duration", "Intel_FP_DP_Percent"]]
    csv_df = csv_df.sort_values(
        ["ThreadID", "Timestamp"],
        key=lambda col: ut.natural_sort_series(col) if col.name == "ThreadID" else col,
    )

    roof_csv_filepath = os.path.join(output_dir, "carm_colors.csv")
    with open(roof_csv_filepath, "w") as f:
        f.write(metadata_line + "\n")
        csv_df.to_csv(f, index=False, header=False, sep="\t")

    print("carm_colors.csv file written.", flush=True)

    return


@app.callback(
    Input("button-carm-gflops", "n_clicks"),
    Input("graph-lines", "data"),
    prevent_initial_call=True,
)
def generate_gflops_csv(n_clicks, lines):
    global full_base_statistics_df, prv_trace_path, time_unit
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id != "button-carm-gflops":
        raise PreventUpdate

    if lines is None:
        print("Graph lines data is None, cannot generate GFLOPS CSV.", flush=True)
        return

    df: pd.DataFrame = full_base_statistics_df.copy()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    min_gflops = df["GFLOPS"].min()
    max_gflops = df["GFLOPS"].max()
    metadata_line = (
        f"#{timestamp}:CSV:RUNAPP:{prv_trace_path}:{time_unit}:{WindowMode.GRADIENT.value}:{min_gflops}:{max_gflops}"
    )

    csv_df = df[["ThreadID", "Timestamp", "Duration", "GFLOPS"]].copy()
    csv_df["GFLOPS"] = csv_df["GFLOPS"].apply(lambda x: f"{x:.10f}")
    csv_df = csv_df.sort_values(
        ["ThreadID", "Timestamp"],
        key=lambda col: ut.natural_sort_series(col) if col.name == "ThreadID" else col,
    )

    output_dir = os.path.dirname(prv_trace_path)
    csv_filepath = os.path.join(output_dir, "carm_gflops.csv")
    with open(csv_filepath, "w") as f:
        f.write(metadata_line + "\n")
        csv_df.to_csv(f, index=False, header=False, sep="\t")

    print("carm_gflops.csv file written.", flush=True)

    return


@app.callback(
    Input("button-carm-ai", "n_clicks"),
    Input("graph-lines", "data"),
    prevent_initial_call=True,
)
def generate_ai_csv(n_clicks, lines):
    global full_base_statistics_df, prv_trace_path, time_unit
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id != "button-carm-ai":
        raise PreventUpdate

    if lines is None:
        print("Graph lines data is None, cannot generate AI CSV.", flush=True)
        return

    df: pd.DataFrame = full_base_statistics_df.copy()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    min_ai = df["Arithmetic_Intensity"].min()
    max_ai = df["Arithmetic_Intensity"].max()
    metadata_line = (
        f"#{timestamp}:CSV:RUNAPP:{prv_trace_path}:{time_unit}:{WindowMode.GRADIENT.value}:{min_ai}:{max_ai}"
    )

    csv_df = df[["ThreadID", "Timestamp", "Duration", "Arithmetic_Intensity"]].copy()
    csv_df["Arithmetic_Intensity"] = csv_df["Arithmetic_Intensity"].apply(lambda x: f"{x:.10f}")
    csv_df = csv_df.sort_values(
        ["ThreadID", "Timestamp"],
        key=lambda col: ut.natural_sort_series(col) if col.name == "ThreadID" else col,
    )

    output_dir = os.path.dirname(prv_trace_path)
    csv_filepath = os.path.join(output_dir, "carm_ai.csv")
    with open(csv_filepath, "w") as f:
        f.write(metadata_line + "\n")
        csv_df.to_csv(f, index=False, header=False, sep="\t")

    print("carm_ai.csv file written.", flush=True)

    return


@app.callback(
    Input("button-carm-roof-proximity", "n_clicks"),
    Input("graph-lines", "data"),
    prevent_initial_call=True,
)
def generate_roof_proximity_csv(n_clicks, lines):
    global full_base_statistics_df, prv_trace_path, time_unit
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id != "button-carm-roof-proximity":
        raise PreventUpdate

    if lines is None:
        print("Graph lines data is None, cannot generate roof proximity CSV.", flush=True)
        return

    df: pd.DataFrame = full_base_statistics_df.copy()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.dirname(prv_trace_path)

    ai = df["Arithmetic_Intensity"].values
    perf = df["GFLOPS"].values

    level_names = {"L1": "l1", "L2": "l2", "L3": "l3", "DRAM": "dram"}

    for level, suffix in level_names.items():
        if level not in lines:
            continue

        roof = lines[level]
        start_x, start_y = roof["start"]
        ridge_x, ridge_y = roof["ridge"]
        end_x, end_y = roof["end"]

        roof_vals = np.zeros_like(ai)

        left = ai <= ridge_x
        if np.any(left):
            if ridge_x == start_x:
                roof_vals[left] = start_y
            else:
                slope = (ridge_y - start_y) / (ridge_x - start_x)
                roof_vals[left] = start_y + slope * (ai[left] - start_x)

        right = ai > ridge_x
        if np.any(right):
            if end_x == ridge_x:
                roof_vals[right] = ridge_y
            else:
                slope = (end_y - ridge_y) / (end_x - ridge_x)
                roof_vals[right] = ridge_y + slope * (ai[right] - ridge_x)

        valid = (ai > 0) & (perf > 0) & (roof_vals > 0)
        ratios = np.where(valid, np.minimum(perf / roof_vals, 1.0), 0.0)

        metadata_line = f"#{timestamp}:CSV:RUNAPP:{prv_trace_path}:{time_unit}:{WindowMode.GRADIENT.value}:0.0:1.0"

        rel_df = pd.DataFrame(
            {
                "ThreadID": df["ThreadID"],
                "Timestamp": df["Timestamp"],
                "Duration": df["Duration"],
                "Ratio": ratios,
            }
        )
        rel_df["Ratio"] = rel_df["Ratio"].apply(lambda x: f"{x:.10f}")
        rel_df = rel_df.sort_values(
            ["ThreadID", "Timestamp"],
            key=lambda col: ut.natural_sort_series(col) if col.name == "ThreadID" else col,
        )

        csv_filepath = os.path.join(output_dir, f"carm_rel_{suffix}.csv")
        with open(csv_filepath, "w") as f:
            f.write(metadata_line + "\n")
            rel_df.to_csv(f, index=False, header=False, sep="\t")

        print(f"carm_rel_{suffix}.csv file written.", flush=True)

    return


@app.callback(
    Output("slider-components", "style"),
    Output("paraver-sync-check", "disabled"),
    Input("graph-lines", "data"),
    State("filename", "value"),
)
def toggle_components(lines, selected_file):
    if not selected_file:
        return {"display": "none"}, True
    else:
        return {"display": "block"}, False


@app.callback(
    Output("warning-modal", "is_open"),
    Input("close-warning-modal", "n_clicks"),
    State("warning-modal", "is_open"),
)
def toggle_modal_warning(close_clicks, current_state):
    if close_clicks:
        return not current_state
    return current_state


@app.callback(
    [
        Output("point-edit-modal", "is_open"),
        Output("clicked-trace-index", "data"),
        Output("clicked-point-index", "data"),
    ],
    [
        Input("graphs", "clickData"),
        Input("close-dot-modal", "n_clicks"),
        Input("dot-submit-button", "n_clicks"),
    ],
    [State("point-edit-modal", "is_open")],
)
def open_modal_on_click(click_data, close_clicks, submit_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id in ["dot-submit-button", "close-dot-modal"]:
        return False, -1, -1

    if click_data:
        trace_idx = click_data["points"][0]["curveNumber"]
        point_idx = click_data["points"][0]["pointIndex"]
        return True, trace_idx, point_idx

    return is_open, -1, -1


@app.callback(
    [
        Output("graphs", "figure", allow_duplicate=True),
        Output("point-edit-modal", "is_open", allow_duplicate=True),
    ],
    [Input("dot-submit-button", "n_clicks")],
    [
        State("dot-color-picker", "value"),
        State("dot-size-input", "value"),
        State("dot-symbol-dropdown", "value"),
        State("clicked-trace-index", "data"),
        State("clicked-point-index", "data"),
        State("graphs", "figure"),
    ],
    prevent_initial_call=True,
)
def update_point_style(
    n_submit,
    chosen_color,
    chosen_size,
    chosen_symbol,
    trace_idx,
    point_idx,
    current_fig,
):
    if not n_submit:
        raise PreventUpdate
    if trace_idx < 0 or point_idx < 0:
        raise PreventUpdate

    trace_data = current_fig["data"][trace_idx]
    markers = trace_data["marker"]

    x_vals = trace_data["x"]
    n_points = len(x_vals) if x_vals else 0

    if n_points == 0:
        raise PreventUpdate

    color_array = ut.ensure_list(markers, "color", "blue", n_points)
    size_array = ut.ensure_list(markers, "size", 10, n_points)
    symbol_array = ut.ensure_list(markers, "symbol", "circle", n_points)

    color_array[point_idx] = chosen_color["hex"]
    size_array[point_idx] = chosen_size
    symbol_array[point_idx] = chosen_symbol

    trace_data["marker"]["color"] = color_array
    trace_data["marker"]["size"] = size_array
    trace_data["marker"]["symbol"] = symbol_array

    return current_fig, False


@app.callback(
    Output("annotation-modal", "is_open"),
    [
        Input("create-annotation-button", "n_clicks"),
        Input("submit-annotation", "n_clicks"),
    ],
    [State("annotation-modal", "is_open")],
)
def toggle_modal_annotation(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [
        Output("graphs", "figure", allow_duplicate=True),
        Output("annotations-store", "data"),
    ],
    [Input("submit-annotation", "n_clicks")],
    [
        State("annotation-text-input", "value"),
        State("graphs", "figure"),
        State("annotations-store", "data"),
    ],
    prevent_initial_call=True,
)
def add_annotation(n_clicks, text, figure, annotations):
    if n_clicks and text:
        x = math.log10(1)
        y = math.log10(1)

        new_annotation = {
            "x": x,
            "y": y,
            "xref": "x",
            "yref": "y",
            "text": text,
            "showarrow": False,
            "bgcolor": "white",
            "bordercolor": "black",
            "borderwidth": 1,
        }

        if "annotations" in figure["layout"]:
            figure["layout"]["annotations"].append(new_annotation)
        else:
            figure["layout"]["annotations"] = [new_annotation]

        if annotations is None:
            annotations = []
        annotations.append(new_annotation)

        return figure, annotations

    return figure, annotations


@app.callback(
    [
        Output("graphs", "figure", allow_duplicate=True),
        Output("disable-annotation-button", "children"),
    ],
    [
        Input("disable-annotation-button", "n_clicks"),
        Input("disable-annotation-button", "children"),
    ],
    [State("graphs", "figure")],
    prevent_initial_call=True,
)
def toggle_annotations(n_clicks, button_text, current_fig):
    if not n_clicks:
        return current_fig, "Disable Annotations"

    if "annotations" in current_fig["layout"] and current_fig["layout"]["annotations"]:
        current_fig["layout"]["annotations"] = []

    if button_text == "Disable Annotations":
        button_text = "Enable Annotations"
    else:
        button_text = "Disable Annotations"

    return current_fig, button_text


def build_annotation_card(index, annotation):
    return dbc.Card(
        [
            dbc.CardHeader(
                f"{annotation.get('text')}",
                style={
                    "color": "white",
                    "fontWeight": "bold",
                    "margin": "0px",
                    "padding": "2px 0px 0px 2px",
                },
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                "Plot:",
                                                style={
                                                    "color": "white",
                                                    "marginRight": "10px",
                                                    "alignSelf": "center",
                                                },
                                            ),
                                            dbc.Checkbox(
                                                id={"type": "annotation-enable", "index": index},
                                                className="mb-0",
                                                style={"alignSelf": "center"},
                                                value=annotation.get("opacity", 1) == 1,
                                            ),
                                        ],
                                        style={"display": "flex", "alignItems": "center"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                "Angle:",
                                                style={
                                                    "color": "white",
                                                    "marginRight": "10px",
                                                    "marginLeft": "30px",
                                                    "alignSelf": "center",
                                                },
                                            ),
                                            dbc.Input(
                                                type="number",
                                                placeholder="Angle",
                                                value=round(annotation.get("textangle", 0)),
                                                id={"type": "angle-input", "index": index},
                                                style={"width": "80px", "height": "25px"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "marginRight": "30px",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "flex-start",
                                },
                            ),
                        ],
                        className="mb-0",
                        align="center",
                    ),
                ],
                style={"margin": "0px", "padding": "0px 0px 2px 2px"},
            ),
        ],
        className="mb-1",
        style={
            "margin": "0px",
            "padding": "0px 0px 2px 2px",
            "backgroundColor": "#6c757d",
            "Color": annotation.get("bordercolor"),
        },
    )


def _build_annotation_accordion_item(title, item_id, indexed_annotations):
    cards = [build_annotation_card(index, annotation) for index, annotation in indexed_annotations]
    return dbc.AccordionItem(title=title, children=cards, item_id=item_id)


@app.callback(
    Output("annotation-accordion", "children"),
    Input("graphs", "figure"),
    prevent_initial_call=True,
)
def generate_angle_inputs(graph):
    if not graph or "annotations" not in graph["layout"]:
        return []

    annotations = graph["layout"]["annotations"]

    group_suffixes = ["_1", "_2"]
    grouped_annotations = {suffix: [] for suffix in group_suffixes}
    ungrouped_annotations = []
    accordion_items = []

    for i, ann in enumerate(annotations):
        name = ann.get("name", "")
        matched = False
        for suffix in group_suffixes:
            if name.endswith(suffix):
                grouped_annotations[suffix].append((i, ann))
                matched = True
                break
        if not matched:
            ungrouped_annotations.append((i, ann))

    grouped_sections = [
        (f"CARM Results {suffix[-1]}", f"group_{suffix}", anns) for suffix, anns in grouped_annotations.items() if anns
    ]
    if ungrouped_annotations:
        grouped_sections.append(("Custom Annotations", "other_annotations", ungrouped_annotations))

    for title, item_id, indexed_annotations in grouped_sections:
        accordion_items.append(_build_annotation_accordion_item(title, item_id, indexed_annotations))

    return accordion_items


@app.callback(
    Output("graphs", "figure", allow_duplicate=True),
    [Input({"type": "annotation-enable", "index": ALL}, "value")],
    [State("graphs", "figure")],
    prevent_initial_call=True,
)
def update_annotations_visibility(checkbox_values, figure):
    fig = go.Figure(figure)
    annotations = fig["layout"]["annotations"]

    if annotations:
        for i, ann in enumerate(annotations):
            if i < len(checkbox_values):
                if checkbox_values[i]:
                    ann["opacity"] = 1  # Visible
                else:
                    ann["opacity"] = 0  # Hidden

        fig.update_layout(annotations=annotations)

    return fig


@app.callback(
    Output("graphs", "figure", allow_duplicate=True),
    [Input({"type": "angle-input", "index": ALL}, "value")],
    State("graphs", "figure"),
    prevent_initial_call=True,
)
def update_annotation_angles(input_angles, figure):
    # Callback to control annotations angle individually
    if not figure or not input_angles:
        raise dash.exceptions.PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    annotations = figure.get("layout", {}).get("annotations", [])

    for i, angle in enumerate(input_angles):
        if i < len(annotations):
            annotations[i]["textangle"] = angle

    new_figure = copy.deepcopy(figure)
    new_figure["layout"]["annotations"] = annotations

    return new_figure


@app.callback(Output("machine-selected", "data"), Input("filename", "value"))
def update_machine_selected(filename):
    # Callback to update the machine selected
    if filename:
        return True
    return False


@app.callback(
    [Output("initial-image", "style"), Output("initial-text", "style")],
    Input("machine-selected", "data"),
)
def toggle_initial_content(machine_selected):
    # Callback to control the visibility of the initial image and text
    if machine_selected:
        return {"display": "none"}, {"display": "none"}
    return {
        "width": "99%",
        "height": "90%",
        "display": "block",
        "background": "transparent",
        "marginLeft": "40px",
    }, {"text-align": "center", "margin-top": "10px"}


@app.callback(
    Output("offcanvas", "is_open"),
    [Input("open-offcanvas", "n_clicks")],
    [State("offcanvas", "is_open")],
)
def toggle_graph_options(n, is_open):
    # Toggle visibility of the sidebar
    if n:
        return not is_open
    return


@app.callback(
    Output("offcanvas2", "is_open"),
    [Input("open-offcanvas2", "n_clicks")],
    [State("offcanvas2", "is_open")],
)
def toggle_paraver_functions(n, is_open):
    # Toggle visibility of the sidebar
    if n:
        return not is_open
    return is_open


@app.callback(
    [
        Output("graphs", "figure", allow_duplicate=True),
        Output("graphs", "config", allow_duplicate=True),
        Output("button-CARM", "children"),
    ],
    [Input("button-CARM", "n_clicks")],
    [State("graphs", "figure"), State("graphs", "config")],
    prevent_initial_call=True,
)
def toggle_editable(n_clicks, figure, config):
    # Toggle editable state of the graph
    new_figure = copy.deepcopy(figure)
    if n_clicks % 2 == 0:
        config["editable"] = True
        return new_figure, config, "Save Text Changes"
    else:
        config["editable"] = False
        return new_figure, config, "Edit Graph Text"


@app.callback(Output("additional-dropdowns", "children"), [Input("filename", "value")])
def update_additional_dropdowns(selected_file):
    # Update the CARM results filter dropdowns
    return _build_additional_dropdowns_card(selected_file, "row1")


@app.callback(Output("additional-dropdowns2", "children"), [Input("filename", "value")])
def update_additional_dropdowns2(selected_file):
    # Update the CARM results filter dropdowns (line2)
    return _build_additional_dropdowns_card(selected_file, "row2")


ADDITIONAL_DROPDOWN_FIELDS = [
    "ISA",
    "Precision",
    "Threads",
    "Loads",
    "Stores",
    "Interleaved",
    "DRAM Bytes",
    "FP Inst",
    "Date",
]

ADDITIONAL_DROPDOWN_WIDTHS = {
    "Date": 250,
    "ISA": 200,
}

ADDITIONAL_DROPDOWN_ROWS = {
    "row1": {
        "suffix": "",
        "label": "CARM Results 1:",
        "label_style": {
            "marginRight": "10px",
            "alignSelf": "center",
            "fontWeight": "bold",
            "minWidth": "125px",
        },
    },
    "row2": {
        "suffix": "2",
        "label": "CARM Results 2:",
        "label_style": {
            "marginRight": "10px",
            "alignSelf": "center",
            "fontWeight": "bold",
            "color": "red",
            "minWidth": "125px",
        },
    },
}


def _build_additional_dropdown_options(df: pd.DataFrame | None, field: str):
    if df is None or df.empty:
        return {}

    sort_desc = field == "Date"
    unique_values = sorted(df[field.replace(" ", "")].unique(), reverse=sort_desc)
    return [{"label": value, "value": value} for value in unique_values]


def _build_additional_dropdowns_list(df: pd.DataFrame | None, suffix: str):
    dropdowns = []

    for field in ADDITIONAL_DROPDOWN_FIELDS:
        options = _build_additional_dropdown_options(df, field)
        width = ADDITIONAL_DROPDOWN_WIDTHS.get(field, 160)
        dropdowns.append(
            html.Div(
                dcc.Dropdown(
                    id=f"{field.lower().replace(' ', '')}-dynamic-dropdown{suffix}",
                    placeholder=field,
                    options=options,
                    multi=False,
                ),
                style={
                    "flex": "1 0 auto",
                    "minWidth": width,
                    "margin": "5px",
                },
            )
        )

    return dropdowns


def _build_additional_dropdowns_card(selected_file, row_key: str):
    row_cfg = ADDITIONAL_DROPDOWN_ROWS[row_key]
    df = None
    if selected_file:
        _, _, _, _, data_list = ut.read_csv_file(selected_file)
        df = pd.DataFrame(data_list)

    dropdowns = _build_additional_dropdowns_list(df, row_cfg["suffix"])

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Div(
                            row_cfg["label"],
                            style=row_cfg["label_style"],
                        ),
                        html.Div(
                            dropdowns,
                            style={
                                "display": "flex",
                                "width": "100%",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "margin": "-10px auto auto auto",
                    },
                )
            ]
        ),
        style={
            "margin": "0px auto 10px auto",
            "padding": "0px",
            "textAlign": "center",
            "display": "flex",
            "height": "60px",
        },
    )


ROOFLINE_FILTER_FIELD_ID_STEMS = {
    "ISA": "isa",
    "Precision": "precision",
    "Threads": "threads",
    "Loads": "loads",
    "Stores": "stores",
    "Interleaved": "interleaved",
    "DRAMBytes": "drambytes",
    "FPInst": "fpinst",
    "Date": "date",
}
ROOFLINE_SORT_DESC_FIELDS = {"Date"}


def _roofline_filter_dropdown_id(field_name: str, suffix: str = "") -> str:
    return f"{ROOFLINE_FILTER_FIELD_ID_STEMS[field_name]}-dynamic-dropdown{suffix}"


def _build_roofline_filter_input_group(suffix: str = "") -> dict[str, Input]:
    return {
        field_name: Input(_roofline_filter_dropdown_id(field_name, suffix), "value")
        for field_name in ROOFLINE_FILTER_FIELD_ID_STEMS
    }


ANALYSIS_CONTROL_INPUTS = {
    "timestamps_range": Input("value-slider", "value"),
    "timestamps_max_range": Input("time-slider", "value"),
    "timestamps_grouper": Input("input-number", "value"),
    "average": Input("average-checkbox", "value"),
    "n_clicks": Input("play-pause-button", "n_clicks"),
    "n_intervals": Input("interval-component", "n_intervals"),
    "ISA_timestamp": Input("isa-checklist", "value"),
    "Precision_timestamp": Input("precision-checklist", "value"),
    "Threads_timestamp": Input("thread-checklist", "value"),
    "color_radio": Input("color-radio", "value"),
    "plot_total": Input("total-checklist", "value"),
    "exponent": Input("exponent-switch", "value"),
    "line_legend": Input("line-legend-switch", "value"),
    "normalize": Input("normalize-switch", "value"),
    "lower_filter": Input("lower-filter", "value"),
    "duration_filter": Input("duration-filter", "value"),
    "line_size": Input("line-size", "value"),
    "title_size": Input("title-size", "value"),
    "axis_size": Input("axis-size", "value"),
    "tick_size": Input("tick-size", "value"),
    "tooltip_size": Input("tooltip-size", "value"),
    "legend_size": Input("legend-size", "value"),
    "dot_size": Input("dot-size", "value"),
    "mask_button": Input("button-paraver-mask", "n_clicks"),
    "accum_button": Input("button-paraver-accumulate", "n_clicks"),
    "paraver_color_button": Input("button-paraver-colors", "n_clicks"),
}
ANALYSIS_CALLBACK_INPUTS = {
    "filters_primary": _build_roofline_filter_input_group(),
    "filters_secondary": _build_roofline_filter_input_group("2"),
    "selected_file": Input("filename", "value"),
    "controls": ANALYSIS_CONTROL_INPUTS,
}


def _compute_filtered_dropdown_options(
    target_field: str, selected_filters: dict[str, Any], selected_file: str | None
) -> list[dict[str, Any]]:
    if not selected_file:
        return []

    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)
    if df.empty:
        return []

    query_conditions = []
    query_locals = {}
    for field_name, value in selected_filters.items():
        if not value:
            continue
        local_name = f"selected_{field_name.lower()}"
        query_conditions.append(f"{field_name} == @{local_name}")
        query_locals[local_name] = value

    if query_conditions:
        df = df.query(" and ".join(query_conditions), local_dict=query_locals)

    values = sorted(df[target_field].dropna().unique(), reverse=target_field in ROOFLINE_SORT_DESC_FIELDS)
    return [{"label": value, "value": value} for value in values]


def _make_chained_dropdown_callback(target_field: str, suffix: str):
    source_fields = [field_name for field_name in ROOFLINE_FILTER_FIELD_ID_STEMS if field_name != target_field]

    def callback(*callback_values):
        *selected_values, selected_file = callback_values
        selected_filters = dict(zip(source_fields, selected_values, strict=False))
        return _compute_filtered_dropdown_options(target_field, selected_filters, selected_file)

    callback.__name__ = f"chained_callback_{target_field}{suffix}"
    return callback


def _register_chained_dropdown_callbacks():
    for suffix in ("", "2"):
        for target_field in ROOFLINE_FILTER_FIELD_ID_STEMS:
            input_list = [
                Input(_roofline_filter_dropdown_id(field_name, suffix), "value")
                for field_name in ROOFLINE_FILTER_FIELD_ID_STEMS
                if field_name != target_field
            ]
            input_list.append(Input("filename", "value"))

            app.callback(
                Output(_roofline_filter_dropdown_id(target_field, suffix), "options"),
                *input_list,
                prevent_initial_call=True,
            )(_make_chained_dropdown_callback(target_field, suffix))


_register_chained_dropdown_callbacks()


@app.callback(
    [
        Output(component_id="graphs", component_property="figure"),
        Output("graphs", "style"),
        Output("graph-size-update", "children"),
        Output("graph-lines", "data"),
        Output("graph-lines2", "data"),
        Output("graph-values", "data"),
        Output("graph-values2", "data"),
        Output("graph-isa", "data"),
        Output("graph-xrange", "data"),
        Output("graph-yrange", "data"),
        Output("change-annon", "data"),
    ],
    inputs=ANALYSIS_CALLBACK_INPUTS,
    state={"figure": State("graphs", "figure")},
    prevent_initial_call=True,
)
def analysis(
    filters_primary,
    filters_secondary,
    selected_file,
    controls,
    figure,
):  # , intervals):
    timestamps_range = controls["timestamps_range"]
    timestamps_max_range = controls["timestamps_max_range"]
    timestamps_grouper = controls["timestamps_grouper"]
    average = controls["average"]
    n_clicks = controls["n_clicks"]
    n_intervals = controls["n_intervals"]
    ISA_timestamp = controls["ISA_timestamp"]
    Precision_timestamp = controls["Precision_timestamp"]
    Threads_timestamp = controls["Threads_timestamp"]
    color_radio = controls["color_radio"]
    plot_total = controls["plot_total"]
    exponent = controls["exponent"]
    line_legend = controls["line_legend"]
    normalize = controls["normalize"]
    lower_filter = controls["lower_filter"]
    duration_filter = controls["duration_filter"]
    line_size = controls["line_size"]
    title_size = controls["title_size"]
    axis_size = controls["axis_size"]
    tick_size = controls["tick_size"]
    tooltip_size = controls["tooltip_size"]
    legend_size = controls["legend_size"]
    dot_size = controls["dot_size"]
    mask_button = controls["mask_button"]
    accum_button = controls["accum_button"]
    paraver_color_button = controls["paraver_color_button"]

    # Callback to draw the CARM graph and plot everything
    top_flops2 = 0
    smallest_ai = 1000
    smallest_gflops = 1000
    graph_width = 1900
    graph_height = 675
    change_annotation = 0

    global data_points
    global lines_origin
    global lines_origin2
    intel_ISA = ["avx512", "avx2", "sse", "scalar"]

    if not selected_file:
        return (
            go.Figure(),
            {"display": "none"},
            "",
            None,
            None,
            None,
            None,
            None,
            [0, 0],
            [0, 0],
            None,
        )

    # Get trigger-id
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    annotations = {}
    if figure is not None:
        annotations = figure.get("layout", {}).get("annotations", [])

    if trigger_id not in ["graphs", "interval-component"]:
        figure = go.Figure()
    use_paraver_mask, use_accumulate, use_paraver_colors = resolve_analysis_paraver_toggles(
        mask_button,
        accum_button,
        paraver_color_button,
        mask_button_offset,
        ac_button_offset,
        color_button_offset,
    )

    # Read roofline data and create DataFrame
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)
    filtered_base, filtered_intel = filter_base_and_intel_data(
        base_statistics_df,
        intel_statistics_df2,
        lower_filter,
        duration_filter,
        use_paraver_mask,
        ut.is_valid_paraver_value,
    )

    # Get timestamp range to display and filter timestamps dataframe accordingly.
    timestamp_start, timestamp_end = resolve_timestamp_slice_bounds(
        timestamps_range,
        timestamps_max_range,
        timestamps_grouper,
    )
    df_filter = filtered_base.iloc[timestamp_start:timestamp_end]
    df_intel_filter = filtered_intel.iloc[timestamp_start:timestamp_end]

    # Filter timestamps again to display based on the filter options
    df_intel_filter2 = ut.construct_query_timestamp(
        df_intel_filter, ISA_timestamp, Precision_timestamp, Threads_timestamp
    )
    df_filter = df_filter[df_filter.index.isin(df_intel_filter2.index)]
    columns_to_check = df_intel_filter2.drop(columns=["ThreadID", "Paraver_Label", "Timestamp"], errors="ignore")

    # Check what ISAs are still being used to adjust the roofline plot shown.
    ISA = infer_effective_isa_from_timestamp_columns(
        columns_to_check,
        lower_filter,
        filters_primary["ISA"],
        "ThreadID" in df_intel_filter2.columns,
    )
    primary_filters_for_query = dict(filters_primary)
    primary_filters_for_query["ISA"] = ISA

    # Get queries for both sets of inputs for the roofline data
    query2 = ut.construct_query(filters_secondary)
    query1 = ut.construct_query(primary_filters_for_query)
    # If user selects nothing yet, use the most recent roofline result
    filtered_df1 = filter_roofline_df_by_query(df, query1)

    # If there is no available ISA that matches what the timestamps are using, cycle through them
    if filtered_df1.empty:
        filtered_df1 = fallback_roofline_df_by_isa(
            df,
            intel_ISA,
            ut.construct_query,
            primary_filters_for_query,
        )

    # If the user selects anything from the second set of dropdowns, get the matching roofline data
    filtered_df2 = (
        filter_roofline_df_by_query(df, query2) if query2 and any(filters_secondary.values()) else pd.DataFrame()
    )

    # Totals from the timestamps for plotting
    threads_app = total_threads
    ai = total_ai
    gflops = total_GFLOPS
    name_app = appname

    # Plot Timestamps, if its a zoom we skip this
    if timestamps_range is not None:
        first = True
        timestamp_series, min_ai, min_gflops = prepare_timestamp_series(
            df_filter,
            df_intel_filter2,
            average,
            timestamps_grouper,
            timestamp_start,
            use_accumulate,
        )
        extra_average = timestamp_series.extra_average

        if min_ai is not None:
            smallest_ai = min_ai
        if min_gflops is not None:
            smallest_gflops = min_gflops

        n = timestamp_series.count
        data_points = n

        def add_timestamp_point_trace(
            point,
            color_context,
            trace_name,
            showlegend,
            legendgroup=None,
        ):
            color = select_timestamp_color(point, color_context)
            tooltip_text = ut.build_timestamp_tooltip_text(*build_timestamp_tooltip_args(point, window_name))
            return go.Scatter(
                **build_timestamp_scatter_trace(
                    point.ai_value,
                    point.gflops_value,
                    trace_name,
                    dot_size,
                    color,
                    tooltip_text,
                    showlegend,
                    legendgroup=legendgroup,
                )
            )

        # If the play function is activated
        if trigger_id == "play-pause-button" and data_points > 0:
            if n_clicks != 1:
                if figure is not None:
                    figure = go.Figure(figure)
        elif (trigger_id == "interval-component") and data_points > 0:
            if figure is not None:
                figure = copy.deepcopy(figure)
                figure = go.Figure(figure)
            indexer, first = resolve_interval_point_index_and_legend(n_intervals, data_points)
            point = get_timestamp_point(timestamp_series, indexer)
            color_context = TimestampColorContext(
                use_paraver_colors=False,
                color_radio=color_radio,
                index=indexer,
                n_points=n,
                start_color=start_color,
                end_color=end_color,
                blend_colors_fn=ut.blend_colors,
                interpolate_color_fn=ut.interpolate_color,
            )
            figure.add_trace(
                add_timestamp_point_trace(
                    point,
                    color_context,
                    f"{name_app}{extra_average}Timestamps",
                    first,
                    legendgroup="1",
                )
            )
        # If we are just doing regular plotting
        else:
            first = True
            plabel_aux = set()
            timestamp_traces = []
            common_name_prefix = f"{name_app}{extra_average}"
            color_context = TimestampColorContext(
                use_paraver_colors=use_paraver_colors,
                color_radio=color_radio,
                index=0,
                n_points=n,
                start_color=start_color,
                end_color=end_color,
                blend_colors_fn=ut.blend_colors,
                interpolate_color_fn=ut.interpolate_color,
            )
            for index, point in enumerate(iter_timestamp_points(timestamp_series)):
                if not should_plot_timestamp_point(use_paraver_mask, point):
                    continue

                color_context.index = index

                showlegend, legend_plabel, first = resolve_timestamp_legend_state(
                    use_paraver_colors,
                    point.plabel,
                    plabel_aux,
                    first,
                    window_mode,
                )

                timestamp_traces.append(
                    add_timestamp_point_trace(
                        point,
                        color_context,
                        f"{common_name_prefix}{legend_plabel} Timestamps",
                        showlegend,
                    )
                )

            if timestamp_traces:
                figure.add_traces(timestamp_traces)

    if trigger_id not in ["interval-component"]:
        # If we want to plot the total dot
        if plot_total:
            tooltip = ut.build_total_tooltip_text(name_app, threads_app, totals, total_FP_inst, total_mem_inst)
            smallest_gflops = min(gflops, smallest_gflops)
            smallest_ai = min(ai, smallest_ai)

            figure.add_trace(
                go.Scatter(
                    x=[ai],
                    y=[gflops],
                    mode="markers",
                    name=f"{name_app} Total",
                    marker={"size": dot_size, "color": "red"},
                    text=[tooltip],
                    hovertemplate="<b>%{text}</b><br>(%{x}, %{y})<br><extra></extra>",
                )
            )
    # If we want to reset the zoom
    if trigger_id not in ["graphs", "interval-component"]:
        figure.update_layout(
            hoverlabel={
                "font_size": tooltip_size,
            },
            title={
                "text": "Cache Aware Roofline Model" + " (per thread)" if normalize else "",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"family": "Arial", "size": title_size, "color": "black"},
            },
            height=graph_height,
            width=graph_width,
            xaxis={
                "title": {
                    "text": "Arithmetic Intensity (flop/byte)",
                    "font": {"family": "Arial", "size": axis_size, "color": "black"},
                },
                "type": "log",
                "dtick": "0.30102999566",
                "title_standoff": 0,
                "automargin": True,
                "tickfont_size": tick_size,
            },
            yaxis={
                "title": {
                    "text": f"Performance (GFLOP/s{' per thread' if normalize else ''})",
                    "font": {"family": "Arial", "size": axis_size, "color": "black"},
                },
                "type": "log",
                "dtick": "0.30102999566",
                "title_standoff": 0,
                "automargin": True,
                "tickfont_size": tick_size,
            },
            legend={
                "font": {"size": legend_size},
                "orientation": "h",
                "x": 0.5,
                "y": 0,
                "xanchor": "center",
                "yanchor": "bottom",
                "yref": "container",
            },
            showlegend=True,
            plot_bgcolor="#e9ecef",
            paper_bgcolor="#e9ecef",
            clickmode="event",
        )
        figure.update_xaxes(showspikes=True)
        figure.update_yaxes(showspikes=True)

    # Plot the roofline lines if possible, based on the data range and calculate angles for the annotations
    lines = {}
    lines2 = {}
    values1 = []
    values2 = []
    isa_labels = []
    top_flops = 0
    top_flops2 = 0
    x_min_angle = 0
    x_max_angle = 0
    y_min_angle = 0
    y_max_angle = 0

    def process_roofline_profile(profile, previous_lines, suffix, require_existing_old_lines=False):
        if profile is None:
            reset_annotations = should_reset_annotations_for_lines({}, previous_lines, trigger_id)
            return {
                "values": [],
                "isa": None,
                "lines": {},
                "top_flops": 0,
                "min_gflops": None,
                "reset_annotations": reset_annotations,
            }

        values, isa, profile_lines, profile_top_flops, profile_min_gflops = profile
        reset_annotations = should_reset_annotations_for_lines(
            profile_lines,
            previous_lines,
            trigger_id,
            require_existing_old_lines=require_existing_old_lines,
        )

        # If its just a zoom we dont plot the lines again, just re-calculate annotation angles.
        if trigger_id not in ["graphs", "interval-component"]:
            figure.add_traces(ut.plot_roofline(values, profile_lines, suffix, isa, line_legend, int(line_size)))

        return {
            "values": values,
            "isa": isa,
            "lines": profile_lines,
            "top_flops": profile_top_flops,
            "min_gflops": profile_min_gflops,
            "reset_annotations": reset_annotations,
        }

    profile1 = calculate_roofline_profile(filtered_df1, normalize, smallest_ai, ut.calculate_roofline)
    profile1_result = process_roofline_profile(
        profile1,
        lines_origin,
        "",
        require_existing_old_lines=True,
    )
    values1 = profile1_result["values"]
    lines = profile1_result["lines"]
    top_flops = profile1_result["top_flops"]
    if profile1_result["isa"] is not None:
        isa_labels.append(profile1_result["isa"])
    if profile1_result["min_gflops"] is not None:
        smallest_gflops = min(smallest_gflops, profile1_result["min_gflops"])
    if profile1_result["reset_annotations"]:
        change_annotation = 1
        annotations = {}
    lines_origin = lines

    if query2 is not None:
        profile2 = calculate_roofline_profile(filtered_df2, normalize, smallest_ai, ut.calculate_roofline)
    else:
        profile2 = None

    profile2_result = process_roofline_profile(profile2, lines_origin2, "2")
    values2 = profile2_result["values"]
    lines2 = profile2_result["lines"]
    top_flops2 = profile2_result["top_flops"]
    if profile2_result["isa"] is not None:
        isa_labels.append(profile2_result["isa"])
    if profile2_result["min_gflops"] is not None:
        smallest_gflops = min(smallest_gflops, profile2_result["min_gflops"])
    if profile2_result["reset_annotations"]:
        change_annotation = 1
        annotations = {}
    lines_origin2 = lines2

    angle_source_lines = lines if lines else lines2
    if angle_source_lines:
        x_min_angle, x_max_angle, y_min_angle, y_max_angle = resolve_roofline_angle_bounds(
            figure.layout.xaxis.range,
            figure.layout.yaxis.range,
            smallest_ai,
            angle_source_lines["DRAM"]["start"][1],
            max(angle_source_lines["L1"]["ridge"][1], max(top_flops2, top_flops)),
        )

    base_lines = lines if lines else lines2
    if exponent and base_lines:
        x_min, x_max = resolve_roofline_x_bounds(smallest_ai)

        y_min = min(smallest_gflops / 5, base_lines["DRAM"]["start"][1] / 5)
        y_max = max(base_lines["L1"]["ridge"][1], max(top_flops2, top_flops)) * 1.3

        x_tickvals, x_ticktext = ut.make_power_of_two_ticks(x_min, x_max)
        y_tickvals, y_ticktext = ut.make_power_of_two_ticks(y_min, y_max)

        # Update axes to show 2^X notation
        figure.update_xaxes(tickmode="array", tickvals=x_tickvals, ticktext=x_ticktext)
        figure.update_yaxes(tickmode="array", tickvals=y_tickvals, ticktext=y_ticktext)
    else:
        # Revert to normal formatting
        figure.update_yaxes(exponentformat=None, tickformat=None)
        figure.update_xaxes(exponentformat=None, tickformat=None)

    timestamp = datetime.datetime.now().isoformat()
    if annotations:
        figure["layout"]["annotations"] = annotations

    return (
        figure,
        {"display": "block"},
        f"Update: {timestamp}",
        lines,
        lines2,
        values1,
        values2,
        isa_labels,
        [x_min_angle, x_max_angle],
        [y_min_angle, y_max_angle],
        change_annotation,
    )  # , 'width': '100%', 'height' : '100%'}


@app.callback(
    [
        Output(component_id="graphs", component_property="figure", allow_duplicate=True),
        Output("change-annon", "data", allow_duplicate=True),
    ],
    [
        Input("graph-size-data", "children"),
        Input("graph-lines", "data"),
        Input("graph-lines2", "data"),
        Input("graph-values", "data"),
        Input("graph-values2", "data"),
        Input("graph-isa", "data"),
        Input("graph-xrange", "data"),
        Input("graph-yrange", "data"),
        Input("graphs", "relayoutData"),
        Input("change-annon", "data"),
        Input("disable-annotation-button", "children"),
        Input("annotation-size", "value"),
    ],
    State("graphs", "figure"),
    prevent_initial_call=True,
)
def angle_updater(
    size,
    lines,
    lines2,
    values1,
    values2,
    ISA,
    xrange,
    yrange,
    relayout_data,
    change_anon,
    disable_anon,
    anon_size,
    figure,
):
    # Callback to update the annotations angles when the graph scale/zoom changes
    if disable_anon != "Enable Annotations" and ISA:
        if figure:
            xaxis_range = figure["layout"]["xaxis"]["range"]

            yaxis_range = figure["layout"]["yaxis"]["range"]

        new_figure = go.Figure(figure)

        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        if size and len(size) >= 2:
            liner = size.split("\n")
            width = float(liner[0].replace("Plot area width:", "").replace("px", "").strip())
            height = float(liner[1].replace("Plot area height:", "").replace("px", "").strip())

            annotations = new_figure["layout"]["annotations"]
            cache_levels = ["L1", "L2", "L3", "DRAM"]
            cache_level_suffix = [
                "L1_1",
                "L2_1",
                "L3_1",
                "DRAM_1",
                "FP_1",
                "FP_FMA_1",
                "L1_2",
                "L2_2",
                "L3_2",
                "DRAM_2",
                "FP_2",
                "FP_FMA_2",
            ]
            for ann in annotations:
                ann_name = ann["name"]

                if ann_name in cache_level_suffix:
                    if ann_name[:-2] in cache_levels:
                        if ann_name[-1] == "1":
                            liner = lines
                        elif ann_name[-1] == "2" and lines2:
                            liner = lines2
                        else:
                            continue
                        log_x1, log_x2 = (
                            math.log10(liner[ann_name[:-2]]["start"][0]),
                            math.log10(liner[ann_name[:-2]]["ridge"][0]),
                        )
                        log_y1, log_y2 = (
                            math.log10(liner[ann_name[:-2]]["start"][1]),
                            math.log10(liner[ann_name[:-2]]["ridge"][1]),
                        )

                        log_xmin, log_xmax = xaxis_range[0], xaxis_range[1]
                        log_ymin, log_ymax = yaxis_range[0], yaxis_range[1]

                        # Compute pixel coordinates based on log scale
                        x1_pixel = ((log_x1 - log_xmin) / (log_xmax - log_xmin)) * width
                        x2_pixel = ((log_x2 - log_xmin) / (log_xmax - log_xmin)) * width
                        y1_pixel = height - ((log_y1 - log_ymin) / (log_ymax - log_ymin)) * height
                        y2_pixel = height - ((log_y2 - log_ymin) / (log_ymax - log_ymin)) * height

                        # Pixel slope
                        pixel_slope = (y2_pixel - y1_pixel) / (x2_pixel - x1_pixel)
                        ann["textangle"] = round(math.degrees(math.atan(pixel_slope)), 2)

            for cache_level in ["L1", "L2", "L3", "DRAM", "FP", "FMA"]:
                if not annotations or change_anon == 1:
                    new_figure.add_annotation(
                        ut.draw_annotation(
                            values1,
                            lines,
                            "1",
                            ISA[0],
                            cache_level,
                            width,
                            height,
                            x_range=[xaxis_range[0], xaxis_range[1]],
                            y_range=[yaxis_range[0], yaxis_range[1]],
                        )
                    )
            if len(lines2) > 0:
                for cache_level in ["L1", "L2", "L3", "DRAM", "FP", "FMA"]:
                    if not annotations or change_anon == 1:
                        new_figure.add_annotation(
                            ut.draw_annotation(
                                values2,
                                lines2,
                                "2",
                                ISA[1],
                                cache_level,
                                width,
                                height,
                                x_range=[xaxis_range[0], xaxis_range[1]],
                                y_range=[yaxis_range[0], yaxis_range[1]],
                            )
                        )
            if change_anon == 1:
                change_anon = 0

            return new_figure, change_anon
    else:
        return figure, change_anon


# Callback to toggle the Play/Pause button and enable/disable the interval
@app.callback(
    [
        Output("play-pause-button", "children"),
        Output("interval-component", "disabled"),
        Output("interval-component", "n_intervals"),
        Output("play-pause-button", "n_clicks"),
    ],
    [
        Input("play-pause-button", "n_clicks"),
        Input("interval-component", "n_intervals"),
        Input("input-number", "value"),
        Input("value-slider", "value"),
        Input("time-slider", "value"),
        Input("average-checkbox", "value"),
    ],
    [
        State("play-pause-button", "children"),
    ],
)
def toggle_play_pause(
    n_clicks,
    n_intervals,
    group_value,
    current_values,
    time_values,
    average,
    current_state,
):
    # Callback to control the play/pause button
    ctx = dash.callback_context
    triggered_input = ctx.triggered[0]["prop_id"].split(".")[0]

    if n_clicks == 0:
        # Initial state
        return "▶️", True, 0, n_clicks

    elif triggered_input in [
        "value-slider",
        "time-slider",
        "input-number",
        "average-checkbox",
    ]:
        n_intervals = 0
        n_clicks = 0
        return "▶️", True, n_intervals, n_clicks
    else:
        if triggered_input == "play-pause-button":
            if current_state == "▶️":
                return "⏸️", False, n_intervals, n_clicks
            else:
                return "▶️", True, n_intervals, n_clicks
        else:
            if current_state == "▶️":
                return "▶️", True, n_intervals, n_clicks
            else:
                if n_intervals >= data_points:
                    # All data points have been displayed; reset n_intervals
                    n_intervals = 0
                    n_clicks = 0
                    return "▶️", True, n_intervals, n_clicks
                else:
                    return "⏸️", False, n_intervals, n_clicks


@app.callback(
    [Output("input-number", "value"), Output("input-number", "max")],
    [
        Input("button-divide", "n_clicks"),
        Input("button-multiply", "n_clicks"),
        Input("time-slider", "value"),
        Input("lower-filter", "value"),
        Input("duration-filter", "value"),
    ],
    [State("input-number", "value")],
)
def update_number(
    divide_clicks,
    multiply_clicks,
    time_values,
    lower_filter,
    duration_filter,
    current_value,
):
    # Callback to update the grouping number
    ctx = dash.callback_context
    start_index = time_values[0]
    end_index = time_values[1]
    filtered_base = base_statistics_df[
        (base_statistics_df["Arithmetic_Intensity"] >= float(lower_filter))
        & (base_statistics_df["GFLOPS"] >= float(lower_filter))
        & (base_statistics_df["Duration"] >= float(duration_filter))
    ]
    filtered_base = filtered_base.reset_index(drop=True)
    selected_segments = filtered_base.loc[start_index:end_index, "Timestamp"].tolist()

    n_segments = len(selected_segments)

    if not ctx.triggered:
        if current_value < 0:
            return 1, n_segments
        return current_value, n_segments
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    new_value = current_value
    if button_id == "button-divide" and current_value > 1:
        new_value = max(1, current_value // 2)
    elif button_id == "button-multiply":
        new_value = min(current_value * 2, n_segments)

    return new_value, n_segments


def _get_timestamp_segments(lower_filter, duration_filter, use_paraver_mask, start_index=None, end_index=None):
    filtered_base, _ = filter_base_and_intel_data(
        base_statistics_df,
        intel_statistics_df2,
        lower_filter,
        duration_filter,
        use_paraver_mask,
        ut.is_valid_paraver_value,
    )

    if start_index is not None and end_index is not None:
        return filtered_base.loc[start_index:end_index, "Timestamp"].tolist()

    return filtered_base["Timestamp"].tolist()


def _group_slider_segments(segments, group_value):
    grouped_segments = []
    for i in range(0, len(segments), group_value):
        current_group = segments[i : i + group_value]
        if len(current_group) > 1:
            grouped_segments.append(f"{current_group[0]}...{current_group[-1]}")
        else:
            grouped_segments.append(f"{current_group[0]}")
    return grouped_segments


def _build_slider_marks(grouped_segments):
    return {i: {"label": value} for i, value in enumerate(grouped_segments)}


def _apply_alternating_mark_styles(marks):
    for i in marks:
        marks[i]["style"] = {"margin-top": "0px"} if i % 2 == 0 else {"margin-top": "-35px"}


def _select_current_range_marks(marks, current_values, grouped_segments):
    if isinstance(current_values, list) and len(current_values) == 2 and len(grouped_segments) > 1:
        return {
            current_values[0]: marks[current_values[0]],
            current_values[1]: marks[current_values[1]],
        }
    if isinstance(current_values, list) and len(current_values) == 2:
        return {current_values[0]: marks[current_values[0]]}
    return {current_values: marks[current_values]}


def _apply_two_mark_style(filtered_marks):
    sorted_keys = sorted(filtered_marks.keys())
    if len(sorted_keys) == 2:
        filtered_marks[sorted_keys[0]]["style"] = {"margin-top": "0px"}
        filtered_marks[sorted_keys[1]]["style"] = {"margin-top": "-35px"}


def _resolve_slider_marks_result(
    segments,
    group_value,
    current_values,
    max_marks,
    initial_range,
    reset_view,
):
    if len(segments) == 0:
        safe_marks = {0: {"label": "No data", "style": {"margin-top": "0px"}}}
        return safe_marks, 0, [0, 0]

    grouped_segments = _group_slider_segments(segments, group_value)
    marks = _build_slider_marks(grouped_segments)
    max_index = len(grouped_segments) - 1

    if max_index <= max_marks:
        _apply_alternating_mark_styles(marks)

    if reset_view:
        if max_index > max_marks:
            filtered_marks = {
                initial_range[0]: marks[initial_range[0]],
                initial_range[1]: marks[initial_range[1]],
            }
            _apply_two_mark_style(filtered_marks)
            return filtered_marks, max_index, initial_range
        return marks, max_index, initial_range

    filtered_marks = _select_current_range_marks(marks, current_values, grouped_segments)

    if max_index > max_marks:
        _apply_two_mark_style(filtered_marks)
        return filtered_marks, max_index, current_values

    return marks, max_index, current_values


SLIDER_MARKS_CONFIG = {
    "time": {
        "max_marks": 15,
        "reset_triggers": {"lower-filter", "duration-filter", "button-paraver-mask"},
    },
    "value": {
        "reset_triggers": {"input-number", "lower-filter", "duration-filter", "time-slider", "button-paraver-mask"},
    },
}


@app.callback(
    [
        Output("time-slider", "marks"),
        Output("time-slider", "max"),
        Output("time-slider", "value"),
    ],
    [
        Input("time-slider", "value"),
        Input("lower-filter", "value"),
        Input("duration-filter", "value"),
        Input("button-paraver-mask", "n_clicks"),
    ],
)
def update_slider_marks2(current_values, lower_filter, duration_filter, mask_button):
    global mask_button_offset
    if mask_button_offset != -1:
        use_paraver_mask = resolve_toggle_enabled(mask_button, mask_button_offset)
    else:
        use_paraver_mask = False

    # Callback to update the timestamp slider marks
    segments = _get_timestamp_segments(lower_filter, duration_filter, use_paraver_mask)

    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    reset_view = current_values is None or triggered_id in SLIDER_MARKS_CONFIG["time"]["reset_triggers"]

    max_index = max(len(segments) - 1, 0)
    initial_range = [0, max(min(max_index, 1), 1)]

    return _resolve_slider_marks_result(
        segments,
        group_value=1,
        current_values=current_values,
        max_marks=SLIDER_MARKS_CONFIG["time"]["max_marks"],
        initial_range=initial_range,
        reset_view=reset_view,
    )


@app.callback(
    [
        Output("value-slider", "marks"),
        Output("value-slider", "max"),
        Output("value-slider", "value"),
    ],
    [
        Input("input-number", "value"),
        Input("value-slider", "value"),
        Input("time-slider", "value"),
        Input("input-number", "value"),
        Input("lower-filter", "value"),
        Input("duration-filter", "value"),
        Input("button-paraver-mask", "n_clicks"),
    ],
)
def update_slider_marks(
    group_value,
    current_values,
    time_values,
    timestamps_grouper,
    lower_filter,
    duration_filter,
    mask_button,
):
    # Callback to update the timestamp slider marks
    global max_dots_auto, mask_button_offset
    start_index = time_values[0]
    end_index = time_values[1]

    max_marks = 8 if timestamps_grouper > 1 else 15
    if mask_button_offset != -1:
        use_paraver_mask = resolve_toggle_enabled(mask_button, mask_button_offset)
    else:
        use_paraver_mask = False

    selected_segments = _get_timestamp_segments(
        lower_filter,
        duration_filter,
        use_paraver_mask,
        start_index,
        end_index,
    )

    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    reset_view = current_values is None or triggered_id in SLIDER_MARKS_CONFIG["value"]["reset_triggers"]

    if selected_segments:
        grouped_count = len(_group_slider_segments(selected_segments, group_value))
        initial_range = [0, max(grouped_count - 1, 0)]
    else:
        initial_range = [0, 0]

    return _resolve_slider_marks_result(
        selected_segments,
        group_value=group_value,
        current_values=current_values,
        max_marks=max_marks,
        initial_range=initial_range,
        reset_view=reset_view,
    )


# Callback to extract the graphs dimensions directly from the component
app.clientside_callback(
    """
    function(relayoutData) {
        // If no relayoutData, don't update (you can adjust this logic based on when you want to trigger this callback)
        if (!relayoutData) {
            return window.dash_clientside.no_update;
        }

        function getPlotSize(attempts) {
            const graphDiv = document.getElementById('graphs');
            if (!graphDiv) return null;

            const plotRect = graphDiv.querySelector('rect.nsewdrag[data-subplot="xy"]');
            if (plotRect) {
                const width = parseFloat(plotRect.getAttribute('width'));
                const height = parseFloat(plotRect.getAttribute('height'));
                return {width, height};
            } else {
                // Retry logic with delay
                if (attempts > 0) {
                    return new Promise(resolve => {
                        setTimeout(() => {
                            resolve(getPlotSize(attempts - 1));
                        }, 100);  // each retry is delayed by 200ms
                    });
                } else {
                    return null;
                }
            }
        }

        // Introduce an initial delay before making the first size query
        return new Promise(resolve => {
            setTimeout(() => {
                resolve(Promise.resolve(getPlotSize(5)).then(size => {
                    if (size) {
                        return `Plot area width: ${size.width}px\\nPlot area height: ${size.height}px`;
                    } else {
                        return 'Plot area not found after multiple attempts.';
                    }
                }));
            }, 100);  // initial delay of 300ms before starting the measurement process
        });
    }
    """,
    Output("graph-size-data", "children"),
    Input("graph-size-update", "children"),
)


def run_server() -> None:
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    logging.getLogger("dash.dash").setLevel(logging.ERROR)

    # Force the host to a loopback address instead of letting Dash/Flask resolve the local hostname, which seems to
    # cause issues in some distributions.
    host = "127.0.0.1"
    print(f"Starting Dash app on http://{host}:{SELECTED_PORT}/")

    from werkzeug.middleware.profiler import ProfilerMiddleware

    # Enable profiling when either PROFILE env var is set.
    # Use `profile_dir` so ProfilerMiddleware writes per-request .prof files
    # instead of printing results for every request
    profiler_env = os.getenv("PROFILE")
    if profiler_env:
        profile_dir = "profiles"
        try:
            os.makedirs(profile_dir, exist_ok=True)
        except Exception:
            pass
        app.server.wsgi_app = ProfilerMiddleware(
            app.server.wsgi_app,
            profile_dir=profile_dir,
            sort_by=("cumtime",),
            restrictions=[50],
        )

    # use run_server for Dash apps (wrapper around Flask.run)
    app.run(debug=False, port=SELECTED_PORT, host=host)


if __name__ == "__main__":
    run_server()
