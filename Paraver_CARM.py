#!/usr/bin/env python3

import os
import datetime
import math
import copy
import argparse
import subprocess
import datetime
import sys
import re
import logging

#Third Party Libraries
#Run: pip install dash dash-bootstrap-components dash-daq numpy pandas plotly 
#To get all of the Libraries in case requirements.txt method fails
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.exceptions import PreventUpdate
from dash import Input, Output, State, html, ALL, dcc, no_update, callback_context

#Local Python Scripts
import GUI_utils as ut

VERSION = "1.0.0"

script_dir = os.path.dirname(os.path.abspath(__file__))
carm_pathway = os.path.join(script_dir, 'carm_results', 'roofline')

#Global Variables
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

#Define the start and end colors for the age coloring
start_color = (135, 206, 250)  #Light Blue
end_color = (0, 0, 139)  #Dark Blue


#CONSTANTS
scaling_factors = {
    'seconds': 1000000,
    'milliseconds': 1000,
    'microseconds': 1,
    'nanoseconds': 0.001
}

prev_range = [0, 1] 


intel_performance_counters = {
        'Intel_FP_Scalar_DP': 1,
        'Intel_FP_Scalar_SP': 1,
        'Intel_FP_SSE_DP': 2,
        'Intel_FP_SSE_SP': 4,
        'Intel_FP_AVX2_DP': 4,
        'Intel_FP_AVX2_SP': 8,
        'Intel_FP_AVX512_DP': 8,
        'Intel_FP_AVX512_SP': 16,
        'Intel_Loads': 1,
        'Intel_Stores': 1,
        'Intel_Loads_Stores': 1
}

intel_performance_counters_mapping = {
        'Intel_FP_Scalar_DP': 'FP_ARITH_INST_RETIRED:SCALAR_DOUBLE',
        'Intel_FP_Scalar_SP': 'FP_ARITH_INST_RETIRED:SCALAR_SINGLE',
        'Intel_FP_SSE_DP': 'FP_ARITH_INST_RETIRED:128B_PACKED_DOUBLE',
        'Intel_FP_SSE_SP': 'FP_ARITH_INST_RETIRED:128B_PACKED_SINGLE',
        'Intel_FP_AVX2_DP': 'FP_ARITH_INST_RETIRED:256B_PACKED_DOUBLE',
        'Intel_FP_AVX2_SP': 'FP_ARITH_INST_RETIRED:256B_PACKED_SINGLE',
        'Intel_FP_AVX512_DP': 'FP_ARITH_INST_RETIRED:512B_PACKED_DOUBLE',
        'Intel_FP_AVX512_SP': 'FP_ARITH_INST_RETIRED:512B_PACKED_SINGLE',
        'Intel_Loads': 'MEM_INST_RETIRED:ALL_LOADS',
        'Intel_Stores': 'MEM_INST_RETIRED:ALL_STORES',
        'Intel_Loads_Stores': 'MEM_INST_RETIRED:ALL'
}

intel_configs = [
    os.path.join(script_dir, 'paraver_carm_configs', 'Intel', 'Intel_FP_Scalar_DP.cfg'),
    os.path.join(script_dir, 'paraver_carm_configs', 'Intel', 'Intel_FP_SSE_DP.cfg'),
    os.path.join(script_dir, 'paraver_carm_configs', 'Intel', 'Intel_FP_AVX2_DP.cfg'),
    os.path.join(script_dir, 'paraver_carm_configs', 'Intel', 'Intel_FP_AVX512_DP.cfg'),
    os.path.join(script_dir, 'paraver_carm_configs', 'Intel', 'Intel_Loads.cfg'),
    os.path.join(script_dir, 'paraver_carm_configs', 'Intel', 'Intel_Stores.cfg')
]

amd_performance_counters = {
    'retired_sse_avx_operations:dp_mult_add_flops': 1,
    'retired_sse_avx_operations:dp_add_sub_flops': 1,
    'retired_sse_avx_operations:dp_mult_flops': 1,
    'retired_sse_avx_operations:dp_div_flops': 1,
    'retired_sse_avx_operations:sp_mult_add_flops': 1,
    'retired_sse_avx_operations:sp_add_sub_flops': 1,
    'retired_sse_avx_operations:sp_mult_flops': 1,
    'retired_sse_avx_operations:sp_div_flops': 1,
    'ls_dispatch:ld_dispatch': 1,
    'ls_dispatch:store_dispatch': 1
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


#Extract counter data
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("-v", "--version", action="store_true")
pre_args, remaining_args = pre_parser.parse_known_args()

if pre_args.version:
    print(f"Paraver_CARM version {VERSION}")
    sys.exit(0)
parser = argparse.ArgumentParser(description="Paraver CARM Dash App")

parser.add_argument("--min_dur", type=float, default=1, help="Minimum duration filter")
parser.add_argument("--color_csv", action='store_true', help="Use color CSV (.legend.csv) corresponding to the mask CSV")
parser.add_argument("--mask_csv", action='store_true', help="Use mask CSV")
parser.add_argument("-ac", action='store_true', help="Optional flag for accumulate values mode")
parser.add_argument("--csv", type=str, required=True, help="Path to the mask CSV")
parser.add_argument("trace_path", type=str, help="Path to the .prv file")

args = parser.parse_args()

min_dur = args.min_dur
use_paraver_coloring = args.color_csv
use_mask_csv = args.mask_csv
ac_mode = args.ac
mask_csv_path = args.csv
path = args.trace_path

data_source_directory = os.path.dirname(path)
color_csv_path = ""

if mask_csv_path != "":
    if mask_csv_path.endswith('.csv'):
        color_csv_path = mask_csv_path.replace(".csv", ".legend.csv")
        sync_csv_path = mask_csv_path.replace(".csv", ".paraver_sync.csv")
        if not os.path.isfile(sync_csv_path):
            no_sync = False

if color_csv_path != "":
    if not color_csv_path.endswith('.legend.csv'):
        print(f"Error: Expected a legend file ending with '.legend.csv', got: {color_csv_path}")
        sys.exit(1)

if mask_csv_path != "" and mask_csv_path.endswith('.csv'):

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

    with open(mask_csv_path, 'r') as f:
        first_line = f.readline().strip()
    
    parts = first_line.split(':')
    prv_filename = os.path.basename(parts[3])
    prv_stem = os.path.splitext(prv_filename)[0]

    time_unit = parts[4] if len(parts) > 4 else "Unknown"

    if legend_filename.endswith('.legend.csv'):
        legend_stem = legend_filename.replace('.legend.csv', '')
    else:
        legend_stem = os.path.splitext(legend_filename)[0]

    if legend_stem.endswith(prv_stem):
        window_name = legend_stem[:-len(prv_stem)].rstrip('_')
    else:
        window_name = legend_stem

    color_sync_button_style = {'width': '100%'}
    timeline_warning_style = {'display': 'none'}
    timeline_warning_card_style = {'display': 'none'}
else:
    window_name = ""
    coloring_button_text = ""
    mask_button_text = ""
    ac_button_text = ""
    mask_button_offset = -1
    color_button_offset = -1
    ac_button_offset = -1
    
    color_sync_button_style = {'width': '100%', 'display': 'none'}
    timeline_warning_style = {'color': 'black',
                    'textAlign': 'center',
                    'fontSize': '16px'}
    timeline_warning_card_style={'margin': '0px auto 15px auto', 'padding': '0px', 'text-align': 'center'}

    time_unit = "Microseconds"
    use_mask_csv = False
    use_paraver_coloring = False

scaling_unit = scaling_factors.get(time_unit.lower(), 1)

if not os.path.exists(path):
    print(f"Error: The path '{path}' does not exist")
    sys.exit(1)

ok = ut.find_and_run("paramedir")
if not ok:
    print("Paramedir not found, please add the path to Paramedir in your PATH (usually found in the Paraver bin directory).")
    sys.exit(1)

if path.endswith('.prv') or path.endswith('.gz'):
    print(f"Executing Paramedir to parse the trace in {path}",flush=True)
    subprocess.run([
        "paramedir",
        path,
        *intel_configs
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print("Paramedir execution finished, calculating CARM metrics.",flush=True)

#Get CARM results
if os.path.exists(carm_pathway):
    csv_files = [f for f in os.listdir(carm_pathway) if f.endswith('_roofline.csv')]
else:
    print("ERROR: No CARM results found. Please add them to the ./carm-results/roofline folder.")
    sys.exit(1)

#Extract machine names from filenames
machine_names = [file.replace('_roofline.csv', '') for file in csv_files]

merged_df = None
missing_files = []
found_files = []
sp_counters_available = False
dp_counters_available = False

#Loop through each counter
for counter_name, value in intel_performance_counters.items():
    filename = f"{counter_name}.csv"
    if os.path.exists(filename):
        found_files.append(filename[:-4])
        df = pd.read_csv(filename, sep='\t', header=None, skiprows=1, names=['ThreadID', 'Timestamp', 'Duration', counter_name])
        if merged_df is None:
            merged_df = df
        else:
            #Merge with the existing DataFrame on ThreadID, Timestamp, and Duration
            merged_df = pd.merge(merged_df, df, on=['ThreadID', 'Timestamp', 'Duration'], how='outer')
    else:
        missing_files.append(filename[:-4])
        #If the file is missing, create a DataFrame with zeros for this counter
        if merged_df is None:
            merged_df = pd.DataFrame(columns=['ThreadID', 'Timestamp', 'Duration', counter_name])
            merged_df[counter_name] = 0
        else:
            merged_df[counter_name] = 0
    
if "Intel_Loads" not in missing_files and "Intel_Stores" not in missing_files and "Intel_Loads_Stores" in missing_files:
    missing_files.remove("Intel_Loads_Stores")

if any("SP" in s for s in found_files):
    sp_counters_available = True
if any("DP" in s for s in found_files):
    dp_counters_available = True


missing_msg = (
    "\nAdd these counters to your XML file to monitor all possible events:\n\n  "
    + "\n  ".join(
        [
            f"{f.replace('_', ' ')} -> {intel_performance_counters_mapping.get(f, 'No mapping found')}"
            for f in missing_files
        ]
    )
)

is_modal_open = len(missing_files) > 0

ordered_df = merged_df.sort_values(by='Timestamp', ascending=True)
biggest_timestamp = ordered_df["Timestamp"].max()

total_time = (biggest_timestamp - ordered_df["Timestamp"].min()) * scaling_unit

total_threads = ordered_df["ThreadID"].nunique()
unique_threadIDs = ordered_df["ThreadID"].unique().tolist()
unique_threadIDs_checkbox = [{'label': thread_id, 'value': thread_id} for thread_id in unique_threadIDs]


filename_with_ext = os.path.basename(path)
appname = os.path.splitext(filename_with_ext)[0]

if ordered_df is not None:
    
    #Calculate totals for each counter column
    for column in ordered_df.columns:
        #Exclude non-counter columns
        if column not in ['ThreadID', 'Timestamp', 'Duration']: 
            totals[column] = ordered_df[column].sum()
    
    for counter, total in totals.items():
        #Check if the counter name contains "FP"
        if "FP" in counter:
            total_FP_inst += total
            total_FP_ops += total*intel_performance_counters[counter]
        else:
            total_mem_inst += total

else:
    print("No data to calculate totals.")

#Calculate approximate size of memory instructions based on the FP instructions present
bytes_modifier = 4*(totals["Intel_FP_Scalar_SP"]/total_FP_inst) + 8*(totals["Intel_FP_Scalar_DP"]/total_FP_inst) + 16*((totals["Intel_FP_SSE_SP"]+totals["Intel_FP_SSE_DP"])/total_FP_inst) + 32*((totals["Intel_FP_AVX2_SP"]+totals["Intel_FP_AVX2_DP"])/total_FP_inst) + 64*((totals["Intel_FP_AVX512_SP"]+totals["Intel_FP_AVX512_DP"])/total_FP_inst)
#Calculate totals for the trace
total_ai = total_FP_ops / (total_mem_inst*bytes_modifier)
total_GFLOPS = total_FP_ops / (total_time * 1e3)

if color_csv_path != "":
    legend = []
    with open(color_csv_path, 'r') as f:
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

                legend.append({
                    'value_start': value_start,
                    'value_end': value_end,
                    'value_label': label,
                    'R': r,
                    'G': g,
                    'B': b
                })
            else:
                print(f"Skipping malformed line: {stripped}")
            
            if use_paraver_coloring:
                coloring_button_text = "Use CARM GUI Colors"
                color_button_offset = 0
            else:
                coloring_button_text = "Use Paraver Timeline Colors"
                color_button_offset = 1

    legend_df = pd.DataFrame(legend)
    legend_df['value_start'] = legend_df['value_start'].astype(float)
    legend_df['value_end'] = legend_df['value_end'].astype(float)

    with open(mask_csv_path, 'r') as f:
        lines = f.readlines()

        data_lines = [line for line in lines if not line.startswith('#')]

        trace_df = pd.DataFrame([
            line.strip().split('\t') for line in data_lines
        ], columns=['ThreadID', 'Timestamp', 'Duration', 'LegendValue'])

        trace_df['Timestamp'] = trace_df['Timestamp'].astype(float)
        trace_df['LegendValue'] = trace_df['LegendValue'].astype(float)

    trace_df = trace_df.sort_values('LegendValue').reset_index(drop=True)
    legend_df = legend_df.sort_values('value_start').reset_index(drop=True)

    color_df = pd.merge_asof(
        trace_df,
        legend_df,
        left_on='LegendValue',
        right_on='value_start',
        direction='backward'
    )

    in_range_mask = (color_df['LegendValue'] >= color_df['value_start']) & (color_df['LegendValue'] <= color_df['value_end'])
    color_df = color_df[in_range_mask].copy()

    color_df = color_df[['ThreadID', 'Timestamp', 'R', 'G', 'B', 'LegendValue', 'value_label']]

    nonzero_colors = color_df[(color_df['R'] > 0) | (color_df['G'] > 0) | (color_df['B'] > 0)]


#Calculate metrics for each trace timestamp
no_flops = 0
match = 0
columns_to_check = ['Intel_FP_Scalar_SP', 'Intel_FP_Scalar_DP', 'Intel_FP_SSE_SP', 'Intel_FP_SSE_DP', 'Intel_FP_AVX2_SP', 'Intel_FP_AVX2_DP', 'Intel_FP_AVX512_SP', 'Intel_FP_AVX512_DP']
ordered_df = ordered_df.fillna(0)
for index, row in ordered_df.iterrows():
    duration = row["Duration"]*scaling_unit
    timestamp = row["Timestamp"]
    if all(pd.isnull(row[col]) or row[col] == 0 for col in columns_to_check):
        no_flops +=1
        full_base_statistics["ThreadID"].append(row["ThreadID"])
        full_base_statistics["Timestamp"].append(timestamp)
        full_base_statistics["Duration"].append(duration)
        full_base_statistics["GFLOPS"].append(0)
        full_base_statistics["Arithmetic_Intensity"].append(0)
        full_base_statistics['Paraver_Label'].append("")
        continue

    
    fp_inst = (
        row["Intel_FP_Scalar_SP"]
        + row["Intel_FP_Scalar_DP"]
        + row["Intel_FP_SSE_SP"]
        + row["Intel_FP_SSE_DP"]
        + row["Intel_FP_AVX2_SP"]
        + row["Intel_FP_AVX2_DP"]
        + row["Intel_FP_AVX512_SP"]
        + row["Intel_FP_AVX512_DP"]
    )

    sp_ops = (row["Intel_FP_Scalar_SP"] * 1 + row["Intel_FP_SSE_SP"] * 4 + row["Intel_FP_AVX2_SP"] * 8 + row["Intel_FP_AVX512_SP"] * 16)
    dp_ops = (row["Intel_FP_Scalar_DP"] * 1 + row["Intel_FP_SSE_DP"] * 2 + row["Intel_FP_AVX2_DP"] * 4 + row["Intel_FP_AVX512_DP"] * 8)
    fp_ops = sp_ops + dp_ops

    mem_ops = row["Intel_Loads"] + row["Intel_Stores"]
    #Calculate approximate size of memory instructions based on the FP instructions present
    bytes_modifier = 4*(row["Intel_FP_Scalar_SP"]/fp_inst) + 8*(row["Intel_FP_Scalar_DP"]/fp_inst) + 16*((row["Intel_FP_SSE_SP"]+row["Intel_FP_SSE_DP"])/fp_inst) + 32*((row["Intel_FP_AVX2_SP"]+row["Intel_FP_AVX2_DP"])/fp_inst) + 64*((row["Intel_FP_AVX512_SP"]+row["Intel_FP_AVX512_DP"])/fp_inst)
    memory_bytes = mem_ops * bytes_modifier

    #Calculate General Statistics
    duration_percent = (duration / total_time) * 100

    if pd.isna(duration) or duration <= 0:
        gflops = 0.0
    else:
        gflops = fp_ops / (duration * 1e3)
    #gflops = fp_ops / (duration * 1e3) if duration > 0 else 0
    fP_percent = (fp_ops / total_FP_ops)

    bandwidth = memory_bytes / (duration * 1e3) if duration > 0 else 0
    memory_percent = mem_ops / total_mem_inst


    if mem_ops > 0:
        load_percentage = ut.custom_round((row["Intel_Loads"] / mem_ops) * 100, 1)
        if load_percentage < 0.1:
            load_percentage = 0.1
    else:
        load_percentage = 0

    dp_percentage = ut.custom_round((dp_ops / fp_ops)*100, 1)
    if dp_percentage < 0.1:
        dp_percentage = 0.1

    arithmethic_intensity = fp_ops / memory_bytes

    base_statistics["ThreadID"].append(row["ThreadID"])
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
        color_match = color_df[(color_df['Timestamp'] == timestamp) & (color_df['ThreadID'].astype(str) == row["ThreadID"])]
        if not color_match.empty:
            match+=1
            base_statistics["Paraver_Value"].append(color_match['LegendValue'].values[0])
            base_statistics["Paraver_Label"].append(color_match['value_label'].values[0])
            base_statistics["R"].append(int(color_match['R'].values[0]))
            base_statistics["G"].append(int(color_match['G'].values[0]))
            base_statistics["B"].append(int(color_match['B'].values[0]))

            intel_statistics2["Paraver_Label"].append(color_match['value_label'].values[0])
            full_base_statistics['Paraver_Label'].append(color_match['value_label'].values[0])
        else:
            base_statistics["Paraver_Value"].append("")
            base_statistics["Paraver_Label"].append("No Label")
            base_statistics["R"].append(0)
            base_statistics["G"].append(0)
            base_statistics["B"].append(0)

            intel_statistics2["Paraver_Label"].append("")
            full_base_statistics['Paraver_Label'].append("")
    else:
        base_statistics["Paraver_Value"].append("")
        base_statistics["Paraver_Label"].append("No Label")
        base_statistics["R"].append(0)
        base_statistics["G"].append(0)
        base_statistics["B"].append(0)

        intel_statistics2["Paraver_Label"].append("")
        full_base_statistics['Paraver_Label'].append("")

    full_base_statistics["ThreadID"].append(row["ThreadID"])
    full_base_statistics["Timestamp"].append(timestamp)
    full_base_statistics["Duration"].append(duration)
    full_base_statistics["GFLOPS"].append(float(gflops))
    full_base_statistics["Arithmetic_Intensity"].append(float(arithmethic_intensity))

    intel_statistics2["ThreadID"].append(row["ThreadID"])
    intel_statistics2["Timestamp"].append(timestamp)
    intel_statistics2["Intel_FP_Scalar_SP"].append(row["Intel_FP_Scalar_SP"])
    intel_statistics2["Intel_FP_Scalar_DP"].append(row["Intel_FP_Scalar_DP"])
    intel_statistics2["Intel_FP_SSE_SP"].append(row["Intel_FP_SSE_SP"]*4)
    intel_statistics2["Intel_FP_SSE_DP"].append(row["Intel_FP_SSE_DP"]*2)
    intel_statistics2["Intel_FP_AVX2_SP"].append(row["Intel_FP_AVX2_SP"]*8)
    intel_statistics2["Intel_FP_AVX2_DP"].append(row["Intel_FP_AVX2_DP"]*4)
    intel_statistics2["Intel_FP_AVX512_SP"].append(row["Intel_FP_AVX512_SP"]*16)
    intel_statistics2["Intel_FP_AVX512_DP"].append(row["Intel_FP_AVX512_DP"]*8)
    intel_statistics2["Intel_FP_SP"].append(sp_ops)
    intel_statistics2["Intel_FP_DP"].append(dp_ops)
    intel_statistics2["Intel_FP_Total"].append(fp_ops)
    intel_statistics2["Intel_FP_DP_Percent"].append(dp_percentage)
    intel_statistics2["Intel_Load"].append(row["Intel_Loads"])
    intel_statistics2["Intel_Store"].append(row["Intel_Stores"])
    intel_statistics2["Intel_Load_Percent"].append(load_percentage)
    
base_statistics_df = pd.DataFrame(base_statistics)
full_base_statistics_df = pd.DataFrame(full_base_statistics)
intel_statistics_df2 = pd.DataFrame(intel_statistics2)
n_segments = base_statistics_df.shape[0]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

#Sidebar Layour Definition
sidebar = dbc.Offcanvas(
    html.Div([
        html.P("Data Filtering", className="mb-2", 
            style={
                'color': 'white',
                'textAlign': 'center',
                'fontSize': '20px'
            }),
        dbc.Card(
            dbc.CardBody([
                html.P("Filter by Vector Extension:", className="mb-1", style={'color': 'black'}),
                dbc.Checklist(
                    id='isa-checklist',
                    options=[
                        {'label': ' Scalar', 'value': 'Scalar'},
                        {'label': ' SSE', 'value': 'SSE'},
                        {'label': ' AVX2', 'value': 'AVX2'},
                        {'label': ' AVX512', 'value': 'AVX512'},
                    ],
                    value=['Scalar', 'SSE', 'AVX2', 'AVX512'],
                    inline=True,
                    className="mb-2",
                    style={'color': 'black'}
                ),
                dbc.Row([
                    dbc.Col([
                        html.P("Filter by Precision:", className="mb-1", style={'color': 'black'}),
                        dbc.Checklist(
                            id='precision-checklist',
                            options=[
                                {'label': ' SP', 'value': 'SP'},
                                {'label': ' DP', 'value': 'DP'}
                            ],
                            value=['SP', 'DP'],
                            inline=True,
                            className="mb-2",
                            style={'color': 'black'}
                        )
                    ], width=6),
                    dbc.Col([
                        html.P("Toggle Total:", 
                        className="mb-1", 
                        style={'color': 'black', 'margin-right': '10px', 'display': 'inline-block'}),
                        dbc.Checklist(
                            id='total-checklist',
                            options=[{'label': '', 'value': 'Total'}],
                            inline=True,
                            className="mb-1",
                            style={'color': 'black', 'display': 'inline-block'}
                        ),       
                    ], width=6),
                ]),
                html.P("Filter by Thread ID:", className="mb-1", style={'color': 'black'}),
                dbc.Checklist(
                    id='thread-checklist',
                    options=unique_threadIDs_checkbox,
                    value=unique_threadIDs,
                    inline=True,
                    className="mb-2",
                    style={'color': 'black'}
                ),
                html.Div([
                    html.P("Cut values lower than:", className="mb-1", style={'color': 'black', 'margin-right': '10px', 'flex': 'none'}),
                    dcc.Input(
                        id='lower-filter',
                        type='number',
                        value=0.0000001,
                        min=0,
                        placeholder="Enter value",
                        debounce=True,
                        style={'flex': '1', 'width': '100%'}
                    ),
                ], style={'display': 'flex','alignItems': 'center'}),
                html.Div([
                    html.P("Minimum Duration (ns):", className="mb-1", style={'color': 'black', 'margin-right': '10px', 'flex': 'none'}),
                    dcc.Input(
                        id='duration-filter',
                        type='number',
                        value=min_dur,
                        min=0,
                        placeholder="Enter value",
                        debounce=True,
                        style={'flex': '1', 'width': '100%'}
                    ),
                ], style={'display': 'flex','alignItems': 'center'})
            ]),
        
            style={'backgroundColor': 'white'},
            className="mb-2"
        ),
        html.P("Graph Customization", className="mb-2", 
            style={
                'color': 'white',
                'textAlign': 'center',
                'fontSize': '20px'
        }),
        dbc.Card(
            dbc.CardBody([
                html.P("Color timestamps based on:", className="mb-1", style={'color': 'black'}),
                dbc.RadioItems(
                    id='color-radio',
                    options=[
                        {'label': ' Youngest', 'value': 'Youngest'},
                        {'label': ' Oldest', 'value': 'Oldest'},
                        {'label': ' Duration', 'value': 'Duration'},
                        {'label': ' Thread ID', 'value': 'Thread ID'},
                        {'label': ' Precision', 'value': 'Precision'},
                        {'label': ' LD/ST Percentage', 'value': 'LD/ST Percentage'},
                        {'label': ' Vector ISA', 'value': 'ISA'}, 
                    ],
                    inline=True,
                    value='Youngest',
                    className="mb-2",
                    style={'color': 'black'}
                ),
                html.Div([
                    dbc.Label("Use Exponent Notation", html_for="exponent-switch", 
                            style={"marginRight": "40px"}),
                    dbc.Switch(
                        id="exponent-switch",
                        label="",
                        value=True
                    )
                    ],
                    style={"display": "flex", "alignItems": "center"}
                ),
                html.Div([
                    dbc.Label("Show Lines Legend", html_for="line-legend-switch", 
                            style={"marginRight": "70px"}),
                    dbc.Switch(
                        id="line-legend-switch",
                        label="",
                        value=True
                    )
                    ],
                    style={"display": "flex", "alignItems": "center"}
                ),
            ]),
            style={'backgroundColor': 'white'},
            className="mb-2"
        ),
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Row(
            [

                    html.P("Lines Width:", className="mb-1", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                    html.P("Dots Size:", className="mb-1", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
            ]),
            dbc.Row(
            [

                    dcc.Input(id='line-size', type='number', min=1, className="mb-2", max=100,step=1, value=3, style={'flex': '1', 'margin-right': '50px', 'width': '70px'}),
                     
                    dcc.Input(id='dot-size', type='number', min=1, className="mb-2", max=100,step=1, value=10, style={'flex': '1', 'margin-right': '50px'}),
            ]),

                dbc.Row(
            [

                    html.P("Title Font:", className="mb-1", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                    html.P("Axis Font:", className="mb-1", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
            ]),
            dbc.Row(
            [

                    dcc.Input(id='title-size', type='number', className="mb-2", min=1, max=100,step=1, value=20, style={'flex': '1', 'margin-right': '50px'}),
                     
                    dcc.Input(id='axis-size', type='number', className="mb-2", min=1, max=100,step=1, value=20, style={'flex': '1', 'margin-right': '50px'}),
            ]),
                dbc.Row(
            [

                    html.P("Legend Font:", className="mb-1", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                    html.P("Ticks Font:", className="mb-1", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
            ]),
            dbc.Row(
            [

                    dcc.Input(id='legend-size', type='number', className="mb-2", min=1, max=100,step=1, value=14, style={'flex': '1', 'margin-right': '50px'}),
                     
                    dcc.Input(id='tick-size', type='number', className="mb-2", min=1, max=100,step=1, value=18, style={'flex': '1', 'margin-right': '50px'}),
            ]),
                dbc.Row(
            [

                    html.P("Annotations:", className="mb-1", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                    html.P("Tooltip Font:", className="mb-1", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
            ]),
            dbc.Row(
            [

                    dcc.Input(id='annotation-size', type='number', className="mb-2", min=1, max=100,step=1, value=10, style={'flex': '1', 'margin-right': '50px'}),
                     
                    dcc.Input(id='tooltip-size', type='number', className="mb-2", min=1, max=100,step=1, value=14, style={'flex': '1', 'margin-right': '50px'}),
            ]),
                        
                        ], title="Change Font/Line Sizes",
                    ),

                    ],id='font-accordion', start_collapsed=True, always_open=True, flush=True, style={'backgroundColor': '#1a1a1a'}, className="mb-2"
                ),

        dbc.Button("Edit Graph Text", id="button-CARM", className="mb-2", style={'width': '100%'}, n_clicks=1),
        html.P("Notations Configuration", className="mb-2", 
            style={
                'color': 'white',
                'textAlign': 'center',
                'fontSize': '20px'
            }),
        html.Div([
            dbc.Accordion([], id='annotation-accordion', start_collapsed=True, always_open=True, flush=True, style={'backgroundColor': '#1a1a1a'})
        ], id='angle-inputs-container', style={'marginBottom': '15px'}),
        dbc.Button("Create Annotation", id="create-annotation-button", className="mb-2", style={'width': '100%'}),
        dbc.Button("Disable Annotations", id="disable-annotation-button", className="mb-2", style={'width': '100%'}),
    ], style={'backgroundColor': '#1a1a1a'}),
    id="offcanvas",
    title=html.H5("Graph Options", style={'color': 'white', 'fontsize': '30px'}),
    is_open=False,
    placement= "end",
    style={'backgroundColor': '#1a1a1a'},
)

sidebar2 = dbc.Offcanvas(
    children=[
        html.P("Paraver -> CARM", className="mb-2", 
                style={
                    'color': 'white',
                    'textAlign': 'center',
                    'fontSize': '20px'
                }),
        dbc.Card(
                            dbc.CardBody([
        html.P("To use the Paraver -> CARM features please launch CARM from a Paraver timeline window", className="mb-0", 
                style=timeline_warning_style),
        ]),
                            style=timeline_warning_card_style
                        ),   

        dbc.Button(coloring_button_text, id="button-paraver-colors", className="mb-2", style=color_sync_button_style, n_clicks=0),
        dbc.Button(mask_button_text, id="button-paraver-mask", className="mb-2", style=color_sync_button_style, n_clicks=0),
        dbc.Button(ac_button_text, id="button-paraver-accumulate", className="mb-4", style=color_sync_button_style, n_clicks=0),
        dbc.Button("Re-Sync Timeline With Paraver", id="button-paraver-sync", className="mb-5", style=color_sync_button_style, n_clicks=0),

        html.P("CARM -> Paraver", className="mb-2", 
                style={
                    'color': 'white',
                    'textAlign': 'center',
                    'fontSize': '20px'
                }),
        dbc.Button("Send Timestamps Roof Labels", id="button-roof-labels", className="mb-2", style={'width': '100%'}, n_clicks=0),
        dbc.Button("Send Timestamps LD/ST Percentage Colors", id="button-carm-ldst-colors", className="mb-2", style={'width': '100%'}, n_clicks=0),
        dbc.Button("Send Timestamps SP/DP Percentage Colors", id="button-carm-spdp-colors", className="mb-2", style={'width': '100%'}, n_clicks=0),
    ],
    id="offcanvas2",
    title=html.H5("Paraver Functions", style={'color': 'white', 'fontsize': '30px'}),
    is_open=False,
    placement= "start",
    style={'backgroundColor': '#1a1a1a'})

#Main app layout
app.layout = dbc.Container([
    dcc.Interval(
        id='interval-component',
        interval=1000,
        n_intervals=0,
        disabled=True
    ),
    dcc.Interval(
        id='paraver-sync-check',
        interval=1000,
        n_intervals=0,
        disabled=True
    ),
    dcc.Store(id='paraver-sync-timestamps', data=[]),
    dcc.Download(id="download-csv"),
    dbc.Row([
        dbc.Col(
            dbc.Button(
                html.Img(src="/assets/bsc.svg", height="30px"),
                id="open-offcanvas2",
                n_clicks=0,
                className="btn-sm",
                style={'border': 'none', 'background': 'transparent', 'padding': '0', 'margin': '0'}
            ),
            width="auto",
            style={'padding-right': '5px', 'padding-left': '5px'}
        ),
        dbc.Col(
            dcc.Dropdown(
                id='filename',
                options=[{'label': machine_name, 'value': os.path.join(carm_pathway, file)} for machine_name, file in zip(machine_names, csv_files)],
                multi=False,
                placeholder="Select Machine Results..."
            ),
            width=True
        ),
        dbc.Col(
            dbc.Button(
                html.Img(src="/assets/CARM_icon3.svg", height="30px"),
                id="open-offcanvas",
                n_clicks=0,
                className="btn-sm",
                style={'border': 'none', 'background': 'transparent', 'padding': '0', 'margin': '0'}
            ),
            width="auto",
            style={'padding-right': '5px', 'padding-left': '5px'}
        ),
    ],
    align="center", 
    style={'margin-top': '1px'}
    ),
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div(id='additional-dropdowns', style={'margin-top': '10px'}),
                html.Div(id='additional-dropdowns2'),
                ])
            ]),
            
        dbc.Row([
            dbc.Col([
                html.Div(
                    html.Div([
                        dbc.Card(
                            dbc.CardBody([
                                html.P(f"Execution Timestamp Range Selection ({time_unit})", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin-right': '0px', 'align-self': 'center', 'margin-top': '-6px'}),
                                html.Div([
                                    dcc.RangeSlider(
                                        id='time-slider',
                                        min=0,
                                        max=None,
                                        step=1,
                                        value=[0, 1],
                                        marks=None,
                                        tooltip=None,
                                        allowCross=False,
                                        pushable=2
                                    ),
                                ], style={'display': 'inline-block', 'width': '770px', 'margin': '0px 20px auto 15px'}),
                            ]),
                            style={'height': '100px', 'margin': '0px auto 10px auto', 'padding': '0px', 'text-align': 'center'}
                        ),              
                        dbc.Card(
                            dbc.CardBody([
                                html.Div([
                                    html.P(f"Execution Timestamps To Plot ({time_unit})", style={
                                        'textAlign': 'center',
                                        'fontWeight': 'bold',
                                        'margin': '0 210px',
                                        'margin-top': '-6px'
                                    }),
                                    html.P("Grouping", style={
                                        'textAlign': 'center',
                                        'fontWeight': 'bold',
                                        'margin': '0 40px'
                                    }),
                                    html.P("Average", style={
                                        'textAlign': 'center',
                                        'fontWeight': 'bold',
                                        'margin': '0 5px'
                                    }),
                                ], style={
                                    'display': 'flex',
                                    'justify-content': 'center',
                                    'align-items': 'center',
                                    'margin-bottom': '10px'
                                }),
                                dcc.Store(id='data-points-store'),
                                    html.Div([
                                        html.Button(
                                            "▶️",
                                            id='play-pause-button',
                                            n_clicks=0,
                                            style={
                                                'border': 'none',
                                                'outline': 'none',
                                                'fontSize': '24px',
                                                'backgroundColor': 'transparent',
                                                'cursor': 'pointer',
                                                'margin-top': '-22px',
                                                'margin-left': '-30px',
                                                'margin-right': '10px'
                                            }),
                                        html.Div([
                                            dcc.RangeSlider(
                                                id='value-slider',
                                                min=0,
                                                max=None,
                                                step=1,
                                                value=[0, 1],
                                                marks=None,
                                                tooltip=None,
                                                allowCross=False,
                                            ),
                                        ], style={'display': 'inline-block', 'width': '710px', 'margin': '0px'}),
                                        html.Div([
                                            html.Button('⬇️', id='button-divide', n_clicks=0),
                                            dcc.Input(id='input-number', type='number', min=1, max=n_segments,step=1, value=1, style={'width': '70px', 'margin': '0 0px'}),
                                            html.Button('⬆️', id='button-multiply', n_clicks=0),
                                        ], style={'display': 'inline-block', 'margin-top': '-15px', 'margin-left': '15px'}),
                                        dbc.Checkbox(
                                            id='average-checkbox',
                                            label='',
                                            style={'margin-top': '-15px', 'margin-left': '40px'}
                                        ),
                                        ], style={
                                        'display': 'flex',
                                        'align-items': 'center',
                                        'justify-content': 'center'
                                    }),
                            ]),
                        style={'height': '100px', 'width': '1070px', 'margin': '0px 10px 10px 10px', 'padding': '0px', 'text-align': 'center'}
                        ),      
                    ],style={'display': 'flex', 'align-items': 'center', 'height': '100%'}),
                ),
            ]),
        ]),
        
        dbc.Row([
            dbc.Col(
                dcc.Graph(id='graphs', style={'display': 'none'}, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'CARM_Tool'},
                    'editable': False,
                    'displaylogo': False,
                    'edits': {
                        'annotationPosition': True,

                    }
                }), 
            width=11),
            ], className="g-0",
        ),
        ],
        id="slider-components",
        style={"display": "none"}
    ),
    html.Div(id='graph-size-data', style={'whiteSpace': 'pre-wrap', 'display': 'none'}),
    html.Div(id='graph-size-update', style={'whiteSpace': 'pre-wrap', 'display': 'none'}),
    dcc.Store(id='store-dimensions'),
    dcc.Store(id='graph-lines'),
    dcc.Store(id='graph-lines2'),
    dcc.Store(id='graph-values'),
    dcc.Store(id='graph-values2'),
    dcc.Store(id='graph-isa'),
    dcc.Store(id='graph-xrange'),
    dcc.Store(id='graph-yrange'),
    dcc.Store(id='change-annon'),
    dcc.Store(id='clicked-point-index', data=-1),
    dcc.Store(id="clicked-trace-index", data=-1),
    dbc.Row([
        dbc.Col([
            html.Div(
                [
                    html.Span("⬆", style={'fontSize': '24px', 'color': '#6c757d', 'marginRight': '10px'}),
                    html.Span("Select a Machine to View CARM Results", style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#6c757d'}),
                    html.Span("⬆", style={'fontSize': '24px', 'color': '#6c757d', 'marginLeft': '10px'})
                ],
                id="initial-text",
                style={'textAlign': 'center', 'marginTop': '10px'}
            ),
            html.Img(
                src="/assets/carm_bsc.svg",
                id="initial-image",
                style={'width': '99%', 'height': '90%', 'background': 'transparent', 'marginLeft': '40px'}
            ),
        ], width=10, style={'backgroundColor': '#e9ecef', 'textAlign': 'center'})
    ], id="initial-content", justify="center", style={'backgroundColor': '#e9ecef', 'textAlign': 'center'}),
    dcc.Store(id='machine-selected', data=False),
    sidebar,
    sidebar2,
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Edit Point Style", style={'text-align': 'center', 'color': 'white'}), style={'backgroundColor': '#6c757d'}),
        dbc.ModalBody([
            daq.ColorPicker(
                label=' ',
                id="dot-color-picker",
                value={"hex": "#0000FF"},  #blue
            ),
            html.Hr(),
            dbc.Col([
                html.Div([
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
                            "width": "45px"
                        },
                    ),
                    html.Label("Shape:", style={"marginRight": "10px"}),
                    dcc.Dropdown(
                        id="dot-symbol-dropdown",
                        options=[
                            {"label": "Circle", "value": "circle"},
                            {"label": "Square", "value": "square"},
                            {"label": "Diamond", "value": "diamond"},
                            {"label": "Cross", "value": "cross"},
                            {"label": "X", "value": "x"},
                            {"label": "Triangle-Up", "value": "triangle-up"},
                            {"label": "Triangle-Down", "value": "triangle-down"},
                        ],
                        value="circle",
                        style={"width": "170px"}
                    ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center"
                    }
                )
                ],
                width="auto",
            ),
            ],
            style={'backgroundColor': '#e9ecef'},
            id="modal-body",
        ),
        dbc.ModalFooter(
            [
                dbc.Button("Submit", id="dot-submit-button", className="ms-auto", n_clicks=0, style={'margin-right': 'auto'}),
                dbc.Button("Close", id="close-dot-modal", className="me-auto", n_clicks=0, style={'margin-left': 'auto'}),
            ],
            className="w-100 d-flex",
            style={'backgroundColor': '#6c757d'}
        ),
        ],
        id="point-edit-modal",
        is_open=False,
        style={"width": "auto", "centered": "true"},
    ),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Warning - Missing Counter Data", className="text-center w-100", style={'text-align': 'center', 'color': 'white'}), style={'text-align': 'center', 'backgroundColor': '#6c757d'}),
        dbc.ModalBody([
            html.Pre(missing_msg)
            ],
            style={
                'backgroundColor': '#e9ecef',
                'text-align': 'center'
            }
        ),
        dbc.ModalFooter(
            [
                dbc.Button("Close", id="close-warning-modal", className="me-auto", n_clicks=0, style={'margin-left': 'auto'}),
            ],
            className="w-100 d-flex",
            style={'backgroundColor': '#6c757d'}
        ),
        ],
        size="lg",
        id="warning-modal",
        is_open=is_modal_open
    ),
    dbc.Modal([
        dbc.ModalHeader("Create a New Annotation"),
        dbc.ModalBody([
            dbc.Label("Annotation Text"),
            dbc.Input(type="text", id="annotation-text-input", placeholder="Enter annotation text"),
        ]),
        dbc.ModalFooter(
            dbc.Button("Submit", id="submit-annotation", className="ms-auto", n_clicks=0)
        ),
        ],
        id="annotation-modal",
        is_open=False,
    ),
    dcc.Store(id='annotations-store', data=[]),
    ],
    fluid=True,
    className="p-3",
    style={'backgroundColor': '#e9ecef'}
)

#App Callbacks
@app.callback(
    Output("button-paraver-colors", "children"),
    Input("button-paraver-colors", "n_clicks"),
)
def toggle_button_label(n_clicks):
    if (n_clicks + color_button_offset) % 2 == 1:
        return "Use Paraver Timeline Colors"
    else:
        return "Use CARM GUI Colors"
    
@app.callback(
    Output("button-paraver-mask", "children"),
    Input("button-paraver-mask", "n_clicks"),
)
def toggle_button_label(n_clicks):
    if (n_clicks + mask_button_offset) % 2 == 1:
        return "Use Semantic Window"
    else:
        return "Use All Timestamps"
    
@app.callback(
    Output("button-paraver-accumulate", "children"),
    Input("button-paraver-accumulate", "n_clicks"),
)
def toggle_button_label(n_clicks):
    if (n_clicks + ac_button_offset) % 2 == 1:
        return "Plot Accumulated Values"
    else:
        return "Plot Raw Values"


@app.callback(
    Output('duration-filter', 'value'),
    Input('duration-filter', 'value'),
    prevent_initial_call=True
)
def enforce_non_null(value):
    if value is None:
        return 0
    return value

@app.callback(
    Output('time-slider', 'value', allow_duplicate=True),
    Output('value-slider', 'value', allow_duplicate=True),
    Output('paraver-sync-timestamps', 'data'),
    Input('paraver-sync-check', 'n_intervals'),
    Input('button-paraver-sync', 'n_clicks'),
    Input('lower-filter', 'value'),
    Input('duration-filter', 'value'),
    Input("button-paraver-mask", "n_clicks"),
    State('paraver-sync-timestamps', 'data'),
    State("filename", "value"),
    prevent_initial_call=True
)
def update_slider_from_csv(n_intervals, button_clicks, lower_filter, duration_filter, mask_button, current_values, selected_file):
    global sync_csv_path
    global current_file_timestamps
    if mask_button_offset == -1:
        raise PreventUpdate
    if not selected_file:
        raise PreventUpdate
    else:
        global no_sync
        global first_load
        try:
            csv_df = pd.read_csv(sync_csv_path)
            time_unit = csv_df.columns[0]
            new_timestamps = [float(csv_df.iloc[0, 0]), float(csv_df.iloc[1, 0])]
            
        except Exception as e:
            first_load +=1
            raise PreventUpdate
        
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if (new_timestamps != current_file_timestamps or trigger_id in ["button-paraver-sync"]):
            first_load +=1
            current_file_timestamps = new_timestamps
            if first_load > 1:
                try:
                    start_index = (full_base_statistics_df['Timestamp'] - new_timestamps[0]).abs().idxmin()
                    end_index = (full_base_statistics_df['Timestamp'] - new_timestamps[1]).abs().idxmin()
                    if mask_button_offset != -1:
                        if (mask_button + mask_button_offset) % 2 == 1:
                            use_paraver_mask = False
                        else:
                            use_paraver_mask = True
                    else:
                        use_paraver_mask = False

                    adjusted_start_index = ut.find_nearest_positive(full_base_statistics_df, start_index, float(lower_filter), float(duration_filter), use_paraver_mask, min_bound=0)
                    adjusted_end_index = ut.find_nearest_positive(full_base_statistics_df, end_index, float(lower_filter), float(duration_filter), use_paraver_mask, min_bound=adjusted_start_index)
                    
                    matching_start_timestamp = full_base_statistics_df.loc[adjusted_start_index, 'Timestamp']
                    matching_end_timestamp = full_base_statistics_df.loc[adjusted_end_index, 'Timestamp']

                    if use_paraver_mask:
                        filtered_base = base_statistics_df[
                            (base_statistics_df['Arithmetic_Intensity'] >= float(lower_filter)) &
                            (base_statistics_df['GFLOPS'] >= float(lower_filter)) &
                            (base_statistics_df['Duration'] >= float(duration_filter)) &
                            (base_statistics_df['Paraver_Value'].apply(ut.is_valid_paraver_value))
                        ]
                    else:
                        filtered_base = base_statistics_df[
                            (base_statistics_df['Arithmetic_Intensity'] >= float(lower_filter)) &
                            (base_statistics_df['GFLOPS'] >= float(lower_filter)) &
                            (base_statistics_df['Duration'] >= float(duration_filter))
                        ]
                    filtered_base = filtered_base.reset_index(drop=True)

                    new_start_index = filtered_base[filtered_base['Timestamp'] == matching_start_timestamp].index[0]
                    new_end_index = filtered_base[filtered_base['Timestamp'] == matching_end_timestamp].index[0]

                except Exception as e:
                    if no_sync:
                        print("Error finding indices in main_df:", e, flush=True)
                        print('Check if the "Cut values lower than" option is not too high for the current region of interest', flush=True)
                        no_sync = False
                    raise dash.exceptions.PreventUpdate

                new_slider_indices = [int(new_start_index), int(new_end_index)]
                
                if trigger_id == "button-paraver-sync":
                    print("----------------------------------------------", flush=True)
                    print("Sync Button Clicked, updating slider to timestamp range {} - {}".format(filtered_base.loc[new_start_index, 'Timestamp'], filtered_base.loc[new_end_index, 'Timestamp']))
                    
                    if adjusted_start_index != start_index:
                        print("INFO: Adjusted Start Timestamp to {} from {} to allow for CARM plotting".format(
                            filtered_base.loc[new_start_index, 'Timestamp'], (full_base_statistics_df.loc[start_index, 'Timestamp']))
                        , flush=True)
                    
                    if adjusted_end_index != end_index:
                        print("INFO: Adjusted End Timestamp to {} from {} to allow for CARM plotting".format(
                            filtered_base.loc[new_end_index, 'Timestamp'], (full_base_statistics_df.loc[end_index, 'Timestamp']))
                        , flush=True)
                    
                    print("----------------------------------------------", flush=True)
                    no_sync = True
                    return new_slider_indices, new_slider_indices, new_timestamps
                
                if new_slider_indices != current_values:
                    print("----------------------------------------------", flush=True)
                    print("Sync CSV values changed, updating slider to timestamp range {} - {}".format(filtered_base.loc[new_start_index, 'Timestamp'], filtered_base.loc[new_end_index, 'Timestamp']))
                    
                    if adjusted_start_index != start_index:
                        print("INFO: Adjusted Start Timestamp to {} from {} to allow for CARM plotting".format(
                            filtered_base.loc[new_start_index, 'Timestamp'], (full_base_statistics_df.loc[start_index, 'Timestamp']))
                        , flush=True)
                    
                    if adjusted_end_index != end_index:
                        print("INFO: Adjusted End Timestamp to {} from {} to allow for CARM plotting".format(
                            filtered_base.loc[new_end_index, 'Timestamp'], (full_base_statistics_df.loc[end_index, 'Timestamp']))
                        , flush=True)

                    print("----------------------------------------------", flush=True)
                    no_sync = True
                    return new_slider_indices, new_slider_indices, new_timestamps
            else:
                raise dash.exceptions.PreventUpdate
        else:
            raise dash.exceptions.PreventUpdate
    
@app.callback(
    Input("button-roof-labels", "n_clicks"),
    Input("graph-lines", "data"),
    prevent_initial_call=True
)
def generate_csv(n_clicks, lines):
    global full_base_statistics_df, path, time_unit
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id != "button-roof-labels":
        raise PreventUpdate
    
    df = full_base_statistics_df.copy()
    df["Roof Label"] = df.apply(lambda row: ut.label_cache_level(row, lines), axis=1)
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    metadata_line = f'#{timestamp}:CSV:RUNAPP:{path}:{time_unit}:window_in_code_mode:1:6'


    csv_df = df[["ThreadID", "Timestamp", "Duration", "Roof Label"]]
    csv_df = csv_df.sort_values(["ThreadID", "Timestamp"])

    output_dir = os.path.dirname(path)
    roof_csv_filepath = os.path.join(output_dir, "carm_roofs.csv")
    with open(roof_csv_filepath, 'w') as f:
        f.write(metadata_line + '\n')
        csv_df.to_csv(f, index=False, header=False, sep="\t")

    roof_labels_filepath = os.path.join(output_dir, "carm_roofs.legend.csv")
    labels_data = [
        [1, "L1",   0,   255,   0],    # Green
        [2, "L2",     0, 0,   255],    # Blue
        [3, "L3",     255, 165, 0],    # Orange
        [4, "DRAM",   255, 0,   0],     # Red
        [5, "No Floating Point Oprations Found",   75,   0, 130],    # Indigo
        [6, "Above L1",   255, 192, 203]     # Pink
    ]
    with open(roof_labels_filepath, 'w') as f:
        for row in labels_data:
            label_line = f'{row[0]} "{row[1]}",{row[2]},{row[3]},{row[4]}\n'
            f.write(label_line)
    print("carm_roofs.csv file written.", flush=True)

    return

@app.callback(
    Input("button-carm-ldst-colors", "n_clicks"),
    Input("button-carm-spdp-colors", "n_clicks"),
    Input(component_id='graphs', component_property='figure'),
    prevent_initial_call=True
)
def generate_color_csv(n_clicks_ldst, n_clicks_spdp, graph):
    global full_base_statistics_df, path, time_unit, intel_statistics_df2
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id not in ["button-carm-ldst-colors", "button-carm-spdp-colors"]:
        raise PreventUpdate

    df = full_base_statistics_df.copy()

    ai_mean = df["Arithmetic_Intensity"].tolist()
    n = len(ai_mean)


    if trigger_id == "button-carm-ldst-colors":
        df = df.merge(
            intel_statistics_df2[["Timestamp", "ThreadID", "Intel_Load_Percent"]],
            on=["Timestamp", "ThreadID"],
            how="left"
        )
        unique_percentages = df["Intel_Load_Percent"].dropna().unique()
        df["Intel_Load_Percent"] = df["Intel_Load_Percent"].fillna(int(0))
    elif trigger_id == "button-carm-spdp-colors":
        df = df.merge(
            intel_statistics_df2[["Timestamp", "ThreadID", "Intel_FP_DP_Percent"]],
            on=["Timestamp", "ThreadID"],
            how="left"
        )
        unique_percentages = df["Intel_FP_DP_Percent"].dropna().unique()
        df["Intel_FP_DP_Percent"] = df["Intel_FP_DP_Percent"].fillna(int(0))
    
    unique_percentages.sort()
    color_map = []

    for percentage in unique_percentages:
        if trigger_id == "button-carm-ldst-colors":
            r, g, b = ut.blend_colors(0, 0, 0, 0, 0, percentage, 0, "LD/ST Percentage", True)
            extra_string = "Loads"
            
        elif trigger_id == "button-carm-spdp-colors":
            r, g, b = ut.blend_colors(0, 0, 0, 0, percentage, 0, 0, "Precision", True)
            extra_string = "DP"

        color_map.append({
                "percentage": percentage,
                "percentage_string": f"{percentage}% {extra_string}",
                "r": r,
                "g": g,
                "b": b
            })
        
    color_map_df = pd.DataFrame(color_map)
    output_dir = os.path.dirname(path)
    roof_labels_filepath = os.path.join(output_dir, "carm_colors.legend.csv")
    ut.format_ld_st_csv(color_map_df, roof_labels_filepath)

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    metadata_line = f'#{timestamp}:CSV:RUNAPP:{path}:{time_unit}:window_in_code_mode:{color_map_df["percentage"].min()}:{color_map_df["percentage"].max()}'

    if trigger_id == "button-carm-ldst-colors":
        csv_df = df[["ThreadID", "Timestamp", "Duration", "Intel_Load_Percent"]]
    elif trigger_id == "button-carm-spdp-colors":
        csv_df = df[["ThreadID", "Timestamp", "Duration", "Intel_FP_DP_Percent"]]
    csv_df = csv_df.sort_values(["ThreadID", "Timestamp"])
    
    roof_csv_filepath = os.path.join(output_dir, "carm_colors.csv")
    with open(roof_csv_filepath, 'w') as f:
        f.write(metadata_line + '\n')
        csv_df.to_csv(f, index=False, header=False, sep="\t")

    print("carm_colors.csv file written.", flush=True)

    return

@app.callback(
    Output("slider-components", "style"),
    Output('paraver-sync-check', 'disabled'),
     Input('graph-lines', 'data'),
     State("filename", "value"),
)
def toggle_components(lines, selected_file):
    if not selected_file:
        return {'display': 'none'}, True
    else:
        return {'display': 'block'}, False

@app.callback(
    Output("warning-modal", "is_open"),
    Input("close-warning-modal", "n_clicks"),
    State("warning-modal", "is_open"),
)
def toggle_modal(close_clicks, current_state):
    if close_clicks:
        return not current_state
    return current_state

@app.callback(
    [Output("point-edit-modal", "is_open"),
     Output("clicked-trace-index", "data"),
     Output("clicked-point-index", "data")],
    [
     Input("graphs", "clickData"),
     Input("close-dot-modal", "n_clicks"),
     Input("dot-submit-button", "n_clicks")
    ],
    [State("point-edit-modal", "is_open")]
)
def open_modal_on_click(click_data, close_clicks, submit_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id in ["dot-submit-button", "close-dot-modal"]:
        return False, -1, -1

    if click_data:
        trace_idx = click_data["points"][0]["curveNumber"]
        point_idx = click_data["points"][0]["pointIndex"]
        return True, trace_idx, point_idx

    return is_open, -1, -1

@app.callback(
    [Output('graphs', 'figure', allow_duplicate=True),
     Output('point-edit-modal', 'is_open', allow_duplicate=True)],
    [Input('dot-submit-button', 'n_clicks')],
    [
        State('dot-color-picker', 'value'),
        State('dot-size-input', 'value'),
        State('dot-symbol-dropdown', 'value'),
        State('clicked-trace-index', 'data'),
        State('clicked-point-index', 'data'),
        State('graphs', 'figure'),
    ],
    prevent_initial_call=True
)
def update_point_style(n_submit,
                       chosen_color, chosen_size, chosen_symbol,
                       trace_idx, point_idx,
                       current_fig):
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
    [Input("create-annotation-button", "n_clicks"), Input("submit-annotation", "n_clicks")],
    [State("annotation-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    [Output('graphs', 'figure', allow_duplicate=True),
     Output('annotations-store', 'data')],
    [Input('submit-annotation', 'n_clicks')],
    [State('annotation-text-input', 'value'),
     State('graphs', 'figure'),
     State('annotations-store', 'data')],
    prevent_initial_call=True
)
def add_annotation(n_clicks, text, figure, annotations):
    if n_clicks and text:
        x = math.log10(1)
        y = math.log10(1)

        new_annotation = {
            'x': x,
            'y': y,
            'xref': 'x',
            'yref': 'y',
            'text': text,
            'showarrow': False,
            'bgcolor': "white",
            'bordercolor': 'black',
            'borderwidth': 1,
        }

        if 'annotations' in figure['layout']:
            figure['layout']['annotations'].append(new_annotation)
        else:
            figure['layout']['annotations'] = [new_annotation]

        if annotations is None:
            annotations = []
        annotations.append(new_annotation)

        return figure, annotations

    return figure, annotations

@app.callback(
    [Output("graphs", "figure", allow_duplicate=True), 
     Output("disable-annotation-button", "children")],
    [Input("disable-annotation-button", "n_clicks"),
     Input("disable-annotation-button", "children")],
    [State("graphs", "figure")],
    prevent_initial_call=True
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

@app.callback(
    Output('annotation-accordion', 'children'),
    Input('graphs', 'figure'),
    prevent_initial_call=True
)
def generate_angle_inputs(graph):
    if not graph or 'annotations' not in graph['layout']:
        return []

    annotations = graph['layout']['annotations']
    
    group_suffixes = ['_1', '_2']
    grouped_annotations = {suffix: [] for suffix in group_suffixes}
    ungrouped_annotations = []
    accordion_items = []

    for i, ann in enumerate(annotations):
        name = ann.get('name', '')
        matched = False
        for suffix in group_suffixes:
            if name.endswith(suffix):
                grouped_annotations[suffix].append((i, ann))
                matched = True
                break
        if not matched:
            ungrouped_annotations.append((i, ann))

    for suffix, anns in grouped_annotations.items():
        if not anns:
            continue

        cards = []
        for i, ann in anns:
            card = dbc.Card(
                [
                    dbc.CardHeader(
                        f"{ann.get('text')}",
                        style={
                            'color': 'white',
                            'fontWeight': 'bold',
                            'margin': '0px',
                            'padding': '2px 0px 0px 2px'
                        }
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
                                                            'color': 'white',
                                                            'marginRight': '10px',
                                                            'alignSelf': 'center'
                                                        }
                                                    ),
                                                    dbc.Checkbox(
                                                        id={'type': 'annotation-enable', 'index': i},
                                                        className="mb-0",
                                                        style={'alignSelf': 'center'},
                                                        value=ann.get('opacity', 1) == 1
                                                    ),
                                                ],
                                                style={'display': 'flex', 'alignItems': 'center'}
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Angle:",
                                                        style={
                                                            'color': 'white',
                                                            'marginRight': '10px',
                                                            'marginLeft': '30px',
                                                            'alignSelf': 'center'
                                                        }
                                                    ),
                                                    dbc.Input(
                                                        type="number",
                                                        placeholder="Angle",
                                                        value=round(ann.get('textangle', 0)),
                                                        id={'type': 'angle-input', 'index': i},
                                                        style={'width': '80px', 'height': '25px'}
                                                    ),
                                                ],
                                                style={
                                                    'display': 'flex',
                                                    'alignItems': 'center',
                                                    'marginRight': '30px'
                                                }
                                            ),
                                        ],
                                        style={
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'flex-start'
                                        }
                                    ),
                                ],
                                className="mb-0",
                                align="center"
                            ),
                        ],
                        style={'margin': '0px', 'padding': '0px 0px 2px 2px'}
                    ),
                ],
                className="mb-1",
                style={
                    'margin': '0px',
                    'padding': '0px 0px 2px 2px',
                    'backgroundColor': '#6c757d',
                    'Color': ann.get('bordercolor')
                }
            )
            cards.append(card)

        group_title = f"CARM Results {suffix[-1]}"
        accordion_item = dbc.AccordionItem(
            title=group_title,
            children=cards,
            item_id=f"group_{suffix}",
        )
        accordion_items.append(accordion_item)

    if ungrouped_annotations:
        cards = []
        for i, ann in ungrouped_annotations:
            card = dbc.Card(
                [
                    dbc.CardHeader(
                        f"{ann.get('text')}",
                        style={
                            'color': 'white',
                            'fontWeight': 'bold',
                            'margin': '0px',
                            'padding': '2px 0px 0px 2px'
                        }
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
                                                            'color': 'white',
                                                            'marginRight': '10px',
                                                            'alignSelf': 'center'
                                                        }
                                                    ),
                                                    dbc.Checkbox(
                                                        id={'type': 'annotation-enable', 'index': i},
                                                        className="mb-0",
                                                        style={'alignSelf': 'center'},
                                                        value=ann.get('opacity', 1) == 1
                                                    ),
                                                ],
                                                style={'display': 'flex', 'alignItems': 'center'}
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Angle:",
                                                        style={
                                                            'color': 'white',
                                                            'marginRight': '10px',
                                                            'marginLeft': '30px',
                                                            'alignSelf': 'center'
                                                        }
                                                    ),
                                                    dbc.Input(
                                                        type="number",
                                                        placeholder="Angle",
                                                        value=round(ann.get('textangle', 0)),
                                                        id={'type': 'angle-input', 'index': i},
                                                        style={'width': '80px', 'height': '25px'}
                                                    ),
                                                ],
                                                style={
                                                    'display': 'flex',
                                                    'alignItems': 'center',
                                                    'marginRight': '30px'
                                                }
                                            ),
                                        ],
                                        style={
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'flex-start'
                                        }
                                    ),
                                ],
                                className="mb-0",
                                align="center"
                            ),
                        ],
                        style={'margin': '0px', 'padding': '0px 0px 2px 2px'}
                    ),
                ],
                className="mb-1",
                style={
                    'margin': '0px',
                    'padding': '0px 0px 2px 2px',
                    'backgroundColor': '#6c757d',
                    'Color': ann.get('bordercolor')
                }
            )
            cards.append(card)

        accordion_item = dbc.AccordionItem(
            title='Custom Annotations',
            children=cards,
            item_id='other_annotations'
        )
        accordion_items.append(accordion_item)

    return accordion_items



@app.callback(
    Output('graphs', 'figure', allow_duplicate=True),
    [Input({'type': 'annotation-enable', 'index': ALL}, 'value')],
    [State('graphs', 'figure')],
    prevent_initial_call=True
)
def update_annotations_visibility(checkbox_values, figure):
    fig = go.Figure(figure)
    annotations = fig['layout']['annotations']

    if annotations:
        for i, ann in enumerate(annotations):
            if i < len(checkbox_values):
                if checkbox_values[i]:
                    ann['opacity'] = 1  #Visible
                else:
                    ann['opacity'] = 0  #Hidden

        fig.update_layout(annotations=annotations)

    return fig

@app.callback(
    Output('graphs', 'figure', allow_duplicate=True),
    [Input({'type': 'angle-input', 'index': ALL}, 'value')],
    State('graphs', 'figure'),
    prevent_initial_call=True
)
def update_annotation_angles(input_angles, figure):
    #Callback to control annotations angle individually
    if not figure or not input_angles:
        raise dash.exceptions.PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    annotations = figure.get('layout', {}).get('annotations', [])

    for i, angle in enumerate(input_angles):
        if i < len(annotations):
            annotations[i]['textangle'] = angle
    
    new_figure = copy.deepcopy(figure)
    new_figure['layout']['annotations'] = annotations

    return new_figure

@app.callback(
    Output('machine-selected', 'data'),
    Input('filename', 'value')
)
def update_machine_selected(filename):
    #Callback to update the machine selected
    if filename:
        return True
    return False

@app.callback(
    [Output('initial-image', 'style'),
     Output('initial-text', 'style')],
    Input('machine-selected', 'data')
)
def toggle_initial_content(machine_selected):
    #Callback to control the visibility of the initial image and text
    if machine_selected:
        return {'display': 'none'}, {'display': 'none'}
    return {'width': '99%', 'height': '90%', 'display': 'block', 'background': 'transparent', 'marginLeft': '40px'}, {'text-align': 'center', 'margin-top': '10px'}

@app.callback(
    Output("offcanvas", "is_open"),
    [Input("open-offcanvas", "n_clicks")],
    [State("offcanvas", "is_open")]
)
def toggle_offcanvas(n, is_open):
    #Toggle visibility of the sidebar
    if n:
        return not is_open
    return 


@app.callback(
    Output("offcanvas2", "is_open"),
    [Input("open-offcanvas2", "n_clicks")],
    [State("offcanvas2", "is_open")]
)
def toggle_offcanvas(n, is_open):
    #Toggle visibility of the sidebar
    if n:
        return not is_open
    return is_open

@app.callback(
    [Output('graphs', 'figure', allow_duplicate=True),
     Output('graphs', 'config', allow_duplicate=True),
     Output('button-CARM', 'children'),
     ],
    [Input('button-CARM', 'n_clicks')],
    [State('graphs', 'figure'),
     State('graphs', 'config')],
     prevent_initial_call=True
)
def toggle_editable(n_clicks, figure, config):
    #Toggle editable state of the graph
    new_figure = copy.deepcopy(figure)
    if n_clicks % 2 == 0:
        config['editable'] = True
        return new_figure, config, "Save Text Changes"
    else:
        config['editable'] = False
        return new_figure, config, "Edit Graph Text"

@app.callback(
    Output('additional-dropdowns', 'children'),
    [Input('filename', 'value')]
)
def update_additional_dropdowns(selected_file):
    #Update the CARM results filter dropdowns
    if selected_file:
        _, _, _, _, data_list = ut.read_csv_file(selected_file)
        df = pd.DataFrame(data_list)

    fields = ['ISA', 'Precision', 'Threads', 'Loads', 'Stores', 'Interleaved', 'DRAM Bytes', 'FP Inst', 'Date']
    dropdowns = []

    for field in fields:
        if selected_file:
            if not df.empty:
                if field == "Date":
                    unique_values = sorted(df[field.replace(" ", "")].unique(), reverse=True)  
                else:
                    unique_values = sorted(df[field.replace(" ", "")].unique())  
                options = [{'label': value, 'value': value} for value in unique_values]
                display = "flex"
            else:
                options = {}
                display = "none"
        else:
            options = {}
            display = "none"

        if field == "Date":
            width = 250
        elif field == "ISA":
            width = 200
        else:
            width = 160
        
        dropdowns.append(
            html.Div(
                dcc.Dropdown(
                    id=f'{field.lower().replace(" ", "")}-dynamic-dropdown',
                    placeholder=field,
                    options=options,
                    multi=False
                ),
                style = {
                'flex': '1 0 auto',
                'minWidth': width,
                'margin': '5px',
            }
            )
        )
    
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                [
                    html.Div(
                        "CARM Results 1:", 
                        style={
                            'marginRight': '10px', 
                            'alignSelf': 'center', 
                            'fontWeight': 'bold',
                            'minWidth': '125px', 
                        }
                    ),
                    html.Div(
                        dropdowns, 
                        style={
                            'display': 'flex',
                            'width': '100%',
                            'justifyContent': 'space-between',
                            'alignItems': 'center',
                        }
                    )
                ],
                style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'margin': '-10px auto auto auto'
                }
            )
        ]),
        style={
            'margin': '0px auto 10px auto',
            'padding': '0px',
            'textAlign': 'center',
            'display': 'flex',
            'height': '60px'
        }
    )

@app.callback(
    Output('additional-dropdowns2', 'children'),
    [Input('filename', 'value')]
)
def update_additional_dropdowns2(selected_file):
    #Update the CARM results filter dropdowns (line2)
    if selected_file:
        _, _, _, _, data_list = ut.read_csv_file(selected_file)
        df = pd.DataFrame(data_list)

    fields = ['ISA', 'Precision', 'Threads', 'Loads', 'Stores', 'Interleaved', 'DRAM Bytes', 'FP Inst', 'Date']
    dropdowns = []

    for field in fields:
        if selected_file:
            if not df.empty:
                if field == "Date":
                    unique_values = sorted(df[field.replace(" ", "")].unique(), reverse=True)  
                else:
                    unique_values = sorted(df[field.replace(" ", "")].unique())  
                options = [{'label': value, 'value': value} for value in unique_values]
                display = "flex"
            else:
                options = {}
                display = "none"
        else:
            options = {}
            display = "none"

        if field == "Date":
            width = 250
        elif field == "ISA":
            width = 200
        else:
            width = 160
        
        dropdowns.append(
            html.Div(
                dcc.Dropdown(
                    id=f'{field.lower().replace(" ", "")}-dynamic-dropdown2',
                    placeholder=field,
                    options=options,
                    multi=False
                ),
                style = {
                'flex': '1 0 auto',
                'minWidth': width,
                'margin': '5px',
            }
            )
        )
    
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                [
                    html.Div(
                        "CARM Results 2:", 
                        style={
                            'marginRight': '10px', 
                            'alignSelf': 'center', 
                            'fontWeight': 'bold',
                            'color': 'red',
                            'minWidth': '125px', 
                        }
                    ),
                    html.Div(
                        dropdowns, 
                        style={
                            'display': 'flex',
                            'width': '100%',
                            'justifyContent': 'space-between',
                            'alignItems': 'center',
                        }
                    )
                ],
                style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'margin': '-10px auto auto auto'
                }
            )
        ]),
        style={
            'margin': '0px auto 10px auto',
            'padding': '0px',
            'textAlign': 'center',
            'display': 'flex',
            'height': '60px'
        }
    )

@app.callback(
    Output("isa-dynamic-dropdown", "options"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_ISA(Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['ISA'].unique())]

@app.callback(
    Output("precision-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Precision(ISA, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return []

    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)
    if df.empty:
        return []

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Precision'].unique())]

@app.callback(
    Output("threads-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Threads(ISA, Precision, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Threads'].unique())]

@app.callback(
    Output("loads-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Loads(ISA, Precision, Threads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Loads'].unique())]

@app.callback(
    Output("stores-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Stores(ISA, Precision, Threads, Loads, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Stores'].unique())]

@app.callback(
    Output("interleaved-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Interleaved(ISA, Precision, Threads, Loads, Stores, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Interleaved'].unique())]

@app.callback(
    Output("drambytes-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_DRAMBytes(ISA, Precision, Threads, Loads, Stores, Interleaved, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['DRAMBytes'].unique())]

@app.callback(
    Output("fpinst-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_FPInst(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['FPInst'].unique())]

@app.callback(
    Output("date-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Date(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Date'].unique(), reverse=True)]

@app.callback(
    Output("isa-dynamic-dropdown2", "options"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_ISA2(Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['ISA'].unique())]

@app.callback(
    Output("precision-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Precision2(ISA, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return []

    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)
    if df.empty:
        return []

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Precision'].unique())]

@app.callback(
    Output("threads-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Threads2(ISA, Precision, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Threads'].unique())]

@app.callback(
    Output("loads-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Loads2(ISA, Precision, Threads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Loads'].unique())]

@app.callback(
    Output("stores-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Stores2(ISA, Precision, Threads, Loads, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Stores'].unique())]

@app.callback(
    Output("interleaved-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Interleaved2(ISA, Precision, Threads, Loads, Stores, DRAMBytes, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Interleaved'].unique())]

@app.callback(
    Output("drambytes-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_DRAMBytes2(ISA, Precision, Threads, Loads, Stores, Interleaved, FPInst, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['DRAMBytes'].unique())]

@app.callback(
    Output("fpinst-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_FPInst2(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, Date, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['FPInst'].unique())]

@app.callback(
    Output("date-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Date2(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, selected_file):
    #Cross filtering of dropdowns callback, based on available results that respect the other dropdowns selections
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Date'].unique(), reverse=True)]

@app.callback(
    [
    Output(component_id='graphs', component_property='figure'),
    Output('graphs', 'style'),
    Output('graph-size-update', 'children'),
    Output('graph-lines', 'data'),
    Output('graph-lines2', 'data'),
    Output('graph-values', 'data'),
    Output('graph-values2', 'data'),
    Output('graph-isa', 'data'),
    Output('graph-xrange', 'data'),
    Output('graph-yrange', 'data'),
    Output('change-annon', 'data'),
    ],
    [
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    Input('value-slider', 'value'),
    Input('time-slider', 'value'),
    Input('input-number', 'value'),
    Input('average-checkbox', 'value'),
    Input('play-pause-button', 'n_clicks'),
    Input('interval-component', 'n_intervals'),
    Input('isa-checklist', 'value'),
    Input('precision-checklist', 'value'),
    Input('thread-checklist', 'value'),
    Input('color-radio', 'value'),
    Input('total-checklist', 'value'),
    Input('exponent-switch', 'value'),
    Input('line-legend-switch', 'value'),
    Input('lower-filter', 'value'),
    Input('duration-filter', 'value'),
    Input('line-size', 'value'),
    Input('title-size', 'value'),
    Input('axis-size', 'value'),
    Input('tick-size', 'value'),
    Input('tooltip-size', 'value'),
    Input('legend-size', 'value'),
    Input('dot-size', 'value'),
    Input("button-paraver-mask", "n_clicks"),
    Input("button-paraver-accumulate", "n_clicks"),
    Input("button-paraver-colors", "n_clicks"),
    ],
     State('graphs', 'figure'),
    prevent_initial_call=True,
)
def analysis(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date,
             ISA2, Precision2, Threads2, Loads2, Stores2, Interleaved2, DRAMBytes2, FPInst2, Date2, selected_file, timestamps_range, timestamps_max_range, timestamps_grouper, average, n_clicks, n_intervals, ISA_timestamp, Precision_timestamp, Threads_timestamp, color_radio, plot_total, exponant, line_legend, lower_filter, duration_filter, line_size, title_size, axis_size, tick_size, tooltip_size, legend_size, dot_size, mask_button, accum_button, paraver_color_button, figure):#, intervals):
    #Callback to draw the CARM graph and plot everything
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
    zoom_ID = []
    no_zoom_ID = []

    if not selected_file:
        return go.Figure(), {'display': 'none'}, "", None, None, None, None, None, [0, 0], [0, 0], None
    
    #Get trigger-id
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    annotations = {}
    if not figure is None:
        fig = go.Figure(figure)
        annotations = fig['layout']['annotations']

    if trigger_id not in ['graphs', 'interval-component']:
        figure = go.Figure()
    if mask_button_offset != -1:
        if (mask_button + mask_button_offset) % 2 == 1:
            use_paraver_mask = False
        else:
            use_paraver_mask = True
    else:
        use_paraver_mask = False

    if (accum_button + ac_button_offset) % 2 == 1:
        use_accumulate = False
    else:
        use_accumulate = True

    if (paraver_color_button + color_button_offset) % 2 == 1:
        use_paraver_colors = False
    else:
        use_paraver_colors = True

    #Read roofline data and create DataFrame
    _, _, _, _, data_list = ut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)
    if use_paraver_mask:
        filtered_base = base_statistics_df[
            (base_statistics_df['Arithmetic_Intensity'] >= float(lower_filter)) &
            (base_statistics_df['GFLOPS'] >= float(lower_filter)) &
            (base_statistics_df['Duration'] >= float(duration_filter)) &
            (base_statistics_df['Paraver_Value'].apply(ut.is_valid_paraver_value))
        ]
    else:
        filtered_base = base_statistics_df[
            (base_statistics_df['Arithmetic_Intensity'] >= float(lower_filter)) &
            (base_statistics_df['GFLOPS'] >= float(lower_filter)) &
            (base_statistics_df['Duration'] >= float(duration_filter))
        ]

    filtered_intel = intel_statistics_df2.loc[filtered_base.index]
    filtered_base = filtered_base.reset_index(drop=True)
    filtered_intel = filtered_intel.reset_index(drop=True)

    #Get timestamp range to display and filter timestamps dataframe accodingly
    timestampls_real_range = [(x * timestamps_grouper+timestamps_max_range[0])  for x in timestamps_range]
    if timestamps_range[1] == 0:
        timestampls_real_range[1] = timestamps_grouper - 1

    if (timestampls_real_range[1]+timestamps_grouper) > (timestamps_max_range[1] + 1):
        df_filter = filtered_base.iloc[timestampls_real_range[0]:timestamps_max_range[1]+1]
        df_intel_filter = filtered_intel.iloc[timestampls_real_range[0]:timestamps_max_range[1]+1]
    else:
        df_filter = filtered_base.iloc[timestampls_real_range[0]:timestampls_real_range[1]+timestamps_grouper]
        df_intel_filter = filtered_intel.iloc[timestampls_real_range[0]:timestampls_real_range[1]+timestamps_grouper]

    #Filter timestamps again to display based on the filter options
    df_intel_filter2 = ut.construct_query_timestamp(df_intel_filter, ISA_timestamp, Precision_timestamp, Threads_timestamp)
    df_filter = df_filter[df_filter.index.isin(df_intel_filter2.index)]
    columns_to_check = df_intel_filter2.drop(columns=['ThreadID', 'Paraver_Label', 'Timestamp'], errors='ignore')
    

    #Check what ISAs are still being used to adjust the roofline plot shown
    if float(lower_filter) > 0:
        positive_columns = columns_to_check.columns[(columns_to_check >= float(lower_filter)).any()].tolist()
    else:
        positive_columns = columns_to_check.columns[(columns_to_check > 0).any()].tolist()
    if 'ThreadID' in df_intel_filter2.columns:
        positive_columns.append('ThreadID')
    if ISA == None:
        if any("AVX512" in col for col in positive_columns):
            ISA = "avx512"
        elif any("AVX2" in col for col in positive_columns):
            ISA = "avx2"
        elif any("SSE" in col for col in positive_columns):
            ISA = "sse"
        else:
            ISA = "scalar"

    #Get queries for both sets of inputs for the roofline data
    query2 = ut.construct_query(ISA2, Precision2, Threads2, Loads2, Stores2, Interleaved2, DRAMBytes2, FPInst2, Date2)
    query1 = ut.construct_query(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date)
    #If user selects nothing yet, use the most recent roofline result
    filtered_df1 = df.query(query1) if query1 else df

    #If there is no available ISA that matches what the timestamps are using, cycle through them
    i = 0
    while filtered_df1.empty:
        query1 = ut.construct_query(intel_ISA[i], Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date)
        filtered_df1 = df.query(query1) if query1 else df
        i+=1
    #If the user selects anything from the second set of dropdowns, get the matching roofline data
    filtered_df2 = df.query(query2) if query2 and any([ISA2, Precision2, Threads2, Loads2, Stores2, Interleaved2, DRAMBytes2, FPInst2, Date2]) else pd.DataFrame()

    #Totals from the timestamps for plotting
    threads_app = total_threads
    ai = total_ai
    gflops = total_GFLOPS
    name_app = appname

    #Plot Timestamps, if its a zoom we skip this
    if not timestamps_range == None:        
        first=True
        #If we are averaging the timestamps calculate the values to plot
        if average and timestamps_grouper > 1:
            extra_average = " Averaged "
            #Create a grouping variable based on 'timestamps_grouper'
            df_filter = df_filter.copy()
            df_filter['group'] = ((df_filter.index - timestampls_real_range[0]) // timestamps_grouper).astype(int)
            df_filter['ai'] = df_filter["Arithmetic_Intensity"]
            df_filter['gflops'] = df_filter["GFLOPS"]
            df_filter['timestamp'] = df_filter["Timestamp"]
            durations = df_filter["Duration"].tolist()
            thread_IDs = df_filter["ThreadID"].tolist()
            
            #Group the data and compute the mean for 'ai' and 'gflops'
            grouped = df_filter.groupby('group')
            ai_mean = grouped['ai'].mean().reset_index(drop=True)
            if len(ai_mean) > 0:
                smallest_ai = min(ai_mean)
            gflops_mean = grouped['gflops'].mean().reset_index(drop=True)
            if len(gflops_mean) > 0:
                smallest_gflops = min(gflops_mean)
            
            #Create labels by joining the timestamps involved in each group
            timestamps_grouped = grouped['timestamp'].apply(
                lambda x: f"{x.iloc[0]}" if len(x) == 1 else f"{x.iloc[0]}...{x.iloc[-1]}"
            ).reset_index(drop=True)

            df_intel_filter2 = df_intel_filter2.copy()
            df_intel_filter2['group'] = ((df_intel_filter2.index - timestampls_real_range[0]) // timestamps_grouper).astype(int)
            grouped_intel = df_intel_filter2.groupby('group')

            scalar_sp_mean = grouped_intel['Intel_FP_Scalar_SP'].mean().reset_index(drop=True)
            scalar_dp_mean = grouped_intel['Intel_FP_Scalar_DP'].mean().reset_index(drop=True)
            sse_sp_mean = grouped_intel['Intel_FP_SSE_SP'].mean().reset_index(drop=True)
            sse_dp_mean = grouped_intel['Intel_FP_SSE_DP'].mean().reset_index(drop=True)
            avx2_sp_mean = grouped_intel['Intel_FP_AVX2_SP'].mean().reset_index(drop=True)
            avx2_dp_mean = grouped_intel['Intel_FP_AVX2_DP'].mean().reset_index(drop=True)
            avx512_sp_mean = grouped_intel['Intel_FP_AVX512_SP'].mean().reset_index(drop=True)
            avx512_dp_mean = grouped_intel['Intel_FP_AVX512_DP'].mean().reset_index(drop=True)
            dp_mean = grouped_intel['Intel_FP_DP'].mean().reset_index(drop=True)
            fp_total_mean = grouped_intel['Intel_FP_Total'].mean().reset_index(drop=True)
            load_mean = grouped_intel['Intel_Load'].mean().reset_index(drop=True)
            store_mean =grouped_intel['Intel_Store'].mean().reset_index(drop=True)

            scalar_perc = (((scalar_sp_mean+scalar_dp_mean)/fp_total_mean)*100).tolist()
            sse_perc = (((sse_sp_mean+sse_dp_mean)/fp_total_mean)*100).tolist()
            avx2_perc = (((avx2_sp_mean+avx2_dp_mean)/fp_total_mean)*100).tolist()
            avx512_perc = (((avx512_sp_mean+avx512_dp_mean)/fp_total_mean)*100).tolist()
            dp_perc = ((dp_mean/fp_total_mean)*100).tolist()
            load_perc = ((load_mean/(load_mean+store_mean))*100).tolist()

            n = len(ai_mean)
        #If we are not averaging the timestamps calculate the values to plot
        else:
            #If we are accumulating values based on the Paraver mask
            if use_accumulate:
                df = df_filter.copy()
                df_intel = df_intel_filter2.copy()
                df = df.sort_values(by=['ThreadID', 'Timestamp']).reset_index(drop=True)
                df_intel = df_intel.sort_values(by=['ThreadID', 'Timestamp']).reset_index(drop=True)

                df['label_shift'] = df.groupby('ThreadID')['Paraver_Label'].shift()
                df['label_changed'] = df['Paraver_Label'] != df['label_shift']
                df_intel['label_shift'] = df_intel.groupby('ThreadID')['Paraver_Label'].shift()
                df_intel['label_changed'] = df_intel['Paraver_Label'] != df_intel['label_shift']

                df['group'] = df.groupby('ThreadID')['label_changed'].cumsum()
                df_intel['group'] = df_intel.groupby('ThreadID')['label_changed'].cumsum()

                df_filter = df.groupby(['ThreadID', 'group']).agg({
                    'ThreadID': 'first',
                    'Timestamp': lambda x: f"{x.min()}" if x.min() == x.max() else f"{x.min()} - {x.max()}",
                    'Duration': 'sum',
                    'Paraver_Label': 'first',
                    'Paraver_Value': 'first',
                    'R': 'first', 'G': 'first', 'B': 'first',
                    'FLOP': 'sum', 'Bytes': 'sum'
                }).reset_index(drop=True)

                df_intel_filter2 = df_intel.groupby(['ThreadID', 'group']).agg({
                    'ThreadID': 'first',
                    'Timestamp': 'min',
                    'Intel_FP_Scalar_SP': 'sum',
                    'Intel_FP_Scalar_DP': 'sum',
                    'Intel_FP_SSE_SP': 'sum',
                    'Intel_FP_SSE_DP': 'sum',
                    'Intel_FP_AVX2_SP': 'sum',
                    'Intel_FP_AVX2_DP': 'sum',
                    'Intel_FP_AVX512_SP': 'sum',
                    'Intel_FP_AVX512_DP': 'sum',
                    'Intel_FP_SP': 'sum',
                    'Intel_FP_DP': 'sum',
                    'Intel_FP_Total': 'sum',
                    'Intel_Load': 'sum',
                    'Intel_Store': 'sum',
                    'Paraver_Label': 'first',
                }).reset_index(drop=True)

                df_filter['GFLOPS'] = df_filter['FLOP'] / (df_filter['Duration'] * 1e3)
                df_filter['Bandwidth'] = df_filter['Bytes'] / df_filter['Duration']
                df_filter['Arithmetic_Intensity'] = df_filter['FLOP'] / df_filter['Bytes']

                df.drop(columns=['group'], inplace=True)
            
            extra_average = " "
            ai_mean = df_filter["Arithmetic_Intensity"].tolist()
            if len(ai_mean) > 0:
                smallest_ai = min(ai_mean)
            n = len(ai_mean)
            data_points = n
            gflops_mean = df_filter["GFLOPS"].tolist()
            if len(gflops_mean) > 0:
                smallest_gflops = min(gflops_mean)
            timestamps_grouped = df_filter["Timestamp"].tolist()
            durations = df_filter["Duration"].tolist()
            thread_IDs = df_filter["ThreadID"].tolist()
            reds = df_filter["R"].tolist()
            greens = df_filter["G"].tolist()
            blues = df_filter["B"].tolist()
            pvalues = df_filter["Paraver_Value"].tolist()
            plabels = df_filter["Paraver_Label"].tolist()
            

            scalar_perc = (((df_intel_filter2["Intel_FP_Scalar_SP"]+df_intel_filter2["Intel_FP_Scalar_DP"])/df_intel_filter2["Intel_FP_Total"])*100).tolist()
            sse_perc = (((df_intel_filter2["Intel_FP_SSE_SP"]+df_intel_filter2["Intel_FP_SSE_DP"])/df_intel_filter2["Intel_FP_Total"])*100).tolist()
            avx2_perc = (((df_intel_filter2["Intel_FP_AVX2_SP"]+df_intel_filter2["Intel_FP_AVX2_DP"])/df_intel_filter2["Intel_FP_Total"])*100).tolist()
            avx512_perc = (((df_intel_filter2["Intel_FP_AVX512_SP"]+df_intel_filter2["Intel_FP_AVX512_DP"])/df_intel_filter2["Intel_FP_Total"])*100).tolist()
            if use_accumulate:
                dp_perc = ((df_intel_filter2["Intel_FP_DP"]/df_intel_filter2["Intel_FP_Total"])*100).tolist()
                load_perc = ((df_intel_filter2["Intel_Load"]/(df_intel_filter2["Intel_Load"]+df_intel_filter2["Intel_Store"]))*100).tolist()
            else:
                dp_perc = ((df_intel_filter2["Intel_FP_DP"]/df_intel_filter2["Intel_FP_Total"])*100).tolist()
                load_perc = (df_intel_filter2["Intel_Load_Percent"]).tolist()

        #If the play function is activated
        if trigger_id == 'play-pause-button' and data_points>0:

            if n_clicks != 1:
                if not figure is None:
                    figure = go.Figure(figure)
        elif (trigger_id == 'interval-component') and data_points>0:
            if not figure is None:
                figure = copy.deepcopy(figure)
                figure = go.Figure(figure)
            if n_intervals == 0:
                indexer = data_points - 1
            else:
                indexer = n_intervals - 1
            
            if indexer == 0:
                first = True
            else:
                first = False
                
            if (color_radio in ["ISA", "Precision", "LD/ST Percentage", "Thread ID"]):
                    color = ut.blend_colors(scalar_perc[indexer], sse_perc[indexer], avx2_perc[indexer], avx512_perc[indexer], dp_perc[indexer], load_perc[indexer], thread_IDs[indexer], color_radio, False)
            elif color_radio == "Paraver":
                color = f'rgb({reds[indexer]},{greens[indexer]},{blues[indexer]})'
            else:
                color = ut.interpolate_color(start_color, end_color, indexer / n)
            tooltip_text = ut.build_timestamp_tooltip_text(scalar_perc[indexer], sse_perc[indexer], avx2_perc[indexer], avx512_perc[indexer], dp_perc[indexer], load_perc[indexer], timestamps_grouped[indexer], thread_IDs[indexer], durations[indexer], pvalues[indexer], plabels[indexer], window_name)
            
            figure.add_trace(go.Scatter(
                x=[ai_mean[indexer]],
                y=[gflops_mean[indexer]],
                mode='markers',
                name=f'{name_app}{extra_average}Timestamps',
                marker=dict(size=dot_size, color=color),
                legendgroup='1',
                showlegend=first,
                text=[tooltip_text],
                hovertemplate='<b>%{text}</b><br>(%{x}, %{y})<br><extra></extra>',
            ))
        #If we are just doing regular plotting
        else:
            first = True
            plabel_aux = []
            for index, (ai_value, gflops_value, timestamp_label, scalar, sse, avx2, avx512, dp, load, thread_ID, duration, red, green, blue, pvalue, plabel) in enumerate(zip(ai_mean, gflops_mean, timestamps_grouped, scalar_perc, sse_perc, avx2_perc, avx512_perc, dp_perc, load_perc, thread_IDs, durations, reds, greens, blues, pvalues, plabels)):
                if use_paraver_mask:
                    if pvalue > 0:
                        if use_paraver_colors:
                            color = f'rgb({red},{green},{blue})'
                            if plabel in plabel_aux:
                                first = False
                            else:
                                plabel_aux.append(plabel)
                                first = True
                        else:
                            if (color_radio in ["ISA", "Precision", "LD/ST Percentage", "Thread ID"]):
                                color = ut.blend_colors(scalar, sse, avx2, avx512, dp, load, thread_ID, color_radio, False)
                            else:
                                color = ut.interpolate_color(start_color, end_color, index / n)

                        tooltip_text = ut.build_timestamp_tooltip_text(scalar, sse, avx2, avx512, dp, load, timestamp_label, thread_ID, duration, pvalue, plabel, window_name)
                        
                        if not use_paraver_colors:
                            plabel = ""

                        figure.add_trace(go.Scatter(
                            x=[ai_value],
                            y=[gflops_value],
                            mode='markers',
                            name=f'{name_app}{extra_average}{plabel} Timestamps',
                            marker=dict(size=dot_size, color=color),
                            showlegend=first,
                            text=[tooltip_text],
                            hovertemplate='<b>%{text}</b><br>(%{x}, %{y})<br><extra></extra>',
                        ))
                        if not use_paraver_colors:
                            first = False

                else:
                    if use_paraver_colors:
                        color = f'rgb({red},{green},{blue})'
                        if plabel in plabel_aux:
                            first = False
                        else:
                            plabel_aux.append(plabel)
                            first = True
                    else:
                        
                        if (color_radio in ["ISA", "Precision", "LD/ST Percentage", "Thread ID"]):
                            color = ut.blend_colors(scalar, sse, avx2, avx512, dp, load, thread_ID, color_radio, False)
                        else:
                            color = ut.interpolate_color(start_color, end_color, index / n)
                    tooltip_text = ut.build_timestamp_tooltip_text(scalar, sse, avx2, avx512, dp, load, timestamp_label, thread_ID, duration, pvalue, plabel, window_name)
                    
                    if not use_paraver_colors:
                        plabel = ""

                    figure.add_trace(go.Scatter(
                        x=[ai_value],
                        y=[gflops_value],
                        mode='markers',
                        name=f'{name_app}{extra_average}{plabel} Timestamps',
                        marker=dict(size=dot_size, color=color),
                        showlegend=first,
                        text=[tooltip_text],
                        hovertemplate='<b>%{text}</b><br>(%{x}, %{y})<br><extra></extra>',
                    ))
                    if not use_paraver_colors:
                        first = False
    
    if  not trigger_id in ['interval-component']:
        #If we want to plot the total dot
        if plot_total:
            tooltip = ut.build_total_tooltip_text(name_app, threads_app, totals, total_FP_inst, total_mem_inst)
            smallest_gflops = min(gflops, smallest_gflops)
            smallest_ai = min(ai, smallest_ai)

            figure.add_trace(go.Scatter(
                                x=[ai], 
                                y=[gflops], 
                                mode='markers', 
                                name=f'{name_app} Total',
                                marker=dict(size=dot_size, color="red"),
                                text=[tooltip],
                                hovertemplate='<b>%{text}</b><br>(%{x}, %{y})<br><extra></extra>',
                            ))
    #If we want to reset the zoom
    if trigger_id not in ['graphs', 'interval-component']:
        figure.update_layout(
            hoverlabel={
                'font_size': tooltip_size,
            },
        title={
            'text': 'Cache Aware Roofline Model',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'family': "Arial",
                'size': title_size,
                'color': "black"
            }
        },
        height=graph_height,
        width=graph_width,
        xaxis={
            'title':{
                'text': 'Arithmetic Intensity (flop/byte)',
                'font': {
                'family': "Arial",
                'size': axis_size,
                'color': "black"
            },
            },
            'type': 'log',
            'dtick': '0.30102999566',
            'title_standoff': 0,
            'automargin': True,
            'tickfont_size': tick_size,
        },
        yaxis={
            'title':{
                'text': 'Performance (GFLOP/s)',
                'font': {
                'family': "Arial",
                'size': axis_size,
                'color': "black"
            },
            },
            'type': 'log',
            'dtick': '0.30102999566',
            'title_standoff': 0,
            'automargin': True,
            'tickfont_size': tick_size,
        },
        
        legend={
            'font': {'size': legend_size},
            'orientation': 'h',
            'x': 0.5,
            'y': 0,
            'xanchor': 'center',
            'yanchor': 'bottom',
            'yref': 'container',
        },
        showlegend=True,

        plot_bgcolor='#e9ecef',
        paper_bgcolor='#e9ecef',
        clickmode='event',
        )
        figure.update_xaxes(showspikes=True)
        figure.update_yaxes(showspikes=True)

    #Plot the roofline lines if possible, based on the data range and calculate angles for the annotations
    if not filtered_df1.empty:
        values1 = filtered_df1.iloc[-1][['L1', 'L2', 'L3', 'DRAM', 'FP', 'FP_FMA', 'FPInst']].tolist()
        ISA = filtered_df1.iloc[-1][['ISA']].tolist()
        lines = ut.calculate_roofline(values1, smallest_ai/5)
        if lines != lines_origin and len(lines_origin) > 0 and trigger_id != 'interval-component':
            change_annotation = 1
            annotations = {}
        lines_origin = lines
        top_flops = lines['L1']['ridge'][1]
        smallest_gflops = min(smallest_gflops, lines['DRAM']['start'][1])
        #If its just a zoom we dont plot the lines again, just re-calculate the angles for the annotations
        if not trigger_id in ['graphs', 'interval-component']:
            figure.add_traces(ut.plot_roofline(values1, lines, '', ISA[0], line_legend, int(line_size)))

        #Grab the axis range of the plot, after its reset or not
        xaxis_range = figure.layout.xaxis.range
        if xaxis_range:
            x_min_angle = 10**xaxis_range[0]
            x_max_angle = 10**xaxis_range[1]
        else:
            x_min_angle = min(0.00390625, smallest_ai/5)
            x_max_angle = 256

        #Extract the current y-axis range if available, otherwise use the data's min/max
        yaxis_range = figure.layout.yaxis.range
        
        if yaxis_range:
            y_min_angle = 10**yaxis_range[0]
            y_max_angle = 10**yaxis_range[1]
        else:
            y_min_angle = lines['DRAM']['start'][1]*0.5
            y_max_angle = max(lines['L1']['ridge'][1], max(top_flops2,top_flops))*2
        
    lines2 = {}
    values2 = []
    if not filtered_df2.empty and not query2 == None:
        values2 = filtered_df2.iloc[-1][['L1', 'L2', 'L3', 'DRAM', 'FP', 'FP_FMA', 'FPInst']].tolist()
        ISA.append(filtered_df2.iloc[-1][['ISA']].tolist()[0])
        lines2 = ut.calculate_roofline(values2, smallest_ai/5)

        if lines2 != lines_origin2 and trigger_id != 'interval-component':
            change_annotation = 1
            annotations = {}
        lines_origin2 = lines2

        top_flops2 = lines2['L1']['ridge'][1]
        if not trigger_id in ['graphs', 'interval-component']:
            figure.add_traces(ut.plot_roofline(values2, lines2, '2', ISA[1], line_legend, int(line_size)))
    else:
        if lines2 != lines_origin2 and trigger_id != 'interval-component':
            change_annotation = 1
            annotations = {} 
        lines_origin2 = lines2
        
    if exponant:
        xaxis_range = figure.layout.xaxis.range
        x_min = min(0.00390625, smallest_ai/5)
        x_max = 256

        #Extract the current y-axis range if available, otherwise use the data's min/max
        yaxis_range = figure.layout.yaxis.range
        y_min = min(smallest_gflops/5, lines['DRAM']['start'][1]/5)
        y_max = max(lines['L1']['ridge'][1], max(top_flops2,top_flops))*1.3

        x_tickvals, x_ticktext = ut.make_power_of_two_ticks(x_min, x_max)
        y_tickvals, y_ticktext = ut.make_power_of_two_ticks(y_min, y_max)

        #Update axes to show 2^X notation
        figure.update_xaxes(tickmode='array', tickvals=x_tickvals, ticktext=x_ticktext)
        figure.update_yaxes(tickmode='array', tickvals=y_tickvals, ticktext=y_ticktext)
    else:
        #Revert to normal formatting
        figure.update_yaxes(exponentformat=None, tickformat=None)
        figure.update_xaxes(exponentformat=None, tickformat=None)

    timestamp = datetime.datetime.now().isoformat()
    if annotations:
        figure['layout']['annotations'] = annotations

    return figure, {'display': 'block'}, f"Update: {timestamp}", lines, lines2, values1, values2, ISA, [x_min_angle, x_max_angle], [y_min_angle, y_max_angle], change_annotation#, 'width': '100%', 'height' : '100%'}


@app.callback(
    [Output(component_id='graphs', component_property='figure', allow_duplicate=True),
     Output('change-annon', 'data', allow_duplicate=True)],
    [
    Input('graph-size-data', 'children'),
    
    Input('graph-lines', 'data'),
    Input('graph-lines2', 'data'),
    Input('graph-values', 'data'),
    Input('graph-values2', 'data'),
    Input('graph-isa', 'data'),
    Input('graph-xrange', 'data'),
    Input('graph-yrange', 'data'),
    Input('graphs', 'relayoutData'),
    Input('change-annon', 'data'),
    Input("disable-annotation-button", "children"),
    Input('annotation-size', 'value'),
    
    ],
    State('graphs', 'figure'),
    prevent_initial_call=True,
)
def angle_updater(size, lines, lines2, values1, values2, ISA, xrange, yrange, relayout_data, change_anon, disable_anon, anon_size, figure):
    #Callback to update the annotations angles when the graph scale/zoom changes
    if disable_anon != "Enable Annotations" and ISA:
        if figure:
            xaxis_range = figure["layout"]["xaxis"]["range"]

            yaxis_range = figure["layout"]["yaxis"]["range"]
        
        new_figure = go.Figure(figure)

        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        if size and len(size) >= 2:
            liner = size.split('\n')
            width = float(liner[0].replace('Plot area width:', '').replace('px','').strip())
            height = float(liner[1].replace('Plot area height:', '').replace('px','').strip())
        
            annotations = new_figure['layout']['annotations']
            cache_levels = ['L1', 'L2', 'L3', 'DRAM']
            cache_level_suffix = ['L1_1', 'L2_1', 'L3_1', 'DRAM_1', 'FP_1', 'FP_FMA_1', 'L1_2', 'L2_2', 'L3_2', 'DRAM_2', 'FP_2', 'FP_FMA_2']
            for ann in annotations:
                    ann_name = ann["name"]

                    if ann_name in cache_level_suffix:
                        if ann_name[:-2] in cache_levels:
                            if ann_name[-1] == '1':
                                liner = lines
                            elif ann_name[-1] == '2' and lines2:
                                liner = lines2
                            else:
                                continue
                            log_x1, log_x2 = math.log10(liner[ann_name[:-2]]['start'][0]), math.log10(liner[ann_name[:-2]]['ridge'][0])
                            log_y1, log_y2 = math.log10(liner[ann_name[:-2]]['start'][1]), math.log10(liner[ann_name[:-2]]['ridge'][1])

                            log_xmin, log_xmax = xaxis_range[0], xaxis_range[1]
                            log_ymin, log_ymax = yaxis_range[0], yaxis_range[1]

                            #Compute pixel coordinates based on log scale
                            x1_pixel = ( (log_x1 - log_xmin) / (log_xmax - log_xmin) ) * width
                            x2_pixel = ( (log_x2 - log_xmin) / (log_xmax - log_xmin) ) * width
                            y1_pixel = height - ( (log_y1 - log_ymin) / (log_ymax - log_ymin) ) * height
                            y2_pixel = height - ( (log_y2 - log_ymin) / (log_ymax - log_ymin) ) * height

                            #Pixel slope
                            pixel_slope = (y2_pixel - y1_pixel) / (x2_pixel - x1_pixel)
                            ann['textangle'] = round(math.degrees(math.atan(pixel_slope)), 2)

            for cache_level in ['L1', 'L2', 'L3', 'DRAM', 'FP', 'FMA']:
                if not annotations or change_anon == 1:
                    new_figure.add_annotation(ut.draw_annotation(values1, lines, '1', ISA[0], cache_level, width, height, x_range=[xaxis_range[0], xaxis_range[1]], y_range=[yaxis_range[0], yaxis_range[1]]))
            if len(lines2) > 0:
                for cache_level in ['L1', 'L2', 'L3', 'DRAM', 'FP', 'FMA']:
                    if not annotations or change_anon == 1:
                        new_figure.add_annotation(ut.draw_annotation(values2, lines2, '2', ISA[1], cache_level, width, height, x_range=[xaxis_range[0], xaxis_range[1]], y_range=[yaxis_range[0], yaxis_range[1]]))
            if change_anon == 1:
                change_anon = 0
                
            return new_figure, change_anon
    else:
        return figure, change_anon

# Callback to toggle the Play/Pause button and enable/disable the interval
@app.callback(
    [Output('play-pause-button', 'children'),
     Output('interval-component', 'disabled'),
     Output('interval-component', 'n_intervals'),
     Output('play-pause-button', 'n_clicks')],
    [Input('play-pause-button', 'n_clicks'),
     Input('interval-component', 'n_intervals'),
     Input('input-number', 'value'),
     Input('value-slider', 'value'),
     Input('time-slider', 'value'),
     Input('average-checkbox', 'value'),],
    [State('play-pause-button', 'children'),
     ]
)
def toggle_play_pause(n_clicks, n_intervals, group_value, current_values, time_values, average, current_state):
    #Callback to control the play/pause button
    ctx = dash.callback_context
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

    if n_clicks == 0:
        #Initial state
        return '▶️', True, 0, n_clicks
    
    elif triggered_input in ["value-slider", "time-slider", "input-number", 'average-checkbox']:
        n_intervals = 0
        n_clicks = 0
        return '▶️', True, n_intervals, n_clicks
    else:
        if triggered_input == "play-pause-button":
            if current_state == '▶️':
                return '⏸️', False, n_intervals, n_clicks
            else:
                return '▶️', True, n_intervals, n_clicks
        else:
            if current_state == '▶️':
                return '▶️', True, n_intervals, n_clicks
            else:
                if n_intervals >= data_points:
                    #All data points have been displayed; reset n_intervals
                    n_intervals = 0
                    n_clicks = 0
                    return '▶️', True, n_intervals, n_clicks
                else:
                    return '⏸️', False, n_intervals, n_clicks

@app.callback(
    [Output('input-number', 'value'),
    Output('input-number', 'max')],
    [Input('button-divide', 'n_clicks'),
     Input('button-multiply', 'n_clicks'),
     Input('time-slider', 'value'),
     Input('lower-filter', 'value'),
     Input('duration-filter', 'value')],
    [State('input-number', 'value')]
)
def update_number(divide_clicks, multiply_clicks, time_values, lower_filter, duration_filter, current_value):
    #Callback to update the grouping number
    ctx = dash.callback_context
    start_index = time_values[0]
    end_index = time_values[1]
    filtered_base = base_statistics_df[
        (base_statistics_df['Arithmetic_Intensity'] >= float(lower_filter)) &
        (base_statistics_df['GFLOPS'] >= float(lower_filter)) &
        (base_statistics_df['Duration'] >= float(duration_filter))
    ]
    filtered_base = filtered_base.reset_index(drop=True)
    selected_segments = filtered_base.loc[start_index:end_index, "Timestamp"].tolist()

    n_segments = len(selected_segments)

    if not ctx.triggered:
        if current_value < 0:
            return 1, n_segments
        return current_value, n_segments
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    new_value = current_value
    if button_id == 'button-divide' and current_value > 1:
        new_value = max(1, current_value // 2)
    elif button_id == 'button-multiply':
        new_value = min(current_value * 2, n_segments)

    return new_value, n_segments

@app.callback(
    [
        Output('time-slider', 'marks'),
        Output('time-slider', 'max'),
        Output('time-slider', 'value')
    ],
    [
        Input('time-slider', 'value'),
        Input('lower-filter', 'value'),
        Input('duration-filter', 'value'),
        Input("button-paraver-mask", "n_clicks")
    ],
)
def update_slider_marks2(current_values, lower_filter, duration_filter, mask_button):
    global mask_button_offset
    if mask_button_offset != -1:
        if (mask_button + mask_button_offset) % 2 == 1:
            use_paraver_mask = False
        else:
            use_paraver_mask = True
    else:
        use_paraver_mask = False

    #Callback to update the timestamp slider marks
    if use_paraver_mask:
        filtered_base = base_statistics_df[
            (base_statistics_df['Arithmetic_Intensity'] >= float(lower_filter)) &
            (base_statistics_df['GFLOPS'] >= float(lower_filter)) &
            (base_statistics_df['Duration'] >= float(duration_filter)) &
            (base_statistics_df['Paraver_Value'].apply(ut.is_valid_paraver_value)) 
        ]
    else:
        filtered_base = base_statistics_df[
            (base_statistics_df['Arithmetic_Intensity'] >= float(lower_filter)) &
            (base_statistics_df['GFLOPS'] >= float(lower_filter)) &
            (base_statistics_df['Duration'] >= float(duration_filter))
        ]

    filtered_base = filtered_base.reset_index(drop=True)
    segments = filtered_base["Timestamp"].tolist()
    n_segments = len(segments)
    if n_segments == 0:
        safe_marks = {0: {'label': 'No data', 'style': {'margin-top': '0px'}}}
        return safe_marks, 0, [0, 0]
    #Calculate the number of items per group
    grouped_segments = []
    group_value = 1

    for i in range(0, n_segments, group_value):
        current_group = segments[i:i + group_value]
        if len(current_group) > 1:
            grouped_segment = f"{current_group[0]}...{current_group[-1]}"
        else:
            grouped_segment = f"{current_group[0]}"
        grouped_segments.append(grouped_segment)

    marks = {i: {'label': v} for i, v in enumerate(grouped_segments)}
    max_index = len(grouped_segments) - 1

    if max_index <= 15:
        for i in marks:
            if i % 2 == 0:
                marks[i]['style'] = {'margin-top': '0px'}
            else:
                marks[i]['style'] = {'margin-top': '-35px'}

    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'input-number' or current_values is None or triggered_id in ["lower-filter", "duration-filter", "button-paraver-mask"]:
        initial_range = [0, max(min(max_index, 1), 1)]

        if max_index > 15:
            #Only show two marks and apply top/bottom logic to these two
            filtered_marks = {
                initial_range[0]: marks[initial_range[0]],
                initial_range[1]: marks[initial_range[1]]
            }
            #First mark on bottom, second on top
            filtered_marks[initial_range[0]]['style'] = {'margin-top': '0px'}
            filtered_marks[initial_range[1]]['style'] = {'margin-top': '-35px'}
            return filtered_marks, max_index, initial_range
        else:
            return marks, max_index, initial_range

    #Update to only show the first and last marks within the selected range
    if isinstance(current_values, list) and len(current_values) == 2 and len(grouped_segments) > 1:
        filtered_marks = {
            current_values[0]: marks[current_values[0]],
            current_values[1]: marks[current_values[1]]
        }
    elif isinstance(current_values, list) and len(current_values) == 2:
        # If there is only one item in the range, show it only
        filtered_marks = {current_values[0]: marks[current_values[0]]}
    else:
        # For single value sliders, just show the selected mark
        filtered_marks = {current_values: marks[current_values]}

    if max_index > 15:
        #Only two marks, first bottom, second top
        sorted_keys = sorted(filtered_marks.keys())
        if len(sorted_keys) == 2:
            filtered_marks[sorted_keys[0]]['style'] = {'margin-top': '0px'}
            filtered_marks[sorted_keys[1]]['style'] = {'margin-top': '-35px'}
        return filtered_marks, max_index, current_values
    else:
        return marks, max_index, current_values


@app.callback(
    [
        Output('value-slider', 'marks'),
        Output('value-slider', 'max'),
        Output('value-slider', 'value')
    ],
    [
        Input('input-number', 'value'),
        Input('value-slider', 'value'),
        Input('time-slider', 'value'),
        Input('input-number', 'value'),
        Input('lower-filter', 'value'),
        Input('duration-filter', 'value'),
        Input("button-paraver-mask", "n_clicks")
    ],
)
def update_slider_marks(group_value, current_values, time_values, timestamps_grouper, lower_filter, duration_filter, mask_button):
    #Callback to update the timestamp slider marks
    global max_dots_auto, mask_button_offset
    start_index = time_values[0]
    end_index = time_values[1]
    
    if timestamps_grouper > 1:
        max_marks = 8
    else:
        max_marks = 15
    if mask_button_offset != -1:
        if (mask_button + mask_button_offset) % 2 == 1:
            use_paraver_mask = False
        else:
            use_paraver_mask = True
    else:
        use_paraver_mask = False
    
    if use_paraver_mask:
        filtered_base = base_statistics_df[
            (base_statistics_df['Arithmetic_Intensity'] >= float(lower_filter)) &
            (base_statistics_df['GFLOPS'] >= float(lower_filter)) &
            (base_statistics_df['Duration'] >= float(duration_filter)) &
            (base_statistics_df['Paraver_Value'].apply(ut.is_valid_paraver_value))
        ]
    else:
        filtered_base = base_statistics_df[
            (base_statistics_df['Arithmetic_Intensity'] >= float(lower_filter)) &
            (base_statistics_df['GFLOPS'] >= float(lower_filter)) &
            (base_statistics_df['Duration'] >= float(duration_filter))
        ]
    filtered_base = filtered_base.reset_index(drop=True)
    #Extract the timestamps and scale them
    selected_segments = filtered_base.loc[start_index:end_index, "Timestamp"].tolist()
    n_segments = len(selected_segments)

    if n_segments == 0:
        safe_marks = {0: {'label': 'No data', 'style': {'margin-top': '0px'}}}
        return safe_marks, 0, [0, 0]

    grouped_segments = []

    for i in range(0, n_segments, group_value):
        current_group = selected_segments[i:i + group_value]
        if len(current_group) > 1:
            grouped_segment = f"{current_group[0]}...{current_group[-1]}"
        else:
            grouped_segment = f"{current_group[0]}"
        grouped_segments.append(grouped_segment)

    marks = {i: {'label': v} for i, v in enumerate(grouped_segments)}
    max_index = len(grouped_segments) - 1

    #Apply styling depending on the number of marks
    if max_index <= max_marks:
        for i in marks:
            if i % 2 == 0:
                marks[i]['style'] = {
                    'margin-top': '0px'
                }
            else:
                marks[i]['style'] = {
                    'margin-top': '-35px'
                }

    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'input-number' or current_values is None or triggered_id in ["lower-filter", "duration-filter", 'time-slider', "button-paraver-mask"]:
        
        if len(grouped_segments) < max_dots_auto:
            if max_index > 0:
                initial_range = [0, max_index]
            else:
                initial_range = [0, 0]
        else:
            if max_index > 0:
                initial_range = [0, min(max_index, 1)]
            else:
                initial_range = [0, 0]

        filtered_marks = {
            initial_range[0]: marks[initial_range[0]],
            initial_range[1]: marks[initial_range[1]]
        }

        if max_index > max_marks:
            filtered_marks[initial_range[0]]['style'] = {
                'margin-top': '0px'
            }
            filtered_marks[initial_range[1]]['style'] = {
                'margin-top': '-35px'
            }
            return filtered_marks, max_index, initial_range
        else:
            return marks, max_index, initial_range

    #Update to only show the first and last marks within the selected range
    if isinstance(current_values, list) and len(current_values) == 2 and len(grouped_segments) > 1:
        filtered_marks = {
            current_values[0]: marks[current_values[0]],
            current_values[1]: marks[current_values[1]]
        }
    elif isinstance(current_values, list) and len(current_values) == 2:
        #If there is only one item in the range, show it only
        filtered_marks = {current_values[0]: marks[current_values[0]]}
    else:
        #For single value sliders, just show the selected mark
        filtered_marks = {current_values: marks[current_values]}

    #If more than max_marks, show only these two and style them accordingly
    if max_index > max_marks:
        sorted_keys = sorted(filtered_marks.keys())
        if len(sorted_keys) == 2:
            filtered_marks[sorted_keys[0]]['style'] = {
                'margin-top': '0px'
            }
            filtered_marks[sorted_keys[1]]['style'] = {
                'margin-top': '-35px'
            }
        return filtered_marks, max_index, current_values
    else:
        return marks, max_index, current_values

#Callback to extract the graphs dimensions directly from the component
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
    Output('graph-size-data', 'children'),
    Input('graph-size-update', 'children'),
)


if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(debug=False)