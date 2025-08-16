import csv
import os
import math
import shutil
import sys
import subprocess

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import hashlib

CONFIG_FILE = "./config/auto_config/config.txt"



intel_ISA_colors = {"avx512": "blue", "avx2": "green", "sse": "purple", "scalar": "black"}
color_map = {
    "blue":   (0, 0, 255),
    "green":  (0, 255, 0),
    "purple": (128, 0, 128),
    "black":  (0, 0, 0)
}

#Colors for SP/DP and Load/Store
precision_color_map = {
    "sp": (0, 0, 255),    # orange
    "dp": (0, 255, 0)       # red
}

loadstore_color_map = {
    "load": (0, 0, 255),
    "store": (255, 0, 0)
}


def find_and_run(command):
    cmd_path = shutil.which(command)
    if cmd_path is None:
        print(f"ERROR: '{command}' not found on PATH.", file=sys.stderr)
        return False, None

    try:
        completed = subprocess.run(
            cmd_path,
            check=True,       
            capture_output=True,
            text=True
        )
        return True

    except subprocess.CalledProcessError as e:
        print(f"'{command}' failed with exit {e.returncode}", file=sys.stderr)
        print("STDERR:", e.stderr, file=sys.stderr)
        return False, None

def carm_eq(ai, bw, fp):
    return np.minimum(ai*bw, fp)

def custom_round(value, digits=4):
    if value == 0:
        return 0  #Directly return 0 if the value is 0
    str_value = str(value)
    if abs(value) >= 1 or 'e' in str_value or 'E' in str_value or '.' not in str_value:
        #For numbers greater than or equal to 1, round normally
        return round(value, digits)
    
    decimal_part = str_value.split('.')[1]
    leading_zeros = 0
    for char in decimal_part:
        if char == '0':
            leading_zeros += 1
        else:
            break
    
    #Adjust the number of digits based on the position of the first significant digit
    total_digits = digits + leading_zeros
    return round(value, total_digits)

def is_valid_paraver_value(val):
    val_str = str(val).strip()
    if val_str == "":
        return False
    try:
        return float(val_str) > 0
    except ValueError:
        return True 

def find_nearest_positive(df, index, lower_filter, duration_filter, use_paraver_mask, min_bound=0):
    max_index = len(df) - 1

    def row_is_valid(i):
        if use_paraver_mask:
            return (
                df.loc[i, 'GFLOPS'] >= lower_filter and
                df.loc[i, 'Arithmetic_Intensity'] >= lower_filter and
                df.loc[i, 'Duration'] >= duration_filter and
                is_valid_paraver_value(df.loc[i, 'Paraver_Label'])
            )

        else:
            return (
                df.loc[i, 'GFLOPS'] >= lower_filter and
                df.loc[i, 'Arithmetic_Intensity'] >= lower_filter and
                df.loc[i, 'Duration'] >= duration_filter
            )

    if index >= min_bound and row_is_valid(index):
        return index
    distance = 1
    while (index - distance >= min_bound) or (index + distance <= max_index):
        left_index = index - distance
        if left_index >= min_bound and row_is_valid(left_index):
            return left_index

        right_index = index + distance
        if right_index <= max_index and row_is_valid(right_index):
            return right_index

        distance += 1
    raise ValueError(f"No row with valid GFLOPS, Arithmetic_Intensity, and Duration found from index {index} with lower bound {min_bound}.")



def read_library_path(tag):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as file:
            for line in file:
                if line.strip() == "":
                    continue
                parts = line.strip().split("=")
                if len(parts) == 2:
                    key, value = parts
                    if key == tag:
                        return value
    return None

def write_library_path(tag, path):
    with open(CONFIG_FILE, "a") as file:
        file.write(f"{tag}={path}\n")

def read_csv_file(file_path):
    data_list = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        machine_name = header[1]
        l1_size = int(header[3])
        l2_size = int(header[5])
        l3_size = int(header[7])

        header2 = next(reader)
        for row in reader:
            if not row or not ''.join(row).strip():
                continue
            data = {}
            data['Date'] = row[0]
            data['ISA'] = row[1]
            data['Precision'] = row[2]
            data['Threads'] = int(row[3])
            data['Loads'] = int(row[4])
            data['Stores'] = int(row[5])
            data['Interleaved'] = row[6]
            data['DRAMBytes'] = int(row[7])
            data['FPInst'] = row[8]
            data['L1'] = float(row[9])
            data['L2'] = float(row[11])
            data['L3'] = float(row[13])
            data['DRAM'] = float(row[15])
            data['FP'] = float(row[17])
            data['FP_FMA'] = float(row[19])
            data_list.append(data)

    return machine_name, l1_size, l2_size, l3_size, data_list

def read_application_csv_file(file_path):
    if not os.path.exists(file_path):
        print("Application file does not exist:", file_path)
        return False

    data_list = []
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)

            if header is None:
                print("File is empty:", file_path)
                return False
            
            for row in reader:
                if row:
                    data = {
                        'Date': row[0],
                        'Method': row[1],
                        'Name': row[2],
                        'ISA': row[3],
                        'Precision': row[4],
                        'Threads': row[5],
                        'AI': float(row[6]),
                        'GFLOPS': float(row[7]),
                        'Bandwidth': float(row[8]),
                        'Time': float(row[9])
                    }
                    data_list.append(data)

    except Exception as e:
        print("Failed to read the file:", file_path, "Error:", e)
        return False
    return data_list if data_list else False

def ensure_list(marker_dict, attr_name, default_value, n_points):
        #If marker[attr_name] doesn't exist or is not a list, convert it to a repeated list.
        if attr_name not in marker_dict:
            return [default_value] * n_points
        
        val = marker_dict[attr_name]
        if isinstance(val, list):
            return val
        else:
            return [val] * n_points

def make_power_of_two_ticks(min_val, max_val):
    min_val = max(min_val, 0.0000000001)
    max_val = max(max_val, 0.0000000001)
    start_exp = math.floor(math.log2(min_val))
    end_exp = math.ceil(math.log2(max_val))
    tickvals = [2**i for i in range(start_exp, end_exp+1)]
    ticktext = [f"2<sup>{i}</sup>" for i in range(start_exp, end_exp+1)]
    return tickvals, ticktext

def extract_last_segment(s):
    return s.split("_")[-1] if "_" in s else s

def extract_prefix(s):
    if "_" in s:
        return s.rsplit("_", 1)[0]
    return s

def interpolate_color(start_color, end_color, factor):
    r = int(start_color[0] + factor * (end_color[0] - start_color[0]))
    g = int(start_color[1] + factor * (end_color[1] - start_color[1]))
    b = int(start_color[2] + factor * (end_color[2] - start_color[2]))
    return f'rgb({r},{g},{b})'

def construct_query(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date):
    query_parts = []
    if ISA:
        query_parts.append(f"ISA == '{ISA}'")
    if Precision:
        query_parts.append(f"Precision == '{Precision}'")
    if Threads:
        query_parts.append(f"Threads == {Threads}")
    if Loads:
        query_parts.append(f"Loads == {Loads}")
    if Stores:
        query_parts.append(f"Stores == {Stores}")
    if Interleaved:
        query_parts.append(f"Interleaved == '{Interleaved}'")
    if DRAMBytes:
        query_parts.append(f"DRAMBytes == {DRAMBytes}")
    if FPInst:
        query_parts.append(f"FPInst == '{FPInst}'")
    if Date:
        query_parts.append(f"Date == '{Date}'")

    return " and ".join(query_parts) if query_parts else None

def construct_query_timestamp(df, ISA_list, Precision_list, Threads_list):

    selected_columns = []
    for isa in ISA_list:
        for precision in Precision_list:
            column_name = f'Intel_FP_{isa}_{precision}'
            if column_name in df.columns:
                selected_columns.append(column_name)
    
    if not selected_columns:
        print("No matching columns found for the selected ISA and Precision.")
        return pd.DataFrame()
    
    condition = (df[selected_columns] > 1).any(axis=1)
    thread_condition = df['ThreadID'].isin(Threads_list)
    combined_condition = condition & thread_condition
    df_filtered = df[combined_condition]
    
    return df_filtered

def roof_value_at_x(roof, x):
    start, ridge, end = roof["start"], roof["ridge"], roof["end"]
    if x <= ridge[0]:
        if ridge[0] == start[0]:
            return start[1]
        slope = (ridge[1] - start[1]) / (ridge[0] - start[0])
        return start[1] + slope * (x - start[0])
    else:
        if end[0] == ridge[0]:
            return ridge[1]
        slope = (end[1] - ridge[1]) / (end[0] - ridge[0])
        return ridge[1] + slope * (x - ridge[0])

def label_cache_level(row, roofs):
    """
    Determine the cache level at which a performance point (Arithmetic Intensity, GFLOPS)
    lies below the roofline. The first matching roof (highest bandwidth) is returned.
    """
    x = row["Arithmetic_Intensity"]
    y = row["GFLOPS"]

    roof_priority = ["DRAM", "L3", "L2", "L1"]
    roof_translation = {"DRAM": 4, "L3": 3, "L2": 2, "L1": 1}

    if x <= 0 or y <= 0:
        return 0

    for level in roof_priority:
        if level in roofs:
            roof_y = roof_value_at_x(roofs[level], x)
            if y < roof_y:
                return roof_translation[level]

    return 6  #Not below any roof

def calculate_roofline(values, min_ai):
    aidots = [0]*3
    FPaidots = [0]*2
    FPgflopdots = [0]*2

    ai = np.linspace(min(0.00390625,min_ai), 256, num=200000)
    cache_levels = ['L1', 'L2', 'L3', 'DRAM']
    
    dots = {}

    for cache_level in cache_levels:
        if values[cache_levels.index(cache_level)] > 0:
            aidots = [0, 0, 0]
            # Compute the first point
            y_values = carm_eq(ai, values[cache_levels.index(cache_level)], values[5])


            #Find the point where y_values stops increasing or reaches a plateau
            for i in range(1, len(y_values)):
                if y_values[i - 1] == y_values[i]:
                    aidots[1] = float(ai[i - 1])
                    break
            else:
                aidots[1] = float(ai[-1])
                i = len(y_values) - 12

            mid_ai = np.sqrt(aidots[1]*min(0.00390625,min_ai))
            mid_gflops = np.sqrt(y_values[0]*y_values[i - 1])

            dots[cache_level] = {
                "start": [min(0.00390625,min_ai), y_values[0]],
                "mid": [mid_ai, mid_gflops],
                "ridge": [aidots[1], y_values[i - 1]],
                "end": [ai[-1], y_values[-1]]
            }

    for i in range(4):
        if values[i]:
            top_roof = values[i]
            break

    y_values = carm_eq(ai, top_roof, values[4])

    for i in range(1, len(y_values)):
        if(y_values[i-1] == y_values[i]):
            FPaidots[0] = float(ai[i-1])
            break
    FPgflopdots[0]= y_values[i-1]

    FPaidots[1] = ai[199999]
    FPgflopdots[1] = y_values[199999]

    dots[values[6]] = {
                "ridge": [FPaidots[0], FPgflopdots[0]],
                "end": [FPaidots[1], FPgflopdots[1]]
            }

    return dots

def plot_roofline(values, dots, name_suffix, ISA, line_legend, line_size):
    aidots = [0]*3
    gflopdots = [0]*3

    traces = []
    cache_levels = ['L1', 'L2', 'L3', 'DRAM']
    if name_suffix == "":
        colors = ['black', 'black', 'black', 'black']
        color_inst = 'black'
    else:
        colors = ['red', 'red', 'red', 'red']
        color_inst = 'red'
    linestyles = ['solid', 'solid', 'dash', 'dot']

    for cache_level, color, linestyle in zip(cache_levels, colors, linestyles):
        
        cache_dots = dots.get(cache_level)
        if cache_dots:
            aidots = [
                cache_dots["start"][0],
                cache_dots["ridge"][0],
                cache_dots["end"][0]
            ]
            gflopdots = [
                cache_dots["start"][1],
                cache_dots["ridge"][1],
                cache_dots["end"][1]
            ]
            trace = go.Scatter(
                x=aidots, y=gflopdots,
                mode='lines',
                text=['',f'{cache_level} {ISA.upper()} Peak Bandwidth: {values[cache_levels.index(cache_level)]} GB/s',f'FP FMA {ISA.upper()} Peak: {values[5]} GFLOP/s'],
                hovertemplate='<b>%{text}</b><br>(%{x}, %{y})<br><extra></extra>',
                line=dict(color=color, dash=linestyle, width=line_size),
                name=f'{cache_level} {ISA.upper()}',
                showlegend=line_legend,
            )
            traces.append(trace)
    
    aidots = [
                dots[values[6]]["ridge"][0],
                dots[values[6]]["end"][0]
            ]
    gflopdots = [
                dots[values[6]]["ridge"][1],
                dots[values[6]]["end"][1]
            ]
    
    trace_inst = go.Scatter(
        x=aidots, y=gflopdots,
        mode='lines',
        text=[f'FP {ISA.upper()} {values[6].upper()} Peak: {values[4]} GFLOP/s',f'FP {ISA.upper()} {values[6].upper()} Peak: {values[4]} GFLOP/s'],
        hovertemplate='<b>%{text}</b><br>(%{x}, %{y})<br><extra></extra>',
        line=dict(color=color_inst, dash="dashdot", width=line_size),
        name=f'{values[6].upper()} {ISA.upper()}',
        showlegend=line_legend,
    )
    traces.append(trace_inst)
    
    return traces

def draw_annotation(values, lines, name_suffix, ISA, cache_level, graph_width, graph_height, x_range=None, y_range=None):
    aidots = [0]*3
    gflopdots = [0]*3
    annotation = {}
    cache_levels = ['L1', 'L2', 'L3', 'DRAM']
    angle_degrees = {}

    if cache_level in cache_levels:
        log_x1, log_x2 = math.log10(lines[cache_level]['start'][0]), math.log10(lines[cache_level]['ridge'][0])
        log_y1, log_y2 = math.log10(lines[cache_level]['start'][1]), math.log10(lines[cache_level]['ridge'][1])

        log_xmin, log_xmax = x_range[0], x_range[1]
        log_ymin, log_ymax = y_range[0], y_range[1]

        x1_pixel = ( (log_x1 - log_xmin) / (log_xmax - log_xmin) ) * graph_width
        x2_pixel = ( (log_x2 - log_xmin) / (log_xmax - log_xmin) ) * graph_width

        y1_pixel = graph_height - ( (log_y1 - log_ymin) / (log_ymax - log_ymin) ) * graph_height
        y2_pixel = graph_height - ( (log_y2 - log_ymin) / (log_ymax - log_ymin) ) * graph_height

        pixel_slope = (y2_pixel - y1_pixel) / (x2_pixel - x1_pixel)

        angle_degrees[cache_level] = math.degrees(math.atan(pixel_slope))
    
    ai = np.linspace(0.00390625, 256, num=200000)

    if name_suffix == "1":
        colors = ['black', 'black', 'black', 'black']
        factor = 1.3
    else:
        colors = ['red', 'red', 'red', 'red']
        factor = 0.7

    if cache_level in cache_levels and values[cache_levels.index(cache_level)] > 0:
        aidots[0] = 0.00390625
        y_values = carm_eq(ai, values[cache_levels.index(cache_level)], values[5])
        gflopdots[0]= y_values[0]
        for i in range(1, len(y_values)):
            if(y_values[i-1] == y_values[i]):
                aidots[1] = float(ai[i-1])
                break
        gflopdots[1]= y_values[i-1]

        annotation = go.layout.Annotation(
        x=math.log10(lines[cache_level]['mid'][0]*factor),
        y=math.log10(lines[cache_level]['mid'][1]*factor),
        text=f'{cache_level} {ISA} Bandwidth: {values[cache_levels.index(cache_level)]} GB/s',
        showarrow=False,
        font=dict(
            color=colors[0],
            size=12,
        ),
        align="center",
        bgcolor="white",
        bordercolor=colors[0],
        borderwidth=1,
        textangle=angle_degrees[cache_level],
        name=f"{cache_level}_{name_suffix}"
        )

    if cache_level == "FMA" and values[5] > 0:
        mid_ai = np.sqrt(lines["L1"]['ridge'][0]*lines["L1"]['end'][0])
        mid_gflops = lines["L1"]['ridge'][1]
        annotation = go.layout.Annotation(
            x=math.log10(mid_ai),
            y=math.log10(mid_gflops),
            text=f'FP FMA {ISA} Peak: {values[5]} GFLOP/s',
            showarrow=False,
            font=dict(
                color=colors[0],
                size=12,
            ),
            align="center",
            bgcolor="white",
            bordercolor=colors[0],
            borderwidth=1,
            textangle=0,
            name=f"FP_FMA_{name_suffix}"
            )
        
    if cache_level == "FP" and values[4] > 0:
        mid_ai = np.sqrt(lines["L1"]['ridge'][0]*lines["L1"]['end'][0])
        mid_gflops = values[4]
        annotation = go.layout.Annotation(
            x=math.log10(mid_ai),
            y=math.log10(mid_gflops),
            text=f'FP {ISA} Peak: {values[4]} GFLOP/s',
            showarrow=False,
            font=dict(
                color=colors[0],
                size=12,
            ),
            align="center",
            bgcolor="white",
            bordercolor=colors[0],
            borderwidth=1,
            textangle=0,
            name=f"FP_{name_suffix}"
            )  

    return annotation


def build_total_tooltip_text(name_app, threads_app, totals, total_FP_inst, total_mem_inst):
    lines = [f'{name_app} Total</b><br>Extra Details</b><br>   Threads: {threads_app}']

    metrics = {
        'Scalar Flops': totals["Intel_FP_Scalar_SP"] + totals["Intel_FP_Scalar_DP"],
        'SSE Flops': totals["Intel_FP_SSE_SP"] + totals["Intel_FP_SSE_DP"],
        'AVX2 Flops': totals["Intel_FP_AVX2_SP"] + totals["Intel_FP_AVX2_DP"],
        'AVX512 Flops': totals["Intel_FP_AVX512_SP"] + totals["Intel_FP_AVX512_DP"],
        'SP Flops': (totals["Intel_FP_Scalar_SP"] + totals["Intel_FP_SSE_SP"] +
                     totals["Intel_FP_AVX2_SP"] + totals["Intel_FP_AVX512_SP"]),
        'DP Flops': (totals["Intel_FP_Scalar_DP"] + totals["Intel_FP_SSE_DP"] +
                     totals["Intel_FP_AVX2_DP"] + totals["Intel_FP_AVX512_DP"]),
        'Loads': totals["Intel_Loads"],
        'Stores': totals["Intel_Stores"]
    }

    for label, value in metrics.items():
        if value != 0:
            if 'Flops' in label:
                percentage = custom_round((value / total_FP_inst) * 100, 1)
                value_formatted = f"{value:.2e}"
                lines.append(f'</b><br>   {label}: {value_formatted} ({percentage}%)')
            elif label in ['Loads', 'Stores']:
                percentage = custom_round((value / total_mem_inst) * 100, 1)
                value_formatted = f"{value:.2e}"
                lines.append(f'</b><br>   {label}: {value_formatted} ({percentage}%)')

    tooltip_text = '</b>'.join(lines)
    return tooltip_text

def build_timestamp_tooltip_text(scalar, sse, avx2, avx512, dp, load, timestamp_label, thread_ID, duration, paraver_value, paraver_label, window_name=None):
    metrics = [
                    ('Scalar Flops', scalar),
                    ('SSE Flops', sse),
                    ('AVX2 Flops', avx2),
                    ('AVX512 Flops', avx512),
                    ('SP Flops', 100-dp),
                    ('DP Flops', dp),
                    ('Loads', load),
                    ('Stores', 100-load)
                ]
    tooltip_lines = [f'Timestamp: {timestamp_label}']
    tooltip_lines.append(f'</b><br>   Thread: {thread_ID}</b><br>   Duration(us): {duration}')
    tooltip_lines.append('</b><br><b>Extra Details</b>')

    for label, value in metrics:
        if value > 0.1:
            rounded_value = custom_round(value, 1)
            tooltip_lines.append(f'</b><br>   {label}: {rounded_value}%')
    
    if window_name:
        tooltip_lines.append('</b><br><b>Paraver Data</b>')
        tooltip_lines.append(f'</b><br>   Window: {window_name}')
        tooltip_lines.append(f'</b><br>   Value: {paraver_value}')
        tooltip_lines.append(f'</b><br>   Label: {paraver_label}')

    tooltip_text = ''.join(tooltip_lines)
    return tooltip_text

def hsv_to_rgb(h, s, v):
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i = i % 6

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    elif i == 5:
        r, g, b = v, p, q

    return int(r*255), int(g*255), int(b*255)

def hash_to_color(name):
    isa_hash = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16)
    hue = isa_hash % 361
    saturation = 0.8
    value = 0.9
    return hsv_to_rgb(hue/360.0, saturation, value)


def blend_rgb(weights, color_dict, return_rgb):
    if not weights:
        return "#000000"
    total = sum(weights.values())
    normalized = {k: v/total for k, v in weights.items()}

    r = g = b = 0
    for k, w in normalized.items():
        cr, cg, cb = color_dict[k]
        r += cr * w
        g += cg * w
        b += cb * w

    r = int(round(r))
    g = int(round(g))
    b = int(round(b))
    if return_rgb:
        return r, g, b
    else:
        return f"#{r:02x}{g:02x}{b:02x}"

def blend_colors(scalar, sse, avx2, avx512, dp, load, thread_ID, color_radio, return_rgb):
    if color_radio == "ISA":
        weights = {
            "scalar": scalar,
            "sse": sse,
            "avx2": avx2,
            "avx512": avx512
        }
        active = {k: v for k, v in weights.items() if v > 0}
        if not active:
            if return_rgb:
                return 0,0,0
            else:
                return "#000000"

        isa_colors = {isa: color_map[intel_ISA_colors[isa]] for isa in active}
        return blend_rgb(active, isa_colors, return_rgb)

    elif color_radio == "Precision":
        weights = {}
        if dp > 0:
            weights["dp"] = dp
            weights["sp"] = 100 - dp

        if not weights:
            if return_rgb:
                return 0,0,0
            else:
                return "#000000"
        return blend_rgb(weights, precision_color_map, return_rgb)

    elif color_radio == "LD/ST Percentage":
        weights = {}
        if load > 0:
            weights["load"] = load
            weights["store"] = 100 - load

        if not weights:
            if return_rgb:
                return 0,0,0
            else:
                return "#000000"

        return blend_rgb(weights, loadstore_color_map, return_rgb)

    elif color_radio == "Thread ID":
        r, g, b = hash_to_color(str(thread_ID))
        return f"#{r:02x}{g:02x}{b:02x}"

    else:
        if return_rgb:
            return 0,0,0
        else:
            return "#000000"
    

def group_consecutive_by_rgb(color_map_df):
    grouped_data = []
    current_group = []
    
    for i in range(len(color_map_df)):
        row = color_map_df.iloc[i]

        if not current_group:
            current_group.append(row)
        else:
            prev_row = current_group[-1]
            if (row["r"], row["g"], row["b"]) == (prev_row["r"], prev_row["g"], prev_row["b"]):
                current_group.append(row)
            else:
                grouped_data.append(process_group(current_group))
                current_group = [row]

    if current_group:
        grouped_data.append(process_group(current_group))

    return pd.DataFrame(grouped_data)

def process_group(group_rows):
    if len(group_rows) == 1:
        ratio = group_rows[0]["Load/Store_ratio"]
        ratio_str = group_rows[0]["Load/Store_ratio_string"]
    else:
        min_ratio = min(r["Load/Store_ratio"] for r in group_rows)
        max_ratio = max(r["Load/Store_ratio"] for r in group_rows)
        ratio = f"{min_ratio:.6f}-{max_ratio:.6f}"
        ratio_str = f"LD/ST: {ratio}"

    return {
        "Load/Store_ratio": ratio,
        "Load/Store_ratio_string": ratio_str,
        "r": group_rows[0]["r"],
        "g": group_rows[0]["g"],
        "b": group_rows[0]["b"]
    }


def format_ld_st_csv(color_map_df, output_path):
    with open(output_path, 'w') as f:
        for _, row in color_map_df.iterrows():
            ratio = row["percentage"]
            label = row["percentage_string"]
            r, g, b = row["r"], row["g"], row["b"]
            ratio_str = f"{ratio}"

            line = f"{ratio_str} \"{label}\",{r},{g},{b}\n"
            f.write(line)