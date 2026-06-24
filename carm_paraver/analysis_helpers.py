from __future__ import annotations

import datetime
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

BLEND_COLOR_MODES = ("ISA", "Precision", "LD/ST Percentage", "Thread ID")
ROOFLINE_X_MIN_DEFAULT = 1.0 / 256.0


class WindowMode(Enum):
    CODE = "window_in_code_mode"
    GRADIENT = "window_in_null_gradient_mode"


ROOFLINE_X_MAX_DEFAULT = 256.0


def resolve_toggle_enabled(n_clicks: int, offset: int) -> bool:
    """Map click parity to enabled/disabled state.

    Current UI logic treats even parity as enabled and odd parity as disabled.
    """
    return (n_clicks + offset) % 2 == 0


def resolve_analysis_paraver_toggles(
    mask_button: int,
    accum_button: int,
    paraver_color_button: int,
    mask_button_offset: int,
    ac_button_offset: int,
    color_button_offset: int,
) -> tuple[bool, bool, bool]:
    """Resolve the three Paraver-related toggle states used by analysis()."""
    if mask_button_offset == -1:
        use_paraver_mask = False
    else:
        use_paraver_mask = resolve_toggle_enabled(mask_button, mask_button_offset)

    use_accumulate = resolve_toggle_enabled(accum_button, ac_button_offset)
    use_paraver_colors = resolve_toggle_enabled(paraver_color_button, color_button_offset)
    return use_paraver_mask, use_accumulate, use_paraver_colors


def filter_base_and_intel_data(
    base_statistics_df: pd.DataFrame,
    intel_statistics_df: pd.DataFrame,
    lower_filter: float,
    duration_filter: float,
    use_paraver_mask: bool,
    is_valid_paraver_value: Callable[[object], bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply shared threshold/mask filtering and keep base/intel rows aligned."""
    lower_filter = float(lower_filter)
    duration_filter = float(duration_filter)

    base_filter = (
        (base_statistics_df["Arithmetic_Intensity"] >= lower_filter)
        & (base_statistics_df["GFLOPS"] >= lower_filter)
        & (base_statistics_df["Duration"] >= duration_filter)
    )
    if use_paraver_mask:
        base_filter = base_filter & base_statistics_df["Paraver_Value"].apply(is_valid_paraver_value)

    filtered_base = base_statistics_df[base_filter]
    filtered_intel = intel_statistics_df.loc[filtered_base.index]
    return filtered_base.reset_index(drop=True), filtered_intel.reset_index(drop=True)


@dataclass
class TimestampPlotData:
    extra_average: str
    ai_mean: list[float]
    gflops_mean: list[float]
    timestamps_grouped: list[Any]
    durations: list[Any]
    thread_ids: list[Any]
    reds: list[Any]
    greens: list[Any]
    blues: list[Any]
    pvalues: list[Any]
    plabels: list[Any]
    scalar_perc: list[Any]
    sse_perc: list[Any]
    avx2_perc: list[Any]
    avx512_perc: list[Any]
    dp_perc: list[Any]
    load_perc: list[Any]


@dataclass
class TimestampPoint:
    ai_value: Any
    gflops_value: Any
    timestamp_label: Any
    scalar_perc: Any
    sse_perc: Any
    avx2_perc: Any
    avx512_perc: Any
    dp_perc: Any
    load_perc: Any
    thread_id: Any
    duration: Any
    red: Any
    green: Any
    blue: Any
    pvalue: Any
    plabel: Any


@dataclass
class TimestampSeries:
    points: list[TimestampPoint]
    extra_average: str

    @property
    def count(self) -> int:
        return len(self.points)


@dataclass
class TimestampColorContext:
    use_paraver_colors: bool
    color_radio: str
    index: int
    n_points: int
    start_color: tuple[int, int, int]
    end_color: tuple[int, int, int]
    blend_colors_fn: Callable[..., Any]
    interpolate_color_fn: Callable[..., Any]


@dataclass(frozen=True)
class SliderMarksConfig:
    group_value: int
    max_marks: int
    reset_trigger_ids: frozenset[str]
    default_range_fn: Callable[[int, int], list[int]]


def build_slider_group_labels(segments: list[Any], group_value: int) -> list[str]:
    """Group timestamp segments into slider labels."""
    grouped_segments: list[str] = []
    for i in range(0, len(segments), group_value):
        current_group = segments[i : i + group_value]
        if len(current_group) > 1:
            grouped_segments.append(f"{current_group[0]}...{current_group[-1]}")
        else:
            grouped_segments.append(f"{current_group[0]}")
    return grouped_segments


def _apply_alternating_slider_styles(marks: dict[int, dict[str, Any]]) -> None:
    for mark_index, mark in marks.items():
        if mark_index % 2 == 0:
            mark["style"] = {"margin-top": "0px"}
        else:
            mark["style"] = {"margin-top": "-35px"}


def _style_two_slider_marks(filtered_marks: dict[int, dict[str, Any]]) -> None:
    sorted_keys = sorted(filtered_marks.keys())
    if len(sorted_keys) == 2:
        filtered_marks[sorted_keys[0]]["style"] = {"margin-top": "0px"}
        filtered_marks[sorted_keys[1]]["style"] = {"margin-top": "-35px"}


def _marks_for_current_values(
    current_values: Any,
    marks: dict[int, dict[str, Any]],
    grouped_segments: list[str],
) -> dict[int, dict[str, Any]]:
    if isinstance(current_values, list) and len(current_values) == 2 and len(grouped_segments) > 1:
        return {
            current_values[0]: marks[current_values[0]],
            current_values[1]: marks[current_values[1]],
        }
    if isinstance(current_values, list) and len(current_values) == 2:
        return {current_values[0]: marks[current_values[0]]}
    return {current_values: marks[current_values]}


def generate_slider_marks_state(
    segments: list[Any],
    current_values: Any,
    triggered_id: str,
    config: SliderMarksConfig,
) -> tuple[dict[int, dict[str, Any]], int, Any]:
    """Build slider marks/max/value while preserving callback-specific behavior."""
    n_segments = len(segments)
    if n_segments == 0:
        safe_marks = {0: {"label": "No data", "style": {"margin-top": "0px"}}}
        return safe_marks, 0, [0, 0]

    grouped_segments = build_slider_group_labels(segments, config.group_value)
    marks: dict[int, dict[str, Any]] = {i: {"label": value} for i, value in enumerate(grouped_segments)}
    max_index = len(grouped_segments) - 1

    if max_index <= config.max_marks:
        _apply_alternating_slider_styles(marks)

    should_reset_range = (
        triggered_id == "input-number" or current_values is None or triggered_id in config.reset_trigger_ids
    )
    if should_reset_range:
        initial_range = config.default_range_fn(max_index, len(grouped_segments))

        if max_index > config.max_marks:
            filtered_marks = {
                initial_range[0]: marks[initial_range[0]],
                initial_range[1]: marks[initial_range[1]],
            }
            _style_two_slider_marks(filtered_marks)
            return filtered_marks, max_index, initial_range

        return marks, max_index, initial_range

    filtered_marks = _marks_for_current_values(current_values, marks, grouped_segments)
    if max_index > config.max_marks:
        _style_two_slider_marks(filtered_marks)
        return filtered_marks, max_index, current_values

    return marks, max_index, current_values


def _column_or_default(df: pd.DataFrame, column: str, default_value: Any) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default_value] * len(df), index=df.index)


def prepare_timestamp_plot_data(
    df_filter: pd.DataFrame,
    df_intel_filter2: pd.DataFrame,
    average: bool,
    timestamps_grouper: int,
    timestamp_start_index: int,
    use_accumulate: bool,
) -> tuple[TimestampPlotData, None | float, None | float]:
    """Build timestamp plotting arrays for average/raw/accumulate modes."""
    local_df = df_filter.copy()
    local_intel_df = df_intel_filter2.copy()

    if average and timestamps_grouper > 1:
        extra_average = " Averaged "
        local_df["group"] = ((local_df.index - timestamp_start_index) // timestamps_grouper).astype(int)
        local_df["ai"] = local_df["Arithmetic_Intensity"]
        local_df["gflops"] = local_df["GFLOPS"]
        local_df["timestamp"] = local_df["Timestamp"]
        grouped = local_df.groupby("group")

        ai_mean = grouped["ai"].mean().reset_index(drop=True).tolist()
        gflops_mean = grouped["gflops"].mean().reset_index(drop=True).tolist()
        timestamps_grouped = (
            grouped["timestamp"]
            .apply(lambda x: f"{x.iloc[0]}" if len(x) == 1 else f"{x.iloc[0]}...{x.iloc[-1]}")
            .reset_index(drop=True)
            .tolist()
        )

        durations = grouped["Duration"].sum().reset_index(drop=True).tolist()
        thread_ids = grouped["ThreadID"].first().reset_index(drop=True).tolist()
        reds = grouped["R"].first().reset_index(drop=True).tolist() if "R" in local_df.columns else [0] * len(ai_mean)
        greens = grouped["G"].first().reset_index(drop=True).tolist() if "G" in local_df.columns else [0] * len(ai_mean)
        blues = grouped["B"].first().reset_index(drop=True).tolist() if "B" in local_df.columns else [0] * len(ai_mean)
        pvalues = (
            grouped["Paraver_Value"].first().reset_index(drop=True).tolist()
            if "Paraver_Value" in local_df.columns
            else [0] * len(ai_mean)
        )
        plabels = (
            grouped["Paraver_Label"].first().reset_index(drop=True).tolist()
            if "Paraver_Label" in local_df.columns
            else [""] * len(ai_mean)
        )

        local_intel_df["group"] = ((local_intel_df.index - timestamp_start_index) // timestamps_grouper).astype(int)
        grouped_intel = local_intel_df.groupby("group")

        scalar_sp_mean = grouped_intel["Intel_FP_Scalar_SP"].mean().reset_index(drop=True)
        scalar_dp_mean = grouped_intel["Intel_FP_Scalar_DP"].mean().reset_index(drop=True)
        sse_sp_mean = grouped_intel["Intel_FP_SSE_SP"].mean().reset_index(drop=True)
        sse_dp_mean = grouped_intel["Intel_FP_SSE_DP"].mean().reset_index(drop=True)
        avx2_sp_mean = grouped_intel["Intel_FP_AVX2_SP"].mean().reset_index(drop=True)
        avx2_dp_mean = grouped_intel["Intel_FP_AVX2_DP"].mean().reset_index(drop=True)
        avx512_sp_mean = grouped_intel["Intel_FP_AVX512_SP"].mean().reset_index(drop=True)
        avx512_dp_mean = grouped_intel["Intel_FP_AVX512_DP"].mean().reset_index(drop=True)
        dp_mean = grouped_intel["Intel_FP_DP"].mean().reset_index(drop=True)
        fp_total_mean = grouped_intel["Intel_FP_Total"].mean().reset_index(drop=True)
        load_mean = grouped_intel["Intel_Load"].mean().reset_index(drop=True)
        store_mean = grouped_intel["Intel_Store"].mean().reset_index(drop=True)

        scalar_perc = (((scalar_sp_mean + scalar_dp_mean) / fp_total_mean) * 100).tolist()
        sse_perc = (((sse_sp_mean + sse_dp_mean) / fp_total_mean) * 100).tolist()
        avx2_perc = (((avx2_sp_mean + avx2_dp_mean) / fp_total_mean) * 100).tolist()
        avx512_perc = (((avx512_sp_mean + avx512_dp_mean) / fp_total_mean) * 100).tolist()
        dp_perc = ((dp_mean / fp_total_mean) * 100).tolist()
        load_perc = ((load_mean / (load_mean + store_mean)) * 100).tolist()
    else:
        if use_accumulate:
            local_df = local_df.sort_values(by=["ThreadID", "Timestamp"]).reset_index(drop=True)
            local_intel_df = local_intel_df.sort_values(by=["ThreadID", "Timestamp"]).reset_index(drop=True)

            local_df["label_shift"] = local_df.groupby("ThreadID")["Paraver_Label"].shift()
            local_df["label_changed"] = local_df["Paraver_Label"] != local_df["label_shift"]
            local_intel_df["label_shift"] = local_intel_df.groupby("ThreadID")["Paraver_Label"].shift()
            local_intel_df["label_changed"] = local_intel_df["Paraver_Label"] != local_intel_df["label_shift"]

            local_df["group"] = local_df.groupby("ThreadID")["label_changed"].cumsum()
            local_intel_df["group"] = local_intel_df.groupby("ThreadID")["label_changed"].cumsum()

            local_df = (
                local_df.groupby(["ThreadID", "group"])
                .agg(
                    {
                        "ThreadID": "first",
                        "Timestamp": lambda x: f"{x.min()}" if x.min() == x.max() else f"{x.min()} - {x.max()}",
                        "Duration": "sum",
                        "Paraver_Label": "first",
                        "Paraver_Value": "first",
                        "R": "first",
                        "G": "first",
                        "B": "first",
                        "FLOP": "sum",
                        "Bytes": "sum",
                    }
                )
                .reset_index(drop=True)
            )

            local_intel_df = (
                local_intel_df.groupby(["ThreadID", "group"])
                .agg(
                    {
                        "ThreadID": "first",
                        "Timestamp": "min",
                        "Intel_FP_Scalar_SP": "sum",
                        "Intel_FP_Scalar_DP": "sum",
                        "Intel_FP_SSE_SP": "sum",
                        "Intel_FP_SSE_DP": "sum",
                        "Intel_FP_AVX2_SP": "sum",
                        "Intel_FP_AVX2_DP": "sum",
                        "Intel_FP_AVX512_SP": "sum",
                        "Intel_FP_AVX512_DP": "sum",
                        "Intel_FP_SP": "sum",
                        "Intel_FP_DP": "sum",
                        "Intel_FP_Total": "sum",
                        "Intel_Load": "sum",
                        "Intel_Store": "sum",
                        "Paraver_Label": "first",
                    }
                )
                .reset_index(drop=True)
            )

            local_df["GFLOPS"] = local_df["FLOP"] / (local_df["Duration"] * 1e3)
            local_df["Bandwidth"] = local_df["Bytes"] / local_df["Duration"]
            local_df["Arithmetic_Intensity"] = local_df["FLOP"] / local_df["Bytes"]

        extra_average = " "
        ai_mean = local_df["Arithmetic_Intensity"].tolist()
        gflops_mean = local_df["GFLOPS"].tolist()
        timestamps_grouped = local_df["Timestamp"].tolist()
        durations = local_df["Duration"].tolist()
        thread_ids = local_df["ThreadID"].tolist()
        reds = _column_or_default(local_df, "R", 0).tolist()
        greens = _column_or_default(local_df, "G", 0).tolist()
        blues = _column_or_default(local_df, "B", 0).tolist()
        pvalues = _column_or_default(local_df, "Paraver_Value", 0).tolist()
        plabels = _column_or_default(local_df, "Paraver_Label", "").tolist()

        scalar_perc = (
            (
                (local_intel_df["Intel_FP_Scalar_SP"] + local_intel_df["Intel_FP_Scalar_DP"])
                / local_intel_df["Intel_FP_Total"]
            )
            * 100
        ).tolist()
        sse_perc = (
            ((local_intel_df["Intel_FP_SSE_SP"] + local_intel_df["Intel_FP_SSE_DP"]) / local_intel_df["Intel_FP_Total"])
            * 100
        ).tolist()
        avx2_perc = (
            (
                (local_intel_df["Intel_FP_AVX2_SP"] + local_intel_df["Intel_FP_AVX2_DP"])
                / local_intel_df["Intel_FP_Total"]
            )
            * 100
        ).tolist()
        avx512_perc = (
            (
                (local_intel_df["Intel_FP_AVX512_SP"] + local_intel_df["Intel_FP_AVX512_DP"])
                / local_intel_df["Intel_FP_Total"]
            )
            * 100
        ).tolist()
        dp_perc = ((local_intel_df["Intel_FP_DP"] / local_intel_df["Intel_FP_Total"]) * 100).tolist()
        if use_accumulate:
            load_perc = (
                (local_intel_df["Intel_Load"] / (local_intel_df["Intel_Load"] + local_intel_df["Intel_Store"])) * 100
            ).tolist()
        else:
            load_perc = local_intel_df["Intel_Load_Percent"].tolist()

    timestamp_plot_data = TimestampPlotData(
        extra_average=extra_average,
        ai_mean=ai_mean,
        gflops_mean=gflops_mean,
        timestamps_grouped=timestamps_grouped,
        durations=durations,
        thread_ids=thread_ids,
        reds=reds,
        greens=greens,
        blues=blues,
        pvalues=pvalues,
        plabels=plabels,
        scalar_perc=scalar_perc,
        sse_perc=sse_perc,
        avx2_perc=avx2_perc,
        avx512_perc=avx512_perc,
        dp_perc=dp_perc,
        load_perc=load_perc,
    )

    min_ai = min(ai_mean) if ai_mean else None
    min_gflops = min(gflops_mean) if gflops_mean else None
    return timestamp_plot_data, min_ai, min_gflops


def timestamp_plot_data_to_series(timestamp_data: TimestampPlotData) -> TimestampSeries:
    """Convert parallel timestamp arrays into a point-object series."""
    points = [
        TimestampPoint(
            ai_value=timestamp_data.ai_mean[i],
            gflops_value=timestamp_data.gflops_mean[i],
            timestamp_label=timestamp_data.timestamps_grouped[i],
            scalar_perc=timestamp_data.scalar_perc[i],
            sse_perc=timestamp_data.sse_perc[i],
            avx2_perc=timestamp_data.avx2_perc[i],
            avx512_perc=timestamp_data.avx512_perc[i],
            dp_perc=timestamp_data.dp_perc[i],
            load_perc=timestamp_data.load_perc[i],
            thread_id=timestamp_data.thread_ids[i],
            duration=timestamp_data.durations[i],
            red=timestamp_data.reds[i],
            green=timestamp_data.greens[i],
            blue=timestamp_data.blues[i],
            pvalue=timestamp_data.pvalues[i],
            plabel=timestamp_data.plabels[i],
        )
        for i in range(len(timestamp_data.ai_mean))
    ]
    return TimestampSeries(points=points, extra_average=timestamp_data.extra_average)


def iter_timestamp_points(series: TimestampSeries):
    """Yield timestamp points in plotting order."""
    return iter(series.points)


def get_timestamp_point(series: TimestampSeries, index: int) -> TimestampPoint:
    """Return a timestamp point by index."""
    return series.points[index]


def prepare_timestamp_series(
    df_filter: pd.DataFrame,
    df_intel_filter2: pd.DataFrame,
    average: bool,
    timestamps_grouper: int,
    timestamp_start_index: int,
    use_accumulate: bool,
) -> tuple[TimestampSeries, None | float, None | float]:
    """Build object-based timestamp plotting data with parity to plot-array mode."""
    timestamp_data, min_ai, min_gflops = prepare_timestamp_plot_data(
        df_filter,
        df_intel_filter2,
        average,
        timestamps_grouper,
        timestamp_start_index,
        use_accumulate,
    )
    return timestamp_plot_data_to_series(timestamp_data), min_ai, min_gflops


def should_plot_timestamp_point(use_paraver_mask: bool, point_or_pvalue: Any) -> bool:
    """Return whether a timestamp point should be plotted for current mask mode."""
    if not use_paraver_mask:
        return True

    pvalue = point_or_pvalue.pvalue if isinstance(point_or_pvalue, TimestampPoint) else point_or_pvalue
    try:
        return float(pvalue) > 0
    except (TypeError, ValueError):
        return False


def _select_timestamp_color_scalar(
    use_paraver_colors: bool,
    color_radio: str,
    scalar_perc: Any,
    sse_perc: Any,
    avx2_perc: Any,
    avx512_perc: Any,
    dp_perc: Any,
    load_perc: Any,
    thread_id: Any,
    red: Any,
    green: Any,
    blue: Any,
    index: int,
    n_points: int,
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
    blend_colors_fn: Callable[..., Any],
    interpolate_color_fn: Callable[..., Any],
) -> Any:
    """Select marker color for timestamp points in both interval and regular modes."""
    if use_paraver_colors or color_radio == "Paraver":
        return f"rgb({red},{green},{blue})"

    if color_radio in BLEND_COLOR_MODES:
        return blend_colors_fn(
            scalar_perc,
            sse_perc,
            avx2_perc,
            avx512_perc,
            dp_perc,
            load_perc,
            thread_id,
            color_radio,
            False,
        )

    ratio = (index / n_points) if n_points > 0 else 0
    return interpolate_color_fn(start_color, end_color, ratio)


def select_timestamp_color(*args, **kwargs) -> Any:
    """Select marker color from either object/context or scalar legacy inputs."""
    if args and isinstance(args[0], TimestampPoint):
        point = args[0]
        color_ctx = args[1] if len(args) > 1 else kwargs.get("color_ctx")
        if color_ctx is None:
            raise ValueError("color context is required when using point-based color selection")

        return _select_timestamp_color_scalar(
            color_ctx.use_paraver_colors,
            color_ctx.color_radio,
            point.scalar_perc,
            point.sse_perc,
            point.avx2_perc,
            point.avx512_perc,
            point.dp_perc,
            point.load_perc,
            point.thread_id,
            point.red,
            point.green,
            point.blue,
            color_ctx.index,
            color_ctx.n_points,
            color_ctx.start_color,
            color_ctx.end_color,
            color_ctx.blend_colors_fn,
            color_ctx.interpolate_color_fn,
        )

    if "point" in kwargs:
        point = kwargs["point"]
        color_ctx = kwargs.get("color_ctx")
        if not isinstance(point, TimestampPoint):
            raise TypeError("point must be a TimestampPoint")
        if color_ctx is None:
            raise ValueError("color context is required when using point-based color selection")
        return select_timestamp_color(point, color_ctx)

    return _select_timestamp_color_scalar(*args, **kwargs)


def build_timestamp_scatter_trace(
    ai_value: Any,
    gflops_value: Any,
    name: str,
    dot_size: int,
    color: Any,
    tooltip_text: str,
    showlegend: bool,
    legendgroup: None | str = None,
) -> dict[str, Any]:
    """Build a consistent Scatter trace payload for timestamp points."""
    trace_payload: dict[str, Any] = {
        "x": [ai_value],
        "y": [gflops_value],
        "mode": "markers",
        "name": name,
        "marker": {"size": dot_size, "color": color},
        "showlegend": showlegend,
        "text": [tooltip_text],
        "hovertemplate": "<b>%{text}</b><br>(%{x}, %{y})<br><extra></extra>",
    }
    if legendgroup is not None:
        trace_payload["legendgroup"] = legendgroup
    return trace_payload


def resolve_timestamp_legend_state(
    use_paraver_colors: bool,
    plabel: str,
    seen_paraver_labels: set[str],
    first_non_paraver: bool,
    window_mode: WindowMode = WindowMode.CODE,
) -> tuple[bool, str, bool]:
    """Resolve showlegend/display label and next non-paraver state for timestamp points.

    When using Paraver colors in code mode, each unique plabel gets its own legend entry.
    In gradient mode (or when using CARM colors), only the first point gets a legend entry.
    """
    if use_paraver_colors and window_mode == WindowMode.CODE:
        showlegend = plabel not in seen_paraver_labels
        seen_paraver_labels.add(plabel)
        return showlegend, plabel, first_non_paraver

    return first_non_paraver, "", False


def normalize_roofline_values(values: list[Any], normalize: bool, threads_value: Any) -> list[Any]:
    """Normalize first six roofline values by thread count when requested."""
    normalized = list(values)
    if not normalize:
        return normalized

    try:
        threads = float(threads_value)
    except (TypeError, ValueError):
        threads = 1.0

    if threads <= 0:
        return normalized

    for i in range(min(6, len(normalized))):
        try:
            normalized[i] = normalized[i] / threads
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return normalized


def calculate_roofline_profile(
    filtered_df: pd.DataFrame,
    normalize: bool,
    smallest_ai: float,
    calculate_roofline_fn: Callable[[list[Any], float], dict],
) -> None | tuple[list[Any], str, dict, float, float]:
    """Return normalized roofline values and derived line metrics for one profile row."""
    if filtered_df.empty:
        return None

    row = filtered_df.iloc[-1]
    raw_values = row[["L1", "L2", "L3", "DRAM", "FP", "FP_FMA", "FPInst"]].tolist()
    values = normalize_roofline_values(raw_values, normalize, row.get("Threads", 1.0))
    lines = calculate_roofline_fn(values, smallest_ai / 5)
    top_flops = lines["L1"]["ridge"][1]
    min_gflops = lines["DRAM"]["start"][1]
    isa = str(row["ISA"])
    return values, isa, lines, top_flops, min_gflops


def resolve_roofline_x_bounds(smallest_ai: float) -> tuple[float, float]:
    """Return default X-axis bounds used for roofline plotting.

    Keeps existing behavior while replacing hard-coded constants with a named helper.
    """
    return min(ROOFLINE_X_MIN_DEFAULT, smallest_ai / 5), ROOFLINE_X_MAX_DEFAULT


def resolve_roofline_angle_bounds(
    xaxis_range: None | list[float],
    yaxis_range: None | list[float],
    smallest_ai: float,
    dram_start_gflops: float,
    peak_gflops: float,
    y_min_scale: float = 0.5,
    y_max_scale: float = 2.0,
) -> tuple[float, float, float, float]:
    """Resolve linear-space axis bounds for annotation-angle calculations."""
    if xaxis_range:
        x_min_angle = 10 ** xaxis_range[0]
        x_max_angle = 10 ** xaxis_range[1]
    else:
        x_min_angle, x_max_angle = resolve_roofline_x_bounds(smallest_ai)

    if yaxis_range:
        y_min_angle = 10 ** yaxis_range[0]
        y_max_angle = 10 ** yaxis_range[1]
    else:
        y_min_angle = dram_start_gflops * y_min_scale
        y_max_angle = peak_gflops * y_max_scale

    return x_min_angle, x_max_angle, y_min_angle, y_max_angle


def filter_roofline_df_by_query(df: pd.DataFrame, query: None | str) -> pd.DataFrame:
    """Apply a pandas query when provided; otherwise return the original dataframe."""
    return df.query(query) if query else df


def fallback_roofline_df_by_isa(
    df: pd.DataFrame,
    intel_isa_order: list[str],
    construct_query_fn: Callable[..., str | None],
    filters: dict[str, Any],
) -> pd.DataFrame:
    """Try ISA fallbacks in order and return the first non-empty filtered dataframe."""
    for isa in intel_isa_order:
        fallback_filters = dict(filters)
        fallback_filters["ISA"] = isa
        fallback_query = construct_query_fn(fallback_filters)
        filtered = filter_roofline_df_by_query(df, fallback_query)
        if not filtered.empty:
            return filtered
    return pd.DataFrame()


def should_reset_annotations_for_lines(
    new_lines: dict,
    old_lines: dict,
    trigger_id: str,
    require_existing_old_lines: bool = False,
) -> bool:
    """Determine whether line changes should clear annotations for the current trigger."""
    if trigger_id == "interval-component":
        return False
    if require_existing_old_lines and len(old_lines) == 0:
        return False
    return new_lines != old_lines


def resolve_timestamp_slice_bounds(
    timestamps_range: list[int],
    timestamps_max_range: list[int],
    timestamps_grouper: int,
) -> tuple[int, int]:
    """Compute start/end (exclusive) bounds for timestamp dataframe slicing."""
    range_start = timestamps_range[0] * timestamps_grouper + timestamps_max_range[0]
    range_end = timestamps_range[1] * timestamps_grouper + timestamps_max_range[0]

    if timestamps_range[1] == 0:
        range_end = timestamps_grouper - 1

    max_end_exclusive = timestamps_max_range[1] + 1
    if (range_end + timestamps_grouper) > max_end_exclusive:
        return range_start, max_end_exclusive

    return range_start, range_end + timestamps_grouper


def infer_effective_isa_from_timestamp_columns(
    columns_to_check: pd.DataFrame,
    lower_filter: float,
    current_isa: str | None,
    has_thread_id_column: bool,
) -> str:
    """Infer ISA from timestamp counter columns when no ISA is already selected."""
    threshold = float(lower_filter)
    if threshold > 0:
        positive_columns = columns_to_check.columns[(columns_to_check >= threshold).any()].tolist()
    else:
        positive_columns = columns_to_check.columns[(columns_to_check > 0).any()].tolist()

    if has_thread_id_column:
        positive_columns.append("ThreadID")

    if current_isa is not None:
        return current_isa
    if any("AVX512" in col for col in positive_columns):
        return "avx512"
    if any("AVX2" in col for col in positive_columns):
        return "avx2"
    if any("SSE" in col for col in positive_columns):
        return "sse"
    return "scalar"


def resolve_interval_point_index_and_legend(n_intervals: int, data_points: int) -> tuple[int, bool]:
    """Compute interval index and first-legend flag for animation plotting."""
    if n_intervals == 0:
        indexer = data_points - 1
    else:
        indexer = n_intervals - 1

    return indexer, indexer == 0


def get_timestamp_plot_point(timestamp_data: TimestampPlotData, index: int) -> tuple[Any, ...]:
    """Return all point fields for a timestamp index in a stable order."""
    return (
        timestamp_data.ai_mean[index],
        timestamp_data.gflops_mean[index],
        timestamp_data.timestamps_grouped[index],
        timestamp_data.scalar_perc[index],
        timestamp_data.sse_perc[index],
        timestamp_data.avx2_perc[index],
        timestamp_data.avx512_perc[index],
        timestamp_data.dp_perc[index],
        timestamp_data.load_perc[index],
        timestamp_data.thread_ids[index],
        timestamp_data.durations[index],
        timestamp_data.reds[index],
        timestamp_data.greens[index],
        timestamp_data.blues[index],
        timestamp_data.pvalues[index],
        timestamp_data.plabels[index],
    )


def iter_timestamp_plot_rows(
    timestamp_data: TimestampPlotData,
):
    """Yield synchronized timestamp plotting rows from the point arrays."""
    return zip(
        timestamp_data.ai_mean,
        timestamp_data.gflops_mean,
        timestamp_data.timestamps_grouped,
        timestamp_data.scalar_perc,
        timestamp_data.sse_perc,
        timestamp_data.avx2_perc,
        timestamp_data.avx512_perc,
        timestamp_data.dp_perc,
        timestamp_data.load_perc,
        timestamp_data.thread_ids,
        timestamp_data.durations,
        timestamp_data.reds,
        timestamp_data.greens,
        timestamp_data.blues,
        timestamp_data.pvalues,
        timestamp_data.plabels,
        strict=False,
    )


def _build_timestamp_tooltip_args_scalar(
    scalar_perc: Any,
    sse_perc: Any,
    avx2_perc: Any,
    avx512_perc: Any,
    dp_perc: Any,
    load_perc: Any,
    timestamp_label: Any,
    thread_id: Any,
    duration: Any,
    pvalue: Any,
    plabel: Any,
    window_name: str,
) -> tuple[Any, ...]:
    """Build positional args for ut.build_timestamp_tooltip_text with stable field order."""
    return (
        scalar_perc,
        sse_perc,
        avx2_perc,
        avx512_perc,
        dp_perc,
        load_perc,
        timestamp_label,
        thread_id,
        duration,
        pvalue,
        plabel,
        window_name,
    )


def build_timestamp_tooltip_args(*args, **kwargs) -> tuple[Any, ...]:
    """Build tooltip args from either a point/window pair or legacy scalar fields."""
    if args and isinstance(args[0], TimestampPoint):
        point = args[0]
        if len(args) > 1:
            window_name = args[1]
        else:
            window_name = kwargs.get("window_name")
        if window_name is None:
            raise ValueError("window_name is required when using point-based tooltip args")
        return _build_timestamp_tooltip_args_scalar(
            point.scalar_perc,
            point.sse_perc,
            point.avx2_perc,
            point.avx512_perc,
            point.dp_perc,
            point.load_perc,
            point.timestamp_label,
            point.thread_id,
            point.duration,
            point.pvalue,
            point.plabel,
            window_name,
        )

    if "point" in kwargs:
        point = kwargs["point"]
        if not isinstance(point, TimestampPoint):
            raise TypeError("point must be a TimestampPoint")
        return build_timestamp_tooltip_args(point, kwargs.get("window_name"))

    return _build_timestamp_tooltip_args_scalar(*args, **kwargs)


def build_csv_metadata_line(
    prv_trace_path: str,
    time_unit: str,
    window_mode: WindowMode,
    vmin: float,
    vmax: float,
) -> str:
    """Format the Paraver-compatible CSV header line with timestamp and metadata."""
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"#{ts}:CSV:RUNAPP:{prv_trace_path}:{time_unit}:{window_mode.value}:{vmin}:{vmax}"


def sort_timestamp_df(df: pd.DataFrame, natural_sort_fn: Callable) -> pd.DataFrame:
    """Sort by ThreadID (natural order) then Timestamp."""
    return df.sort_values(
        ["ThreadID", "Timestamp"],
        key=lambda col: natural_sort_fn(col) if col.name == "ThreadID" else col,
    )


def write_csv_file(csv_df: pd.DataFrame, filepath: str, metadata_line: str) -> None:
    """Write the tab-separated CSV with Paraver header line."""
    with open(filepath, "w") as f:
        f.write(metadata_line + "\n")
        csv_df.to_csv(f, index=False, header=False, sep="\t")
