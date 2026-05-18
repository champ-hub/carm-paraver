import pandas as pd

from carm_paraver.analysis_helpers import (
    TimestampColorContext,
    TimestampPlotData,
    TimestampPoint,
    TimestampSeries,
    build_timestamp_scatter_trace,
    build_timestamp_tooltip_args,
    calculate_roofline_profile,
    fallback_roofline_df_by_isa,
    filter_base_and_intel_data,
    filter_roofline_df_by_query,
    get_timestamp_plot_point,
    infer_effective_isa_from_timestamp_columns,
    iter_timestamp_plot_rows,
    normalize_roofline_values,
    prepare_timestamp_plot_data,
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
    timestamp_plot_data_to_series,
)


def test_resolve_toggle_enabled_even_parity_is_enabled() -> None:
    assert resolve_toggle_enabled(0, 0) is True
    assert resolve_toggle_enabled(1, 1) is True


def test_resolve_toggle_enabled_odd_parity_is_disabled() -> None:
    assert resolve_toggle_enabled(1, 0) is False
    assert resolve_toggle_enabled(0, 1) is False


def test_resolve_analysis_paraver_toggles_with_mask_offset_disabled() -> None:
    use_mask, use_accumulate, use_colors = resolve_analysis_paraver_toggles(
        mask_button=3,
        accum_button=2,
        paraver_color_button=1,
        mask_button_offset=-1,
        ac_button_offset=0,
        color_button_offset=0,
    )

    assert use_mask is False
    assert use_accumulate is True
    assert use_colors is False


def test_filter_base_and_intel_data_without_paraver_mask() -> None:
    base_df = pd.DataFrame(
        {
            "Arithmetic_Intensity": [0.1, 2.0, 2.0],
            "GFLOPS": [2.0, 2.0, 0.1],
            "Duration": [2.0, 2.0, 2.0],
            "Paraver_Value": [0, 0, 0],
        },
        index=[10, 11, 12],
    )
    intel_df = pd.DataFrame({"Intel_FP_Total": [10, 11, 12]}, index=[10, 11, 12])

    filtered_base, filtered_intel = filter_base_and_intel_data(
        base_df,
        intel_df,
        lower_filter=1.0,
        duration_filter=1.0,
        use_paraver_mask=False,
        is_valid_paraver_value=lambda _: True,
    )

    assert filtered_base.shape[0] == 1
    assert filtered_intel.shape[0] == 1
    assert filtered_base.iloc[0]["Arithmetic_Intensity"] == 2.0
    assert filtered_intel.iloc[0]["Intel_FP_Total"] == 11


def test_filter_base_and_intel_data_with_paraver_mask() -> None:
    base_df = pd.DataFrame(
        {
            "Arithmetic_Intensity": [2.0, 2.0, 2.0],
            "GFLOPS": [2.0, 2.0, 2.0],
            "Duration": [2.0, 2.0, 2.0],
            "Paraver_Value": [1, 0, -3],
        },
        index=[20, 21, 22],
    )
    intel_df = pd.DataFrame({"Intel_FP_Total": [20, 21, 22]}, index=[20, 21, 22])

    filtered_base, filtered_intel = filter_base_and_intel_data(
        base_df,
        intel_df,
        lower_filter=1.0,
        duration_filter=1.0,
        use_paraver_mask=True,
        is_valid_paraver_value=lambda value: value > 0,
    )

    assert filtered_base.shape[0] == 1
    assert filtered_intel.shape[0] == 1
    assert filtered_base.iloc[0]["Paraver_Value"] == 1
    assert filtered_intel.iloc[0]["Intel_FP_Total"] == 20


def test_prepare_timestamp_plot_data_average_mode_provides_grouped_metadata() -> None:
    df_filter = pd.DataFrame(
        {
            "Arithmetic_Intensity": [1.0, 3.0, 2.0, 6.0],
            "GFLOPS": [10.0, 30.0, 20.0, 40.0],
            "Timestamp": [100, 101, 200, 201],
            "Duration": [4.0, 6.0, 1.0, 9.0],
            "ThreadID": ["1.1.1", "1.1.2", "1.1.1", "1.1.2"],
            "R": [10, 11, 12, 13],
            "G": [20, 21, 22, 23],
            "B": [30, 31, 32, 33],
            "Paraver_Value": [7, 8, 9, 10],
            "Paraver_Label": ["A", "A", "B", "B"],
        }
    )
    df_intel = pd.DataFrame(
        {
            "Intel_FP_Scalar_SP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_Scalar_DP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_SSE_SP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_SSE_DP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_AVX2_SP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_AVX2_DP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_AVX512_SP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_AVX512_DP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_DP": [2.0, 2.0, 2.0, 2.0],
            "Intel_FP_Total": [16.0, 16.0, 16.0, 16.0],
            "Intel_Load": [2.0, 2.0, 2.0, 2.0],
            "Intel_Store": [2.0, 2.0, 2.0, 2.0],
        }
    )

    result, min_ai, min_gflops = prepare_timestamp_plot_data(
        df_filter,
        df_intel,
        average=True,
        timestamps_grouper=2,
        timestamp_start_index=0,
        use_accumulate=False,
    )

    assert result.extra_average == " Averaged "
    assert len(result.ai_mean) == 2
    assert len(result.gflops_mean) == 2
    assert len(result.reds) == 2
    assert len(result.pvalues) == 2
    assert result.timestamps_grouped == ["100...101", "200...201"]
    assert result.durations == [10.0, 10.0]
    assert min_ai == 2.0
    assert min_gflops == 20.0


def test_prepare_timestamp_plot_data_accumulate_mode_aggregates_runs() -> None:
    df_filter = pd.DataFrame(
        {
            "ThreadID": ["1.1.1", "1.1.1", "1.1.2"],
            "Timestamp": [1, 2, 3],
            "Duration": [10.0, 10.0, 5.0],
            "Paraver_Label": ["A", "A", "B"],
            "Paraver_Value": [1, 1, 2],
            "R": [1, 2, 3],
            "G": [4, 5, 6],
            "B": [7, 8, 9],
            "FLOP": [1000.0, 1000.0, 500.0],
            "Bytes": [100.0, 100.0, 50.0],
            "Arithmetic_Intensity": [10.0, 10.0, 10.0],
            "GFLOPS": [0.1, 0.1, 0.1],
        }
    )
    df_intel = pd.DataFrame(
        {
            "ThreadID": ["1.1.1", "1.1.1", "1.1.2"],
            "Timestamp": [1, 2, 3],
            "Intel_FP_Scalar_SP": [1.0, 1.0, 1.0],
            "Intel_FP_Scalar_DP": [1.0, 1.0, 1.0],
            "Intel_FP_SSE_SP": [1.0, 1.0, 1.0],
            "Intel_FP_SSE_DP": [1.0, 1.0, 1.0],
            "Intel_FP_AVX2_SP": [1.0, 1.0, 1.0],
            "Intel_FP_AVX2_DP": [1.0, 1.0, 1.0],
            "Intel_FP_AVX512_SP": [1.0, 1.0, 1.0],
            "Intel_FP_AVX512_DP": [1.0, 1.0, 1.0],
            "Intel_FP_SP": [2.0, 2.0, 2.0],
            "Intel_FP_DP": [2.0, 2.0, 2.0],
            "Intel_FP_Total": [16.0, 16.0, 16.0],
            "Intel_Load": [5.0, 5.0, 2.0],
            "Intel_Store": [5.0, 5.0, 2.0],
            "Paraver_Label": ["A", "A", "B"],
        }
    )

    result, min_ai, min_gflops = prepare_timestamp_plot_data(
        df_filter,
        df_intel,
        average=False,
        timestamps_grouper=1,
        timestamp_start_index=0,
        use_accumulate=True,
    )

    assert result.extra_average == " "
    assert len(result.ai_mean) == 2
    assert len(result.gflops_mean) == 2
    assert result.timestamps_grouped == ["1 - 2", "3"]
    assert result.thread_ids == ["1.1.1", "1.1.2"]
    assert min_ai == 10.0
    assert min_gflops is not None


def test_should_plot_timestamp_point_respects_mask() -> None:
    assert should_plot_timestamp_point(False, "") is True
    assert should_plot_timestamp_point(True, 2) is True
    assert should_plot_timestamp_point(True, 0) is False
    assert should_plot_timestamp_point(True, "") is False


def test_normalize_roofline_values_only_first_six_entries() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, "fpinst"]
    normalized = normalize_roofline_values(values, normalize=True, threads_value=2)
    assert normalized[:6] == [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    assert normalized[6] == "fpinst"


def test_calculate_roofline_profile_returns_expected_shape() -> None:
    df = pd.DataFrame(
        [
            {
                "L1": 100.0,
                "L2": 50.0,
                "L3": 25.0,
                "DRAM": 10.0,
                "FP": 200.0,
                "FP_FMA": 250.0,
                "FPInst": "FMA",
                "Threads": 2,
                "ISA": "avx2",
            }
        ]
    )

    def fake_calculate_roofline(values: list[object], x_start: float) -> dict:
        assert x_start == 0.2
        return {
            "L1": {"ridge": [1.0, 300.0]},
            "DRAM": {"start": [x_start, 12.0]},
        }

    profile = calculate_roofline_profile(
        df, normalize=True, smallest_ai=1.0, calculate_roofline_fn=fake_calculate_roofline
    )
    assert profile is not None
    values, isa, lines, top_flops, min_gflops = profile
    assert values[:6] == [50.0, 25.0, 12.5, 5.0, 100.0, 125.0]
    assert values[6] == "FMA"
    assert isa == "avx2"
    assert lines["L1"]["ridge"][1] == 300.0
    assert top_flops == 300.0
    assert min_gflops == 12.0


def test_select_timestamp_color_prefers_paraver() -> None:
    color = select_timestamp_color(
        use_paraver_colors=True,
        color_radio="ISA",
        scalar_perc=0,
        sse_perc=0,
        avx2_perc=0,
        avx512_perc=0,
        dp_perc=0,
        load_perc=0,
        thread_id="1.1.1",
        red=10,
        green=20,
        blue=30,
        index=0,
        n_points=10,
        start_color=(1, 2, 3),
        end_color=(4, 5, 6),
        blend_colors_fn=lambda *args: "blend",
        interpolate_color_fn=lambda *_: "interp",
    )
    assert color == "rgb(10,20,30)"


def test_select_timestamp_color_blend_mode() -> None:
    color = select_timestamp_color(
        use_paraver_colors=False,
        color_radio="ISA",
        scalar_perc=1,
        sse_perc=2,
        avx2_perc=3,
        avx512_perc=4,
        dp_perc=5,
        load_perc=6,
        thread_id="1.1.2",
        red=0,
        green=0,
        blue=0,
        index=1,
        n_points=4,
        start_color=(1, 2, 3),
        end_color=(4, 5, 6),
        blend_colors_fn=lambda *args: f"blend-{args[7]}",
        interpolate_color_fn=lambda *_: "interp",
    )
    assert color == "blend-ISA"


def test_select_timestamp_color_gradient_mode() -> None:
    color = select_timestamp_color(
        use_paraver_colors=False,
        color_radio="Youngest",
        scalar_perc=0,
        sse_perc=0,
        avx2_perc=0,
        avx512_perc=0,
        dp_perc=0,
        load_perc=0,
        thread_id="1.1.3",
        red=0,
        green=0,
        blue=0,
        index=3,
        n_points=6,
        start_color=(1, 2, 3),
        end_color=(4, 5, 6),
        blend_colors_fn=lambda *_: "blend",
        interpolate_color_fn=lambda _s, _e, ratio: ratio,
    )
    assert color == 0.5


def test_build_timestamp_scatter_trace_without_legend_group() -> None:
    trace = build_timestamp_scatter_trace(
        ai_value=1.5,
        gflops_value=200.0,
        name="trace-name",
        dot_size=9,
        color="rgb(1,2,3)",
        tooltip_text="tooltip",
        showlegend=True,
    )
    assert trace["x"] == [1.5]
    assert trace["y"] == [200.0]
    assert trace["name"] == "trace-name"
    assert trace["marker"] == {"size": 9, "color": "rgb(1,2,3)"}
    assert trace["showlegend"] is True
    assert "legendgroup" not in trace


def test_build_timestamp_scatter_trace_with_legend_group() -> None:
    trace = build_timestamp_scatter_trace(
        ai_value=2.0,
        gflops_value=300.0,
        name="trace-with-group",
        dot_size=10,
        color="blue",
        tooltip_text="tip",
        showlegend=False,
        legendgroup="group-1",
    )
    assert trace["legendgroup"] == "group-1"
    assert trace["showlegend"] is False


def test_filter_roofline_df_by_query_with_none_returns_original() -> None:
    df = pd.DataFrame({"ISA": ["avx2", "scalar"], "Threads": [2, 1]})
    filtered = filter_roofline_df_by_query(df, None)
    assert filtered.equals(df)


def test_filter_roofline_df_by_query_applies_filter() -> None:
    df = pd.DataFrame({"ISA": ["avx2", "scalar"], "Threads": [2, 1]})
    filtered = filter_roofline_df_by_query(df, 'ISA == "avx2"')
    assert filtered.shape[0] == 1
    assert filtered.iloc[0]["ISA"] == "avx2"


def test_fallback_roofline_df_by_isa_returns_first_non_empty() -> None:
    df = pd.DataFrame({"ISA": ["avx2", "scalar"], "Threads": [2, 1]})

    def fake_construct_query(filters: dict[str, object]) -> str:
        return f'ISA == "{filters["ISA"]}"'

    filtered = fallback_roofline_df_by_isa(
        df,
        ["avx512", "avx2", "scalar"],
        fake_construct_query,
        {
            "ISA": None,
            "Precision": None,
            "Threads": None,
            "Loads": None,
            "Stores": None,
            "Interleaved": None,
            "DRAMBytes": None,
            "FPInst": None,
            "Date": None,
        },
    )
    assert filtered.shape[0] == 1
    assert filtered.iloc[0]["ISA"] == "avx2"


def test_resolve_timestamp_legend_state_with_paraver_colors() -> None:
    seen = {"A"}
    showlegend, label, next_first = resolve_timestamp_legend_state(True, "B", seen, True)
    assert showlegend is True
    assert label == "B"
    assert next_first is True
    assert seen == {"A", "B"}


def test_resolve_timestamp_legend_state_without_paraver_colors() -> None:
    seen = {"A"}
    showlegend, label, next_first = resolve_timestamp_legend_state(False, "B", seen, True)
    assert showlegend is True
    assert label == ""
    assert next_first is False
    assert seen == {"A"}


def test_should_reset_annotations_for_lines_respects_interval_trigger() -> None:
    assert should_reset_annotations_for_lines({"a": 1}, {"a": 2}, "interval-component") is False


def test_should_reset_annotations_for_lines_requires_existing_old_lines() -> None:
    assert should_reset_annotations_for_lines({"a": 1}, {}, "graphs", require_existing_old_lines=True) is False
    assert should_reset_annotations_for_lines({"a": 1}, {"a": 2}, "graphs", require_existing_old_lines=True) is True


def test_should_reset_annotations_for_lines_regular_path() -> None:
    assert should_reset_annotations_for_lines({"a": 1}, {"a": 1}, "graphs") is False
    assert should_reset_annotations_for_lines({"a": 2}, {"a": 1}, "graphs") is True


def test_resolve_interval_point_index_and_legend_for_initial_tick() -> None:
    indexer, first = resolve_interval_point_index_and_legend(0, 10)
    assert indexer == 9
    assert first is False


def test_resolve_interval_point_index_and_legend_for_first_point() -> None:
    indexer, first = resolve_interval_point_index_and_legend(1, 10)
    assert indexer == 0
    assert first is True


def test_resolve_timestamp_slice_bounds_standard_case() -> None:
    start, end = resolve_timestamp_slice_bounds(
        timestamps_range=[2, 4],
        timestamps_max_range=[10, 25],
        timestamps_grouper=3,
    )
    assert start == 16
    assert end == 25


def test_resolve_timestamp_slice_bounds_zero_upper_and_clamped() -> None:
    start, end = resolve_timestamp_slice_bounds(
        timestamps_range=[0, 0],
        timestamps_max_range=[5, 6],
        timestamps_grouper=4,
    )
    assert start == 5
    assert end == 7


def test_infer_effective_isa_from_timestamp_columns_uses_threshold_positive() -> None:
    columns_to_check = pd.DataFrame(
        {
            "Intel_FP_AVX2_DP": [0.0, 0.2],
            "Intel_FP_SSE_DP": [0.05, 0.0],
        }
    )
    isa = infer_effective_isa_from_timestamp_columns(
        columns_to_check,
        lower_filter=0.1,
        current_isa=None,
        has_thread_id_column=True,
    )
    assert isa == "avx2"


def test_infer_effective_isa_from_timestamp_columns_keeps_existing_isa() -> None:
    columns_to_check = pd.DataFrame(
        {
            "Intel_FP_AVX512_DP": [10.0],
            "Intel_FP_SSE_DP": [10.0],
        }
    )
    isa = infer_effective_isa_from_timestamp_columns(
        columns_to_check,
        lower_filter=0.0,
        current_isa="scalar",
        has_thread_id_column=False,
    )
    assert isa == "scalar"


def test_infer_effective_isa_from_timestamp_columns_defaults_to_scalar() -> None:
    columns_to_check = pd.DataFrame({"Intel_FP_AVX2_DP": [0.0], "Intel_FP_SSE_DP": [0.0]})
    isa = infer_effective_isa_from_timestamp_columns(
        columns_to_check,
        lower_filter=0.0,
        current_isa=None,
        has_thread_id_column=False,
    )
    assert isa == "scalar"


def test_iter_timestamp_plot_rows_keeps_field_order() -> None:
    timestamp_data = TimestampPlotData(
        extra_average=" ",
        ai_mean=[1.0],
        gflops_mean=[2.0],
        timestamps_grouped=["10...20"],
        durations=[100],
        thread_ids=["1.1.1"],
        reds=[1],
        greens=[2],
        blues=[3],
        pvalues=[4],
        plabels=["label"],
        scalar_perc=[11.0],
        sse_perc=[12.0],
        avx2_perc=[13.0],
        avx512_perc=[14.0],
        dp_perc=[15.0],
        load_perc=[16.0],
    )
    rows = list(iter_timestamp_plot_rows(timestamp_data))
    assert rows == [(1.0, 2.0, "10...20", 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, "1.1.1", 100, 1, 2, 3, 4, "label")]


def test_iter_timestamp_plot_rows_truncates_to_shortest_sequence() -> None:
    timestamp_data = TimestampPlotData(
        extra_average=" ",
        ai_mean=[1, 2],
        gflops_mean=[10],
        timestamps_grouped=["a", "b"],
        durations=[1, 2],
        thread_ids=["t1", "t2"],
        reds=[1, 2],
        greens=[1, 2],
        blues=[1, 2],
        pvalues=[1, 2],
        plabels=["p1", "p2"],
        scalar_perc=[0, 0],
        sse_perc=[0, 0],
        avx2_perc=[0, 0],
        avx512_perc=[0, 0],
        dp_perc=[0, 0],
        load_perc=[0, 0],
    )
    rows = list(iter_timestamp_plot_rows(timestamp_data))
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1] == 10


def test_get_timestamp_plot_point_returns_indexed_fields() -> None:
    timestamp_data = TimestampPlotData(
        extra_average=" ",
        ai_mean=[1.0, 3.0],
        gflops_mean=[2.0, 4.0],
        timestamps_grouped=["a", "b"],
        durations=[10, 20],
        thread_ids=["1.1.1", "1.1.2"],
        reds=[11, 22],
        greens=[33, 44],
        blues=[55, 66],
        pvalues=[7, 8],
        plabels=["L1", "L2"],
        scalar_perc=[1, 2],
        sse_perc=[3, 4],
        avx2_perc=[5, 6],
        avx512_perc=[7, 8],
        dp_perc=[9, 10],
        load_perc=[11, 12],
    )

    point = get_timestamp_plot_point(timestamp_data, 1)
    assert point == (3.0, 4.0, "b", 2, 4, 6, 8, 10, 12, "1.1.2", 20, 22, 44, 66, 8, "L2")


def test_build_timestamp_tooltip_args_order() -> None:
    args = build_timestamp_tooltip_args(
        1,
        2,
        3,
        4,
        5,
        6,
        "100...200",
        "1.1.1",
        123,
        7,
        "Label",
        "window",
    )
    assert args == (1, 2, 3, 4, 5, 6, "100...200", "1.1.1", 123, 7, "Label", "window")


def test_timestamp_plot_data_to_series_preserves_fields() -> None:
    timestamp_data = TimestampPlotData(
        extra_average=" Averaged ",
        ai_mean=[1.0, 2.0],
        gflops_mean=[10.0, 20.0],
        timestamps_grouped=["100...101", "200...201"],
        durations=[3.0, 4.0],
        thread_ids=["1.1.1", "1.1.2"],
        reds=[10, 11],
        greens=[20, 21],
        blues=[30, 31],
        pvalues=[1, 0],
        plabels=["A", "B"],
        scalar_perc=[1.0, 2.0],
        sse_perc=[3.0, 4.0],
        avx2_perc=[5.0, 6.0],
        avx512_perc=[7.0, 8.0],
        dp_perc=[9.0, 10.0],
        load_perc=[11.0, 12.0],
    )

    series = timestamp_plot_data_to_series(timestamp_data)
    assert isinstance(series, TimestampSeries)
    assert series.extra_average == " Averaged "
    assert series.count == 2
    assert series.points[1].timestamp_label == "200...201"
    assert series.points[1].thread_id == "1.1.2"


def test_prepare_timestamp_series_average_parity() -> None:
    df_filter = pd.DataFrame(
        {
            "Arithmetic_Intensity": [1.0, 3.0, 2.0, 6.0],
            "GFLOPS": [10.0, 30.0, 20.0, 40.0],
            "Timestamp": [100, 101, 200, 201],
            "Duration": [4.0, 6.0, 1.0, 9.0],
            "ThreadID": ["1.1.1", "1.1.2", "1.1.1", "1.1.2"],
            "R": [10, 11, 12, 13],
            "G": [20, 21, 22, 23],
            "B": [30, 31, 32, 33],
            "Paraver_Value": [7, 8, 9, 10],
            "Paraver_Label": ["A", "A", "B", "B"],
        }
    )
    df_intel = pd.DataFrame(
        {
            "Intel_FP_Scalar_SP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_Scalar_DP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_SSE_SP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_SSE_DP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_AVX2_SP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_AVX2_DP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_AVX512_SP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_AVX512_DP": [1.0, 1.0, 1.0, 1.0],
            "Intel_FP_DP": [2.0, 2.0, 2.0, 2.0],
            "Intel_FP_Total": [16.0, 16.0, 16.0, 16.0],
            "Intel_Load": [2.0, 2.0, 2.0, 2.0],
            "Intel_Store": [2.0, 2.0, 2.0, 2.0],
        }
    )

    flat_data, flat_min_ai, flat_min_gflops = prepare_timestamp_plot_data(
        df_filter,
        df_intel,
        average=True,
        timestamps_grouper=2,
        timestamp_start_index=0,
        use_accumulate=False,
    )
    series, series_min_ai, series_min_gflops = prepare_timestamp_series(
        df_filter,
        df_intel,
        average=True,
        timestamps_grouper=2,
        timestamp_start_index=0,
        use_accumulate=False,
    )

    assert series.count == len(flat_data.ai_mean)
    assert series.points[0].ai_value == flat_data.ai_mean[0]
    assert series.points[0].gflops_value == flat_data.gflops_mean[0]
    assert series.points[0].timestamp_label == flat_data.timestamps_grouped[0]
    assert series.points[0].thread_id == flat_data.thread_ids[0]
    assert flat_min_ai == series_min_ai
    assert flat_min_gflops == series_min_gflops


def test_should_plot_timestamp_point_supports_object() -> None:
    point = TimestampPoint(
        ai_value=1.0,
        gflops_value=2.0,
        timestamp_label="100",
        scalar_perc=0.0,
        sse_perc=0.0,
        avx2_perc=0.0,
        avx512_perc=0.0,
        dp_perc=0.0,
        load_perc=0.0,
        thread_id="1.1.1",
        duration=1.0,
        red=0,
        green=0,
        blue=0,
        pvalue=1,
        plabel="A",
    )
    masked = TimestampPoint(**{**point.__dict__, "pvalue": 0})

    assert should_plot_timestamp_point(True, point) is True
    assert should_plot_timestamp_point(True, masked) is False
    assert should_plot_timestamp_point(False, masked) is True


def test_select_timestamp_color_supports_object_context() -> None:
    point = TimestampPoint(
        ai_value=1.0,
        gflops_value=2.0,
        timestamp_label="100",
        scalar_perc=10.0,
        sse_perc=20.0,
        avx2_perc=30.0,
        avx512_perc=40.0,
        dp_perc=50.0,
        load_perc=60.0,
        thread_id="1.1.1",
        duration=1.0,
        red=7,
        green=8,
        blue=9,
        pvalue=1,
        plabel="A",
    )
    context = TimestampColorContext(
        use_paraver_colors=True,
        color_radio="ISA",
        index=0,
        n_points=1,
        start_color=(1, 2, 3),
        end_color=(4, 5, 6),
        blend_colors_fn=lambda *args: "blend",
        interpolate_color_fn=lambda *args: "interp",
    )

    assert select_timestamp_color(point, context) == "rgb(7,8,9)"


def test_build_timestamp_tooltip_args_supports_object() -> None:
    point = TimestampPoint(
        ai_value=1.0,
        gflops_value=2.0,
        timestamp_label="100...200",
        scalar_perc=1,
        sse_perc=2,
        avx2_perc=3,
        avx512_perc=4,
        dp_perc=5,
        load_perc=6,
        thread_id="1.1.1",
        duration=123,
        red=0,
        green=0,
        blue=0,
        pvalue=7,
        plabel="Label",
    )

    args = build_timestamp_tooltip_args(point, "window")
    assert args == (1, 2, 3, 4, 5, 6, "100...200", "1.1.1", 123, 7, "Label", "window")


def test_resolve_roofline_x_bounds_uses_defaults_when_small_ai_high() -> None:
    x_min, x_max = resolve_roofline_x_bounds(10.0)
    assert x_min == 1.0 / 256.0
    assert x_max == 256.0


def test_resolve_roofline_x_bounds_uses_small_ai_when_lower() -> None:
    x_min, x_max = resolve_roofline_x_bounds(0.01)
    assert x_min == 0.002
    assert x_max == 256.0


def test_resolve_roofline_angle_bounds_prefers_layout_ranges() -> None:
    bounds = resolve_roofline_angle_bounds(
        xaxis_range=[-2.0, 2.0],
        yaxis_range=[0.0, 3.0],
        smallest_ai=1.0,
        dram_start_gflops=2.0,
        peak_gflops=100.0,
    )
    assert bounds == (0.01, 100.0, 1.0, 1000.0)


def test_resolve_roofline_angle_bounds_falls_back_to_data_ranges() -> None:
    x_min, x_max, y_min, y_max = resolve_roofline_angle_bounds(
        xaxis_range=None,
        yaxis_range=None,
        smallest_ai=0.01,
        dram_start_gflops=20.0,
        peak_gflops=400.0,
    )
    assert x_min == 0.002
    assert x_max == 256.0
    assert y_min == 10.0
    assert y_max == 800.0
