"""
Unit tests for data_models.py, specifically the ForecastV2Request.from_dfs_pre_split method
and all its helper functions.
"""

import numpy as np
import pandas as pd
import pytest
from synthefy.data_models import (
    ForecastV2Request,
)


class TestValidateBacktestingInputs:
    """Tests for _validate_backtesting_inputs method."""

    def test_both_cutoff_date_and_num_target_rows_raises_error(self):
        """Should raise error when both cutoff_date and num_target_rows are provided."""
        with pytest.raises(
            ValueError, match="Only one of cutoff_date or num_target_rows"
        ):
            ForecastV2Request._validate_backtesting_inputs(
                cutoff_date="2023-01-01",
                num_target_rows=10,
                forecast_window=None,
                stride=None,
                target_cols=["sales"],
                metadata_cols=[],
                leak_cols=[],
            )

    def test_neither_cutoff_date_nor_num_target_rows_raises_error(self):
        """Should raise error when neither cutoff_date nor num_target_rows are provided."""
        with pytest.raises(
            ValueError,
            match="Either cutoff_date or num_target_rows must be provided",
        ):
            ForecastV2Request._validate_backtesting_inputs(
                cutoff_date=None,
                num_target_rows=None,
                forecast_window=None,
                stride=None,
                target_cols=["sales"],
                metadata_cols=[],
                leak_cols=[],
            )

    def test_forecast_window_without_stride_raises_error(self):
        """Should raise error when forecast_window is provided without stride."""
        with pytest.raises(
            ValueError,
            match="Forecast window and stride must be provided together",
        ):
            ForecastV2Request._validate_backtesting_inputs(
                cutoff_date="2023-01-01",
                num_target_rows=None,
                forecast_window="7D",
                stride=None,
                target_cols=["sales"],
                metadata_cols=[],
                leak_cols=[],
            )

    def test_stride_without_forecast_window_raises_error(self):
        """Should raise error when stride is provided without forecast_window."""
        with pytest.raises(
            ValueError,
            match="Forecast window and stride must be provided together",
        ):
            ForecastV2Request._validate_backtesting_inputs(
                cutoff_date="2023-01-01",
                num_target_rows=None,
                forecast_window=None,
                stride="1D",
                target_cols=["sales"],
                metadata_cols=[],
                leak_cols=[],
            )

    def test_date_based_with_wrong_forecast_window_type_raises_error(self):
        """Should raise error when using cutoff_date with integer forecast_window."""
        with pytest.raises(
            ValueError,
            match="forecast_window must be a string when using cutoff_date",
        ):
            ForecastV2Request._validate_backtesting_inputs(
                cutoff_date="2023-01-01",
                num_target_rows=None,
                forecast_window=7,
                stride="1D",
                target_cols=["sales"],
                metadata_cols=[],
                leak_cols=[],
            )

    def test_row_based_with_wrong_forecast_window_type_raises_error(self):
        """Should raise error when using num_target_rows with string forecast_window."""
        with pytest.raises(
            ValueError,
            match="forecast_window must be an integer when using num_target_rows",
        ):
            ForecastV2Request._validate_backtesting_inputs(
                cutoff_date=None,
                num_target_rows=30,
                forecast_window="7D",
                stride=1,
                target_cols=["sales"],
                metadata_cols=[],
                leak_cols=[],
            )

    def test_leak_cols_not_in_metadata_cols_raises_error(self):
        """Should raise error when leak_cols contains columns not in metadata_cols."""
        with pytest.raises(
            ValueError, match="leak_cols must be a subset of metadata_cols"
        ):
            ForecastV2Request._validate_backtesting_inputs(
                cutoff_date="2023-01-01",
                num_target_rows=None,
                forecast_window=None,
                stride=None,
                target_cols=["sales"],
                metadata_cols=["temperature"],
                leak_cols=["promotion", "temperature"],
            )

    def test_target_cols_overlap_with_metadata_cols_raises_error(self):
        """Should raise error when target_cols and metadata_cols overlap."""
        with pytest.raises(
            ValueError, match="target_cols and metadata_cols should not overlap"
        ):
            ForecastV2Request._validate_backtesting_inputs(
                cutoff_date="2023-01-01",
                num_target_rows=None,
                forecast_window=None,
                stride=None,
                target_cols=["sales", "temperature"],
                metadata_cols=["temperature", "promotion"],
                leak_cols=[],
            )

    def test_negative_num_target_rows_raises_error(self):
        """Should raise error when num_target_rows is negative."""
        with pytest.raises(
            ValueError, match="num_target_rows must be a positive integer"
        ):
            ForecastV2Request._validate_backtesting_inputs(
                cutoff_date=None,
                num_target_rows=-5,
                forecast_window=None,
                stride=None,
                target_cols=["sales"],
                metadata_cols=[],
                leak_cols=[],
            )

    def test_zero_num_target_rows_raises_error(self):
        """Should raise error when num_target_rows is zero."""
        with pytest.raises(
            ValueError, match="num_target_rows must be a positive integer"
        ):
            ForecastV2Request._validate_backtesting_inputs(
                cutoff_date=None,
                num_target_rows=0,
                forecast_window=None,
                stride=None,
                target_cols=["sales"],
                metadata_cols=[],
                leak_cols=[],
            )

    def test_valid_date_based_single_window(self):
        """Should not raise error for valid date-based single window configuration."""
        # Should not raise
        ForecastV2Request._validate_backtesting_inputs(
            cutoff_date="2023-01-01",
            num_target_rows=None,
            forecast_window=None,
            stride=None,
            target_cols=["sales"],
            metadata_cols=["temperature"],
            leak_cols=["temperature"],
        )

    def test_valid_row_based_backtesting(self):
        """Should not raise error for valid row-based backtesting configuration."""
        # Should not raise
        ForecastV2Request._validate_backtesting_inputs(
            cutoff_date=None,
            num_target_rows=30,
            forecast_window=7,
            stride=1,
            target_cols=["sales"],
            metadata_cols=["temperature"],
            leak_cols=[],
        )


class TestSplitDfToCorrelates:
    """Tests for split_df_to_correlates method."""

    def test_basic_single_target_column(self):
        """Should create one sample for a single target column."""
        history_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "sales": [100.0, 110.0, 120.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-04", periods=2, freq="D"),
                "sales": [130.0, 140.0],
            }
        )

        samples = ForecastV2Request.split_df_to_correlates(
            history_df=history_df,
            target_df=target_df,
            timestamp_col="timestamp",
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
        )

        assert len(samples) == 1
        sample = samples[0]
        assert sample.sample_id == "sales"
        assert sample.forecast is True
        assert sample.metadata is False
        assert sample.leak_target is False
        assert len(sample.history_timestamps) == 3
        assert len(sample.target_timestamps) == 2

    def test_target_and_metadata_columns(self):
        """Should create samples for both target and metadata columns."""
        history_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "sales": [100.0, 110.0, 120.0],
                "temperature": [20.0, 21.0, 22.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-04", periods=2, freq="D"),
                "sales": [130.0, 140.0],
                "temperature": [23.0, 24.0],
            }
        )

        samples = ForecastV2Request.split_df_to_correlates(
            history_df=history_df,
            target_df=target_df,
            timestamp_col="timestamp",
            target_cols=["sales"],
            metadata_cols=["temperature"],
            leak_cols=[],
        )

        assert len(samples) == 2

        # First sample should be target
        assert samples[0].sample_id == "sales"
        assert samples[0].forecast is True
        assert samples[0].metadata is False

        # Second sample should be metadata
        assert samples[1].sample_id == "temperature"
        assert samples[1].forecast is False
        assert samples[1].metadata is True

    def test_leak_target_flag_set_correctly(self):
        """Should set leak_target flag correctly for leak columns."""
        history_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "sales": [100.0, 110.0, 120.0],
                "promotion": [1.0, 1.0, 0.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-04", periods=2, freq="D"),
                "sales": [130.0, 140.0],
                "promotion": [0.0, 1.0],
            }
        )

        samples = ForecastV2Request.split_df_to_correlates(
            history_df=history_df,
            target_df=target_df,
            timestamp_col="timestamp",
            target_cols=["sales"],
            metadata_cols=["promotion"],
            leak_cols=["promotion"],
        )

        assert len(samples) == 2
        assert samples[1].sample_id == "promotion"
        assert samples[1].leak_target is True

    def test_nan_values_converted_to_none(self):
        """Should convert NaN values to None for JSON compatibility."""
        history_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "sales": [100.0, np.nan, 120.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-04", periods=2, freq="D"),
                "sales": [np.nan, 140.0],
            }
        )

        samples = ForecastV2Request.split_df_to_correlates(
            history_df=history_df,
            target_df=target_df,
            timestamp_col="timestamp",
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
        )

        sample = samples[0]
        assert sample.history_values[1] is None
        assert sample.target_values[0] is None
        assert sample.history_values[0] == 100.0
        assert sample.target_values[1] == 140.0


class TestSingleWindowByDate:
    """Tests for _single_window_by_date method."""

    def test_basic_single_window(self):
        """Should split dataframe into single window by date."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
                "sales": range(10),
            }
        )

        windows = ForecastV2Request._single_window_by_date(
            df=df,
            timestamp_col="timestamp",
            cutoff_date="2023-01-05",
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
        )

        assert len(windows) == 1
        samples = windows[0]
        assert len(samples) == 1

        sample = samples[0]
        assert (
            len(sample.history_timestamps) == 5
        )  # Jan 1-5 (inclusive of cutoff)
        assert len(sample.target_timestamps) == 5  # Jan 6-10

    def test_no_history_raises_error(self):
        """Should raise error when cutoff_date is before all data."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-05", periods=5, freq="D"),
                "sales": range(5),
            }
        )

        with pytest.raises(ValueError, match="No history rows found"):
            ForecastV2Request._single_window_by_date(
                df=df,
                timestamp_col="timestamp",
                cutoff_date="2023-01-01",
                target_cols=["sales"],
                metadata_cols=[],
                leak_cols=[],
            )

    def test_no_target_returns_empty(self):
        """Should return empty list when cutoff_date is after all data."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
                "sales": range(5),
            }
        )

        windows = ForecastV2Request._single_window_by_date(
            df=df,
            timestamp_col="timestamp",
            cutoff_date="2023-01-10",
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
        )

        assert len(windows) == 0

    def test_timezone_aware_timestamps(self):
        """Should handle timezone-aware timestamps correctly."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2023-01-01", periods=10, freq="D", tz="UTC"
                ),
                "sales": range(10),
            }
        )

        windows = ForecastV2Request._single_window_by_date(
            df=df,
            timestamp_col="timestamp",
            cutoff_date="2023-01-05",
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
        )

        assert len(windows) == 1


class TestSingleWindowByRows:
    """Tests for _single_window_by_rows method."""

    def test_basic_single_window(self):
        """Should split dataframe into single window by rows."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
                "sales": range(10),
            }
        )

        windows = ForecastV2Request._single_window_by_rows(
            df=df,
            timestamp_col="timestamp",
            num_target_rows=3,
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
        )

        assert len(windows) == 1
        samples = windows[0]
        assert len(samples) == 1

        sample = samples[0]
        assert len(sample.history_timestamps) == 7
        assert len(sample.target_timestamps) == 3

    def test_num_target_rows_exceeds_df_length_raises_error(self):
        """Should raise error when num_target_rows >= dataframe length."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
                "sales": range(5),
            }
        )

        with pytest.raises(
            ValueError, match="num_target_rows must be less than"
        ):
            ForecastV2Request._single_window_by_rows(
                df=df,
                timestamp_col="timestamp",
                num_target_rows=5,
                target_cols=["sales"],
                metadata_cols=[],
                leak_cols=[],
            )


class TestBacktestingByDate:
    """Tests for _backtesting_by_date method."""

    def test_multiple_windows_created(self):
        """Should create multiple windows with given stride."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=20, freq="D"),
                "sales": range(20),
            }
        )

        windows = ForecastV2Request._backtesting_by_date(
            df=df,
            timestamp_col="timestamp",
            cutoff_date="2023-01-05",
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
            forecast_window="3D",
            stride="2D",
        )

        # Data: Jan 1-20. Cutoff: Jan 5 (history <= Jan 5, target > Jan 5). Window: 3D. Stride: 2D
        # Windows: Jan 5 cutoff (target: Jan 6-9 = 3D), Jan 7 cutoff (target: Jan 8-11),
        #          Jan 9 cutoff (target: Jan 10-13), Jan 11 cutoff (target: Jan 12-15),
        #          Jan 13 cutoff (target: Jan 14-17), Jan 15 cutoff (target: Jan 16-19),
        #          Jan 17 cutoff (target: Jan 18-21, but only Jan 18-20 available = 3D),
        #          Jan 19 cutoff (target: Jan 20-23, but only Jan 20 available = 1D not 3D)
        # = 8 windows (cutoff moves by 2D each time, last window has partial data)
        assert len(windows) == 8

        # Each window should have samples
        for window in windows:
            assert len(window) == 1  # One target column

    def test_forecast_window_respected(self):
        """Should respect forecast_window size."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=20, freq="D"),
                "sales": range(20),
            }
        )

        windows = ForecastV2Request._backtesting_by_date(
            df=df,
            timestamp_col="timestamp",
            cutoff_date="2023-01-05",
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
            forecast_window="7D",
            stride="1D",
        )

        # First window should have 7 target timestamps
        first_sample = windows[0][0]
        assert len(first_sample.target_timestamps) == 7

    def test_stops_when_no_more_target_data(self):
        """Should stop creating windows when there's no more target data."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=15, freq="D"),
                "sales": range(15),
            }
        )

        windows = ForecastV2Request._backtesting_by_date(
            df=df,
            timestamp_col="timestamp",
            cutoff_date="2023-01-05",
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
            forecast_window="5D",
            stride="1D",
        )

        # Data: Jan 1-15. Cutoff: Jan 5 (history <= Jan 5, target > Jan 5). Window: 5D. Stride: 1D
        # Windows: Jan 5 cutoff (target: Jan 6-11 = 5D), Jan 6 cutoff (target: Jan 7-12),
        #          Jan 7 cutoff (target: Jan 8-13), Jan 8 cutoff (target: Jan 9-14),
        #          Jan 9 cutoff (target: Jan 10-15), Jan 10 cutoff (target: Jan 11-16, but only Jan 11-15 = 5D),
        #          Jan 11 cutoff (target: Jan 12-17, but only Jan 12-15 = 4D),
        #          Jan 12 cutoff (target: Jan 13-18, but only Jan 13-15 = 3D),
        #          Jan 13 cutoff (target: Jan 14-19, but only Jan 14-15 = 2D),
        #          Jan 14 cutoff (target: Jan 15-20, but only Jan 15 = 1D)
        # = 10 windows (some with partial data)
        assert len(windows) == 10


class TestBacktestingByRows:
    """Tests for _backtesting_by_rows method."""

    def test_multiple_windows_created(self):
        """Should create multiple windows with given stride."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=50, freq="D"),
                "sales": range(50),
            }
        )

        windows = ForecastV2Request._backtesting_by_rows(
            df=df,
            timestamp_col="timestamp",
            num_target_rows=20,
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
            forecast_window=5,
            stride=2,
        )

        # Start at cutoff_idx = 50 - 20 = 30
        # Windows: [30:35], [32:37], [34:39], [36:41], [38:43], [40:45], [42:47], [44:49]
        # Stop when cutoff_idx + forecast_window > 50
        # So: 30+5=35 ✓, 32+5=37 ✓, ..., 44+5=49 ✓, 46+5=51 ✗
        # = 8 windows
        assert len(windows) == 8

        # Each window should have samples
        for window in windows:
            assert len(window) == 1

    def test_forecast_window_size_respected(self):
        """Should respect forecast_window size in rows."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=50, freq="D"),
                "sales": range(50),
            }
        )

        windows = ForecastV2Request._backtesting_by_rows(
            df=df,
            timestamp_col="timestamp",
            num_target_rows=20,
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
            forecast_window=10,
            stride=5,
        )

        # Each window should have forecast_window target rows
        for window in windows:
            sample = window[0]
            assert len(sample.target_timestamps) == 10

    def test_stops_when_exceeds_dataframe_length(self):
        """Should stop when cutoff + forecast_window exceeds dataframe length."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=30, freq="D"),
                "sales": range(30),
            }
        )

        windows = ForecastV2Request._backtesting_by_rows(
            df=df,
            timestamp_col="timestamp",
            num_target_rows=10,
            target_cols=["sales"],
            metadata_cols=[],
            leak_cols=[],
            forecast_window=5,
            stride=2,
        )

        # Start at cutoff_idx = 30 - 10 = 20
        # Windows: [20:25], [22:27], [24:29]
        # Stop when cutoff_idx + forecast_window > 30
        # So: 20+5=25 ✓, 22+5=27 ✓, 24+5=29 ✓, 26+5=31 ✗
        # = 3 windows
        assert len(windows) == 3
        # Last window should not exceed dataframe bounds
        last_sample = windows[-1][0]
        assert (
            len(last_sample.history_timestamps)
            + len(last_sample.target_timestamps)
            <= 30
        )


class TestFromDfsPreSplit:
    """Tests for the main from_dfs_pre_split method."""

    def test_single_window_by_date(self):
        """Should create request with single window split by date."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
                "sales": range(10),
                "temperature": range(20, 30),
            }
        )

        request = ForecastV2Request.from_dfs_pre_split(
            dfs=[df],
            timestamp_col="timestamp",
            target_cols=["sales"],
            model="test-model",
            cutoff_date="2023-01-05",
            metadata_cols=["temperature"],
        )

        assert request.model == "test-model"
        assert len(request.samples) == 1
        assert len(request.samples[0]) == 2  # sales + temperature

    def test_single_window_by_rows(self):
        """Should create request with single window split by rows."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
                "sales": range(10),
            }
        )

        request = ForecastV2Request.from_dfs_pre_split(
            dfs=[df],
            timestamp_col="timestamp",
            target_cols=["sales"],
            model="test-model",
            num_target_rows=3,
        )

        assert request.model == "test-model"
        assert len(request.samples) == 1
        assert len(request.samples[0]) == 1  # Only sales

    def test_backtesting_by_date(self):
        """Should create request with multiple windows for date-based backtesting."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=20, freq="D"),
                "sales": range(20),
            }
        )

        request = ForecastV2Request.from_dfs_pre_split(
            dfs=[df],
            timestamp_col="timestamp",
            target_cols=["sales"],
            model="test-model",
            cutoff_date="2023-01-05",
            forecast_window="5D",
            stride="2D",
        )

        assert request.model == "test-model"
        # Data: Jan 1-20. Cutoff: Jan 5 (history <= Jan 5, target > Jan 5). Window: 5D. Stride: 2D
        # Windows: Jan 5 cutoff (target: Jan 6-11), Jan 7 cutoff (target: Jan 8-13),
        #          Jan 9 cutoff (target: Jan 10-15), Jan 11 cutoff (target: Jan 12-17),
        #          Jan 13 cutoff (target: Jan 14-19), Jan 15 cutoff (target: Jan 16-21, but only to Jan 20 = 5D),
        #          Jan 17 cutoff (target: Jan 18-23, but only to Jan 20 = 3D),
        #          Jan 19 cutoff (target: Jan 20-25, but only Jan 20 = 1D)
        # = 8 windows (some with partial data)
        assert len(request.samples) == 8

    def test_backtesting_by_rows(self):
        """Should create request with multiple windows for row-based backtesting."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=50, freq="D"),
                "sales": range(50),
            }
        )

        request = ForecastV2Request.from_dfs_pre_split(
            dfs=[df],
            timestamp_col="timestamp",
            target_cols=["sales"],
            model="test-model",
            num_target_rows=20,
            forecast_window=5,
            stride=2,
        )

        assert request.model == "test-model"
        # Start at cutoff_idx = 50 - 20 = 30
        # Windows: [30:35], [32:37], [34:39], [36:41], [38:43], [40:45], [42:47], [44:49]
        # Stop when cutoff_idx + forecast_window > 50
        # = 8 windows
        assert len(request.samples) == 8

    def test_multiple_dataframes(self):
        """Should process multiple dataframes and combine results."""
        df1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
                "sales": range(10),
            }
        )
        df2 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
                "sales": range(10, 20),
            }
        )

        request = ForecastV2Request.from_dfs_pre_split(
            dfs=[df1, df2],
            timestamp_col="timestamp",
            target_cols=["sales"],
            model="test-model",
            cutoff_date="2023-01-05",
        )

        # Should have windows from both dataframes (1 window each)
        assert len(request.samples) == 2

    def test_multiple_target_columns(self):
        """Should handle multiple target columns correctly."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
                "sales": range(10),
                "revenue": range(100, 110),
            }
        )

        request = ForecastV2Request.from_dfs_pre_split(
            dfs=[df],
            timestamp_col="timestamp",
            target_cols=["sales", "revenue"],
            model="test-model",
            cutoff_date="2023-01-05",
        )

        assert len(request.samples) == 1
        assert len(request.samples[0]) == 2  # Both target columns
        assert request.samples[0][0].forecast is True
        assert request.samples[0][1].forecast is True

    def test_leak_cols_properly_marked(self):
        """Should properly mark leak columns."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
                "sales": range(10),
                "promotion": [1] * 10,
            }
        )

        request = ForecastV2Request.from_dfs_pre_split(
            dfs=[df],
            timestamp_col="timestamp",
            target_cols=["sales"],
            model="test-model",
            cutoff_date="2023-01-05",
            metadata_cols=["promotion"],
            leak_cols=["promotion"],
        )

        # Find promotion sample
        promotion_sample = [
            s for s in request.samples[0] if s.sample_id == "promotion"
        ][0]
        assert promotion_sample.leak_target is True

    def test_empty_result_raises_error(self):
        """Should raise error when no valid windows can be created."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
                "sales": range(5),
            }
        )

        with pytest.raises(
            ValueError, match="No valid windows could be created"
        ):
            ForecastV2Request.from_dfs_pre_split(
                dfs=[df],
                timestamp_col="timestamp",
                target_cols=["sales"],
                model="test-model",
                cutoff_date="2023-02-01",  # After all data
            )

    def test_timestamps_sorted_automatically(self):
        """Should sort timestamps automatically."""
        # Create unsorted dataframe
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2023-01-05",
                    "2023-01-02",
                    "2023-01-08",
                    "2023-01-01",
                    "2023-01-10",
                ],
                "sales": [5, 2, 8, 1, 10],
            }
        )

        request = ForecastV2Request.from_dfs_pre_split(
            dfs=[df],
            timestamp_col="timestamp",
            target_cols=["sales"],
            model="test-model",
            num_target_rows=2,
        )

        # Should work without error and have sorted data
        assert len(request.samples) == 1
        sample = request.samples[0][0]

        # Values should be sorted by timestamp
        assert sample.history_values == [1.0, 2.0, 5.0]
        assert sample.target_values == [8.0, 10.0]
