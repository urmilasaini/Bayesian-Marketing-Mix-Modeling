"""Unit tests for data_loader module.

Tests the Conjura dataset loading functionality including:
- Time series listing and discovery
- Single organisation data loading
- Active channel detection
- Representative time series selection
"""

import numpy as np
import pandas as pd
import pytest

from hill_mixture_mmm.data_loader import (
    LoadedData,
    TimeSeriesConfig,
    get_active_channels,
    list_timeseries,
    load_timeseries,
    select_representative_timeseries,
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    rng = np.random.default_rng(42)

    rows = []
    for org_id in ["org_001", "org_002"]:
        for territory in ["All Territories", "US"]:
            for date in dates:
                rows.append(
                    {
                        "mmm_timeseries_id": f"{org_id}_{territory}",
                        "organisation_id": org_id,
                        "organisation_vertical": "Electronics"
                        if org_id == "org_001"
                        else "Fashion",
                        "organisation_subvertical": "Phones" if org_id == "org_001" else "Shoes",
                        "territory_name": territory,
                        "date_day": date,
                        "currency_code": "USD",
                        "google_paid_search_spend": rng.lognormal(3, 0.5)
                        if org_id == "org_001"
                        else 0,
                        "google_shopping_spend": rng.lognormal(2, 0.5)
                        if org_id == "org_001"
                        else 0,
                        "google_pmax_spend": 0,
                        "google_display_spend": 0,
                        "google_video_spend": 0,
                        "meta_facebook_spend": rng.lognormal(3, 0.5) if org_id == "org_002" else 0,
                        "meta_instagram_spend": rng.lognormal(2, 0.5) if org_id == "org_002" else 0,
                        "meta_other_spend": 0,
                        "tiktok_spend": 0,
                        "all_purchases": rng.integers(10, 100),
                        "first_purchases": rng.integers(1, 20),
                        "all_purchases_units": rng.integers(15, 150),
                        "first_purchases_units": rng.integers(1, 25),
                        "all_purchases_original_price": rng.lognormal(5, 0.5),
                        "first_purchases_original_price": rng.lognormal(4, 0.5),
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "test_mmm_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestListTimeseries:
    """Tests for list_timeseries function."""

    def test_discovers_all_timeseries(self, sample_csv):
        """Should find all organisation/territory combinations."""
        ts_info = list_timeseries(sample_csv, min_length=100)

        assert len(ts_info) == 4
        assert "organisation_id" in ts_info.columns
        assert "territory_name" in ts_info.columns
        assert "n_days" in ts_info.columns

    def test_min_length_filter(self, sample_csv):
        """Should filter out short time series."""
        ts_info = list_timeseries(sample_csv, min_length=300)
        assert len(ts_info) == 0

    def test_active_channels_detected(self, sample_csv):
        """Should correctly identify active spend channels."""
        ts_info = list_timeseries(sample_csv, min_length=100)

        org1_row = ts_info[ts_info["organisation_id"] == "org_001"].iloc[0]
        assert "google_paid_search_spend" in org1_row["active_channels"]
        assert "google_shopping_spend" in org1_row["active_channels"]

        org2_row = ts_info[ts_info["organisation_id"] == "org_002"].iloc[0]
        assert "meta_facebook_spend" in org2_row["active_channels"]
        assert "meta_instagram_spend" in org2_row["active_channels"]

    def test_sorted_by_length(self, sample_csv):
        """Should return results sorted by n_days descending."""
        ts_info = list_timeseries(sample_csv, min_length=100)
        assert ts_info["n_days"].iloc[0] >= ts_info["n_days"].iloc[-1]


class TestGetActiveChannels:
    """Tests for get_active_channels function."""

    def test_detects_nonzero_channels(self):
        """Should find channels with sufficient non-zero data."""
        df = pd.DataFrame(
            {
                "google_paid_search_spend": [100, 50, 0, 80, 60],
                "google_shopping_spend": [0, 0, 0, 10, 0],
                "meta_facebook_spend": [0, 0, 0, 0, 0],
            }
        )

        active = get_active_channels(df, min_nonzero_ratio=0.5)
        assert "google_paid_search_spend" in active
        assert "google_shopping_spend" not in active
        assert "meta_facebook_spend" not in active

    def test_handles_missing_columns(self):
        """Should ignore columns not in SPEND_COLUMNS."""
        df = pd.DataFrame(
            {
                "google_paid_search_spend": [100, 50, 80],
                "other_column": [1, 2, 3],
            }
        )

        active = get_active_channels(df, min_nonzero_ratio=0.1)
        assert "google_paid_search_spend" in active
        assert "other_column" not in active


class TestLoadTimeseries:
    """Tests for load_timeseries function."""

    def test_loads_organisation_data(self, sample_csv):
        """Should load data for a specific organisation."""
        config = TimeSeriesConfig(
            organisation_id="org_001",
            territory="All Territories",
            target_col="all_purchases",
            aggregate_spend=True,
        )

        data = load_timeseries(sample_csv, config)

        assert isinstance(data, LoadedData)
        assert len(data.x) == 200
        assert len(data.y) == 200
        assert len(data.dates) == 200
        assert data.x.dtype == np.float32
        assert data.y.dtype == np.float32

    def test_metadata_populated(self, sample_csv):
        """Should populate metadata correctly."""
        config = TimeSeriesConfig(
            organisation_id="org_001",
            target_col="all_purchases",
        )

        data = load_timeseries(sample_csv, config)

        assert data.meta["organisation_id"] == "org_001"
        assert data.meta["territory"] == "All Territories"
        assert data.meta["T"] == 200
        assert data.meta["aggregated"] is True
        assert "total_spend" in data.meta
        assert "total_target" in data.meta

    def test_aggregates_spend_by_default(self, sample_csv):
        """Should sum all spend channels when aggregate_spend=True."""
        config = TimeSeriesConfig(
            organisation_id="org_001",
            aggregate_spend=True,
        )

        data = load_timeseries(sample_csv, config)

        assert data.x.ndim == 1
        assert data.x.sum() > 0

    def test_raises_for_missing_org(self, sample_csv):
        """Should raise ValueError for non-existent organisation."""
        config = TimeSeriesConfig(
            organisation_id="nonexistent_org",
        )

        with pytest.raises(ValueError, match="No data found"):
            load_timeseries(sample_csv, config)

    def test_raises_for_none_org_id(self, sample_csv):
        """Should raise ValueError when organisation_id is None."""
        config = TimeSeriesConfig(organisation_id=None)

        with pytest.raises(ValueError, match="organisation_id must be specified"):
            load_timeseries(sample_csv, config)

    def test_raises_for_short_series(self, sample_csv):
        """Should raise ValueError if series is too short."""
        config = TimeSeriesConfig(
            organisation_id="org_001",
            min_series_length=500,
        )

        with pytest.raises(ValueError, match="Series too short"):
            load_timeseries(sample_csv, config)

    def test_custom_target_column(self, sample_csv):
        """Should allow different target columns."""
        config = TimeSeriesConfig(
            organisation_id="org_001",
            target_col="first_purchases",
        )

        data = load_timeseries(sample_csv, config)

        assert data.meta["target_col"] == "first_purchases"
        assert data.y.mean() < 100


class TestSelectRepresentativeTimeseries:
    """Tests for select_representative_timeseries function."""

    def test_selects_requested_count(self, sample_csv):
        """Should return requested number of time series."""
        selected = select_representative_timeseries(sample_csv, n=2, min_length=100, min_channels=1)

        assert len(selected) == 2
        assert all(isinstance(s, str) for s in selected)

    def test_stratifies_by_vertical(self, sample_csv):
        """Should select from different verticals when possible."""
        selected = select_representative_timeseries(
            sample_csv, n=2, min_length=100, min_channels=1, seed=42
        )

        assert len(set(selected)) == 2

    def test_deterministic_with_seed(self, sample_csv):
        """Should produce same results with same seed."""
        selected1 = select_representative_timeseries(
            sample_csv, n=2, min_length=100, min_channels=1, seed=123
        )
        selected2 = select_representative_timeseries(
            sample_csv, n=2, min_length=100, min_channels=1, seed=123
        )

        assert selected1 == selected2

    def test_raises_when_no_series_meet_criteria(self, sample_csv):
        """Should raise ValueError if no series meet criteria."""
        with pytest.raises(ValueError, match="No time series meet the criteria"):
            select_representative_timeseries(
                sample_csv,
                n=5,
                min_length=100,
                min_channels=10,
            )
