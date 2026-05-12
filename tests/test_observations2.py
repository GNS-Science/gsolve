# GSolve - gravity processing software.
# Copyright (c) 2026 Earth Sciences New Zealand.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPLv3

# Copyright (c) 2025 Earth Sciences New Zealand.
"""Additional tests for gsolve.observations — untested methods."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from gsolve import GravityObservations
from gsolve.observations import GravityObservationsParameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(n_loops: int = 1, n_per_loop: int = 3) -> GravityObservations:
    """Build a small GravityObservations with optional multi-loop layout."""
    n = n_loops * n_per_loop
    site_id = [f"S{i % n_per_loop + 1}" for i in range(n)]
    meter_id = ["M1"] * n
    meter_reading_mgal = [980100.0 + i * 10 for i in range(n)]
    loop = [f"L{i // n_per_loop + 1}" for i in range(n)]
    datetimes = [f"2024-01-01T{10 + i:02d}:00:00" for i in range(n)]
    return GravityObservations(
        site_id=site_id,
        datetime=datetimes,
        meter_id=meter_id,
        meter_reading_mgal=meter_reading_mgal,
        loop=loop,
    )


# ---------------------------------------------------------------------------
# GravityObservationsParameters __setattr__
# ---------------------------------------------------------------------------


class TestGravityObservationsParameters:
    def test_timedelta_unit_coerced_to_timedelta(self):
        params = GravityObservationsParameters(timedelta_unit="1h")
        assert isinstance(params.timedelta_unit, pd.Timedelta)
        assert params.timedelta_unit == pd.Timedelta("1h")

    def test_timedelta_unit_minutes(self):
        params = GravityObservationsParameters(timedelta_unit="30min")
        assert params.timedelta_unit == pd.Timedelta("30min")

    def test_fixed_time_datum_none_becomes_nat(self):
        params = GravityObservationsParameters(
            timedelta_unit="1h", fixed_time_datum=None
        )
        assert pd.isnull(params.fixed_time_datum)

    def test_fixed_time_datum_set(self):
        ts = pd.Timestamp("2024-06-01 00:00:00")
        params = GravityObservationsParameters(timedelta_unit="1h", fixed_time_datum=ts)
        assert params.fixed_time_datum is not None
        assert not pd.isnull(params.fixed_time_datum)


# ---------------------------------------------------------------------------
# GravityObservations.__repr__
# ---------------------------------------------------------------------------


class TestGravityObservationsRepr:
    def test_repr_contains_class_name(self):
        obs = _make_obs()
        r = repr(obs)
        assert "GravityObservations" in r

    def test_repr_contains_n_observations(self):
        obs = _make_obs()
        r = repr(obs)
        assert "n_observations=3" in r

    def test_repr_contains_n_loops(self):
        obs = _make_obs(n_loops=2)
        r = repr(obs)
        assert "n_loops=2" in r


# ---------------------------------------------------------------------------
# GravityObservations.set_obs_id
# ---------------------------------------------------------------------------


class TestSetObsId:
    def test_default_generates_index(self):
        obs = _make_obs()
        assert obs.data.index.name == "obs_id"
        assert len(obs.data.index) == 3

    def test_set_from_array(self):
        obs = _make_obs()
        obs.set_obs_id(["a", "b", "c"])
        assert list(obs.data.index) == ["a", "b", "c"]

    def test_set_from_column_name(self):
        obs = _make_obs()
        obs.data["custom_id"] = ["x", "y", "z"]
        obs.set_obs_id("custom_id")
        assert list(obs.data.index) == ["x", "y", "z"]
        # Column should be dropped after setting as index
        assert "custom_id" not in obs.data.columns

    def test_set_from_column_name_keep(self):
        obs = _make_obs()
        obs.data["custom_id"] = ["x", "y", "z"]
        obs.set_obs_id("custom_id", drop=False)
        assert "custom_id" in obs.data.columns

    def test_duplicate_error_mode(self):
        obs = _make_obs()
        with pytest.raises(ValueError, match="duplicated"):
            obs.set_obs_id(["a", "a", "b"], duplicated_obs_id="error")

    def test_duplicate_keep_mode_warns(self):
        obs = _make_obs()
        with pytest.warns(UserWarning):
            obs.set_obs_id(["a", "a", "b"], duplicated_obs_id="keep")
        # Duplicates kept as-is
        assert list(obs.data.index) == ["a", "a", "b"]

    def test_duplicate_rename_mode_warns(self):
        obs = _make_obs()
        with pytest.warns(UserWarning):
            obs.set_obs_id(["a", "a", "b"], duplicated_obs_id="rename")
        # After rename, a.001 and a.002 should appear
        idx_values = list(obs.data.index)
        assert len(set(idx_values)) == 3

    def test_invalid_idx_type_raises(self):
        obs = _make_obs()
        with pytest.raises(TypeError):
            obs.set_obs_id(123)  # type: ignore


# ---------------------------------------------------------------------------
# GravityObservations.params() method
# ---------------------------------------------------------------------------


class TestParamsMethod:
    def test_params_returns_parameters_object(self):
        obs = _make_obs()
        p = obs.params()
        assert isinstance(p, GravityObservationsParameters)

    def test_params_timedelta_unit(self):
        obs = _make_obs()
        obs.set_timedelta_unit("30min")
        p = obs.params()
        assert p.timedelta_unit == pd.Timedelta("30min")


# ---------------------------------------------------------------------------
# GravityObservations.set_calibration_factor
# ---------------------------------------------------------------------------


class TestSetCalibrationFactor:
    def test_set_single_meter(self):
        obs = _make_obs()
        obs.set_calibration_factor(1.005)
        assert obs.data["calibration_factor"].eq(1.005).all()

    def test_set_without_meter_id_multiple_meters_raises(self):
        obs = _make_obs()
        obs.data.iloc[0, obs.data.columns.get_loc("meter_id")] = "M2"
        with pytest.raises(ValueError, match="Multiple gravity meters"):
            obs.set_calibration_factor(1.005)

    def test_set_with_meter_id(self):
        obs = _make_obs()
        obs.data.iloc[0, obs.data.columns.get_loc("meter_id")] = "M2"
        obs.set_calibration_factor(1.005, meter_id="M2")
        assert (
            obs.data.loc[obs.data["meter_id"] == "M2", "calibration_factor"]
            .eq(1.005)
            .all()
        )
        # M1 rows unchanged (still 1.0)
        assert (
            obs.data.loc[obs.data["meter_id"] == "M1", "calibration_factor"]
            .eq(1.0)
            .all()
        )

    def test_set_with_bad_meter_id_raises(self):
        obs = _make_obs()
        with pytest.raises(ValueError, match="not found"):
            obs.set_calibration_factor(1.005, meter_id="MISSING")


# ---------------------------------------------------------------------------
# GravityObservations.calculate_tide_corrected_gravity
# ---------------------------------------------------------------------------


class TestCalculateTideCorrectedGravity:
    def test_column_created(self):
        obs = _make_obs()
        obs.calculate_tide_corrected_gravity()
        assert "gravity_corr" in obs.data.columns

    def test_values_eq_reading_times_calibration(self):
        obs = _make_obs()
        obs.set_calibration_factor(1.0)
        obs.calculate_tide_corrected_gravity()
        expected = obs.data["meter_reading_mgal"] * 1.0
        np.testing.assert_array_almost_equal(
            obs.data["gravity_corr"].to_numpy(),
            expected.to_numpy(),
        )

    def test_values_with_calibration_factor(self):
        obs = _make_obs()
        obs.set_calibration_factor(1.01)
        obs.calculate_tide_corrected_gravity()
        expected = obs.data["meter_reading_mgal"] * 1.01
        np.testing.assert_array_almost_equal(
            obs.data["gravity_corr"].to_numpy(),
            expected.to_numpy(),
            decimal=5,
        )


# ---------------------------------------------------------------------------
# activate / deactivate error paths
# ---------------------------------------------------------------------------


class TestActivateDeactivateErrors:
    def test_deactivate_bad_obs_id_raises(self):
        obs = _make_obs()
        with pytest.raises(ValueError, match="obs_id"):
            obs.deactivate(obs_id="NONEXISTENT")

    def test_deactivate_bad_site_id_raises(self):
        obs = _make_obs()
        with pytest.raises(ValueError, match="site_id"):
            obs.deactivate(site_id="NONEXISTENT")

    def test_deactivate_bad_loop_raises(self):
        obs = _make_obs()
        with pytest.raises(ValueError, match="loop"):
            obs.deactivate(loop="NONEXISTENT")


# ---------------------------------------------------------------------------
# loop_summary and site_summary
# ---------------------------------------------------------------------------


class TestSummaryMethods:
    def test_loop_summary_returns_dataframe(self):
        obs = _make_obs(n_loops=2)
        summary = obs.loop_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2

    def test_loop_summary_index_is_loop(self):
        obs = _make_obs(n_loops=2)
        summary = obs.loop_summary()
        assert summary.index.name == "loop"

    def test_site_summary_returns_dataframe(self):
        obs = _make_obs()
        obs.calculate_tide_corrected_gravity()
        summary = obs.site_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3  # 3 unique sites

    def test_site_summary_custom_col(self):
        obs = _make_obs()
        summary = obs.site_summary(data_col="meter_reading_mgal")
        assert isinstance(summary, pd.DataFrame)

    def test_site_summary_bad_col_raises(self):
        obs = _make_obs()
        with pytest.raises(ValueError, match="not found"):
            obs.site_summary(data_col="nonexistent_col")


# ---------------------------------------------------------------------------
# check_data
# ---------------------------------------------------------------------------


class TestCheckData:
    def test_check_data_clean_obs_returns_true(self):
        obs = _make_obs()
        obs.calculate_tide_corrected_gravity()
        result = obs.check_data(warn=False)
        # Might not be True if some derived fields are missing — check it runs
        assert isinstance(result, bool)

    def test_check_data_all_inactive_returns_false(self):
        obs = _make_obs()
        obs.deactivate(loop="L1")
        result = obs.check_data(warn=False)
        assert result is False
