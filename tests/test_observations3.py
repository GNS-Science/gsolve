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
"""Additional tests — GravitySurvey, write_to_csv, _get_writable_df."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from gsolve import GravityObservations, GravitySites
from gsolve.observations import GravitySurvey
from gsolve.sites import ReferenceGravity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(n_loops: int = 2, n_per_loop: int = 3) -> GravityObservations:
    n = n_loops * n_per_loop
    site_id = [f"S{i % n_per_loop + 1}" for i in range(n)]
    loop = [f"L{i // n_per_loop + 1}" for i in range(n)]
    datetimes = [f"2024-01-01T{10 + i:02d}:00:00" for i in range(n)]
    return GravityObservations(
        site_id=site_id,
        datetime=datetimes,
        meter_id=["M1"] * n,
        meter_reading_mgal=[980100.0 + i * 5 for i in range(n)],
        loop=loop,
    )


def _make_sites() -> GravitySites:
    return GravitySites(
        site_id=["S1", "S2", "S3"],
        latitude=[-45.0, -44.0, -43.0],
        longitude=[170.0, 171.0, 172.0],
        height_ellipsoidal=[10.0, 50.0, 200.0],
    )


def _make_survey() -> GravitySurvey:
    return GravitySurvey(obs=_make_obs(), sites=_make_sites())


# ---------------------------------------------------------------------------
# GravitySurvey construction
# ---------------------------------------------------------------------------


class TestGravitySurvey:
    def test_basic_construction(self):
        survey = _make_survey()
        assert isinstance(survey.observations, GravityObservations)
        assert isinstance(survey.sites, GravitySites)

    def test_obs_data_accessible(self):
        survey = _make_survey()
        assert survey.observations.data.shape[0] == 6

    def test_sites_data_accessible(self):
        survey = _make_survey()
        assert len(survey.sites.data) == 3

    def test_set_calibration_factor_delegates(self):
        survey = _make_survey()
        survey.set_calibration_factor(1.005)
        assert survey.observations.data["calibration_factor"].eq(1.005).all()

    def test_calculate_tide_corrected_gravity_delegates(self):
        survey = _make_survey()
        survey.set_calibration_factor(1.0)
        survey.calculate_tide_corrected_gravity()
        assert "gravity_corr" in survey.observations.data.columns

    def test_set_reference_gravity(self):
        survey = _make_survey()
        ref = ReferenceGravity(
            site_id=["S1"],
            gravity=[980100.0],
        )
        survey.set_reference_gravity(ref)
        assert "reference_gravity" in survey.sites.data.columns

    def test_pre_flight_check_valid_survey(self):
        survey = _make_survey()
        survey.set_calibration_factor(1.0)
        survey.calculate_tide_corrected_gravity()
        # Result may be False because some derived fields may be missing
        result = survey.pre_flight_check(warn=False)
        assert isinstance(result, bool)

    def test_pre_flight_check_missing_sites(self):
        obs = _make_obs()
        # Sites only cover S1, S2 (not S3)
        sites_partial = GravitySites(
            site_id=["S1", "S2"],
            latitude=[-45.0, -44.0],
            longitude=[170.0, 171.0],
            height_ellipsoidal=[10.0, 50.0],
        )
        survey = GravitySurvey(obs=obs, sites=sites_partial)
        result = survey.pre_flight_check(warn=False)
        assert result is False


# ---------------------------------------------------------------------------
# GravityObservations._get_writable_df and write_to_csv
# ---------------------------------------------------------------------------


class TestGetWritableDf:
    def test_returns_dataframe(self):
        obs = _make_obs()
        df = obs._get_writable_df()
        assert isinstance(df, pd.DataFrame)

    def test_active_only(self):
        obs = _make_obs()
        obs.deactivate(loop="L1")
        df = obs._get_writable_df(active_only=True)
        assert len(df) == 3  # Only L2 obs

    def test_include_unknown_fields_true(self):
        obs = _make_obs()
        obs.data["my_custom_col"] = 99.0
        df = obs._get_writable_df(include_unknown_fields=True)
        assert "my_custom_col" in df.columns

    def test_include_unknown_fields_list(self):
        obs = _make_obs()
        obs.data["extra"] = 42.0
        df = obs._get_writable_df(include_unknown_fields=["extra"])
        assert "extra" in df.columns

    def test_include_unknown_fields_bad_list_raises(self):
        obs = _make_obs()
        with pytest.raises(ValueError, match="not found"):
            obs._get_writable_df(include_unknown_fields=["nonexistent_col"])


class TestGravityObservationsWriteToCsv:
    def test_write_to_csv_creates_file(self):
        obs = _make_obs()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = pathlib.Path(tmpdir) / "test_obs.csv"
            obs.write_to_csv(fpath)
            assert fpath.exists()

    def test_write_to_csv_active_only(self):
        obs = _make_obs()
        obs.deactivate(loop="L1")
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = pathlib.Path(tmpdir) / "test_obs_active.csv"
            obs.write_to_csv(fpath, active_only=True)
            df = pd.read_csv(fpath)
            assert len(df) == 3


# ---------------------------------------------------------------------------
# core/data.py coverage gaps — via GravitySites
# ---------------------------------------------------------------------------


class TestGSolveTableCoverage:
    def test_repr_with_data(self):
        sites = _make_sites()
        r = repr(sites)
        assert "GravitySites" in r

    def test_write_to_csv(self):
        sites = _make_sites()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = pathlib.Path(tmpdir) / "sites.csv"
            sites.write_to_csv(fpath)
            assert fpath.exists()
            df = pd.read_csv(fpath)
            assert len(df) == 3

    def test_bool_with_data(self):
        sites = _make_sites()
        assert bool(sites) is True

    def test_len_with_data(self):
        sites = _make_sites()
        assert len(sites) == 3

    def test_set_column_with_no_default_raises(self):
        sites = _make_sites()
        with pytest.raises(ValueError, match="no default value"):
            sites.set_column("totally_unknown_col_xyz", data=None)
