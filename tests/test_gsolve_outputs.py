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
"""Tests for gsolve.gsolve_outputs — GSolveSolutionParameters, GSolveResults."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")  # headless backend; no display required for tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gsolve.gsolve_outputs import (
    GSolveResults,
    GSolveSolutionParameters,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SITE_IDS = ["A", "B"]
LOOPS = ["L1", "L1", "L1", "L2", "L2"]
OBS_SITE_IDS = ["A", "B", "A", "B", "A"]


def _make_obs_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": OBS_SITE_IDS,
            "loop": LOOPS,
            "timedelta": [0.0, 0.5, 1.0, 0.0, 0.5],
        },
        index=pd.Index([10, 11, 12, 13, 14], name="obs_id"),
    )


def _make_ref_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"absolute_gravity": [980100.0, 980200.0]},
        index=pd.Index(["A", "B"], name="site_id"),
    )


def _make_solver_return(
    n_sites: int = 2,
    n_loops: int = 2,
    n_obs: int = 5,
    calibration_factor: float | None = None,
) -> tuple:
    """Minimal fake solver return tuple matching GSolveSolverReturn."""
    site_grav = np.array([980150.0, 980250.0])
    obs_residuals = np.zeros(n_obs)
    site_var = np.ones(n_sites) * 0.01
    drift = np.zeros(n_loops)
    baseline = np.zeros(n_loops)
    mask = np.ones(n_obs, dtype=bool)
    return (
        site_grav,
        obs_residuals,
        site_var,
        drift,
        baseline,
        calibration_factor,
        mask,
    )


def _make_results(
    method: int = 1,
    use_loops: bool = True,
    calculate_calibration_factor: bool = False,
    percentile_clipping: float = 95.0,
) -> GSolveResults:
    results = GSolveResults(
        method=method,
        use_loops=use_loops,
        calculate_calibration_factor=calculate_calibration_factor,
        percentile_clipping=percentile_clipping,
    )
    results.set_inputs(_make_obs_df(), _make_ref_df())
    solver_return = _make_solver_return(
        calibration_factor=1.0 if calculate_calibration_factor else None
    )
    results.set_solutions(solver_return)
    return results


# ---------------------------------------------------------------------------
# GSolveSolutionParameters
# ---------------------------------------------------------------------------


class TestGSolveSolutionParameters:
    def test_defaults_set(self):
        params = GSolveSolutionParameters(
            method=1,
            use_loops=True,
            percentile_clipping=95.0,
            calculate_calibration_factor=False,
        )
        assert params.gsolve_run_datetime is not None
        assert params.gsolve_version is not None

    def test_explicit_datetime_stored(self):
        ts = pd.Timestamp("2024-01-15 12:00:00")
        params = GSolveSolutionParameters(
            method=2,
            use_loops=False,
            percentile_clipping=90.0,
            calculate_calibration_factor=False,
            gsolve_run_datetime=ts,
        )
        assert params.gsolve_run_datetime == ts

    def test_version_string_type(self):
        params = GSolveSolutionParameters(
            method=1,
            use_loops=True,
            percentile_clipping=95.0,
            calculate_calibration_factor=False,
        )
        assert isinstance(params.gsolve_version, str)

    def test_calibration_factor_default_nan(self):
        params = GSolveSolutionParameters(
            method=1,
            use_loops=True,
            percentile_clipping=95.0,
            calculate_calibration_factor=False,
        )
        assert np.isnan(params.calculated_calibration_factor)

    def test_all_fields_stored(self):
        params = GSolveSolutionParameters(
            method=3,
            use_loops=True,
            percentile_clipping=99.0,
            calculate_calibration_factor=True,
            calculated_calibration_factor=1.0023,
        )
        assert params.method == 3
        assert params.use_loops is True
        assert params.percentile_clipping == pytest.approx(99.0)
        assert params.calculate_calibration_factor is True
        assert params.calculated_calibration_factor == pytest.approx(1.0023)


# ---------------------------------------------------------------------------
# GSolveResults — construction
# ---------------------------------------------------------------------------


class TestGSolveResultsConstruction:
    def test_params_created(self):
        r = GSolveResults(
            method=1,
            use_loops=True,
            calculate_calibration_factor=False,
            percentile_clipping=95.0,
        )
        assert isinstance(r.params, GSolveSolutionParameters)

    def test_set_inputs_stores_copies(self):
        r = GSolveResults(
            method=1,
            use_loops=True,
            calculate_calibration_factor=False,
            percentile_clipping=95.0,
        )
        obs = _make_obs_df()
        ref = _make_ref_df()
        r.set_inputs(obs, ref)
        # Verify copies are stored (not same objects)
        assert r.observations_input is not obs
        assert r.reference_sites_input is not ref
        assert list(r.observations_input.columns) == list(obs.columns)

    def test_set_solutions_populates_dataframes(self):
        r = _make_results()
        assert hasattr(r, "obs_solution")
        assert hasattr(r, "site_solution")
        assert hasattr(r, "loop_solution")

    def test_obs_solution_has_expected_columns(self):
        r = _make_results()
        for col in ("site_id", "loop", "residual", "timedelta", "active"):
            assert col in r.obs_solution.columns

    def test_site_solution_has_expected_columns(self):
        r = _make_results()
        for col in ("n_obs", "absolute_gravity", "variance", "stdev", "stderr"):
            assert col in r.site_solution.columns

    def test_loop_solution_has_expected_columns(self):
        r = _make_results()
        for col in ("n_obs", "drift", "baseline"):
            assert col in r.loop_solution.columns

    def test_site_solution_sorted_by_site_id(self):
        r = _make_results()
        assert list(r.site_solution.index) == sorted(r.site_solution.index)

    def test_loop_solution_sorted_by_loop_id(self):
        r = _make_results()
        assert list(r.loop_solution.index) == sorted(r.loop_solution.index)


# ---------------------------------------------------------------------------
# Calibration factor
# ---------------------------------------------------------------------------


class TestCalibrationFactor:
    def test_calibration_factor_property(self):
        r = _make_results(calculate_calibration_factor=True)
        assert isinstance(r.calibration_factor, float)
        assert r.calibration_factor == pytest.approx(1.0)

    def test_calibration_factor_stored_in_params(self):
        r = _make_results(calculate_calibration_factor=True)
        assert r.params.calculated_calibration_factor == pytest.approx(1.0)

    def test_calibration_none_raises_when_expected(self):
        r = GSolveResults(
            method=1,
            use_loops=True,
            calculate_calibration_factor=True,
            percentile_clipping=95.0,
        )
        r.set_inputs(_make_obs_df(), _make_ref_df())
        solver_return = _make_solver_return(calibration_factor=None)
        with pytest.raises(ValueError, match="calibration factor was not calculated"):
            r.set_solutions(solver_return)


# ---------------------------------------------------------------------------
# obs_solution / site_solution / loop_solution values
# ---------------------------------------------------------------------------


class TestSolutionValues:
    """Verify the values (not just columns) propagated into the solution frames."""

    def test_obs_solution_residuals_and_timedelta(self):
        r = GSolveResults(
            method=1,
            use_loops=True,
            calculate_calibration_factor=False,
            percentile_clipping=95.0,
        )
        r.set_inputs(_make_obs_df(), _make_ref_df())
        residuals = np.array([0.01, -0.02, 0.03, 0.0, -0.01])
        solver_return = (
            np.array([980150.0, 980250.0]),
            residuals,
            np.ones(2) * 0.01,
            np.array([0.1, 0.2]),
            np.array([1.0, 2.0]),
            None,
            np.ones(5, dtype=bool),
        )
        r.set_solutions(solver_return)
        np.testing.assert_allclose(r.obs_solution["residual"].to_numpy(), residuals)
        np.testing.assert_allclose(
            r.obs_solution["timedelta"].to_numpy(), [0.0, 0.5, 1.0, 0.0, 0.5]
        )
        assert r.obs_solution["active"].all()

    def test_obs_solution_active_mask_propagated(self):
        r = GSolveResults(
            method=1,
            use_loops=True,
            calculate_calibration_factor=False,
            percentile_clipping=95.0,
        )
        r.set_inputs(_make_obs_df(), _make_ref_df())
        mask = np.array([True, True, True, False, True])
        solver_return = (
            np.array([980150.0, 980250.0]),
            np.zeros(5),
            np.ones(2) * 0.01,
            np.array([0.1, 0.2]),
            np.array([1.0, 2.0]),
            None,
            mask,
        )
        r.set_solutions(solver_return)
        np.testing.assert_array_equal(r.obs_solution["active"].to_numpy(), mask)

    def test_site_solution_stats(self):
        r = GSolveResults(
            method=1,
            use_loops=True,
            calculate_calibration_factor=False,
            percentile_clipping=95.0,
        )
        r.set_inputs(_make_obs_df(), _make_ref_df())
        site_var = np.array([0.04, 0.09])
        solver_return = (
            np.array([980150.0, 980250.0]),
            np.zeros(5),
            site_var,
            np.array([0.1, 0.2]),
            np.array([1.0, 2.0]),
            None,
            np.ones(5, dtype=bool),
        )
        r.set_solutions(solver_return)
        # stdev == sqrt(variance)
        np.testing.assert_allclose(
            r.site_solution["stdev"].to_numpy(), np.sqrt(site_var)
        )
        # variance column stored as-is
        np.testing.assert_allclose(r.site_solution["variance"].to_numpy(), site_var)
        # stderr == sqrt(var) / sqrt(n_obs)
        n_obs = r.site_solution["n_obs"].to_numpy()
        np.testing.assert_allclose(
            r.site_solution["stderr"].to_numpy(),
            np.sqrt(site_var) / np.sqrt(n_obs),
        )

    def test_loop_solution_values(self):
        r = GSolveResults(
            method=1,
            use_loops=True,
            calculate_calibration_factor=False,
            percentile_clipping=95.0,
        )
        r.set_inputs(_make_obs_df(), _make_ref_df())
        drift = np.array([0.1, 0.2])
        baseline = np.array([1.0, 2.0])
        solver_return = (
            np.array([980150.0, 980250.0]),
            np.zeros(5),
            np.ones(2) * 0.01,
            drift,
            baseline,
            None,
            np.ones(5, dtype=bool),
        )
        r.set_solutions(solver_return)
        np.testing.assert_allclose(r.loop_solution["drift"].to_numpy(), drift)
        np.testing.assert_allclose(r.loop_solution["baseline"].to_numpy(), baseline)
        # n_obs per loop: L1 has 3, L2 has 2
        assert r.loop_solution.loc["L1", "n_obs"] == 3
        assert r.loop_solution.loc["L2", "n_obs"] == 2


# ---------------------------------------------------------------------------
# Plotting: plot_residual_cdf
# ---------------------------------------------------------------------------


def _make_plot_results() -> GSolveResults:
    """Results object with non-trivial residuals/drift suitable for plotting."""
    r = GSolveResults(
        method=2,
        use_loops=True,
        calculate_calibration_factor=False,
        percentile_clipping=90.0,
    )
    r.set_inputs(_make_obs_df(), _make_ref_df())
    solver_return = (
        np.array([980150.0, 980250.0]),
        np.array([0.01, -0.02, 0.03, 0.0, -0.01]),
        np.ones(2) * 0.01,
        np.array([0.1, 0.2]),
        np.array([1.0, 2.0]),
        None,
        np.ones(5, dtype=bool),
    )
    r.set_solutions(solver_return)
    return r


class TestPlotResidualCDF:
    def teardown_method(self):
        plt.close("all")

    def test_default_plots_each_loop(self):
        r = _make_plot_results()
        ax = r.plot_residual_cdf(show=False)
        assert isinstance(ax, plt.Axes)
        assert ax.get_title() == "Distribution of residuals for each loop"

    def test_single_loop_title(self):
        r = _make_plot_results()
        ax = r.plot_residual_cdf(loop="L1", show=False)
        assert ax.get_title() == "Distribution of residuals for loop L1"

    def test_list_of_loops(self):
        r = _make_plot_results()
        ax = r.plot_residual_cdf(loop=["L1", "L2"], show=False)
        assert isinstance(ax, plt.Axes)
        # at least one series drawn
        assert ax.get_legend() is not None

    def test_ugal_unit_label(self):
        r = _make_plot_results()
        ax = r.plot_residual_cdf(unit="uGal", show=False)
        assert "μGal" in ax.get_xlabel()

    def test_mgal_unit_label(self):
        r = _make_plot_results()
        ax = r.plot_residual_cdf(unit="mGal", show=False)
        assert "mGal" in ax.get_xlabel()

    def test_invalid_unit_raises(self):
        r = _make_plot_results()
        with pytest.raises(ValueError, match="unrecognized unit"):
            r.plot_residual_cdf(unit="furlongs", show=False)

    def test_unknown_loop_raises(self):
        r = _make_plot_results()
        with pytest.raises(ValueError, match="not found in observation residuals"):
            r.plot_residual_cdf(loop="DOES_NOT_EXIST", show=False)

    def test_invalid_ax_type_raises(self):
        r = _make_plot_results()
        with pytest.raises(TypeError, match="invalid type for ax"):
            r.plot_residual_cdf(ax="not an axes", show=False)

    def test_plots_onto_supplied_ax(self):
        r = _make_plot_results()
        _fig, ax = plt.subplots()
        returned = r.plot_residual_cdf(ax=ax, show=False)
        assert returned is ax

    def test_filename_appends_loop_suffix(self, tmp_path):
        r = _make_plot_results()
        target = tmp_path / "cdf.png"
        r.plot_residual_cdf(loop="L1", filename=str(target), show=False)
        assert (tmp_path / "cdf_loop_L1.png").exists()

    def test_filename_multiple_loops_suffix(self, tmp_path):
        r = _make_plot_results()
        target = tmp_path / "cdf.png"
        r.plot_residual_cdf(loop=["L1", "L2"], filename=str(target), show=False)
        assert (tmp_path / "cdf_loops_L1_L2.png").exists()


# ---------------------------------------------------------------------------
# Plotting: plot_residual_drift
# ---------------------------------------------------------------------------


class TestPlotResidualDrift:
    def teardown_method(self):
        plt.close("all")

    def test_default_drift_plot(self):
        r = _make_plot_results()
        ax = r.plot_residual_drift(loop="L1", show=False)
        assert isinstance(ax, plt.Axes)
        assert "drift" in ax.get_ylabel()
        assert "mGal" in ax.get_ylabel()

    def test_stem_plot_without_drift(self):
        r = _make_plot_results()
        ax = r.plot_residual_drift(loop="L1", plot_drift=False, show=False)
        assert ax.get_ylabel() == "residual (mGal)"

    def test_ugal_unit(self):
        r = _make_plot_results()
        ax = r.plot_residual_drift(loop="L1", unit="uGal", show=False)
        assert "μGal" in ax.get_ylabel()

    def test_invalid_unit_raises(self):
        r = _make_plot_results()
        with pytest.raises(ValueError, match="unrecognized unit"):
            r.plot_residual_drift(loop="L1", unit="furlongs", show=False)

    def test_invalid_ax_type_raises(self):
        r = _make_plot_results()
        with pytest.raises(TypeError, match="invalid type for ax"):
            r.plot_residual_drift(loop="L1", ax=123, show=False)

    def test_plots_onto_supplied_ax(self):
        r = _make_plot_results()
        _fig, ax = plt.subplots()
        returned = r.plot_residual_drift(loop="L1", ax=ax, show=False)
        assert returned is ax

    def test_loop_accepts_non_string(self):
        # loop is coerced to str internally; passing a value that matches the
        # string loop id should work.
        r = _make_plot_results()
        ax = r.plot_residual_drift(loop="L2", show=False)
        assert "L2" in ax.get_title()

    def test_filename_appends_loop_suffix(self, tmp_path):
        r = _make_plot_results()
        target = tmp_path / "drift.png"
        r.plot_residual_drift(loop="L1", filename=str(target), show=False)
        assert (tmp_path / "drift_loop_L1.png").exists()
