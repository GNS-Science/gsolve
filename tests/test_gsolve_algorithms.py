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

"""Tests for solver core logic in gsolve.gsolve_algorithms."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from gsolve.gsolve_algorithms import (
    call_gsolve_lstsq,
    call_gsolve_calibration,
    g_solver_lstsq,
)
from gsolve.gsolve_outputs import GSolveResults


def _obs_df(include_meter_reading: bool = False) -> pd.DataFrame:
    """Create a small, stable observation set for solver tests."""
    data: dict[str, object] = {
        "site_id": ["A", "B", "A", "B", "A", "B"],
        "gravity": [1000.0, 1002.0, 1001.0, 1003.0, 1001.5, 1003.5],
        "timedelta": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        "loop": ["L1", "L1", "L1", "L2", "L2", "L2"],
    }
    if include_meter_reading:
        data["meter_reading_mgal"] = [
            1000.0,
            1002.1,
            1000.9,
            1003.1,
            1001.6,
            1003.4,
        ]
    return pd.DataFrame(data)


def _ref_df(index: list[str] | None = None) -> pd.DataFrame:
    """Create a reference-site table keyed by site_id index."""
    if index is None:
        index = ["A", "B"]
    grav = [1000.2 + i for i in range(len(index))]
    return pd.DataFrame(
        {"reference_gravity": grav},
        index=pd.Index(index, name="site_id"),
    )


def _solver_inputs(n_obs: int = 6, use_two_loops: bool = True) -> dict:
    """Build raw ndarray inputs for direct g_solver_lstsq tests."""
    site_ids = np.array(["A", "B", "A", "B", "A", "B"][:n_obs])
    obs_loop = np.array(["L1", "L1", "L1", "L2", "L2", "L2"][:n_obs])
    if not use_two_loops:
        obs_loop = np.array(["L1"] * n_obs)

    return {
        "obs_g": np.array([1000.0, 1002.0, 1001.0, 1003.0, 1001.5, 1003.5][:n_obs]),
        "obs_site_id": site_ids,
        "obs_timedelta": np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0][:n_obs]),
        "ties_site_id": np.array(["A", "B"]),
        "ties_g": np.array([1000.2, 1001.2]),
        "obs_loop": obs_loop,
    }


def _solver_inputs_robust() -> dict:
    """Build a better-conditioned synthetic dataset for clipping tests."""
    n_obs = 18
    site_ids = np.array(["A", "B", "C"] * 6)
    obs_loop = np.array((["L1"] * 6) + (["L2"] * 6) + (["L3"] * 6))
    obs_timedelta = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5] * 3)
    obs_g = np.array(
        [
            1000.0,
            1001.8,
            1003.4,
            1000.6,
            1002.2,
            1003.8,
            1000.2,
            1002.0,
            1003.6,
            1000.8,
            1002.4,
            1004.0,
            1000.3,
            1002.1,
            1003.7,
            1000.9,
            1002.5,
            1060.0,
        ]
    )

    return {
        "obs_g": obs_g,
        "obs_site_id": site_ids,
        "obs_timedelta": obs_timedelta,
        "ties_site_id": np.array(["A", "B", "C"]),
        "ties_g": np.array([1000.2, 1001.2, 1002.2]),
        "obs_loop": obs_loop,
    }


class TestCallGSolveLstsq:
    def test_no_tie_sites_raises(self) -> None:
        obs = _obs_df()
        ref = _ref_df(index=["X", "Y"])

        with pytest.raises(ValueError, match="no tie sites"):
            call_gsolve_lstsq(obs=obs, ref_sites=ref, method=1)

    def test_missing_input_column_raises(self) -> None:
        obs = _obs_df()
        ref = _ref_df()

        with pytest.raises(KeyError, match="obs dataframe missing required column"):
            call_gsolve_lstsq(
                obs=obs.rename(columns={"gravity": "xxgravity"}),
                ref_sites=ref,
                method=1,
            )

        with pytest.raises(
            KeyError, match="ref_sites dataframe missing required column"
        ):
            call_gsolve_lstsq(
                obs=obs,
                ref_sites=ref.rename(columns={"reference_gravity": "xx"}),
                method=1,
            )

    def test_returns_gsolve_results_object(self) -> None:
        obs = _obs_df()
        ref = _ref_df()

        result = call_gsolve_lstsq(
            obs=obs,
            ref_sites=ref,
            method=2,
            use_loops=True,
            percentile_clipping=100.0,
        )

        assert isinstance(result, GSolveResults)
        assert result.params.method == 2
        assert result.obs_solution.shape[0] == obs.shape[0]
        assert set(result.site_solution.index) == {"A", "B"}


class TestCallGSolveCalibration:
    def test_no_tie_sites_raises(self) -> None:
        obs = _obs_df(include_meter_reading=True)
        ref = _ref_df(index=["X", "Y"])

        with pytest.raises(ValueError, match="no tie sites"):
            call_gsolve_lstsq(obs=obs, ref_sites=ref, method=1)

    def test_returns_gsolve_results_object(self) -> None:
        obs = _obs_df(include_meter_reading=True)
        ref = _ref_df()

        result = call_gsolve_calibration(
            obs=obs,
            ref_sites=ref,
            method=2,
            use_loops=True,
            percentile_clipping=100.0,
        )

        assert isinstance(result, GSolveResults)
        assert result.params.method == 2
        assert result.obs_solution.shape[0] == obs.shape[0]
        assert set(result.site_solution.index) == {"A", "B"}

    def test_returns_calibration_as_float(self) -> None:
        obs = _obs_df(include_meter_reading=True)
        ref = _ref_df()

        result = call_gsolve_calibration(
            obs=obs,
            ref_sites=ref,
            method=1,
        )

        assert isinstance(result.calibration_factor, float)
        assert np.isfinite(result.calibration_factor)

    def test_missing_input_column_raises(self) -> None:
        obs = _obs_df(include_meter_reading=True)
        ref = _ref_df()

        with pytest.raises(KeyError, match="obs dataframe missing required column"):
            call_gsolve_lstsq(
                obs=obs.rename(columns={"gravity": "xxgravity"}),
                ref_sites=ref,
                method=1,
            )

        with pytest.raises(
            KeyError, match="ref_sites dataframe missing required column"
        ):
            call_gsolve_lstsq(
                obs=obs,
                ref_sites=ref.rename(columns={"reference_gravity": "xx"}),
                method=1,
            )


class TestGSolverLstsqValidation:
    def test_invalid_method_raises(self) -> None:
        kwargs = _solver_inputs()

        with pytest.raises(ValueError, match="invalid method"):
            g_solver_lstsq(
                **kwargs,
                method=99,
                use_loops=True,
                calculate_calibration_factor=False,
            )

    @pytest.mark.parametrize("percentile", [-0.1, 100.1])
    def test_invalid_percentile_raises(self, percentile: float) -> None:
        kwargs = _solver_inputs()

        with pytest.raises(ValueError, match="must be between 0 and 100 inclusive"):
            g_solver_lstsq(
                **kwargs,
                method=1,
                use_loops=True,
                calculate_calibration_factor=False,
                percentile_clipping=percentile,
            )

    def test_missing_non_detided_values_raises_for_calibration(self) -> None:
        kwargs = _solver_inputs()

        with pytest.raises(ValueError, match="obs_g_not_detided must be provided"):
            g_solver_lstsq(
                **kwargs,
                method=1,
                use_loops=True,
                calculate_calibration_factor=True,
                obs_g_not_detided=None,
            )


class TestGSolverLstsqBranches:
    def test_use_loops_false_returns_single_loop_terms(self) -> None:
        kwargs = _solver_inputs(use_two_loops=False)
        gravity, residuals, gravity_var, drift, baseline, cal, mask = g_solver_lstsq(
            **kwargs,
            method=1,
            use_loops=False,
            calculate_calibration_factor=False,
            percentile_clipping=100.0,
        )

        assert gravity.shape == (2, 1)
        assert residuals.shape[0] == kwargs["obs_g"].shape[0]
        assert gravity_var.shape[0] == 2
        assert drift.shape == (1,)
        assert baseline.shape == (1,)
        assert cal is None
        assert mask.dtype == bool
        assert np.all(mask)

    def test_method_3_executes_and_returns_expected_shapes(self) -> None:
        kwargs = _solver_inputs(use_two_loops=True)
        gravity, residuals, gravity_var, drift, baseline, cal, mask = g_solver_lstsq(
            **kwargs,
            method=3,
            use_loops=True,
            calculate_calibration_factor=False,
            percentile_clipping=100.0,
        )

        assert gravity.shape[0] == 2
        assert residuals.shape[0] == kwargs["obs_g"].shape[0]
        assert gravity_var.shape[0] == 2
        assert drift.shape == (2,)
        assert baseline.shape == (2,)
        assert cal is None
        assert np.all(np.isfinite(gravity))
        assert np.all(np.isfinite(gravity_var))
        assert mask.shape[0] == kwargs["obs_g"].shape[0]

    def test_percentile_100_keeps_all_observations(self) -> None:
        kwargs = _solver_inputs()
        _, _, _, _, _, _, mask = g_solver_lstsq(
            **kwargs,
            method=2,
            use_loops=True,
            calculate_calibration_factor=False,
            percentile_clipping=100.0,
        )
        assert np.all(mask)

    def test_percentile_clipping_excludes_outlier(self) -> None:
        kwargs = _solver_inputs(n_obs=6)
        kwargs["obs_g"] = np.array([1000.0, 1002.0, 1001.0, 1003.0, 1001.5, 1200.0])

        _, _, _, _, _, _, mask = g_solver_lstsq(
            **kwargs,
            method=1,
            use_loops=True,
            calculate_calibration_factor=False,
            percentile_clipping=80.0,
        )

        assert mask.shape[0] == kwargs["obs_g"].shape[0]
        assert np.sum(~mask) >= 1

    def test_percentile_clipping_robust_geometry_has_finite_outputs(self) -> None:
        kwargs = _solver_inputs_robust()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gravity, residuals, gravity_var, drift, baseline, cal, mask = (
                g_solver_lstsq(
                    **kwargs,
                    method=1,
                    use_loops=True,
                    calculate_calibration_factor=False,
                    percentile_clipping=95.0,
                )
            )

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert not runtime_warnings
        assert gravity.shape[0] == 3
        assert residuals.shape[0] == kwargs["obs_g"].shape[0]
        assert gravity_var.shape[0] == 3
        assert drift.shape == (3,)
        assert baseline.shape == (3,)
        assert cal is None
        assert np.sum(mask) >= 7
        assert np.sum(~mask) >= 1
        assert np.all(np.isfinite(gravity))
        assert np.all(np.isfinite(gravity_var))
