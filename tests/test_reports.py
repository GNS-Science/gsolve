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

"""Tests for report assembly and consistency checks in gsolve.reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import gsolve.reports as reports_module
from gsolve import GravityObservations, GravitySites
from gsolve.gsolve_outputs import GSolveResults
from gsolve.reports import GSolveReport


def _make_observations() -> GravityObservations:
    return GravityObservations(
        site_id=["A", "B", "A", "B", "A", "B"],
        meter_reading_mgal=[1000.0, 1002.0, 1001.0, 1003.0, 1001.5, 1003.5],
        meter_id=["G106"] * 6,
        datetime=[
            "2021-01-01T00:00:00",
            "2021-01-01T00:10:00",
            "2021-01-01T00:20:00",
            "2021-01-01T01:00:00",
            "2021-01-01T01:10:00",
            "2021-01-01T01:20:00",
        ],
        loop=["L1", "L1", "L1", "L2", "L2", "L2"],
        comment=["o1", "o2", "o3", "o4", "o5", "o6"],
    )


def _make_sites() -> GravitySites:
    return GravitySites(
        site_id=["A", "B"],
        latitude=[-41.0, -41.1],
        longitude=[174.0, 174.1],
        height_ellipsoidal=[10.0, 20.0],
    )


def _make_results(obs: GravityObservations) -> GSolveResults:
    obs_input = obs.data.loc[:, ["site_id", "loop", "loop_tdelta"]].copy()
    obs_input = obs_input.rename(columns={"loop_tdelta": "timedelta"})
    ref_input = pd.DataFrame(
        {"reference_gravity": [1000.2, 1001.2]},
        index=pd.Index(["A", "B"], name="site_id"),
    )

    results = GSolveResults(
        method=1,
        use_loops=True,
        calculate_calibration_factor=False,
        percentile_clipping=95.0,
    )
    results.set_inputs(obs_input, ref_input)

    site_grav = np.array([1000.3, 1001.3])
    residuals = np.array([0.01, -0.02, 0.01, -0.01, 0.0, 0.01])
    site_var = np.array([0.01, 0.02])
    drift = np.array([0.01, -0.01])
    baseline = np.array([0.0, 0.0])
    calibration_factor = None
    mask = np.array([True, True, True, True, True, False])

    results.set_solutions(
        (site_grav, residuals, site_var, drift, baseline, calibration_factor, mask)
    )
    return results


@dataclass
class _DummyAnomalies:
    data: pd.DataFrame
    params: dict
    tcorr_params: dict | None = None


@dataclass
class _DummyTerrainCorrections:
    data: pd.DataFrame
    params: dict


class TestGSolveReportAssembly:
    def test_init_requires_required_inputs(self) -> None:
        obs = _make_observations()
        sites = _make_sites()
        results = _make_results(obs)

        with pytest.raises(ValueError, match="must be provided"):
            GSolveReport(observations=None, sites=sites, results=results)  # type: ignore[arg-type]

    def test_assembly_builds_site_obs_loop_tables(self) -> None:
        obs = _make_observations()
        sites = _make_sites()
        results = _make_results(obs)

        report = GSolveReport(observations=obs, sites=sites, results=results)

        assert "n_obs_input" in report.site_data.columns
        assert "n_obs_used" in report.site_data.columns
        assert "in_loops" in report.site_data.columns
        assert "included_in_solution" in report.obs_data.columns
        assert "has_site_solution" in report.obs_data.columns
        assert "n_obs_input" in report.loop_data.columns
        assert "n_obs_used" in report.loop_data.columns
        assert not any(c.endswith("_duplicate") for c in report.site_data.columns)
        assert not any(c.endswith("_duplicate") for c in report.obs_data.columns)

    def test_params_sections_are_populated(self) -> None:
        obs = _make_observations()
        sites = _make_sites()
        results = _make_results(obs)

        report = GSolveReport(observations=obs, sites=sites, results=results)

        assert "observations" in report.params
        assert "solution" in report.params


class TestGSolveReportTerrainCorrectionConsistency:
    def test_inconsistent_terrain_corrections_raise_with_anomaly_tcorr(self) -> None:
        obs = _make_observations()
        sites = _make_sites()
        results = _make_results(obs)

        anomalies = _DummyAnomalies(
            data=pd.DataFrame({"anomaly": [1.0, 2.0]}, index=pd.Index(["A", "B"])),
            params={"method": "dummy"},
            tcorr_params={"tcorr_zone_1": {"min_dist": 0.0, "max_dist": 1000.0}},
        )
        terrain_bad = _DummyTerrainCorrections(
            data=pd.DataFrame({"tcorr_zone_1": [0.1, 0.2]}, index=pd.Index(["A", "B"])),
            params={"tcorr_zone_1": {"min_dist": 0.0, "max_dist": 900.0}},
        )

        with pytest.raises(ValueError, match="terrain_corrections are inconsistent"):
            GSolveReport(
                observations=obs,
                sites=sites,
                results=results,
                anomalies=anomalies,  # type: ignore[arg-type]
                terrain_corrections=terrain_bad,  # type: ignore[arg-type]
            )

    def test_consistent_terrain_corrections_are_accepted(self) -> None:
        obs = _make_observations()
        sites = _make_sites()
        results = _make_results(obs)

        tc_params = {"tcorr_zone_1": {"min_dist": 0.0, "max_dist": 1000.0}}
        anomalies = _DummyAnomalies(
            data=pd.DataFrame({"anomaly": [1.0, 2.0]}, index=pd.Index(["A", "B"])),
            params={"method": "dummy"},
            tcorr_params=tc_params,
        )
        terrain_ok = _DummyTerrainCorrections(
            data=pd.DataFrame({"tcorr_zone_1": [0.1, 0.2]}, index=pd.Index(["A", "B"])),
            params=tc_params,
        )

        report = GSolveReport(
            observations=obs,
            sites=sites,
            results=results,
            anomalies=anomalies,  # type: ignore[arg-type]
            terrain_corrections=terrain_ok,  # type: ignore[arg-type]
        )

        assert report.terrain_correction_data is not None
        assert "tcorr_zone_1" in report.terrain_correction_data.columns


class TestGSolveReportToExcel:
    def test_to_excel_raises_if_workbook_exists_and_error(self, tmp_path: Path) -> None:
        obs = _make_observations()
        sites = _make_sites()
        results = _make_results(obs)
        report = GSolveReport(observations=obs, sites=sites, results=results)

        out_file = tmp_path / "report.xlsx"
        out_file.touch()

        with pytest.raises(ValueError, match="already exists"):
            report.to_excel(out_file, if_workbook_exists="error")

    def test_to_excel_append_raises_if_existing_worksheets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obs = _make_observations()
        sites = _make_sites()
        results = _make_results(obs)
        report = GSolveReport(observations=obs, sites=sites, results=results)

        out_file = tmp_path / "report.xlsx"
        out_file.touch()

        monkeypatch.setattr(
            reports_module,
            "get_excel_worksheets",
            lambda _fname: ["observations", "sites"],
        )

        with pytest.raises(ValueError, match="already exist"):
            report.to_excel(
                out_file,
                if_workbook_exists="append",
                if_sheet_exists="error",
            )

    def test_to_excel_writes_expected_sheets_without_terrain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obs = _make_observations()
        sites = _make_sites()
        results = _make_results(obs)
        report = GSolveReport(observations=obs, sites=sites, results=results)

        calls: list[dict[str, Any]] = []

        def _fake_write_excel_worksheet(df, filename, sheet_name, **kwargs):
            calls.append(
                {"sheet_name": sheet_name, "kwargs": kwargs, "shape": df.shape}
            )

        monkeypatch.setattr(
            reports_module, "write_excel_worksheet", _fake_write_excel_worksheet
        )

        out_file = tmp_path / "report.xlsx"
        report.to_excel(out_file, if_workbook_exists="replace", if_sheet_exists="new")

        assert [c["sheet_name"] for c in calls] == [
            "observations",
            "sites",
            "loops",
            "metadata",
        ]
        assert calls[0]["kwargs"]["if_workbook_exists"] == "replace"
        assert calls[1]["kwargs"]["if_workbook_exists"] == "append"
        assert calls[2]["kwargs"]["if_workbook_exists"] == "append"
        assert calls[3]["kwargs"]["if_workbook_exists"] == "append"

    def test_to_excel_writes_terrain_sheet_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obs = _make_observations()
        sites = _make_sites()
        results = _make_results(obs)

        terrain_ok = _DummyTerrainCorrections(
            data=pd.DataFrame({"tcorr_zone_1": [0.1, 0.2]}, index=pd.Index(["A", "B"])),
            params={"tcorr_zone_1": {"min_dist": 0.0, "max_dist": 1000.0}},
        )
        report = GSolveReport(
            observations=obs,
            sites=sites,
            results=results,
            terrain_corrections=terrain_ok,  # type: ignore[arg-type]
        )
        # Keep this test focused on worksheet-writing flow; in this test setup the
        # dummy terrain parameters are plain dict values (not metadata-serializable).
        report.params = {
            k: v for k, v in report.params.items() if hasattr(v, "to_series")
        }

        sheet_names: list[str] = []

        def _fake_write_excel_worksheet(_df, _filename, sheet_name, **_kwargs):
            sheet_names.append(sheet_name)

        monkeypatch.setattr(
            reports_module, "write_excel_worksheet", _fake_write_excel_worksheet
        )

        out_file = tmp_path / "report.xlsx"
        report.to_excel(out_file, if_workbook_exists="replace", if_sheet_exists="new")

        assert sheet_names == [
            "observations",
            "sites",
            "loops",
            "terrain_corrections",
            "metadata",
        ]
