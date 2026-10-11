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

"""Tests for :class:`EternaTidalParameters` and :class:`EternaPredictTidalCorrection`.

Both classes live in :mod:`gsolve.tide.earth_tide`. The ``EternaPredict`` tests
exercise the ETERNA/pygtide backend, so they require ``pygtide`` to be installed.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from gsolve.tide.earth_tide import (
    EarthTideCorrectionProvider,
    EternaPredictTidalCorrection,
    EternaTidalParameters,
)

# A valid, non-overlapping, well-ordered parameter set (two bands).
_VALID_PARAMS = [
    [0.0, 1.0, 1.16, 0.0],
    [1.5, 2.5, 1.18, 0.5],
]


# ---------------------------------------------------------------------------
# EternaTidalParameters
# ---------------------------------------------------------------------------
class TestEternaTidalParameters:
    """Tests for the tidal-parameter container class."""

    def test_init_builds_dataframe(self) -> None:
        p = EternaTidalParameters(
            freq_start=[0.0, 1.5],
            freq_stop=[1.0, 2.5],
            amplitude=[1.16, 1.18],
            phase_lead=[0.0, 0.5],
        )
        assert isinstance(p.data, pd.DataFrame)
        assert list(p.data.columns) == [
            "freq_start",
            "freq_stop",
            "amplitude",
            "phase_lead",
        ]
        assert p.data.shape == (2, 4)
        # all stored as floats
        for col in ("freq_start", "freq_stop", "amplitude", "phase_lead"):
            assert np.issubdtype(p.data[col].dtype, np.floating)

    def test_init_with_wave_group_and_extra_kwargs(self) -> None:
        p = EternaTidalParameters(
            freq_start=[0.0, 1.5],
            freq_stop=[1.0, 2.5],
            amplitude=[1.16, 1.18],
            phase_lead=[0.0, 0.5],
            wave_group=["Long", "M2"],
            note=["a", "b"],
        )
        assert "wave_group" in p.data.columns
        assert "note" in p.data.columns
        assert p.data["wave_group"].to_list() == ["Long", "M2"]

    def test_init_sorts_by_freq_start(self) -> None:
        p = EternaTidalParameters(
            freq_start=[1.5, 0.0],
            freq_stop=[2.5, 1.0],
            amplitude=[1.18, 1.16],
            phase_lead=[0.5, 0.0],
        )
        assert p.data["freq_start"].to_list() == [0.0, 1.5]
        assert p.data["freq_stop"].to_list() == [1.0, 2.5]

    def test_repr(self) -> None:
        p = EternaTidalParameters.from_array(_VALID_PARAMS)
        assert repr(p) == "EternaTidalParameters(nparam=2)"

    def test_copy_is_independent(self) -> None:
        p = EternaTidalParameters.from_array(_VALID_PARAMS)
        c = p.copy()
        assert isinstance(c, EternaTidalParameters)
        assert c.data is not p.data
        npt.assert_array_equal(c.pygtide_wavegroup_arg(), p.pygtide_wavegroup_arg())
        # mutating the copy does not touch the original
        c.data.loc[0, "amplitude"] = 99.0
        assert p.data.loc[0, "amplitude"] != pytest.approx(99.0)

    # -- validation -------------------------------------------------------
    def test_validate_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="Validation failed"):
            EternaTidalParameters(
                freq_start=[np.nan],
                freq_stop=[1.0],
                amplitude=[1.0],
                phase_lead=[0.0],
            )

    def test_validate_rejects_non_increasing_interval(self) -> None:
        # freq_start >= freq_stop
        with pytest.raises(ValueError, match="Validation failed"):
            EternaTidalParameters(
                freq_start=[1.0],
                freq_stop=[0.5],
                amplitude=[1.0],
                phase_lead=[0.0],
            )

    def test_validate_rejects_overlapping_intervals(self) -> None:
        with pytest.raises(ValueError, match="Validation failed"):
            EternaTidalParameters.from_array(
                [[0.0, 2.0, 1.0, 0.0], [1.0, 3.0, 1.0, 0.0]]
            )

    # -- constructors -----------------------------------------------------
    def test_from_array_1d(self) -> None:
        p = EternaTidalParameters.from_array([0.0, 10.0, 1.16, 0.0])
        assert p.data.shape == (1, 4)
        npt.assert_array_equal(
            p.pygtide_wavegroup_arg(), np.array([[0.0, 10.0, 1.16, 0.0]])
        )

    def test_from_array_2d(self) -> None:
        p = EternaTidalParameters.from_array(_VALID_PARAMS)
        npt.assert_array_equal(
            p.pygtide_wavegroup_arg(), np.asarray(_VALID_PARAMS, dtype=float)
        )

    def test_from_array_bad_shape_raises(self) -> None:
        # fewer than 4 columns is rejected
        with pytest.raises(ValueError, match="2D with at least 4 columns"):
            EternaTidalParameters.from_array([1.0, 2.0, 3.0])

    def test_from_array_more_than_four_columns_raises(self) -> None:
        # The implementation requires exactly 4 columns (arr.shape[1] == 4),
        # so extra columns are rejected rather than preserved.
        with pytest.raises(ValueError, match="2D with at least 4 columns"):
            EternaTidalParameters.from_array([[0.0, 1.0, 1.16, 0.0, 42.0]])

    def test_from_array_passthrough_dataframe(self) -> None:
        df = pd.DataFrame(
            _VALID_PARAMS,
            columns=["freq_start", "freq_stop", "amplitude", "phase_lead"],
        )
        p = EternaTidalParameters.from_array(df)
        npt.assert_array_equal(
            p.pygtide_wavegroup_arg(), np.asarray(_VALID_PARAMS, dtype=float)
        )

    def test_from_dataframe(self) -> None:
        df = pd.DataFrame(
            {
                "freq_start": [0.0, 1.5],
                "freq_stop": [1.0, 2.5],
                "amplitude": [1.16, 1.18],
                "phase_lead": [0.0, 0.5],
                "wave_group": ["Long", "M2"],
            }
        )
        p = EternaTidalParameters.from_dataframe(df)
        assert p.data.shape[0] == 2
        assert "wave_group" in p.data.columns

    def test_from_dataframe_include_index(self) -> None:
        df = pd.DataFrame(
            {
                "freq_start": [0.0],
                "freq_stop": [1.0],
                "amplitude": [1.16],
                "phase_lead": [0.0],
            },
            index=pd.Index(["Long"], name="wave_group"),
        )
        p = EternaTidalParameters.from_dataframe(df, include_index=True)
        assert "wave_group" in p.data.columns
        assert p.data["wave_group"].to_list() == ["Long"]

    def test_from_dataframe_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"freq_start": [0.0], "freq_stop": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            EternaTidalParameters.from_dataframe(df)

    def test_from_excel_round_trip(self, tmp_path) -> None:
        df = pd.DataFrame(
            {
                "freq_start": [0.0, 1.5],
                "freq_stop": [1.0, 2.5],
                "amplitude": [1.16, 1.18],
                "phase_lead": [0.0, 0.5],
            }
        )
        fname = tmp_path / "params.xlsx"
        df.to_excel(fname, index=False)
        p = EternaTidalParameters.from_excel(fname)
        npt.assert_allclose(p.pygtide_wavegroup_arg(), df.to_numpy(dtype=float))

    def test_from_excel_multiple_sheets_raises(self, tmp_path) -> None:
        df = pd.DataFrame(
            {
                "freq_start": [0.0],
                "freq_stop": [1.0],
                "amplitude": [1.16],
                "phase_lead": [0.0],
            }
        )
        fname = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(fname) as writer:
            df.to_excel(writer, sheet_name="one", index=False)
            df.to_excel(writer, sheet_name="two", index=False)
        with pytest.raises(ValueError, match="multiple sheets"):
            EternaTidalParameters.from_excel(fname, sheet_name=[0, 1])

    def test_default(self) -> None:
        p = EternaTidalParameters.default()
        npt.assert_array_equal(
            p.pygtide_wavegroup_arg(), np.array([[0.0, 10.0, 1.16, 0.0]])
        )

    def test_default_custom_values(self) -> None:
        p = EternaTidalParameters.default(
            freq_start=0.1, freq_stop=5.0, amplitude_factor=1.2, phase_lead=1.0
        )
        npt.assert_array_equal(
            p.pygtide_wavegroup_arg(), np.array([[0.1, 5.0, 1.2, 1.0]])
        )

    def test_quick_tide_pro_defaults(self) -> None:
        p = EternaTidalParameters.quick_tide_pro_defaults()
        # 12 wave groups defined in the method
        assert p.data.shape[0] == 12
        assert "wave_group" in p.data.columns
        # all expected named constituents present
        assert {"M2", "O1", "K1", "S2"}.issubset(set(p.data["wave_group"]))
        # resulting intervals must still be valid (sorted, non-overlapping)
        arr = p.pygtide_wavegroup_arg()
        assert np.all(arr[:, 1] > arr[:, 0])  # stop > start
        assert np.all(np.diff(arr[:, 0]) > 0)  # sorted ascending

    def test_pygtide_wavegroup_arg_is_copy(self) -> None:
        p = EternaTidalParameters.from_array(_VALID_PARAMS)
        arr = p.pygtide_wavegroup_arg()
        arr[0, 0] = -999.0
        # mutating the returned array must not affect internal data
        assert p.data.loc[0, "freq_start"] != pytest.approx(-999.0)


# ---------------------------------------------------------------------------
# EternaPredictTidalCorrection
# ---------------------------------------------------------------------------
class TestEternaPredictTidalCorrection:
    """Tests for the ETERNA/pygtide-backed tidal correction provider."""

    def test_satisfies_protocol(self) -> None:
        c = EternaPredictTidalCorrection()
        assert isinstance(c, EarthTideCorrectionProvider)

    def test_default_uses_default_tidal_params(self) -> None:
        c = EternaPredictTidalCorrection()
        assert isinstance(c.tidal_params, EternaTidalParameters)
        npt.assert_array_equal(
            c.tidal_params.pygtide_wavegroup_arg(),
            np.array([[0.0, 10.0, 1.16, 0.0]]),
        )

    def test_repr_contains_pgt_kwargs(self) -> None:
        c = EternaPredictTidalCorrection()
        r = repr(c)
        assert r.startswith("EternaPredictTidalCorrection(")
        for key in (
            "tidalpoten",
            "tidalcompo",
            "amtruncate",
            "poletidecor",
            "lodtidecor",
        ):
            assert key in r

    def test_init_with_eterna_tidal_params_object(self) -> None:
        params = EternaTidalParameters.from_array(_VALID_PARAMS)
        c = EternaPredictTidalCorrection(tidal_params=params)
        npt.assert_array_equal(
            c.tidal_params.pygtide_wavegroup_arg(),
            np.asarray(_VALID_PARAMS, dtype=float),
        )
        # stored params are a copy, not the same object
        assert c.tidal_params is not params

    def test_init_with_array_tidal_params(self) -> None:
        c = EternaPredictTidalCorrection(tidal_params=_VALID_PARAMS)
        assert isinstance(c.tidal_params, EternaTidalParameters)
        npt.assert_array_equal(
            c.tidal_params.pygtide_wavegroup_arg(),
            np.asarray(_VALID_PARAMS, dtype=float),
        )

    def test_set_tidal_params(self) -> None:
        c = EternaPredictTidalCorrection()
        c.set_tidal_params(_VALID_PARAMS)
        npt.assert_array_equal(
            c.tidal_params.pygtide_wavegroup_arg(),
            np.asarray(_VALID_PARAMS, dtype=float),
        )

    def test_identifier_default_matches_repr(self) -> None:
        c = EternaPredictTidalCorrection()
        assert c.identifier() == repr(c)

    def test_identifier_with_kwargs(self) -> None:
        c = EternaPredictTidalCorrection()
        ident = c.identifier(custom=5)
        assert "custom=5" in ident

    def test_time_series_shape_and_columns(self) -> None:
        c = EternaPredictTidalCorrection()
        ts = c.time_series(
            lat=-41.0,
            lon=174.0,
            elev=0.0,
            starttime="2021-07-01T00:00",
            duration=2,
            sample_interval=3600,
        )
        assert isinstance(ts, pd.DataFrame)
        assert list(ts.columns) == ["signal", "tide", "pole_tide", "lod_tide"]
        assert isinstance(ts.index, pd.DatetimeIndex)
        assert len(ts) > 0

    def test_time_series_unit_scaling(self) -> None:
        c = EternaPredictTidalCorrection()
        kwargs = {
            "lat": -41.0,
            "lon": 174.0,
            "elev": 0.0,
            "starttime": "2021-07-01T00:00",
            "duration": 2,
            "sample_interval": 3600,
        }
        ts_native = c.time_series(unit="nm/s^2", **kwargs)
        ts_ugal = c.time_series(unit="ugal", **kwargs)
        ts_mgal = c.time_series(unit="mgal", **kwargs)
        # nm/s^2 -> ugal is x1e-3, nm/s^2 -> mgal is x1e-4
        npt.assert_allclose(
            ts_ugal["signal"].to_numpy(),
            ts_native["signal"].to_numpy() * 1e-3,
            rtol=1e-9,
        )
        npt.assert_allclose(
            ts_mgal["signal"].to_numpy(),
            ts_native["signal"].to_numpy() * 1e-4,
            rtol=1e-9,
        )

    def test_time_series_invalid_unit_raises(self) -> None:
        c = EternaPredictTidalCorrection()
        with pytest.raises(ValueError, match="invalid unit"):
            c.time_series(
                lat=-41.0,
                lon=174.0,
                elev=0.0,
                starttime="2021-07-01T00:00",
                duration=2,
                unit="furlongs",
            )

    def test_time_series_invalid_duration_raises(self) -> None:
        c = EternaPredictTidalCorrection()
        with pytest.raises(ValueError, match="invalid duration"):
            c.time_series(
                lat=-41.0,
                lon=174.0,
                elev=0.0,
                starttime="2021-07-01T00:00",
                duration=0,
            )

    def test_time_series_invalid_sample_interval_raises(self) -> None:
        c = EternaPredictTidalCorrection()
        with pytest.raises(ValueError, match="invalid sample_interval"):
            c.time_series(
                lat=-41.0,
                lon=174.0,
                elev=0.0,
                starttime="2021-07-01T00:00",
                duration=2,
                sample_interval=0,
            )

    def test_tidal_correction_returns_floats(self) -> None:
        c = EternaPredictTidalCorrection()
        dt = pd.DatetimeIndex(["2021-07-01T01:00", "2021-07-01T01:30"])
        out = c.tidal_correction(
            lat=np.array([-41.0, -41.0]),
            lon=np.array([174.0, 174.0]),
            elev=np.array([0.0, 0.0]),
            date_time=dt,
            site_id="A",
            unit="mgal",
        )
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))

    def test_tidal_correction_matches_time_series(self) -> None:
        # A correction at a sample time should match the (negated) time
        # series value interpolated at that same instant.
        c = EternaPredictTidalCorrection()
        lat, lon, elev = -41.0, 174.0, 0.0
        ts = c.time_series(
            lat=lat,
            lon=lon,
            elev=elev,
            starttime="2021-07-01T00:00",
            duration=3,
            sample_interval=60,
        )
        # pick an exact sample instant from the series
        sample_time = ts.index[30]
        out = c.tidal_correction(
            lat=np.array([lat]),
            lon=np.array([lon]),
            elev=np.array([elev]),
            date_time=pd.DatetimeIndex([sample_time]),
            site_id="A",
            unit="mgal",
        )
        # tidal_correction negates the time series signal
        npt.assert_allclose(out[0], -ts.loc[sample_time, "signal"], rtol=1e-6)

    def test_tidal_correction_multiple_sites(self) -> None:
        c = EternaPredictTidalCorrection()
        dt = pd.DatetimeIndex(
            [
                "2021-07-01T01:00",
                "2021-07-01T02:00",
                "2021-07-01T01:00",
                "2021-07-01T02:00",
            ]
        )
        out = c.tidal_correction(
            lat=np.array([-41.0, -41.0, 20.0, 20.0]),
            lon=np.array([174.0, 174.0, -70.0, -70.0]),
            elev=np.array([0.0, 0.0, 100.0, 100.0]),
            date_time=dt,
            site_id=np.array(["A", "A", "B", "B"]),
            unit="mgal",
        )
        assert out.shape == (4,)
        assert np.all(np.isfinite(out))
        # different sites should yield different corrections
        assert not np.allclose(out[:2], out[2:])

    def test_tidal_correction_requires_site_id(self) -> None:
        c = EternaPredictTidalCorrection()
        dt = pd.DatetimeIndex(["2021-07-01T01:00"])
        with pytest.raises(ValueError, match="site_id is a required parameter"):
            c.tidal_correction(
                lat=np.array([-41.0]),
                lon=np.array([174.0]),
                elev=np.array([0.0]),
                date_time=dt,
                site_id=None,
            )
