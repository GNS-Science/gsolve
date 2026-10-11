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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gsolve.tide.ocean_load as _ocean_load
from gsolve import GravityObservations
from gsolve.tide.ocean_load import HardispOceanLoadCorrector

# @pytest.fixture
# def qtp_timeseries_file(tmp_path: str) -> str:
#     """Create a temporary QTP time series file for testing."""


def test_read_qtp_timeseries(shared_datadir) -> None:
    """Read the QTP time series file for testing."""
    qtp_timeseries_file = shared_datadir / "qtp_timeseries_correction_load.txt"
    c = _ocean_load.qtp_to_corrector(qtp_timeseries_file, corr_type="auto")

    # test matches protocol
    assert isinstance(c, _ocean_load.OceanLoadCorrectionProvider)

    # test matches expected type
    assert isinstance(c, _ocean_load.OceanLoadTimeSeries)

    # test errors are trapped
    with pytest.raises(ValueError, match="Format error reading"):
        _ = _ocean_load.qtp_to_corrector(qtp_timeseries_file, corr_type="site-datetime")
    with pytest.raises(ValueError, match="invalid corr_type"):
        _ = _ocean_load.qtp_to_corrector(qtp_timeseries_file, corr_type="invalid")


def test_read_qtp_site_datetime(shared_datadir) -> None:
    """Read the QTP site-datetime file for testing."""
    qtp_site_datetime_file = shared_datadir / "qtp_input_Modified.csv"
    c = _ocean_load.qtp_to_corrector(qtp_site_datetime_file)

    # test matches protocol
    assert isinstance(c, _ocean_load.OceanLoadCorrectionProvider)

    # test matches expected type
    assert isinstance(c, _ocean_load.OceanLoadAtSiteTime)

    # test errors are trapped
    with pytest.raises(ValueError, match="Format error reading"):
        _ = _ocean_load.qtp_to_corrector(qtp_site_datetime_file, corr_type="timeseries")
    with pytest.raises(ValueError, match="invalid corr_type"):
        _ = _ocean_load.qtp_to_corrector(qtp_site_datetime_file, corr_type="invalid")


# ============================================================================
# Tests for HardispOceanLoadCorrector (regression tests for pyhardisp tuple handling)
# ============================================================================


@pytest.fixture
def hardisp_blq_file() -> Path:
    """Path to the TeMaari ocean loading BLQ file used for testing."""
    blq_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "ocean_load"
        / "hardisp"
        / "TeMaari_olmpp_mGal.dat"
    )
    if not blq_path.exists():
        pytest.skip(f"Test BLQ file not found at {blq_path}")
    return blq_path


@pytest.fixture
def dummy_observations_for_ocean_load() -> GravityObservations:
    """Create dummy observations with multiple sites and datetimes."""
    data = {
        "site_id": ["TGKB", "TGM01", "TGKB", "TGM02", "TGKB"],
        "meter_reading_mgal": [100.0, 101.0, 102.0, 103.0, 104.0],
        "meter_id": ["G106", "G106", "G106", "G106", "G106"],
        "datetime": [
            "2013-08-01 00:00:00",
            "2013-08-01 06:00:00",
            "2013-08-01 12:00:00",
            "2013-08-02 00:00:00",
            "2013-08-02 06:00:00",
        ],
        "loop": 1,
        "comment": ["obs1", "obs2", "obs3", "obs4", "obs5"],
    }
    return GravityObservations(**data)


class TestHardispOceanLoadCorrector:
    """Test suite for HardispOceanLoadCorrector."""

    def test_ocean_load_corrector_initialization(self, hardisp_blq_file: Path) -> None:
        """Test that HardispOceanLoadCorrector can be initialized from a BLQ file."""
        corrector = HardispOceanLoadCorrector(hardisp_blq_file)
        assert corrector is not None
        assert len(corrector.stations) > 0
        assert "TGKB" in corrector.stations

    def test_ocean_load_correction_returns_array(
        self,
        hardisp_blq_file: Path,
        dummy_observations_for_ocean_load: GravityObservations,
    ) -> None:
        """Test that ocean_load_correction returns correct shape and dtype."""
        corrector = HardispOceanLoadCorrector(hardisp_blq_file)

        corrections = corrector.ocean_load_correction(
            site_id=dummy_observations_for_ocean_load.data["site_id"],
            date_time=dummy_observations_for_ocean_load.data["datetime"],
            if_not_matched="warn",
        )

        # Check return type and shape
        assert isinstance(corrections, np.ndarray)
        assert corrections.dtype == np.float64
        assert corrections.shape == (len(dummy_observations_for_ocean_load),)
        assert len(corrections) == 5

    def test_ocean_load_correction_values_are_scalar(
        self,
        hardisp_blq_file: Path,
        dummy_observations_for_ocean_load: GravityObservations,
    ) -> None:
        """Test that all correction values are scalars (not arrays)."""
        corrector = HardispOceanLoadCorrector(hardisp_blq_file)

        corrections = corrector.ocean_load_correction(
            site_id=dummy_observations_for_ocean_load.data["site_id"],
            date_time=dummy_observations_for_ocean_load.data["datetime"],
            if_not_matched="warn",
        )

        # Ensure all values are scalar floats, not arrays
        for val in corrections:
            assert np.isscalar(val) or val.ndim == 0
            assert isinstance(float(val), float)

    def test_apply_ocean_load_correction_integration(
        self,
        hardisp_blq_file: Path,
        dummy_observations_for_ocean_load: GravityObservations,
    ) -> None:
        """Test that apply_ocean_load_correction works without raising assignment error.

        This is the main regression test for the bug fix.
        Bug: ValueError: setting an array element with a sequence
        This occurred when pyhardisp returned a tuple of arrays, and the code tried
        to assign v[0] (an array) to a scalar numpy array element.
        """
        corrector = HardispOceanLoadCorrector(hardisp_blq_file)
        obs = dummy_observations_for_ocean_load

        # This should NOT raise:
        # ValueError: setting an array element with a sequence
        obs.apply_ocean_load_correction(corrector=corrector, if_not_matched="warn")

        # Verify that the correction column was added
        assert "ocean_load_corr" in obs.data.columns

        # Verify that the column has the right shape and dtype
        assert obs.data["ocean_load_corr"].shape == (len(obs),)
        assert obs.data["ocean_load_corr"].dtype == np.float64

        # Verify that at least some corrections are non-NaN (matched observations)
        non_nan_count = obs.data["ocean_load_corr"].notna().sum()
        assert non_nan_count > 0, "Expected some matched observations with corrections"

    def test_ocean_load_correction_multiple_epochs_per_observer(
        self, hardisp_blq_file: Path
    ) -> None:
        """Test ocean loading correction with repeated observations at same site."""
        # Create observations with same site, different times
        data = {
            "site_id": ["TGKB"] * 10,
            "meter_reading_mgal": np.arange(10, dtype=float),
            "meter_id": ["G106"] * 10,
            "datetime": pd.date_range("2013-08-01", periods=10, freq="h").astype(str),
            "loop": 1,
            "comment": [f"obs{i}" for i in range(10)],
        }
        obs = GravityObservations(**data)
        corrector = HardispOceanLoadCorrector(hardisp_blq_file)

        # Should work without error
        obs.apply_ocean_load_correction(corrector=corrector, if_not_matched="error")

        # All corrections should be valid (no NaN for matched site)
        assert obs.data["ocean_load_corr"].notna().all()

        # Corrections should vary with time (first and last should be different)
        first_corr = obs.data["ocean_load_corr"].iloc[0]
        last_corr = obs.data["ocean_load_corr"].iloc[-1]
        assert first_corr != last_corr, (
            "Ocean loading corrections should vary with time"
        )

    def test_ocean_load_correction_unmatched_sites_warn(
        self, hardisp_blq_file: Path
    ) -> None:
        """Test handling of observations with sites not in the BLQ file."""
        data = {
            "site_id": ["TGKB", "UNKNOWN_SITE", "TGM01"],
            "meter_reading_mgal": [100.0, 101.0, 102.0],
            "meter_id": ["G106", "G106", "G106"],
            "datetime": [
                "2013-08-01 00:00:00",
                "2013-08-01 06:00:00",
                "2013-08-01 12:00:00",
            ],
            "loop": 1,
            "comment": ["obs1", "obs2", "obs3"],
        }
        obs = GravityObservations(**data)
        corrector = HardispOceanLoadCorrector(hardisp_blq_file)

        # Should warn but not raise with if_not_matched="warn"
        with pytest.warns(UserWarning, match="not found in station loading model"):
            obs.apply_ocean_load_correction(corrector=corrector, if_not_matched="warn")

        # Check that matched sites have values, unmatched have NaN
        assert not np.isnan(obs.data["ocean_load_corr"].iloc[0])  # TGKB matched
        assert np.isnan(obs.data["ocean_load_corr"].iloc[1])  # UNKNOWN_SITE unmatched
        assert not np.isnan(obs.data["ocean_load_corr"].iloc[2])  # TGM01 matched

    def test_ocean_load_correction_unmatched_sites_error(
        self, hardisp_blq_file: Path
    ) -> None:
        """Test that unmatched sites raise error when if_not_matched='error'."""
        data = {
            "site_id": ["TGKB", "UNKNOWN_SITE"],
            "meter_reading_mgal": [100.0, 101.0],
            "meter_id": ["G106", "G106"],
            "datetime": ["2013-08-01 00:00:00", "2013-08-01 06:00:00"],
            "loop": 1,
            "comment": ["obs1", "obs2"],
        }
        obs = GravityObservations(**data)
        corrector = HardispOceanLoadCorrector(hardisp_blq_file)

        # Should raise ValueError for unmatched site
        with pytest.raises(ValueError, match="not found in station loading model"):
            obs.apply_ocean_load_correction(corrector=corrector, if_not_matched="error")


# ============================================================================
# Tests for module-level helpers and timeseries/site-time corrector classes
# ============================================================================


def test_datetimes_to_np_datetime64() -> None:
    # scalar Timestamp -> length-1 datetime64 array
    r = _ocean_load._datetimes_to_np_datetime64(pd.Timestamp("2022-01-01"))
    assert isinstance(r, np.ndarray)
    assert np.issubdtype(r.dtype, np.datetime64)
    assert r.shape == (1,)

    # array-like -> datetime64 array of matching length
    r2 = _ocean_load._datetimes_to_np_datetime64(
        pd.to_datetime(["2022-01-01", "2022-01-02"])
    )
    assert np.issubdtype(r2.dtype, np.datetime64)
    assert len(r2) == 2

    # dtype override
    r3 = _ocean_load._datetimes_to_np_datetime64(
        pd.Timestamp("2022-01-01"), dtype="datetime64[s]"
    )
    assert r3.dtype == np.dtype("datetime64[s]")


def test_validate_timeseries_data() -> None:
    good = pd.DataFrame(
        {"correction": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(
            pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]),
            name="datetime",
        ),
    )
    # valid data returns None without raising
    assert _ocean_load._validate_timeseries_data(good) is None

    with pytest.raises(TypeError, match=r"must be a pandas DataFrame"):
        _ocean_load._validate_timeseries_data("not a dataframe")

    with pytest.raises(TypeError, match=r"not indexed by datetime"):
        _ocean_load._validate_timeseries_data(
            pd.DataFrame({"correction": [1.0]}, index=pd.RangeIndex(1))
        )

    with pytest.raises(ValueError, match=r"at least two rows"):
        _ocean_load._validate_timeseries_data(
            pd.DataFrame(
                {"correction": [1.0]},
                index=pd.DatetimeIndex(pd.to_datetime(["2022-01-01"])),
            )
        )

    with pytest.raises(ValueError, match=r"sorted in increasing order"):
        _ocean_load._validate_timeseries_data(
            pd.DataFrame(
                {"correction": [1.0, 2.0]},
                index=pd.DatetimeIndex(pd.to_datetime(["2022-01-02", "2022-01-01"])),
            )
        )


# ----------------------------------------------------------------------------
# OceanLoadAtSiteTime
# ----------------------------------------------------------------------------


@pytest.fixture
def site_time_provider() -> _ocean_load.OceanLoadAtSiteTime:
    site_ids = ["A", "A", "B"]
    datetimes = pd.to_datetime(
        ["2023-01-01 12:00", "2023-01-01 13:00", "2023-01-01 12:00"]
    )
    corrections = [0.025, 0.030, 0.015]
    return _ocean_load.OceanLoadAtSiteTime(site_ids, datetimes, corrections)


def test_site_time_provider_protocol_and_identifier(
    site_time_provider: _ocean_load.OceanLoadAtSiteTime,
) -> None:
    assert isinstance(site_time_provider, _ocean_load.OceanLoadCorrectionProvider)
    assert site_time_provider.identifier() == "OceanLoadAtSiteTime()"
    assert site_time_provider.data.shape == (3, 1)


def test_site_time_provider_lookup(
    site_time_provider: _ocean_load.OceanLoadAtSiteTime,
) -> None:
    out = site_time_provider.ocean_load_correction(
        ["A"], pd.to_datetime(["2023-01-01 12:00"])
    )
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, np.array([0.025]))

    # scalar site_id is broadcast against the datetimes
    out2 = site_time_provider.ocean_load_correction(
        "A", pd.to_datetime(["2023-01-01 12:00", "2023-01-01 13:00"])
    )
    np.testing.assert_allclose(out2, np.array([0.025, 0.030]))


def test_site_time_provider_unmatched(
    site_time_provider: _ocean_load.OceanLoadAtSiteTime,
) -> None:
    # error mode
    with pytest.raises(ValueError, match=r"not found"):
        site_time_provider.ocean_load_correction(
            ["A"], pd.to_datetime(["2099-01-01 12:00"]), if_not_matched="error"
        )

    # warn mode returns NaN for unmatched pairs
    with pytest.warns(UserWarning, match=r"not found"):
        out = site_time_provider.ocean_load_correction(
            ["A"], pd.to_datetime(["2099-01-01 12:00"]), if_not_matched="warn"
        )
    assert np.isnan(out).all()


def test_site_time_provider_length_mismatch(
    site_time_provider: _ocean_load.OceanLoadAtSiteTime,
) -> None:
    with pytest.raises(ValueError, match=r"same length"):
        site_time_provider.ocean_load_correction(
            ["A", "B"], pd.to_datetime(["2023-01-01 12:00"])
        )


def test_site_time_provider_rejects_2d_site_id() -> None:
    with pytest.raises(ValueError, match=r"1d array|1-dimensional"):
        _ocean_load.OceanLoadAtSiteTime(
            site_id=np.array([["A", "B"], ["C", "D"]]),
            date_time=pd.to_datetime(["2023-01-01", "2023-01-02"]),
            corrections=[1.0, 2.0],
        )


# ----------------------------------------------------------------------------
# OceanLoadTimeSeries
# ----------------------------------------------------------------------------


@pytest.fixture
def timeseries_provider() -> _ocean_load.OceanLoadTimeSeries:
    datetimes = pd.date_range("2023-01-01", periods=5, freq="h")
    return _ocean_load.OceanLoadTimeSeries(
        date_time=datetimes,
        corrections=[1.0, 2.0, 3.0, 4.0, 5.0],
        metadata={"src": "test"},
    )


def test_timeseries_provider_properties(
    timeseries_provider: _ocean_load.OceanLoadTimeSeries,
) -> None:
    assert isinstance(timeseries_provider, _ocean_load.OceanLoadCorrectionProvider)
    assert timeseries_provider.identifier() == "OceanLoadTimeSeries()"
    assert timeseries_provider.starttime == pd.Timestamp("2023-01-01 00:00")
    assert timeseries_provider.endtime == pd.Timestamp("2023-01-01 04:00")
    # five hourly samples span 4 hours -> 3600 s mean interval
    assert timeseries_provider.sample_rate == pytest.approx(3600.0)
    assert "src=test" in repr(timeseries_provider)


def test_timeseries_provider_metadata_default() -> None:
    datetimes = pd.date_range("2023-01-01", periods=3, freq="h")
    ts = _ocean_load.OceanLoadTimeSeries(
        date_time=datetimes, corrections=[1.0, 2.0, 3.0]
    )
    assert ts.metadata == {}


def test_timeseries_provider_out_of_range_raises(
    timeseries_provider: _ocean_load.OceanLoadTimeSeries,
) -> None:
    # Datetimes outside the series span raise before interpolation.
    with pytest.raises(ValueError, match=r"outside time series range"):
        timeseries_provider.ocean_load_correction(
            "ignored",
            pd.to_datetime(["2099-01-01 00:00"]),
            if_not_matched="error",
        )


def test_timeseries_provider_interpolates(
    timeseries_provider: _ocean_load.OceanLoadTimeSeries,
) -> None:
    # corrections are 1..5 at hourly samples from 00:00..04:00
    # linear interpolation at the half-hour midpoint -> 3.5
    mid = timeseries_provider.ocean_load_correction(
        "ignored", pd.to_datetime(["2023-01-01 02:30"])
    )
    np.testing.assert_allclose(mid, np.array([3.5]))

    # exact grid points return the stored values
    ends = timeseries_provider.ocean_load_correction(
        "ignored",
        pd.to_datetime(["2023-01-01 00:00", "2023-01-01 04:00"]),
    )
    np.testing.assert_allclose(ends, np.array([1.0, 5.0]))


def test_timeseries_provider_out_of_range_warn_sets_nan(
    timeseries_provider: _ocean_load.OceanLoadTimeSeries,
) -> None:
    with pytest.warns(UserWarning, match=r"outside time series range"):
        out = timeseries_provider.ocean_load_correction(
            "ignored",
            pd.to_datetime(["2023-01-01 02:00", "2099-01-01 00:00"]),
            if_not_matched="warn",
        )
    # in-range value interpolated, out-of-range set to NaN
    assert np.isclose(out[0], 3.0)
    assert np.isnan(out[1])


# ----------------------------------------------------------------------------
# generate_qtp_input
# ----------------------------------------------------------------------------


def test_generate_qtp_input(tmp_path: Path) -> None:
    out = tmp_path / "qtp_in.csv"
    _ocean_load.generate_qtp_input(
        site_id=["A", "B"],
        datetimes=pd.to_datetime(["2023-01-01 12:00", "2023-01-01 13:00"]),
        latitude=[-41.0, -42.0],
        longitude=[174.0, 175.0],
        elevation=10.0,  # scalar elevation broadcast to all rows
        output_file=out,
    )
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    # site id, formatted datetime, lat, lon, elevation
    first = lines[0].split(",")
    assert first[0] == "A"
    assert first[1] == "01/01/2023 12:00"
    assert float(first[4]) == pytest.approx(10.0)


def test_generate_qtp_input_shape_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"same shape"):
        _ocean_load.generate_qtp_input(
            site_id=["A", "B"],
            datetimes=pd.to_datetime(["2023-01-01 12:00"]),
            latitude=[-41.0],
            longitude=[174.0],
            elevation=10.0,
            output_file=tmp_path / "bad.csv",
        )
