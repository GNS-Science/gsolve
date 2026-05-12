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

import datetime
from pathlib import Path

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pytest

from gsolve import GravityObservations
import gsolve.tide.ocean_load as _ocean_load
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
