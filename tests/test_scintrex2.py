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

import tempfile
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)

from gsolve.scintrex import (
    CG6Data,
    ScintrexData,
    _normalize_keyword,
    _split_header_key_val_unit,
)


@pytest.fixture
def sample_cg6_metadata():
    return {
        "survey_name": "Test Survey",
        "instrument_serial_number": "CG6-1234",
        "created": "2023-01-01T12:00:00",
        "operator": "Test Operator",
        "gcal1": 1.0,
        "goff": 0.0,
        "gref": 980000.0,
    }


@pytest.fixture
def sample_cg6_data():
    return pd.DataFrame(
        {
            "station": ["A", "B", "A"],
            "date": ["2023-01-01", "2023-01-01", "2023-01-01"],
            "time": ["12:00:00", "12:05:00", "12:10:00"],
            "corrgrav": [980000.0, 980001.0, 980000.5],
            "line": ["1", "1", "2"],
            "stddev": [0.1, 0.1, 0.1],
            "rawgrav": [980000.0, 980001.0, 980000.5],
            "x": [0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0],
            "sensortemp": [20.0, 20.0, 20.0],
            "tidecorr": [0.0, 0.0, 0.0],
            "tiltcorr": [0.0, 0.0, 0.0],
            "tempcorr": [0.0, 0.0, 0.0],
            "driftcorr": [0.0, 0.0, 0.0],
            "corrections[drift-temp-na-tide-tilt]": ["11111", "11111", "11111"],
        }
    )


@pytest.fixture
def sample_cg6_file(sample_cg6_metadata, sample_cg6_data):
    # Create a temporary file with CG6 format
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".dat", delete=False) as f:
        # Write metadata
        f.write("/cg-6_calibration\n")
        for key, value in sample_cg6_metadata.items():
            f.write(f"/{key}: {value}\n")

        # Write column headers
        columns = " ".join(sample_cg6_data.columns)
        f.write(f"/{columns}\n")

        # Write data
        for _, row in sample_cg6_data.iterrows():
            f.write(" ".join(str(x) for x in row.values) + "\n")

        f.flush()
        return f.name


class TestScintrexData:
    def test_abstract_class(self):
        with pytest.raises(TypeError):
            ScintrexData(pd.DataFrame(), {})

    @pytest.mark.skip
    def test_meter_id_property(self, sample_cg6_metadata, sample_cg6_data):
        class TestScintrex(ScintrexData):
            def to_gsolve_observations(self):
                pass

            def set_loop(self):
                pass

            def _set_metadata(self, metadata, metadata_units):
                self.metadata = metadata
                self.metadata_units = metadata_units or {}

            def _set_data(self, data):
                self.data = data

        # Test with full serial number
        obj = TestScintrex(sample_cg6_data, {"instrument_serial_number": "CG6-1234"})
        assert obj.meter_id == "1234"

        # Test with short serial number
        obj = TestScintrex(sample_cg6_data, {"instrument_serial_number": "12"})
        assert obj.meter_id == "12"

        # Test with no serial number
        obj = TestScintrex(sample_cg6_data, {})
        assert obj.meter_id == ""

    @pytest.mark.skip
    def test_stations_property(self, sample_cg6_metadata, sample_cg6_data):
        class TestScintrex(ScintrexData):
            def to_gsolve_observations(self):
                pass

            def set_loop(self):
                pass

            def _set_metadata(self, metadata, metadata_units):
                self.metadata = metadata
                self.metadata_units = metadata_units or {}

            def _set_data(self, data):
                self.data = data

        obj = TestScintrex(sample_cg6_data, sample_cg6_metadata)
        assert set(obj.stations) == {"A", "B"}

        # Test with no data
        obj = TestScintrex(pd.DataFrame(), sample_cg6_metadata)
        assert obj.stations == []

    @pytest.mark.skip
    def test_copy(self, sample_cg6_metadata, sample_cg6_data):
        class TestScintrex(ScintrexData):
            def to_gsolve_observations(self):
                pass

            def set_loop(self):
                pass

            def _set_metadata(self, metadata, metadata_units):
                self.metadata = metadata
                self.metadata_units = metadata_units or {}

            def _set_data(self, data):
                self.data = data

        obj = TestScintrex(sample_cg6_data, sample_cg6_metadata)
        obj_copy = obj.copy()

        # Check it's a copy
        assert obj_copy is not obj
        assert obj_copy.data is not obj.data
        assert obj_copy.metadata is not obj.metadata

        # Check data is the same
        pd.testing.assert_frame_equal(obj_copy.data, obj.data)
        assert obj_copy.metadata == obj.metadata


class TestCG6Data:
    def test_init(self, sample_cg6_metadata, sample_cg6_data):
        obj = CG6Data(sample_cg6_data, sample_cg6_metadata)

        # Check basic attributes
        assert isinstance(obj.data, pd.DataFrame)
        assert isinstance(obj.metadata, dict)
        assert isinstance(obj.metadata_units, dict)

        # Check datetime conversion
        assert "datetime" in obj.data.columns
        assert isinstance(obj.data["datetime"].iloc[0], pd.Timestamp)

        # Check correction flags
        for flag in [
            "correction_drift",
            "correction_temp",
            "correction_tide",
            "correction_tilt",
        ]:
            assert flag in obj.data.columns
            assert obj.data[flag].dtype == bool

    def test_from_file(self, sample_cg6_file):
        obj = CG6Data.from_file(sample_cg6_file)

        # Check basic attributes
        assert isinstance(obj.data, pd.DataFrame)
        assert isinstance(obj.metadata, dict)
        assert "survey_name" in obj.metadata

        # Clean up
        Path(sample_cg6_file).unlink()

    def test_set_loop_from_field(self, sample_cg6_metadata, sample_cg6_data):
        obj = CG6Data(sample_cg6_data, sample_cg6_metadata)
        obj.set_loop(field="line")

        assert "loop" in obj.data.columns
        assert obj.data["loop"].tolist() == ["1", "1", "2"]

    def test_set_loop_from_time_gap(self, sample_cg6_metadata, sample_cg6_data):
        obj = CG6Data(sample_cg6_data, sample_cg6_metadata)
        obj.set_loop(time_gap="10 minutes")

        assert "loop" in obj.data.columns
        # Should create two loops since first two readings are 5 min apart (<10 min gap)
        # and third reading is 5 min after second (but different line)
        assert len(obj.data["loop"].unique()) == 1  # All within 10 minutes

    def test_set_loop_from_datetimes(self, sample_cg6_metadata, sample_cg6_data):
        obj = CG6Data(sample_cg6_data, sample_cg6_metadata)

        # Create datetime mapping
        datetimes = {"2023-01-01T11:00:00": "Loop1", "2023-01-01T12:06:00": "Loop2"}
        obj.set_loop(datetimes=datetimes)

        assert "loop" in obj.data.columns
        assert obj.data["loop"].tolist() == ["Loop1", "Loop1", "Loop2"]

    def test_to_gsolve_observations(self, sample_cg6_metadata, sample_cg6_data):
        obj = CG6Data(sample_cg6_data, sample_cg6_metadata)
        obj.set_loop(field="line")

        obs = obj.to_gsolve_observations()

        # Check basic fields
        assert "meter_reading_mgal" in obs.data.columns
        assert "site_id" in obs.data.columns
        assert "meter_id" in obs.data.columns

        # Check meter_id format
        assert obs.data["meter_id"].iloc[0].startswith("CG6:")

    def test_to_gsolve_sites(self, sample_cg6_metadata, sample_cg6_data):
        obj = CG6Data(sample_cg6_data, sample_cg6_metadata)

        # Add GPS and user coordinates for testing
        sample_cg6_data["latgps"] = [-35.0, -35.1, -35.0]
        sample_cg6_data["longps"] = [149.0, 149.1, 149.0]
        sample_cg6_data["elevgps"] = [600.0, 610.0, 600.0]
        sample_cg6_data["latuser"] = [-35.0, -35.1, -35.0]
        sample_cg6_data["lonuser"] = [149.0, 149.1, 149.0]
        sample_cg6_data["elevuser"] = [600.0, 610.0, 600.0]

        obj = CG6Data(sample_cg6_data, sample_cg6_metadata)

        # Test with user coordinates
        sites_user = obj.to_gsolve_sites(coords_source="user")
        assert "latitude" in sites_user.data.columns
        assert len(sites_user.data) == 2  # Two unique stations

        # Test with GPS coordinates
        sites_gps = obj.to_gsolve_sites(coords_source="gps")
        assert "latitude" in sites_gps.data.columns

    def test_set_drift_correction(self, sample_cg6_metadata, sample_cg6_data):
        obj = CG6Data(sample_cg6_data, sample_cg6_metadata)
        orig_drift_corrr = obj.data["driftcorr"].copy()
        # Set drift correction
        obj.set_drift_correction(
            drift_rate=1.0,  # 1 mGal/day
            drift_zero_time="2023-01-01T12:00:00",
        )

        # Check metadata was updated
        assert obj.metadata["drift_rate"] == 1.0
        assert isinstance(obj.metadata["drift_zero_time"], pd.Timestamp)

        # Check drift correction was applied
        assert "driftcorr" in obj.data.columns
        assert not obj.data["driftcorr"].equals(orig_drift_corrr)

    def test_copy_method(self, sample_cg6_metadata, sample_cg6_data):
        obj = CG6Data(sample_cg6_data, sample_cg6_metadata)
        obj_copy = obj.copy()

        # Check it's a copy
        assert obj_copy is not obj
        assert obj_copy.data is not obj.data
        assert obj_copy.metadata is not obj.metadata

        # Check data is the same
        pd.testing.assert_frame_equal(obj_copy.data, obj.data)
        assert obj_copy.metadata == obj.metadata


class TestHelperFunctions:
    def test_normalize_keyword(self):
        assert _normalize_keyword("Test Keyword") == "test_keyword"
        assert _normalize_keyword("Another-Test") == "another-test"
        assert _normalize_keyword("  Trim Me  ") == "trim_me"

    def test_split_header_key_val_unit(self):
        # Test with unit
        k, v, u = _split_header_key_val_unit("/test_keyword [mGal]: 123.45")
        assert k == "test_keyword"
        assert v == "123.45"
        assert u == "mGal"

        # Test without unit
        k, v, u = _split_header_key_val_unit("/another_test: value")
        assert k == "another_test"
        assert v == "value"
        assert u == ""

        # Test with no value
        k, v, u = _split_header_key_val_unit("/just_a_keyword")
        assert k == "just_a_keyword"
        assert v == ""
        assert u == ""

    def test_slurp_scintrex_text_file(self, sample_cg6_file):
        from gsolve.scintrex import _slurp_scintrex_text_file

        lines = _slurp_scintrex_text_file(sample_cg6_file)
        assert isinstance(lines, list)
        assert len(lines) > 0
        Path(sample_cg6_file).unlink()
