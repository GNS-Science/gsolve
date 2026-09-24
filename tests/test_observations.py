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

# Description: Test cases for the GravityObservations class.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from gsolve import GravityObservations, GravitySites
from gsolve.core import data


@pytest.fixture
def legacy_observations(shared_datadir: Path) -> GravityObservations:
    obs_file = shared_datadir / "legacy.xlsx"
    return GravityObservations.from_excel(obs_file)


def dummy_data() -> dict:
    return {
        "site_id": [1, 2, 3],
        "meter_reading": [1.0, 2.0, 3.0],
        "meter_id": ["dummy", "dummy", "dummy"],
        "datetime": [
            "2021-01-01T12:00:00",
            "2021-01-01T12:10:00",
            "2021-01-01T12:20:00",
        ],
        "loop": 1,
        "comment": ["blah 1", "blah 2", "blah 3"],
    }


def dummy_data_mgal() -> dict:
    return {
        "site_id": [1, 2, 3],
        "meter_reading_mgal": [1.0, 2.0, 3.0],
        "meter_id": ["dummy", "dummy", "dummy"],
        "datetime": [
            "2021-01-01T12:00:00",
            "2021-01-01T12:10:00",
            "2021-01-01T12:20:00",
        ],
        "loop": 1,
        "comment": ["blah 1", "blah 2", "blah 3"],
    }


def dummy_observations() -> GravityObservations:
    return GravityObservations(**dummy_data())


def dummy_observation_mgal() -> GravityObservations:
    return GravityObservations(**dummy_data_mgal())


class TestObservationsInit:
    def test_gravity_observations_init_mgal(self) -> None:
        data = dummy_data_mgal()
        obj = GravityObservations(**data)
        n = len(data["site_id"])
        assert obj.data["loop"].to_list() == [str(data["loop"])] * n
        assert obj.data["comment"].to_list() == data["comment"]

    def test_gravity_observations_init_meter_units(self) -> None:
        data = dummy_data()
        obj = GravityObservations(**data)
        n = len(data["site_id"])
        assert obj.data["loop"].to_list() == [str(data["loop"])] * n
        assert obj.data["comment"].to_list() == data["comment"]

    def test_gravity_observations_init_no_meter_readings(self) -> None:
        data = dummy_data()
        data.pop("meter_reading")
        with pytest.raises(
            ValueError, match=r"meter_reading or meter_reading_mgal must be specified"
        ):
            _ = GravityObservations(**data)

    def test_gravity_observations_init_has_obs_id(self) -> None:
        data = dummy_data()

        # obs_id not specified
        obj_unspecified = GravityObservations(**data)
        assert obj_unspecified.data.index.dtype.name in {"object", "str"}

        prefixes = obj_unspecified.data.index.str.partition(".").get_level_values(0)
        assert_index_equal(
            prefixes, pd.Index(obj_unspecified.data["site_id"]), check_names=False
        )

        # obs_id specified but is None
        obj_none = GravityObservations(**data, obs_id=None)
        assert_frame_equal(obj_none.data, obj_unspecified.data)

        # obs_id is a sequence
        idx = ["id1", "id2", "id3"]
        obj_seq = GravityObservations(**data, obs_id=idx)
        assert_index_equal(obj_seq.data.index, pd.Index(idx), check_names=False)

    def test_gravity_observations_init_bad_inputs(self) -> None:
        df = pd.DataFrame(dummy_data())

        # Case 1: missing a required field
        with pytest.raises(ValueError, match=r"site_id"):
            _ = GravityObservations.from_dataframe(df.drop(columns=["site_id"]))

        # Case 2: Test where inputs uneven length
        d = df.to_dict(orient="list")
        d["site_id"] = [1, 2]
        with pytest.raises(ValueError):
            _ = GravityObservations(**d)  # ty:ignore[invalid-argument-type]

        # Case 3: Test inputs of invalid type
        d = df.to_dict(orient="list")
        d["meter_reading"] = ["a", "b", "c"]
        with pytest.raises((TypeError, ValueError)):
            _ = GravityObservations(**d)  # ty:ignore[invalid-argument-type]

    def test_gravity_observations_from_dataframe(self) -> None:
        data = dummy_data()
        n = len(data["site_id"])
        df = pd.DataFrame(data)
        obj = GravityObservations.from_dataframe(df)
        assert obj.data["loop"].to_list() == [str(data["loop"])] * n
        obj = GravityObservations.from_dataframe(df, ignore_unknown_fields=False)
        assert obj.data["comment"].to_list() == data["comment"]

    def test_gravity_observations_from_excel(self, shared_datadir: Path) -> None:
        # test 1 read legacy file
        obj1 = GravityObservations.from_excel(shared_datadir / "legacy_format.xlsx")
        assert obj1.data.shape[0] == 220

        # test read new format
        obs_file = shared_datadir / "current_format.xlsx"
        obj2 = GravityObservations.from_excel(shared_datadir / "current_format.xlsx")
        assert obj2.data.shape[0] == 220

        assert_frame_equal(obj1.data, obj2.data)

        # test read from a non-existent sheet
        with pytest.raises(ValueError, match=r"not found in"):
            _ = GravityObservations.from_excel(obs_file, sheet_name="not found in")

        # test missing datetime column
        with pytest.raises(
            ValueError, match=r"DataFrame missing required columns \[\'datetime\'\]"
        ):
            GravityObservations.from_excel(obs_file, parse_split_datetime=False)

        # GravityObservations.from_excel(obs_file,


class TestObservationsMerge:
    @staticmethod
    def _observations_objects() -> tuple[GravityObservations, GravityObservations]:
        obj_orig = dummy_observations()
        obj_unique = obj_orig.copy()
        obj_unique.data["loop"] = obj_unique.data["loop"] + "xx"
        obj_unique.set_obs_id(obj_orig.data.index + "_xx")
        return obj_orig, obj_unique

    def test_gravity_observations_merge(self) -> None:
        obj_orig, obj_unique = self._observations_objects()
        obj_final = obj_orig.merge(obj_unique)

        assert len(obj_final.data) == len(obj_unique) + len(obj_orig)

        # ensure that parameters are set to orig
        obj_unique.set_fixed_time_datum("2010-10-10T00:00:00")
        obj_unique.set_timedelta_unit("1s")
        obj_final = obj_orig.merge(obj_unique)
        assert obj_final.fixed_time_datum() == obj_orig.fixed_time_datum()
        assert obj_final.timedelta_unit() == obj_orig.timedelta_unit()

    def test_gravity_observations_merge_bad_args(self) -> None:
        obj_orig, obj_unique = self._observations_objects()

        with pytest.raises(TypeError, match=r"invalid type for other"):
            _ = obj_orig.merge("ddd")

        with pytest.raises(ValueError, match=r"invalid if_duplicate_loops arg"):
            _ = obj_orig.merge(obj_unique, if_duplicate_loops="bad_arg")

        with pytest.raises(ValueError, match=r"invalid if_duplicate_obs_ids arg"):
            _ = obj_orig.merge(obj_unique, if_duplicate_obs_ids="bad_arg")

    def test_gravity_observations_merge_duplicate_loop_flags(self) -> None:
        obj_orig, obj_unique = self._observations_objects()
        obj_dupe_loop = obj_unique.copy()
        obj_dupe_loop.data["loop"] = obj_orig.data["loop"].to_numpy()

        with pytest.raises(ValueError):
            _ = obj_orig.merge(
                obj_dupe_loop, if_duplicate_loops="error", if_duplicate_obs_ids="keep"
            )

        assert obj_dupe_loop.data["loop"].to_list() == obj_orig.data["loop"].to_list()
        with pytest.warns(UserWarning, match="keeping"):
            _ = obj_orig.merge(obj_dupe_loop, if_duplicate_loops="keep")

        with pytest.warns(UserWarning, match="dropping"):
            _ = obj_orig.merge(obj_dupe_loop, if_duplicate_loops="drop")

        with pytest.warns(UserWarning, match=r"adding suffix"):
            _ = obj_orig.merge(obj_dupe_loop, if_duplicate_loops="rename")

        # catch duplicate obs_id
        obj_dupe_obs_id = obj_unique.copy()
        obj_dupe_obs_id.data.index = obj_orig.data.index

        with pytest.raises(ValueError, match=r"duplicate obs_id"):
            _ = obj_orig.merge(
                obj_dupe_obs_id,
                if_duplicate_obs_ids="error",
            )

        with pytest.warns(UserWarning, match=r"dropping"):
            _ = obj_orig.merge(obj_dupe_obs_id, if_duplicate_obs_ids="drop")

        with pytest.warns(UserWarning, match=r"adding suffix"):
            _ = obj_orig.merge(obj_dupe_obs_id, if_duplicate_obs_ids="rename")


class TestObservationTimedelta:
    def test_gravity_observations_tdelta(self) -> None:
        obj1 = GravityObservations(**dummy_data())
        obj2 = GravityObservations(**dummy_data())
        obj2.set_column("loop", 2)
        obj2.set_column(
            "datetime", pd.to_datetime(dummy_data()["datetime"]) + pd.Timedelta("1d")
        )
        obj2.set_obs_id()
        obj3 = obj1.merge(obj2)

        # test that timedelta_unit is set correctly
        assert obj1.timedelta_unit() == pd.Timedelta("1h")
        obj1.set_timedelta_unit(pd.Timedelta("1m"))
        assert np.isclose(obj1.timedelta_unit().total_seconds(), 60.0)

        # test that tdelta columns are set correctly
        obj3.set_tdelta()
        assert "survey_tdelta" in obj3.data.columns
        assert "loop_tdelta" in obj3.data.columns
        assert np.isclose(obj3.data["survey_tdelta"].iloc[0], 0.0)
        assert obj3.data["survey_tdelta"].iloc[1:].ne(0.0).all()

        for loop in obj3.loop_ids:
            m = obj3.data["loop"].eq(loop).to_list()
            assert np.isclose(obj3.data[m]["loop_tdelta"].min(), 0.0)

    def test_gravity_observations_timedelta_unit(
        self,
    ) -> None:
        # test that default tdelta unit is set correctly
        obs = dummy_observations()
        assert obs.timedelta_unit() == pd.Timedelta("1h")
        obs.set_tdelta()
        td_hr = obs.data["survey_tdelta"].copy()

        # test that tdelta unit is set correctly
        obs.set_timedelta_unit(pd.Timedelta("1s"))
        assert obs.timedelta_unit() == pd.Timedelta("1s")

        # test that new tdelta unit was applied to data
        assert_series_equal(td_hr, obs.data["survey_tdelta"] / 3600.0)

        # now set it back to default
        obs.set_timedelta_unit(pd.Timedelta("1h"))
        assert obs.timedelta_unit() == pd.Timedelta("1h")
        assert_series_equal(td_hr, obs.data["survey_tdelta"])

    def test_gravity_observations_fixed_time_datum(
        self,
    ) -> None:
        obs = dummy_observations()
        # test that default fixed_time_datum is undefined as expected
        assert pd.isna(obs.fixed_time_datum())

        obs.set_tdelta()
        td1 = obs.data["survey_tdelta"].copy()

        # test that fixed_time_datum is set correctly
        # make fixed_time_datum = datetime.min() - 1 day
        # -> survey_tdelta should be shifted by + 1 day
        dt = pd.Timedelta("1d")
        dt_sec = dt.total_seconds() / obs.timedelta_unit().total_seconds()
        t0 = obs.data["datetime"].min() - dt
        obs.set_fixed_time_datum(t0)
        assert obs.fixed_time_datum() == t0

        # test that new fixed_time_datum was applied to tdelta
        assert_series_equal(td1 + dt_sec, obs.data["survey_tdelta"])

        # now set it back to None
        obs.set_fixed_time_datum(None)
        assert pd.isna(obs.fixed_time_datum())
        assert_series_equal(td1, obs.data["survey_tdelta"])


def test_gravity_observations_properties() -> None:
    obs = dummy_observations()
    assert obs.loop_ids == ["1"]
    assert obs.starttime == obs.data["datetime"].min()
    assert obs.endtime == obs.data["datetime"].max()

    # test that loop_ids are updated correctly
    obs.set_column("loop", 2)
    assert obs.loop_ids == ["2"]

    # test that loop_ids are d in dat_time order rather than lexically
    obs.data.iloc[-1, obs.data.columns.get_loc("loop")] = "1"
    assert obs.loop_ids == ["2", "1"]


def test_gravity_observations_activate_deactivate() -> None:
    obs = dummy_observations()
    active_column = "active"
    assert obs.data[active_column].eq(True).all()

    # case: activate/deactivate loop
    target_column = "loop"
    target_value = "1"
    kwargs = {target_column: target_value}
    expected = obs.data[target_column].ne(target_value)
    obs.deactivate(**kwargs)
    assert_series_equal(obs.data[active_column], expected, check_names=False)
    obs.activate(**kwargs)
    assert obs.data[active_column].eq(True).all()

    # ensure partial deactivation
    idx = obs.data.sample(1).index[0]
    expected = pd.Series(index=obs.data.index, data=True)
    expected[idx] = False
    target_loop_id = "junk"
    obs.data.loc[idx, target_column] = target_loop_id
    kwargs = {target_column: target_loop_id}
    obs.deactivate(**kwargs)
    assert_series_equal(obs.data[active_column], expected, check_names=False)
    obs.activate(**kwargs)
    assert obs.data[active_column].eq(True).all()

    # site
    target_column = "site_id"
    idx = obs.data.sample(1).index[0]
    target_value = obs.data.loc[idx, target_column]
    expected = pd.Series(index=obs.data.index, data=True)
    expected[idx] = False
    kwargs = {target_column: target_value}
    obs.deactivate(**kwargs)
    assert_series_equal(obs.data[active_column], expected, check_names=False)
    obs.activate(**kwargs)
    assert obs.data[active_column].eq(True).all()

    # obs_id
    target_column = "obs_id"
    target_value = obs.data.sample(1).index[0]
    expected = pd.Series(index=obs.data.index, data=True)
    expected[target_value] = False
    kwargs = {target_column: target_value}
    obs.deactivate(**kwargs)
    assert_series_equal(obs.data[active_column], expected, check_names=False)
    obs.activate(**kwargs)
    assert obs.data[active_column].eq(True).all()

    # mixed types
    target_loop_id = "xxxx"
    obs.data.iloc[0, obs.data.columns.get_loc("loop")] = target_loop_id
    target_obs_id = obs.data.index[-1]

    kwargs = {"loop": "xxxx", "obs_id": target_obs_id}
    expected = pd.Series(index=obs.data.index, data=True)
    expected.iloc[[0, -1]] = False

    obs.deactivate(**kwargs)
    assert_series_equal(obs.data[active_column], expected, check_names=False)
    obs.activate(**kwargs)
    assert obs.data[active_column].eq(True).all()


#     # site_data = obs.get_site_data(1)
#     # assert site_data.shape[0] == 3
#     # assert site_data["site_id"].to_list() == [1, 2, 3]

#     # # test that get_site_data returns a copy of the data
#     # site_data.iloc[0, site_data.columns.get_loc("site_id")] = 999
#     # assert obs.data["site_id"].iloc[0] == 1
