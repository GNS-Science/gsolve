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

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from gsolve.core import data
from gsolve.core.data import (
    COMMON_FIELDS,
    TERRAIN_DENSITY,
    WATER_DENSITY,
    DataFieldSpecification,
    GSolveParameters,
    GSolveTable,
)


@pytest.fixture
def gsolve_table_subclass():
    class TestClass(data.GSolveTable):
        _known_fields = {
            "a": data.DataFieldSpecification("a", str, "", True),
            "b": data.DataFieldSpecification("b", float, 0.0, True),
            "c": data.DataFieldSpecification("c", "datetime", pd.NaT, False),
            "d": data.DataFieldSpecification("d", bool, False, False),
        }
        _index_field = "a"

    return TestClass


class TestDataFieldSpecification:
    def test_data_field_specification(self):
        # Test case 1: Create a DataFieldSpecification object

        fs = data.DataFieldSpecification("site_id", str, "", True)
        assert fs.name == "site_id"
        assert fs.dtype == str
        assert fs.default == ""
        assert fs.required is True

        # Test case 2: Create a DataFieldSpecification object with a default value
        fs = data.DataFieldSpecification("longitude", float, 0.0)
        assert fs.name == "longitude"
        assert fs.dtype == float
        assert fs.default == 0.0
        assert fs.required is False

    # def test_data_field_specification_convert(self):
    #     fs = data.DataFieldSpecification("site_id", str, "", True)
    #     assert fs.convert(1) == "1"
    #     fs = data.DataFieldSpecification("site_id", float, "", False)
    #     assert fs.convert(1) == 1.0

    # def test_data_field_specification_convert(self):
    #     dfs = DataFieldSpecification("test", int, default=0)
    #     assert dfs.convert("5") == 5

    #     dfs_dt = DataFieldSpecification("dt", "datetime")
    #     dt = pd.Timestamp("2020-01-01T12:00:00")
    #     assert pd.Timestamp(dfs_dt.convert(dt)) == dt

    #     dfs_td = DataFieldSpecification("td", "timedelta")
    #     td = pd.Timedelta("1 days")
    #     assert dfs_td.convert("1 days") == td

    #     dfs_bool = DataFieldSpecification("flag", bool, converter=lambda x: bool(x))
    #     assert dfs_bool.convert(1) is True

    #     dfs_null = DataFieldSpecification("stupid_converter", dtype=None, default=None)
    #     assert dfs_null.convert("anything") == "anything"
    #     assert dfs_null.convert(123) == 123

    # def test_data_field_specification_create_column(self):
    #     dfs = DataFieldSpecification("test", int, default=1)
    #     idx = pd.RangeIndex(10)
    #     s = dfs.create_column([1, 2, 3])
    #     assert isinstance(s, pd.Series)
    #     pdt.assert_series_equal(s, pd.Series([1, 2, 3]))

    #     pdt.assert_series_equal(
    #         dfs.create_column(None, index=idx),
    #         pd.Series([1] * 10, index=idx),
    #     )


def test_gsolve_table_known_fields(gsolve_table_subclass):
    assert gsolve_table_subclass.known_fields() == ["a", "b", "c", "d"]
    assert gsolve_table_subclass.required_fields() == ["a", "b"]


class DummyTable(GSolveTable):
    _known_fields = COMMON_FIELDS
    _default_excel_sheet_name = "Sheet1"

    def __init__(self, **kwargs) -> None:
        self.data = pd.DataFrame(kwargs)


def test_gsolve_table_bool_and_len():
    t = DummyTable(
        site_id=["A"],
        obs_id=["1"],
        loop=["L"],
        datetime=[pd.Timestamp("2020-01-01")],
        active=[True],
    )
    assert bool(t)
    assert len(t) == 1

    t_empty = DummyTable(site_id=[], obs_id=[], loop=[], datetime=[], active=[])
    assert not bool(t_empty)
    assert len(t_empty) == 0


def test_gsolve_table_known_and_required_fields():
    assert set(DummyTable.known_fields()) == set(COMMON_FIELDS.keys())
    req = DummyTable.required_fields()
    assert "site_id" in req and "obs_id" in req and "loop" in req and "datetime" in req


def test_gsolve_table_set_column():
    t = DummyTable(
        site_id=["A"],
        obs_id=["1"],
        loop=["L"],
        datetime=[pd.Timestamp("2020-01-01")],
        active=[True],
    )
    t.set_column("active", [False])
    assert (t.data["active"] == [False]).all()


def test_gsolve_table_from_dataframe_and_csv(tmp_path):
    df = pd.DataFrame(
        {
            "site_id": ["A"],
            "obs_id": ["1"],
            "loop": ["L"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "active": [True],
        }
    )
    t = DummyTable.from_dataframe(df)
    assert isinstance(t, DummyTable)
    assert t.data.shape[0] == 1

    csv_file = tmp_path / "test.csv"
    df.to_csv(csv_file, index=False)
    t2 = DummyTable.from_csv(csv_file)
    assert isinstance(t2, DummyTable)
    assert t2.data.shape[0] == 1


class TestGSolveParameters:
    def _dummy_params_class(self):
        @dataclass
        class MyParams(GSolveParameters):
            a: int = 1
            b: float = 2.0

        return MyParams

    def test_gsolve_parameters_copy_and_dict(self):
        p = self._dummy_params_class()()
        p2 = p.copy()
        assert p is not p2
        assert p.to_dict() == {"a": 1, "b": 2.0}

    def test_to_series_and_from_series(self):
        _cls = self._dummy_params_class()
        p = self._dummy_params_class()()
        s = p.to_series()
        assert isinstance(s, pd.Series)
        assert s["a"] == 1 and s["b"] == 2.0

        p2 = _cls.from_series(s)
        assert isinstance(p2, _cls)
        assert p2.a == 1 and p2.b == 2.0

    def test_default_parameters(self):
        defaults = self._dummy_params_class().default_values()
        assert defaults == {"a": 1, "b": 2.0}

    def test_summary(self):
        p = self._dummy_params_class()()
        summary_list = p.summary(as_list=True)
        assert isinstance(summary_list, list)
        summary_str = p.summary(as_list=False)
        assert isinstance(summary_str, str)
        assert "a: 1" in summary_str and "b: 2.0" in summary_str
