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

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)

import gsolve.core.utils as utils
from gsolve.core.utils import (
    DEFAULT_TIMESTAMP_COLUMNS,
    check_duplicate_index,
    columns_to_timestamp,
    expand_datetime_column,
    merge_datetime_columns,
    normalize_field_names,
    timestamp_to_columns,
    to_naive_utc_datetime,
)


@pytest.fixture
def ymd_datetime_index() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", "2022-01-03", freq="h")


@pytest.fixture
def ymd_series(ymd_datetime_index: pd.DatetimeIndex) -> pd.Series:
    ds = ymd_datetime_index.to_series()
    ds.index = pd.RangeIndex(len(ds))
    return ds


@pytest.fixture
def ymd_dataframe(ymd_series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "year": ymd_series.dt.year.to_numpy(),
            "month": ymd_series.dt.month.to_numpy(),
            "day": ymd_series.dt.day.to_numpy(),
            "hour": ymd_series.dt.hour.to_numpy(),
            "minute": ymd_series.dt.minute.to_numpy(),
            "second": ymd_series.dt.second.to_numpy(),
        }
    )


def test_to_naive_utc_datetime(
    ymd_datetime_index: pd.DatetimeIndex, ymd_series: pd.Series
) -> None:
    # Test case 1: Convert a single timestamp to naive UTC timestamp
    t1 = datetime(2022, 1, 1, 12, 0, 0)
    assert to_naive_utc_datetime(t1) == pd.Timestamp(t1, tz=None)

    # Test case 2: Convert a list of timestamps to naive UTC timestamps
    t2 = [datetime(2022, 1, 1, 12, 0, 0), datetime(2022, 1, 2, 12, 0, 0)]
    assert all(to_naive_utc_datetime(t2) == pd.to_datetime(t2))

    # test that UTC tz correctly stripped
    t3 = "2023-01-01T12:00:00"
    assert to_naive_utc_datetime(pd.Timestamp(t3, tz="UTC")) == pd.Timestamp(
        t3, tz=None
    )

    # Test case 4: Convert a pandas Series of timestamps to naive UTC timestamps
    t4a = pd.date_range("2022-01-01", "2022-01-10", freq="D", tz="UTC")
    t4b = pd.date_range("2022-01-01", "2022-01-10", freq="D")
    assert_index_equal(to_naive_utc_datetime(t4a), t4b)

    # test that non standard tz correctly stripped
    t5a = "2023-01-01T12:00:00+01:00"
    t5b = "2023-01-01T11:00:00"
    # single value
    assert to_naive_utc_datetime(t5a) == pd.Timestamp(t5b)
    # array like
    assert all(to_naive_utc_datetime([t5a, t5a]) == pd.DatetimeIndex([t5b, t5b]))

    # test pandas datetime index is returned
    assert_index_equal(to_naive_utc_datetime(ymd_datetime_index), ymd_datetime_index)

    with pytest.raises(ValueError):
        to_naive_utc_datetime("not a date")
    with pytest.raises(ValueError):
        to_naive_utc_datetime([datetime(2022, 1, 1), "not a date"])

    with pytest.raises(ValueError):
        to_naive_utc_datetime(pd.NA, allow_nat=False)
    with pytest.raises(ValueError):
        to_naive_utc_datetime([pd.Timestamp("2022-01-01"), None], allow_nat=False)


def test_to_1d_ndarray() -> None:
    # Test case 1: Convert a scalar to a 1D ndarray
    assert np.array_equal(utils.to_1d_ndarray(1), np.array([1]))

    # Test case 2: Convert a list to a 1D ndarray
    assert np.array_equal(utils.to_1d_ndarray([1, 2, 3]), np.array([1, 2, 3]))

    # Test case 3: Convert a numpy array to a 1D ndarray
    assert np.array_equal(
        utils.to_1d_ndarray(np.array([[1], [2], [3]])), np.array([1, 2, 3])
    )

    # Test case 4: Extend a length 1 array to expected size
    assert np.array_equal(
        utils.to_1d_ndarray(1, expected_size=3, extend_len_1_array=True),
        np.array([1, 1, 1]),
    )

    # Test case 5: Check that an error is raised if input cannot be converted to 1D array
    with pytest.raises(ValueError, match="input not convertible to 1d array"):
        _ = utils.to_1d_ndarray([[1, 2], [3, 4]])

    # Test case 6: Check that an error is raised if array is not expected size
    with pytest.raises(ValueError, match="expected array of size 2, got size = 1"):
        _ = utils.to_1d_ndarray(1, expected_size=2)

    # Test case 7: Check that an error is raised if expected size is None
    # but extend_len_1_array is True
    with pytest.raises(
        ValueError,
        match="expected_size must be specified if extend_len_1_array is True",
    ):
        _ = utils.to_1d_ndarray(1, extend_len_1_array=True)


def test_to_1d_ndarray_or_float():
    v1 = utils.to_1d_ndarray_or_float(1)
    assert isinstance(v1, np.float64) and v1 == 1.0

    v2 = utils.to_1d_ndarray_or_float([0, 1, 2])
    assert isinstance(v2, np.ndarray) and v2.dtype == np.float64
    assert np.array_equal(v2, np.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError):
        _ = utils.to_1d_ndarray_or_float(np.array(["a", "b"]))


def test_check_duplicate_index() -> None:
    # Test case 1: Check for duplicate index in a pandas Index
    idx1 = pd.Index(["a", "b", "c"])
    check_duplicate_index(idx1)

    # Test case 2: Check for duplicate index in a pandas DataFrame
    df2 = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        },
        index=["a", "b", "c"],
    )
    check_duplicate_index(df2)

    # Test case 3: Check for duplicate index in a pandas Series
    s3 = pd.Series([1, 2, 3], index=["a", "b", "c"])
    check_duplicate_index(s3)

    # Test case 4: Check for duplicate index in a pandas DataFrame with
    # duplicate index
    df4 = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        },
        index=["a", "b", "b"],
    )
    with pytest.raises(ValueError):
        check_duplicate_index(df4)

    # Catch no index error
    with pytest.raises(TypeError, match=r"idx must be a pandas Index"):
        check_duplicate_index([1, 2, 3])
    with pytest.raises(TypeError, match=r"idx must be a pandas Index"):
        check_duplicate_index(1)


def test_normalize_columns_names() -> None:
    # Test case 1: Normalize column names of a DataFrame with no special
    # characters
    cols_bad = ["Aa", "bB", 1, "aa bb"]
    cols_normalised = ["aa", "bb", "1", "aa_bb"]
    idx_name_bad = "test indeXX"
    idx_name_normalised = "test_indexx"

    df1 = pd.DataFrame(
        data=1, columns=cols_bad, index=pd.RangeIndex(4, name=idx_name_bad)
    )
    df2 = normalize_field_names(df1)
    assert (
        df2.columns.tolist() == cols_normalised
        and df2.index.name == idx_name_normalised
    )

    ds1 = pd.Series(data=1, index=pd.Index(cols_bad, name=idx_name_bad))
    assert normalize_field_names(ds1).index.name == idx_name_normalised

    with pytest.raises(TypeError, match=r"must be a pandas"):
        _ = normalize_field_names([1, 2, 3])
    with pytest.raises(TypeError, match=r"must be a pandas"):
        _ = normalize_field_names(1)


def test_columns_to_timestamp(
    ymd_series: pd.Series, ymd_dataframe: pd.DataFrame
) -> None:
    # Test case 1: Convert columns of a DataFrame to timestamps
    std_col_names = ymd_dataframe.columns.to_list()

    ds1 = columns_to_timestamp(ymd_dataframe)
    assert_series_equal(
        ds1,
        ymd_series,
        check_names=False,
        check_index=False,
        check_index_type=False,
    )

    # Test ensure ts_columns arg parsed correctly
    new_col_names = ["y", "m", "d", "H", "M", "S"]
    df2 = ymd_dataframe.rename(columns=dict(zip(std_col_names, new_col_names)))
    assert_series_equal(
        columns_to_timestamp(df2, ts_columns=new_col_names),
        ymd_series,
        check_index=False,
    )

    # Test catch missing columns
    with pytest.raises(ValueError, match=r"matched columns"):
        _ = columns_to_timestamp(ymd_dataframe.drop(columns="hour"))

    # Test insufficient columns
    with pytest.raises(ValueError, match=r"must be >= 3"):
        df3 = ymd_dataframe.drop(columns=ymd_dataframe.columns[2:])
        _ = columns_to_timestamp(df3)

    # Test missing columns
    with pytest.raises(ValueError, match=r"expected columns"):
        _ = columns_to_timestamp(ymd_dataframe.drop(columns="hour"))

    # check ts_columns arg correctly passed
    new_names = {c: f"xx{c}" for c in ymd_dataframe.columns}
    df_odd_names = ymd_dataframe.rename(columns=new_names)
    assert_series_equal(
        columns_to_timestamp(df_odd_names, ts_columns=list(new_names.values())),
        ymd_series,
        check_index=False,
    )
    with pytest.raises(TypeError, match=r"ts_columns must be list-like"):
        _ = columns_to_timestamp(df_odd_names, ts_columns="not_list_like")
    with pytest.raises(ValueError, match=r"length of ts_columns"):
        _ = columns_to_timestamp(df_odd_names, ts_columns=[1, 2])
    with pytest.raises(ValueError, match=r"length of ts_columns"):
        _ = columns_to_timestamp(df_odd_names, ts_columns=[1] * 25)


def test_merge_datetime_columns(
    ymd_series: pd.Series, ymd_dataframe: pd.DataFrame
) -> None:
    df = merge_datetime_columns(ymd_dataframe)
    assert_series_equal(
        df["datetime"], ymd_series, check_index=False, check_names=False
    )

    # Test case 2: Merge datetime columns with custom name
    df2 = merge_datetime_columns(ymd_dataframe, name="timestamp")
    assert_series_equal(
        df2["timestamp"], ymd_series, check_index=False, check_names=False
    )

    # Test case 3: Merge datetime columns with custom columns
    ds3 = merge_datetime_columns(ymd_dataframe, ts_columns=["year", "month", "day"])
    assert_series_equal(
        ds3["datetime"],
        ymd_series.dt.normalize(),
        check_index=False,
        check_names=False,
    )

    # Test case 4: Merge datetime columns and drop original columns
    df4 = merge_datetime_columns(ymd_dataframe, drop=True)
    assert df4.columns.tolist() == ["datetime"]
    assert all([c not in df4.columns for c in ymd_dataframe.columns])

    with pytest.raises(ValueError, match=r"already exists in dataframe"):
        _ = merge_datetime_columns(ymd_dataframe, name="hour")


def test_timestamp_to_columns(
    ymd_datetime_index: pd.DatetimeIndex,
    ymd_series: pd.Series,
    ymd_dataframe: pd.DataFrame,
) -> None:
    # Test case 1: Convert a pandas Series of timestamps to columns
    # ts = ps.series(ymd_series
    assert_frame_equal(timestamp_to_columns(ymd_series), ymd_dataframe)

    # Test case: Convert datetime index to columns
    df2 = ymd_dataframe.set_index(ymd_datetime_index)
    assert_frame_equal(timestamp_to_columns(ymd_datetime_index), df2)

    # Test case 3: Check bad input types caught
    with pytest.raises(TypeError, match=r"Input data are not datetime-like"):
        _ = timestamp_to_columns(pd.RangeIndex(100))  # type: ignore
    with pytest.raises(TypeError, match=r"Input data are not datetime-like"):
        _ = timestamp_to_columns(["a", "b", "c"])  # type: ignore
    with pytest.raises(TypeError, match=r"Input data are not datetime-like"):
        _ = timestamp_to_columns([1, 2, 3])  # type: ignore
    with pytest.raises(TypeError, match=r"Input data are not datetime-like"):
        _ = timestamp_to_columns(1)  # type: ignore

    # Test case 4: Check bad minimum resolution arg caught
    with pytest.raises(ValueError):
        _ = timestamp_to_columns(ymd_series, resolution="bad")  # type: ignore

    # Test case 5: Check minimum resolution arg will truncate output
    ds5 = ymd_series.dt.floor("1h")
    assert_frame_equal(
        timestamp_to_columns(ds5, resolution="day", round_method="floor"),
        ymd_dataframe.iloc[:, :3],
    )

    # assert_series_equal(
    #     timestamp_to_columns(ds5, resolution="day", round_method="ceil")["day"],
    #     ymd_dataframe["day"] + 1,
    # )

    with pytest.raises(ValueError, match=r"unreconised rounding method"):
        _ = timestamp_to_columns(ymd_series, resolution="day", round_method="bad")

    # Test case 7: ensure prefix is set
    prefix = "test_"
    df1 = timestamp_to_columns(ymd_series, prefix=prefix)
    assert all([str(c).startswith(prefix) for c in df1.columns])

    # deal with NaT's
    ds_nat = ymd_series.copy()
    ds_nat.iloc[0] = pd.NaT
    assert timestamp_to_columns(ds_nat).iloc[0, :].isna().all()
    assert timestamp_to_columns(ds_nat, fill_nat=99).iloc[0, :].eq(99).all()


def test_expand_datetime_column(
    ymd_series: pd.Series, ymd_dataframe: pd.DataFrame
) -> None:
    dt_cols = DEFAULT_TIMESTAMP_COLUMNS[:6]
    df = pd.DataFrame(
        {"datetime": ymd_series, "hello": "world", "goodbye": 1}, index=ymd_series.index
    )

    # Test case 1: Expand a datetime column to multiple columns
    df1 = expand_datetime_column(df)
    assert_frame_equal(df1.loc[:, dt_cols], ymd_dataframe)

    # Test case 2: Ensure datetime column is dropped
    df2 = expand_datetime_column(df, drop=True)
    assert "datetime" not in df2.columns

    # Test case 3: Check error raised if name not in dataframe
    # todo
    # with pytest.raises(KeyError):
    #     _ = expand_datetime_column(df.drop(columns="datetime"), column_name="datetime")
    # with pytest.raises(KeyError):
    #     _ = expand_datetime_column(df, column_name="missing")

    # Test case 4: Check error raised if columns already exist
    # with pytest.raises(ValueError, match=r"columns already exist in dataframe"):
    #     _ = expand_datetime_column(df.assign(hour=1))

    # Test Case 5: Check min_resolution arg correctly passed to timestamp_to_columns
    df5 = expand_datetime_column(df, resolution="microsecond")
    assert "microsecond" in df5.columns and df5["microsecond"].eq(0).all()


def test_round_coords():
    a = np.arange(-5.5, 6.0)
    b = np.arange(-5.0, 6.5)
    assert np.all(utils.round_coords(a) == b)
