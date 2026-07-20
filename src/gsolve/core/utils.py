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

"""Utility functions used across the gsolve codebase."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pandas.api.types import is_datetime64_any_dtype, is_dict_like, is_list_like
from pandas.api.typing import NaTType, NAType

from gsolve.core._typing import (
    AllowedTimestampResolution,
    AllowedTimestampRoundingMethods,
    DatetimeArray,
    DatetimeScalar,
    StringArray,
    TimedeltaScalar,
)

__all__ = [
    "is_list_like",
    "is_dict_like",
    "to_naive_utc_datetime",
    "check_duplicate_index",
    "normalize_field_names",
    "normalize_str",
    "DEFAULT_TIMESTAMP_COLUMNS",
    "merge_datetime_columns",
    "columns_to_timestamp",
    "timestamp_to_columns",
    "expand_datetime_column",
    "prepare_writable_df",
    "GSolveDataWarning",
    "generate_loop_intervals",
    "identify_loop_blocks",
    "loops_from_gaps",
    "generate_loop_names",
    "round_coords",
]


@overload
def to_naive_utc_datetime(
    t: DatetimeScalar, allow_nat: Literal[True], **kwargs
) -> pd.Timestamp | NaTType: ...


@overload
def to_naive_utc_datetime(
    t: DatetimeScalar, allow_nat: Literal[False], **kwargs
) -> pd.Timestamp: ...


@overload
def to_naive_utc_datetime(
    t: DatetimeScalar, allow_nat: bool = True, **kwargs
) -> pd.Timestamp | NaTType: ...


@overload
def to_naive_utc_datetime(
    t: pd.Series,
    allow_nat: bool = True,
    **kwargs,
) -> pd.Series: ...


@overload
def to_naive_utc_datetime(
    t: list | tuple | NDArray | pd.Index | pd.DatetimeIndex,
    allow_nat: bool = True,
    **kwargs,
) -> pd.DatetimeIndex: ...


def to_naive_utc_datetime(
    t: DatetimeScalar | DatetimeArray | NaTType,
    allow_nat: bool = True,
    **kwargs,
) -> pd.Timestamp | pd.Series | pd.DatetimeIndex | NaTType:
    """
    Convert inputs to UTC time, but with timezone information set to None.

    Datetime-like inputs that are timezone-aware are converted to UTC and
    then stripped of their timezone information. All other inputs are
    converted using ``pandas.to_datetime``

    Parameters
    ----------
    t : datetime-like or array-like
        The timestamp(s) to convert.
    allow_nat : bool, default True
        If False, raise ValueError if any input resolves to NaT.
    kwargs :
        Additional keyword arguments to be passed to ``pandas.to_datetime()``.

    Returns
    -------
    datetimes : pandas.Timestamp, pandas.Series, or pandas.DatetimeIndex
        A copy of the inputs converted to UTC but without timezone information.

    See Also
    --------
    pandas.to_datetime
        The underlying function used to convert the input to a timestamp.
    """
    # scalars
    if isinstance(t, (NaTType, NAType)):
        rval = pd.NaT

    elif isinstance(t, pd.Timestamp):
        rval = t if t.tz is None else t.tz_convert("UTC").tz_localize(None)

    elif isinstance(t, pd.DatetimeIndex):
        rval = t if t.tz is None else t.tz_convert("UTC").tz_localize(None)
    elif isinstance(t, pd.Series):
        ds = t if t.dtype == "datetime64[ns]" else pd.to_datetime(t, **kwargs)
        rval = ds if ds.dt.tz is None else ds.dt.tz_convert("UTC").dt.tz_localize(None)

    # arrays
    elif isinstance(t, DatetimeArray):
        idx = pd.DatetimeIndex(pd.to_datetime(t, **kwargs))
        rval = idx if idx.tz is None else idx.tz_convert("UTC").tz_localize(None)

    else:
        return_scalar = False

        if not is_list_like(t):
            t = [t]
            return_scalar = True
        try:
            idx = pd.to_datetime(t, **kwargs)
        except Exception:
            raise ValueError(
                f"unable to convert input '{t}' of type {type(t).__name__} "
                "to Timestamp or DateTimeIndex"
            )

        rval = idx if idx.tz is None else idx.tz_convert("UTC").tz_localize(None)
        rval = rval[0] if return_scalar else rval

    if not allow_nat:
        if isinstance(rval, (pd.DatetimeIndex, pd.Series)):
            if any(rval.isna()):
                raise ValueError(
                    "input contains values that resolve to NaT and allow_nat=False"
                )
        elif rval is pd.NaT:
            raise ValueError("input resolves to NaT and allow_nat=False")

    return rval


def to_1d_ndarray(
    a: ArrayLike,
    expected_size: int | None = None,
    extend_len_1_array: bool = False,
) -> NDArray:
    _a = np.atleast_1d(a)
    if _a.ndim > 1:
        _a = np.squeeze(_a)
    if _a.ndim != 1:
        raise ValueError(f"input not convertible to 1d array")

    if extend_len_1_array and _a.size == 1:
        if expected_size is not None:
            _a = np.full(expected_size, _a[0])
        else:
            raise ValueError(
                "expected_size must be specified if extend_len_1_array is True"
            )
    if expected_size is not None and _a.size != expected_size:
        raise ValueError(
            f"expected array of size {expected_size}, got size = {_a.size}"
        )

    return _a


def to_1d_ndarray_or_float(a: ArrayLike) -> NDArray[np.float64] | np.float64:
    _a = to_1d_ndarray(a).astype(np.float64)
    if _a.size == 1:
        return _a[0]
    else:
        return _a


# Remove in future release
def check_duplicate_index(idx: pd.Index | pd.DataFrame | pd.Series) -> None:
    """Raise a ValueError if the index contains duplicate values."""
    if isinstance(idx, (pd.DataFrame, pd.Series)):
        _idx = idx.index
    elif isinstance(idx, pd.Index):
        _idx = idx
    else:
        raise TypeError(
            f"idx must be a pandas Index, DataFrame, or Series, not {type(idx).__name__}"
        )

    if _idx.duplicated().any():
        idx_name = _idx.name if _idx.name else "index"
        idx_dupes = _idx[_idx.duplicated().tolist()]
        raise ValueError(f"duplicate index values: {idx_dupes.unique().to_list()}")


@overload
def normalize_field_names(df: pd.DataFrame) -> pd.DataFrame: ...


@overload
def normalize_field_names(df: pd.Series) -> pd.Series: ...


def normalize_field_names(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Return a copy of ``df`` with column values and axis.names converted to snake_case.

    If ``df`` is a DataFrame, ``df.columns``, ``df.columns.name``, and ``df.index.name``
    are normalized. If ``df`` is a Series, then only ``df.name`` and ``df.index.name`` are normalized.

    Parameters
    ----------
    df : DataFrame or Series
        The object to normalize.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        The same type as ``df``.
    """
    if isinstance(df, pd.DataFrame):
        return df.rename(columns=normalize_str).rename_axis(
            columns=normalize_str, index=normalize_str
        )
    elif isinstance(df, pd.Series):
        df = df.rename_axis(index=normalize_str)
        df.name = normalize_str(str(df.name))
        return df
    else:
        raise TypeError(
            f"df must be a pandas DataFrame or Series, not {type(df).__name__}"
        )


def normalize_str(s: str | int | float | bool | None) -> str | None:
    """Convert ``s`` to str and format it to snake_case.

    Parameters
    ----------
    s : str, int, float, bool, or None
        The value to convert and normalize. Do nothing if ``s`` is None

    Returns
    -------
    str or None
        The normalized string or None if the input was None.
    """
    if s is not None:
        return str(s).strip().lower().replace(" ", "_")
    return None


DEFAULT_TIMESTAMP_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
)

AVAIL_TRUNCATION_COLUMNS = DEFAULT_TIMESTAMP_COLUMNS[2:-1]
_TIMESTAMP_COLUMNS_TO_RESOLUTION: dict[str, str] = {
    "day": "1D",
    "hour": "1h",
    "minute": "1m",
    "second": "1s",
    "microsecond": "1us",
    "nanosecond": "1ns",
}


def merge_datetime_columns(
    df: pd.DataFrame,
    name: str = "datetime",
    ts_columns: str | Sequence[str] | None = None,
    drop: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Merge discrete datetime columns into ``pandas.Timestamp``.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to be converted.
    name : str, default = "datetime"
        The label for the output timestamp column.
    ts_columns : array-like, optional
        An ordered sequence of column names representing the date and time
        components to be parsed.  The default is
        ("year", "month", "day", "hour", "minute", "second", "microsecond").
        ``ts_columns`` is parsed in order from largest time unit (year) down
        to the smallest (microseconds). ``ts_columns`` must define at least
        year, month, and day columns, and no 'gaps' are permitted.
    drop : bool, default = False
        Drop the date and time columns after conversion.
    kwargs
        Additional keyword arguments to be passed to ``pandas.to_datetime()``.

    Returns
    -------
    pandas.DataFrame
        Modifed copy of the input DataFrame.

    See Also
    --------
    columns_to_timestamp
    pandas.to_datetime
        The underlying function used to convert the columns to a timestamp.
    """
    if name in df.columns:
        raise ValueError(f"column '{name}' already exists in dataframe")

    ts_columns = ts_columns or list(DEFAULT_TIMESTAMP_COLUMNS)
    timestamps = columns_to_timestamp(df, ts_columns=ts_columns, **kwargs)
    rval = df.assign(**{name: timestamps})
    if drop:
        return rval.drop(columns=ts_columns, errors="ignore")
    return rval


def columns_to_timestamp(
    df: pd.DataFrame,
    ts_columns: str | Sequence[str] | None = None,
    **kwargs,
) -> pd.Series:
    """
    Generate a Series of Timestamps from discrete date & time columns in ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to be converted.
    ts_columns : array-like of str, optional
        An ordered sequence of column names representing the date and time
        components to be parsed.  The elements correspond to the default columns
        ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond').
        ``ts_columns`` is evaluated in order from largest time unit (year) down to
        (microseconds). ``ts_columns`` must at  define at least 'year',
        'month', and 'day' columns. E.g. ts_columns = ['yy', 'mm', 'dd', 'HH', 'MM', "SS"]
    kwargs
        Additional keyword arguments passed to ``pandas.to_datetime``.

    Returns
    -------
    pandas.Series
        Series of pandas.Timestamp objects.

    See Also
    --------
    merge_datetime_columns

    timestamp_to_columns
        The inverse function to ``columns_to_timestamp``.
    pandas.to_datetime
        The underlying function used to convert the columns to a timestamp.
    """
    # input sanity checks

    ts_columns = ts_columns or DEFAULT_TIMESTAMP_COLUMNS
    if not is_list_like(ts_columns):
        raise TypeError(
            f"ts_columns must be list-like, not {type(ts_columns).__name__}"
        )

    n_ts_columns = len(ts_columns)
    if not 3 <= n_ts_columns <= len(DEFAULT_TIMESTAMP_COLUMNS):
        raise ValueError(
            f"length of ts_columns is {n_ts_columns}, "
            "must be between 3 (=ymd) and 8 (=ymdHMSun)"
        )
    matched_columns = [c for c in ts_columns if c in df.columns]
    n_matched = len(matched_columns)

    # columns matching sanity checks
    if not 2 < n_matched <= n_ts_columns:
        raise ValueError(
            f"found {n_matched} columns matching ts_columns, "
            f"must be >= 3 (ymd) and <= {n_ts_columns} the length of ts_columns"
        )
    if matched_columns != list(ts_columns[:n_matched]):
        raise ValueError(
            f"expected columns {ts_columns[:n_matched]} "
            f"!= matched columns {matched_columns}"
        )
    rename_map = dict(zip(matched_columns, DEFAULT_TIMESTAMP_COLUMNS))
    return pd.to_datetime(
        df.loc[:, matched_columns].rename(columns=rename_map), **kwargs
    )


def timestamp_to_columns(
    ds: pd.Series | pd.DatetimeIndex,
    resolution: AllowedTimestampResolution | None = "second",
    round_method: AllowedTimestampRoundingMethods = "round",
    fill_nat: int | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Convert a Series of ``pandas.Timestamp`` to discrete date & time columns.

    Parameters
    ----------
    ds : pandas.Series
        The Series or array like  containing the timestamps to be converted.
    resolution : default "second"
        Truncate output columns to ``resolution``. The truncation method is
    round_method : {"round", "floor", "ceil"}, default "round"
        Control how datetimes are truncated to the specified resolution.
        The default 'round' is appropriate where ``resolution`` is a time increment.
        'floor' might be a better choice where ``resolution`` is a date increment.
    fill_nat : int, default = None
        If not None, fill ``pandas.NaT`` values with ``fill_nat``. NaT's are
        correctly split to NaN's, hoever a side effect is that the
        dataframe dtype will be `float` dtype rather than `int`
    prefix : str, default = ""
        Prepend ``prefix`` to output column names. This is useful when
        splitting multiple datetime columns, to ensure that the output
        columns are uniquely named.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the discrete time components.

    See Also
    --------
    columns_to_timestamp
    expand_datetime_column
    """
    if not is_datetime64_any_dtype(ds):
        raise TypeError("Input data are not datetime-like")
    if isinstance(ds, pd.Index):
        ds = ds.to_series()

    if resolution is not None:
        if resolution not in AVAIL_TRUNCATION_COLUMNS:
            raise ValueError(
                f"resolution '{resolution}' is not in {AVAIL_TRUNCATION_COLUMNS}"
            )

        _res = _TIMESTAMP_COLUMNS_TO_RESOLUTION[resolution]

        if round_method == "floor":
            ds = ds.dt.floor(_res)
        elif round_method == "round":
            ds = ds.dt.round(_res)
        elif round_method == "ceil":
            ds = ds.dt.ceil(_res)
        else:
            raise ValueError("unreconised rounding method '{round_method}'")

    df = pd.DataFrame(
        data={
            "year": ds.dt.year,
            "month": ds.dt.month,
            "day": ds.dt.day,
            "hour": ds.dt.hour,
            "minute": ds.dt.minute,
            "second": ds.dt.second,
            "microsecond": ds.dt.microsecond,
            "nanosecond": ds.dt.nanosecond,
        },
        index=ds.index,
    )

    if resolution is not None:
        iloc_resolution = DEFAULT_TIMESTAMP_COLUMNS.index(resolution)
        df = df.iloc[:, : iloc_resolution + 1]

    if fill_nat:
        df = df.fillna(fill_nat)
    if prefix:
        df = df.rename(columns={c: f"{prefix}{c}" for c in df.columns})
    return df


def expand_datetime_column(
    df: pd.DataFrame,
    column_name: str | Sequence[str] = "",
    resolution: AllowedTimestampResolution | None = "second",
    round_method: AllowedTimestampRoundingMethods = "round",
    prefix: str | StringArray = "",
    insert_after: bool = True,
    drop: bool = False,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Expand datetime column(s) to discrete date and time component columns.

    The output columns will be named 'year', 'month', 'day', 'hour', 'minute', 'second',
    'microsecond', 'nanosecond', depending on the specied ``resolution`` and ``prefix``
    parameters.

    This method facillitates the export of data for reading by Microsoft Excel and
    other spreadsheet software.  These applications arbitrarily mutilate date and time
    data on import. Splitting such data into discrete date and time component columns
    ensures that all information is preserved.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to be converted.
    column_name : str, list-like,  default is ""
        The columns to be split. If column_name not specified, split all datetime
        columns.
    drop : bool, default is False
        Drop ``column_name`` column after conversion.
    resolution : {'year', 'month', 'day', 'hour', 'minute', 'second',
        'microsecond', 'nanosecond'}, default is "second"
        Output datetime components down to specified resolution.
    round_method : {'floor', 'ceil', 'round'}, default is 'round':
        Control how datetimes are truncated to the specified resolution. The default
        'round' is appropriate where ``resolution`` is a time increment. 'floor'
        might be a better choice where ``resolution`` is 'year', 'month' or 'day'.
    prefix : str, list-like, default is ""
        Prepend ``prefix`` to output column names. Behaviour depends on how many columns
        are to be split::

            - If ``prefix`` is empty and 1 column to split, then use standard column names
            "year", "month",...
            - If ``prefix`` is empty and multiple columns to split, then prepend source
            column name to output "ColA_year", "ColA_month"..., "ColB_year", ... etc.
            - If ``prefix`` is defined, then a ``prefix`` must be specified for each column
            to split
    insert_after : bool, default is True
        If True, insert new columns after the column being split. If False,
        append the new columns to the end of the DataFrame.
    drop : bool, default is False
        If True, drop the source datetime columns from the output.
    overwrite : bool, default False
        If False, raise a ValueError if splitting columns would overwrite existing
        columns. Note that if overwrite is True, existing columns will be deleted
        prior to splitting to ensure the order of output columns is as expected.

    Returns
    -------
    pandas.DataFrame
        Modifed copy of the input DataFrame.

    See Also
    --------
    timestamp_to_columns

    """
    candidate_columns = [str(c) for c in df.columns if is_datetime64_any_dtype(df[c])]
    if not candidate_columns:
        return df.copy()

    if column_name == "":
        cols_to_split = candidate_columns
    elif isinstance(column_name, str):
        # if column_name not in candidate_columns:
        #     raise ValueError(
        #         f"column '{column_name}' not found or not datetime-like in dataframe"
        #     )
        cols_to_split = [column_name]
    elif isinstance(column_name, Sequence):
        cols_to_split = list(column_name)
    else:
        raise TypeError(
            f"column_name must be a string or list-like, not {type(column_name).__name__}"
        )

    cols_to_split = [str(c) for c in cols_to_split if str(c) in candidate_columns]
    if not cols_to_split:
        raise ValueError(
            f"the specified column_name(s) are either missing or or are "
            "not datetime-like columns"
        )

    if prefix is None or prefix == "":
        if len(cols_to_split) > 1:
            prefixes = [f"{n}_" for n in cols_to_split]
        else:
            prefixes = [""]
    else:
        prefixes = prefix if is_list_like(prefix) else [prefix]
        if len(prefixes) != len(cols_to_split):
            raise ValueError(
                f"inconsisitent 'column_name' and 'prefix' arg lengths: "
                f"{len(cols_to_split)} != {len(prefixes)}"
            )

    for col, pre in zip(cols_to_split, prefixes):
        ts_df = timestamp_to_columns(
            df[col], resolution=resolution, round_method=round_method, prefix=pre
        )
        existing = ts_df.columns.intersection(df.columns).to_list()
        if existing:
            if not overwrite:
                raise ValueError(
                    f"overwrite=False and splitting would overwrite columns: {existing}"
                )
            else:
                df = df.drop(columns=existing)

        if not insert_after:
            df = pd.concat([df, ts_df], axis=1)
        else:
            i_dt = df.columns.get_loc(col)
            if not isinstance(i_dt, int):
                raise TypeError(
                    "unexpected error: could not locate resolution column in output dataframe"
                )
            i_dt += 1
            df = pd.concat([df.iloc[:, :i_dt], ts_df, df.iloc[:, i_dt:]], axis=1)

        if drop:
            return df.drop(columns=col)

    return df


def prepare_writable_df(
    df: pd.DataFrame,
    normalize_column_names: bool = True,
    expand_datetime: str | None = None,
    datetime_resolution: AllowedTimestampResolution | None = "second",
    datetime_round_method: Literal["floor", "ceil", "round"] = "floor",
    datetime_prefix: str = "",
    drop_datetime: bool = False,
    bool_to_int: bool = True,
) -> pd.DataFrame:
    """Prepare a DataFrame for writing to a file.

    The primary purpose of this function is to ensure that data are formatted
    in a consistent manner. This is particularly important when writing to Excel
    spreadsheets, which will destructively modify datetime data on reading.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be prepared.
    normalize_column_names : bool, default True
        Make column names lowercase with no spaces.
    expand_datetime : str, default None
        Name of the column holding datetime to expand.
    datetime_resolution : str | None, default 'second'
        The resolution to which datettime column are to be epaneded. See
        ``expand_datetime_column`` for explanation.
    datetime_round_method : {'floor', 'ceil', 'round'}, default 'floor'
        Control how datetimes are truncated to the specified resolution.
        See ``expand_datetime_column`` for explanation.
    datetime_prefix : str, default ""
        Prefix to prepend to expanded datetime column names.
    drop_datetime : bool, default True
        If True and ``expand_datetime`` is not None, drop the expanded
        datetime column from output DataFrame.
    bool_to_int : bool, default True
        Whether to convert boolean columns to integers {False: 0, True: 1}.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if expand_datetime is not None and expand_datetime:
        df = expand_datetime_column(
            df,
            column_name=expand_datetime,
            drop=drop_datetime,
            resolution=datetime_resolution,
            round_method=datetime_round_method,
            prefix=datetime_prefix,
        )
    if normalize_column_names:
        df = normalize_field_names(df)
    if bool_to_int:
        df = df.astype({c: int for c in df.select_dtypes(include=bool)})
    return df


class GSolveDataWarning:
    """A convenience class for collecting and displaying warning messages.

    Its primary purpose is to aggregate warning messages, display them
    in a controlled manner, and provide a summary of errors/warnings ecountered.

    Parameters
    ----------
    prefix : str
        A prefix string to prepend to each warning message.
    show : bool, default True
        Whether to print warning messages immediately or store them for later display.

    Attributes
    ----------
    prefix : str
        The prefix string to prepend to each warning message.
    show : bool
        Whether to print warning messages immediately or store them for later display.
    messages : list of str
        A list of warning messages.
    count: int
        The total number of warning messages stored

    Examples
    --------
    >>> warner = GSolveDataWarning(prefix="oops!")
    >>> warner("Something Bad")
    oops!: Something Bad
    >>> warner.final_msg()
    oops!: 1 problem(s) encountered

    """

    def __init__(
        self,
        prefix: str,
        show: bool = True,
    ) -> None:
        self.prefix: str = prefix
        self.show: bool = bool(show)
        self.messages: list[str] = []

    @property
    def count(self) -> int:  # noqa: D102
        return len(self.messages)

    def __call__(self, msg: str) -> None:  # noqa: D102
        self.messages.append(msg)
        self._display(msg)

    def _display(self, msg: str) -> None:
        if self.show:
            print(f"{self.prefix}: {msg}", file=sys.stderr)

    def print_msgs(self) -> None:
        """Print all stored warning messages."""
        for msg in self.messages:
            print(f"{self.prefix}: {msg}")

    def final_msg(self) -> None:  # noqa: D102
        if self.count > 0:
            self._display(f"{self.count} problem(s) encountered")


def generate_loop_intervals(
    datetime_bounds: DatetimeArray,
) -> pd.IntervalIndex:
    """Generate time intervals from a sequence of datetimes.

    Parameters
    ----------
    datetime_bounds : array-like of datetime-like
        Array of sorted datetime-like values defining interval boundaries.

    Returns
    -------
    pandas.IntervalIndex
        Interval index object.
    """
    db = to_naive_utc_datetime(datetime_bounds, allow_nat=False)
    if not isinstance(db, (pd.Series, pd.DatetimeIndex)) or len(db) < 2:
        raise ValueError(
            "datetimes must be an array-like object with at least two elements."
        )
    db = pd.Series(db)

    if not db.is_monotonic_increasing:
        raise ValueError("datetimes must be sorted in increasing order.")

    return pd.IntervalIndex.from_tuples(
        list(zip(datetime_bounds[:-1], datetime_bounds[1:])), closed="left"
    )


def identify_loop_blocks(
    datetimes: DatetimeArray,
    gap: TimedeltaScalar,
    as_intervals: bool = False,
) -> pd.DatetimeIndex | pd.IntervalIndex:
    """Identify loop blocks in a sequence of datetimes based on gaps.

    Parameters
    ----------
    datetimes : array of datetime-like
        Array of datetimes in which to identify gaps. Must be sorted in increasing order.
    gap : TimedeltaScalar
        Loop start/ends are identified where the difference between datetimes exceeds
        ``gap``.
    as_intervals : bool, default False
        If True, return loop blocks as `pd.IntervalIndex`. If False, return
        loop block start/stop datetimes as `pd.DatetimeIndex`.

    Returns
    -------
    pd.DatetimeIndex | pd.IntervalIndex
        Array start / stop datetimes for loop blocks.
    """
    dt = to_naive_utc_datetime(datetimes, allow_nat=False)
    if not isinstance(dt, (pd.Series, pd.DatetimeIndex)) or len(dt) < 2:
        raise ValueError(
            "datetimes must be an array-like object with at least two elements."
        )
    dt = pd.Series(dt)  # ensure we have a Series for diff() and indexing
    if not dt.is_monotonic_increasing:
        raise ValueError("datetimes must be sorted in increasing order.")
    # if isinstance(_datetimes, pd.DatetimeIndex):
    #     _datetimes = _datetimes.to_series()
    gap = pd.to_timedelta(gap)
    gaps = dt.diff().gt(gap)

    one_sec = pd.Timedelta("1s")
    gap_bounds: list[pd.Timestamp] = (
        [dt.iloc[0] - one_sec] + dt.loc[gaps].to_list() + [dt.iloc[-1] + one_sec]
    )
    if as_intervals:
        return pd.IntervalIndex.from_tuples(
            list(zip(gap_bounds[:-1], gap_bounds[1:])), closed="left"
        )
    else:
        return pd.DatetimeIndex(gap_bounds)


def loops_from_gaps(
    datetimes: DatetimeArray,
    gap: TimedeltaScalar,
    loop_start: int = 1,
    loop_step: int = 1,
    loop_format: str = "{LOOP}",
) -> np.ndarray:
    """Generate loops from gaps in a sequence of datetimes and assign loop id's.

    Parameters
    ----------
    datetimes : array of datetime-like
        The observation datetimes to be split into loops. Must be sorted in
        increasing order.
    gap : TimedeltaScalar
        Loop start/ends are identified where the difference between
        datetimes exceeds ``gap``.
    loop_start : int, default 1
        Loop identifier start value.
    loop_step : int, default 1
        Increment loop identifier by ``step``.
    loop_format : str, default "{LOOP}"
        Format loop identifiers using ``loop_format``. Use "LOOP" as a
        placeholder for the loop number. Default `"{LOOP}"` is
        effectively no formatting. For example, `"x_{'LOOP':02d}_y"`
        would produce loop id's `x_01_y`, `x_02_y`, etc.

    Returns
    -------
    np.ndarray
        Array of loop id's corresponding to input ``datetimes``.
    """
    dt = to_naive_utc_datetime(datetimes, allow_nat=False)
    if not isinstance(dt, (pd.Series, pd.DatetimeIndex)) or len(dt) < 2:
        raise ValueError(
            "datetimes must be an array-like object with at least two elements."
        )
    loop_intervals = identify_loop_blocks(dt, gap, as_intervals=True)
    loop_ids = generate_loop_names(
        len(loop_intervals), start=loop_start, step=loop_step, format_str=loop_format
    )
    ds = pd.Series(loop_ids, index=loop_intervals)
    return ds[dt].to_numpy()


def generate_loop_names(
    n: int,
    start: int = 1,
    step: int = 1,
    format_str: str = "{LOOP}",
) -> list[str]:
    """Generate loop identifier strings.

    Parameters
    ----------
    n : int or array-like
        Number of loop identifiers to generate if ``n`` is an integer. If ``n`` is an
        array-like object, then the number of loop identifiers is set to the length of ``n``.
    start : int, default 1
        Loop identifier start value.
    step : int, default 1
        Increment loop identifier by ``step``.
    format_str : str, default "{LOOP}"
        Format loop identifiers using ``format_str``. Use "LOOP" as a placeholder for
        the loop number. The default "{LOOP}" is effectively no formatting.

    Returns
    -------
    list[str]
        List of loop identifier strings.
    """
    return [format_str.format(LOOP=i) for i in range(start, start + n * step, step)]


def round_coords(arr: np.typing.ArrayLike) -> np.ndarray:
    """Round values in ``arr``, with halfway cases rounded towards positive inifinty.

    Used in converting coords to indices, which is essentially a binning
    operation.

    Parameters
    ----------
    arr : array_like of float
        Data to be rounded. Can be a scalar or array_like of any shape.

    Returns
    -------
    rounded_array: ndarray
        An array or scalar of the same type as ``arr``.

    """
    return np.trunc(np.add(arr, 0.5))


def dms2rad(
    d: ArrayLike, m: ArrayLike, s: ArrayLike
) -> np.float64 | NDArray[np.float64]:
    """Convert angles from degree-minutes-seconds to radians.

    This function makes no attempt to round inputs to 0-359 or 0-59 range.

    Warning
    -------
    Direction of the angle is determined from ``d``, so:

        - if ``d`` is negative, then ``m`` and ``s`` are assumed to be negative.
        - Negative zero is **not** handled correctly for integer types.
          This is a limitation of NumPy.

    Parameters
    ----------
    d, m, s : int, float, array_like
        The degrees ``d``, mintes ``m`` and seconds ``s`` of the angle(s) to
        be converted. Scalar and array_like inputs can be mixed, but
        arrays must be of the same shape.

    Returns
    -------
    angle : ndarray or float
        The angle in radians. A scalar float if ``d``, ``m`` & ``s`` are
        all scalar.

    """
    deg = np.copysign(np.absolute(d) + np.divide(m, 60.0) + np.divide(s, 3600.0), d)
    return np.deg2rad(deg)
