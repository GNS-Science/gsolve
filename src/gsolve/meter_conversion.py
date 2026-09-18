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

"""Module for converting Lacoste-Romberg G and D meter readings to mGal."""

import pathlib
from collections.abc import Sequence
from io import StringIO
from typing import Protocol, Self, TextIO, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.typing import NaTType

from gsolve.core._typing import DatetimeArray, DatetimeScalar, FilePath
from gsolve.core.utils import to_naive_utc_datetime

__all__ = ["LaCosteRombergDialConverter"]


@runtime_checkable
class MeterReadingConverter(Protocol):
    def convert_readings(
        self,
        readings: npt.ArrayLike,
        meter_id: str | Sequence[str] | None = None,
        date_time: DatetimeScalar | DatetimeArray | None = None,
    ) -> npt.NDArray[np.float64]: ...

    def converter_id(self) -> str: ...


class LaCosteRombergDialConverter:
    """
    Convert Lacoste-Romberg G and D meter readings to mGal.

    Implements table based linear interpolation using the "Calibration Table"
    provided with each L&R meter. Readings may be filtered by ``meter_id`` and
    date range.

    Parameters
    ----------
    meter_id : str
        The gravity meter name.
    counter_reading : ArrayLike
        Array of counter readings. This will typically be an array of floats
        from 0.0 to 7000.0 in increments of 100.0 for G meters or
        0.0 to 200.0 in incremenets of 10.0 for D meters
    value_mgal : ArrayLike
        Gravity in milligals at each ``counter_reading``.
    interval_factor : ArrayLike, optional
        The gradient of mGal/counter_reading for each interval.
    starttime : datetimelike, optional
        Date from which correction parameters are valid, default is
        :attr:`pandas.Timestamp.min`.
    endtime : datetimelike, optional
        Date up to which correction parameters are valid. Defaults to
        :attr:`pandas.Timestamp.max`.

    Attributes
    ----------
    table: DataFrame
        The conversion table, with columns ``counter_reading``, ``value_mgal``,
        ``interval_factor`` and ``value_mgal_from_ifactor``.

    Notes
    -----
    If ``interval_factor`` is provided, then ``value_mgal`` will be recalculated
    and stored in the ``value_mgal_from_ifactor`` column. L&R calibration tables
    typically provide ``value_mgal`` rounded to 2 dp (10 ugal resolution)
    whereas ``interval_factor`` is specified to 5 dp (1 ugal resolution).
    Corrections are interpolated using ``value_mgal_from_ifactor`` where
    possible to minimise loss of precision.
    """

    _table_header_keys = ("meter_id", "starttime", "endtime")
    _table_column_labels = ("counter_reading", "value_mgal", "interval_factor")

    def __init__(
        self,
        *,
        meter_id: str,
        counter_reading: npt.ArrayLike,
        value_mgal: npt.ArrayLike,
        interval_factor: npt.ArrayLike | None = None,
        starttime: DatetimeScalar | NaTType | None = None,
        endtime: DatetimeScalar | NaTType | None = None,
    ) -> None:
        self.table: pd.DataFrame
        self.meter_id: str = meter_id
        self._starttime: pd.Timestamp | None
        self._endtime: pd.Timestamp | None

        self.set_datetime_range(starttime, endtime)

        c_reading = np.atleast_1d(np.array(counter_reading, dtype=np.float64))
        value_mgal = np.atleast_1d(np.array(value_mgal, dtype=np.float64))

        if c_reading.ndim != 1 or c_reading.size == 0:
            msg = "counter_reading must be a non-empty 1-dimensional array."
            raise ValueError(msg)
        if np.isnan(c_reading).any():
            msg = "counter_reading contains NaN."
            raise ValueError(msg)
        if value_mgal.ndim != 1 or value_mgal.size == 0:
            msg = "value_mgal must be a non-empty 1-dimensional array."
            raise ValueError(msg)
        if np.isnan(value_mgal).any():
            msg = "value_mgal contains NaN."
            raise ValueError(msg)

        if c_reading.size != value_mgal.size:
            msg = "counter_reading and value_mgal arrays must be the same shape."
            raise ValueError(msg)

        nrows: int = c_reading.size

        if interval_factor is not None:
            interval_factor = np.atleast_1d(interval_factor).astype(float)
            if interval_factor.ndim != 1 or interval_factor.size == 0:
                msg = "if specified, interval_factor must be a non-empty 1-dimensional array."
                raise ValueError(msg)

            if interval_factor.size == nrows:
                interval_factor[-1] = np.nan
            elif interval_factor.size == nrows - 1:
                interval_factor = np.append(interval_factor, np.nan)
            else:
                msg = (
                    f"invalid interval_factor: array size {interval_factor.size} is not "
                    f"the same as or 1 less than counter_reading ({nrows})."
                )
                raise ValueError(msg)
            if np.isnan(interval_factor[:-1]).any():
                msg_0 = "interval_factor is specified, but contains NaN values."
                raise ValueError(msg_0)
            recalc_value_mgal = True
        else:
            interval_factor = np.full_like(c_reading, np.nan)
            recalc_value_mgal = False

        self.table = pd.DataFrame(
            data={
                "counter_reading": c_reading,
                "value_mgal": value_mgal,
                "interval_factor": interval_factor,
                "value_mgal_from_ifactor": np.nan,
            },
            dtype=float,
        ).set_index("counter_reading")

        if (
            not self.table.index.is_monotonic_increasing
            or not self.table.index.is_unique
        ):
            msg_0 = "counter_reading values must be unique and in ascending order."
            raise ValueError(msg_0)

        if recalc_value_mgal:
            ifac = self.table["interval_factor"].astype(float).to_numpy()

            # Use interval factor to re-calculate gravity values if possible
            # L&R tables typically are intended for 'human' use, so
            # - 'value_mgal' is listed to 2 dp (i.e. 10 ugal precision) for space/legibility
            # - 'interval_factor' is listed to 4 dp (. 0.1 ugal precision)
            # -> using 'value_mgal' as listed will result in a loss of precision of up to 7 ugal

            mgal_ifac = self.table["value_mgal_from_ifactor"].to_numpy(copy=True)

            mgal_ifac[0] = self.table["value_mgal"].iloc[0]
            mgal_ifac[1:] = np.diff(counter_reading) * ifac[:-1]
            self.table["value_mgal_from_ifactor"] = np.cumsum(mgal_ifac)

    @property
    def meter_id(self) -> str:
        """The ID/serial number of the meter."""
        return getattr(self, "_meter_id", "")

    @meter_id.setter
    def meter_id(self, val: str) -> None:
        if val is None or not str(val).strip():
            msg = "meter_id must be specified."
            raise ValueError(msg)
        self._meter_id = str(val).strip()

    def set_datetime_range(
        self,
        starttime: DatetimeScalar | NaTType | None,
        endtime: DatetimeScalar | NaTType | None,
    ) -> None:
        """Set the start and end times defining the converter's valid date range.

        The meter conversion values for a given LaCoste-Romberg gravity meter may change
        over time due to, say, upgrades or physical damage. The starttime and endtime
        properties allow for a conversion table to be assigned a date range for which
        it is valid. Conversion will only be applied to readings that fall within the
        valid date range.

        Parameters
        ----------
        starttime : datetimelike, NaT or None
            Date from which correction parameters are valid, default is None (i.e.
            no start date).
        endtime : datetimelike, NaT or None
            Date up to which correction parameters are valid. Defaults to None (i.e.
            no end date).

        Raises
        ------
        ValueError
            If starttime or endtime cannot be converted to a ``pandas.Timestamp``, or if
            starttime is >= endtime.
        TypeError
            If starttime or endtime is not datetimelike, NaT or None.
        """
        if starttime is pd.NaT or starttime is None:
            st = None
        elif isinstance(starttime, DatetimeScalar):
            try:
                st = to_naive_utc_datetime(starttime, allow_nat=False)
            except ValueError as e:
                msg = f"Error setting starttime: {e}"
                raise ValueError(msg) from None
        else:
            msg = f"invalid starttime type {type(starttime)}. Should be datetimelike or None."
            raise TypeError(msg)

        if endtime is pd.NaT or endtime is None:
            et = None
        elif isinstance(endtime, DatetimeScalar):
            try:
                et = to_naive_utc_datetime(endtime, allow_nat=False)
            except ValueError as e:
                msg = f"Error setting endtime: {e}"
                raise ValueError(msg) from None
        else:
            msg = (
                f"invalid endtime type {type(endtime)}. Should be datetimelike or None."
            )
            raise TypeError(msg)

        if st is not None and et is not None and st >= et:
            msg = f"invalid time combination: ({st}) is >= endtime ({et})"
            raise ValueError(msg)

        self._starttime = st
        self._endtime = et

    @property
    def starttime(self) -> pd.Timestamp | None:
        """The date from which correction parameters are valid."""
        st = getattr(self, "_starttime", None)
        if st is not None and not isinstance(st, pd.Timestamp):
            msg = f"invalid starttime type {type(st)}. Should be pandas.Timestamp or None."
            raise TypeError(msg)
        return st

    @property
    def endtime(self) -> pd.Timestamp | None:
        """The date up to which correction parameters are valid."""
        r = getattr(self, "_endtime", None)
        if r is not None and not isinstance(r, pd.Timestamp):
            msg = f"invalid endtime type {type(r)}. Should be pandas.Timestamp or None."
            raise TypeError(msg)
        return r

    def convert_readings(
        self,
        readings: npt.ArrayLike,
        meter_id: npt.ArrayLike | None = None,
        date_time: DatetimeScalar | DatetimeArray | None = None,
    ) -> npt.NDArray[np.float64]:
        """Convert meter readings to milligal.

        Parameters
        ----------
        readings : float, array_like
            The readings to be converted.
        meter_id : str, array_like, optional
            The meter id/name associated with the readings. If provided, only readings
            with ``meter_id`` matching the converter's ``meter_id`` will be converted.
        date_time : datetimelike, array_like, optional
            The date/time of the readings. If provided, only readings
            with ``date_time`` falling within converter's ``valid_date_range``
            are converted.

        Returns
        -------
        float, ndarray
            The converted readings. Readings where ``meter_id`` or ``date_time``
            do not match the converter's ``meter_id`` or ``valid_date_range``
            will be returned as NaN.

        Raises
        ------
        ValueError
            Where reading(s) are outside the limits of the conversion table.
        TypeError
            If ``meter_id`` is not a string or array of strings, or if ``date_time``
            is not datetimelike or array of datetimelike.
        """
        interval_bounds: npt.NDArray[np.float64] = self.table.index.to_numpy(np.float64)

        readings = np.atleast_1d(readings).astype(float)
        if (readings < interval_bounds.min()).any() | (
            readings > interval_bounds.max()
        ).any():
            msg = (
                "1 or more readings are outside range of convertible values: "
                f"{interval_bounds.min()} - {interval_bounds.max()}."
            )
            raise ValueError(msg)

        if meter_id is not None:
            m_meter_id = np.atleast_1d(meter_id).astype(str) == self.meter_id

            if m_meter_id.size == 0:
                msg_0 = "invalid meter_id arg: empty array."
                raise ValueError(msg_0)
            if m_meter_id.ndim != 1:
                msg_0 = "invalid meter_id arg: must be a scalar or 1-dimensional array."
                raise ValueError(msg_0)

            if m_meter_id.size == 1 and readings.size > 1:
                m_meter_id = np.full(readings.shape, m_meter_id[0])
            elif m_meter_id.size != readings.size:
                msg_0 = "invalid meter_id arg: length must match readings array."
                raise ValueError(msg_0)
        else:
            m_meter_id = np.full(readings.shape, True)

        if date_time is not None:
            dt = to_naive_utc_datetime(date_time)
            if isinstance(dt, pd.Timestamp):
                date_time = pd.DatetimeIndex([dt] * readings.size)
            elif isinstance(dt, (pd.Series, pd.DatetimeIndex)):
                date_time = pd.DatetimeIndex(dt)
            else:
                msg_0 = (
                    "date_time could not be converted to a Timestamp or DatetimeIndex."
                )
                raise TypeError(msg_0)

            if date_time.size != readings.size:
                msg_0 = "invalid date_time array: date_time values must be the same length as readings."
                raise ValueError(msg_0)
            if any(date_time.isna()):
                msg_0 = "date_time contains NaT values."
                raise ValueError(msg_0)

            m_datetime = np.full_like(readings, True, dtype=bool)
            if self.starttime is not None:
                m_datetime &= date_time >= self.starttime.asm8
            if self.endtime is not None:
                m_datetime &= date_time <= self.endtime.asm8
        else:
            m_datetime = np.full_like(readings, True, dtype=bool)

        interval_mgal: npt.NDArray[np.float64]
        if self.table["value_mgal_from_ifactor"].notna().any():
            interval_mgal = self.table["value_mgal_from_ifactor"].to_numpy(np.float64)
        else:
            interval_mgal: npt.NDArray[np.float64] = self.table["value_mgal"].to_numpy(
                np.float64
            )
        converted: npt.NDArray[np.float64] = np.interp(
            readings, interval_bounds, interval_mgal
        )

        m = np.logical_and(m_meter_id, m_datetime)
        converted[~m] = np.nan

        return converted

    def converter_id(self) -> str:
        """
        Return identifier for this meter conversion table.

        Returns
        -------
        str
            Identifier label of form 'meter_id:{starttime}_to_endtime'.
        """
        st = (
            "from_" + self.starttime.strftime("%Y-%m-%d")
            if self.starttime is not None
            else "undefined"
        )

        et = (
            "to_" + self.endtime.strftime("%Y-%m-%d")
            if self.endtime is not None
            else "undefined"
        )

        return f"LR_{self.meter_id}:{st}_{et}"

    @classmethod
    def from_dataframe(
        cls,
        meter_id: str,
        table: pd.DataFrame,
        starttime: DatetimeScalar = pd.Timestamp.min,
        endtime: DatetimeScalar = pd.Timestamp.max,
    ) -> Self:
        """
        Generate a LaCosteRombergDialConverter object from a standard L&R G-meter table.

        The input table data must have at least 3 columns, which are assumed to be
        "interval_start", "interval_end", "interval_factor".

        Parameters
        ----------
        meter_id : str
            Meter id/name.
        table : _pd.DataFrame | _npt.ArrayLike
            The correction table data.
        starttime : datetimelike
            Date from which correction parameters are valid, default is
            ``pandas.Timestamp.min``.
        endtime : datetimelike
            Date up to which correction parameters are valid. Defaults to
            ``pandas.Timestamp.max``.

        Returns
        -------
        LaCosteRombergDialConverter
        """
        return cls(
            meter_id=meter_id,
            counter_reading=table["counter_reading"],
            value_mgal=table["value_mgal"],
            interval_factor=table["interval_factor"],
            starttime=starttime,
            endtime=endtime,
        )

    @classmethod
    def from_csv(cls, fname: FilePath, **kwargs) -> Self:
        """
        Generate a LaCosteRombergDialConverter object from a csv file.

        Parameters
        ----------
        fname : str or TextIO
            Path to the csv file, or a file-like object containing the csv data.
        kwargs : dict
            Additional keyword arguments to be passed to ``pandas.read_csv``.

        Returns
        -------
        LaCosteRombergDialConverter

        Raises
        ------
        ValueError
            Raised if:

            - the file is empty
            - required header keys are missing
            - column labels are missing or invalid
            - readings are outside the limits of the conversion table
        """
        if isinstance(fname, TextIO):
            data = fname.readlines()
        else:
            with pathlib.Path(fname).open(encoding="utf-8-sig", mode="r") as fh:
                data = fh.readlines()
        if not data:
            msg = "empty file"
            raise ValueError(msg)
        hdr = {}
        while data[0].startswith("#"):
            hdr_line = data.pop(0).lstrip("#").strip()
            if not hdr_line:
                continue

            h = [v.strip() for v in hdr_line.split(",")]
            if len(h) == 1:
                msg = (
                    "reading csv header: "
                    f"header key '{h[0]}' has no corresponding value"
                )
                raise ValueError(msg)
            if len(h) > 2 and "".join(h[2:]):
                msg = (
                    "reading csv header: "
                    f"key '{h[0]}' has multiple corresponding values '{hdr_line}'"
                )
                raise ValueError(msg)
            if h[0] not in cls._table_header_keys:
                msg = f"reading csv header: invalid header key name '{h[0]}'"
                raise ValueError(msg)

            hdr[h[0]] = h[1]

        if "meter_id" not in hdr:
            msg = f"reading '{fname}': meter_id not specified in header"
            raise ValueError(msg)

        # if have column labels
        if data[0].strip() == ",".join(cls._table_column_labels):
            data.pop(0)

        with StringIO("\n".join(data)) as buffer:
            df = pd.read_csv(
                buffer, dtype=float, names=cls._table_column_labels, **kwargs
            )

        return cls.from_dataframe(**hdr, table=df)
