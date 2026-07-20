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

import warnings
from io import StringIO
from typing import Any, Protocol, Sequence, TextIO, runtime_checkable

import numpy as _np
import numpy.typing as _npt
import pandas as _pd
from pandas.api.typing import NaTType

from gsolve.core._typing import DatetimeArray, DatetimeScalar, FilePath
from gsolve.core.utils import is_list_like, to_naive_utc_datetime

__all__ = ["LaCosteRombergDialConverter"]


@runtime_checkable
class MeterReadingConverter(Protocol):
    def convert_readings(
        self,
        readings: _npt.ArrayLike,
        meter_id: str | Sequence[str] | None = None,
        date_time: DatetimeScalar | DatetimeArray | None = None,
    ) -> _npt.NDArray[_np.float64]: ...

    def converter_id(self) -> str: ...


# class MeterReadingConverterABC(abc.ABC):
#     """Base class for Lacoste Romberg meter reading converters"""

#     @abc.abstractmethod
#     def convert_readings(
#         self,
#         readings: Sequence[float],
#         meter_id: str | Sequence[Any] | None = None,
#         date_time: DatetimeScalar | DatetimeArray | None = None,
#     ) -> _npt.NDArray[_np.float64]:
#         pass

#     @abc.abstractmethod
#     def converter_id(self) -> str:
#         pass

#     @property
#     def meter_id(self) -> str:
#         return getattr(self, "_meter_id")

#     @meter_id.setter
#     def meter_id(self, val: str) -> None:
#         if val is None or not str(val).strip():
#             raise ValueError("meter_id must be specified.")
#         self._meter_id = str(val).strip()

#     @property
#     def starttime(self) -> _pd.Timestamp:
#         """The date from which correction parameters are valid."""
#         st = getattr(self, "_starttime", None)
#         return st if isinstance(st, _pd.Timestamp) else _pd.Timestamp.min

#     @starttime.setter
#     def starttime(self, t: DatetimeScalar | None | NaTType) -> None:
#         if _pd.isna(t) or t is None:
#             _t = None
#         else:
#             _t = to_naive_utc_datetime(t)
#         if not isinstance(_t, _pd.Timestamp):
#             _t = None

#         if _t is not None and self.endtime is not None and _t >= self.endtime:
#             raise ValueError(f"starttime ({t}) must be < endtime ({self.endtime})")
#         self._starttime = _t

#     @property
#     def endtime(self) -> _pd.Timestamp | None:
#         """The date up to which correction parameters are valid."""
#         t = getattr(self, "_endtime", None)
#         return t if isinstance(t, _pd.Timestamp) else _pd.Timestamp.max

#     @endtime.setter
#     def endtime(self, t: DatetimeScalar | NaTType) -> None:
#         if isinstance(t, NaTType) or t is None:
#             _t = _pd.Timestamp.max
#         else:
#             _t = to_naive_utc_datetime(t)
#         if not isinstance(_t, _pd.Timestamp):
#             _t = _pd.Timestamp.max
#         if _t < self.starttime:
#             raise ValueError(f"endtime ({t}) must be > starttime ({self.starttime})")
#         self._endtime = _t

#     @property
#     def valid_date_range(self) -> _pd.Interval:
#         # TODO: consider whether to make this a closed interval, or half-open with end exclusive
#         return _pd.Interval(self.starttime, self.endtime, closed="left")


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
        meter_id: str,
        counter_reading: _npt.ArrayLike,
        value_mgal: _npt.ArrayLike,
        interval_factor: _npt.ArrayLike | None = None,
        starttime: DatetimeScalar | NaTType | None = None,
        endtime: DatetimeScalar | NaTType | None = None,
    ) -> None:
        self.table: _pd.DataFrame
        self.meter_id: str = meter_id
        self._starttime: _pd.Timestamp | None
        self._endtime: _pd.Timestamp | None

        self.set_datetime_range(starttime, endtime)

        _c_reading = _np.atleast_1d(_np.array(counter_reading, dtype=_np.float64))
        _value_mgal = _np.atleast_1d(_np.array(value_mgal, dtype=_np.float64))

        if _c_reading.ndim != 1 or _c_reading.size == 0:
            raise ValueError("counter_reading must be a non-empty 1-dimensional array.")
        if _np.isnan(_c_reading).any():
            raise ValueError("counter_reading contains NaN.")
        if _value_mgal.ndim != 1 or _value_mgal.size == 0:
            raise ValueError("value_mgal must be a non-empty 1-dimensional array.")
        if _np.isnan(_value_mgal).any():
            raise ValueError("value_mgal contains NaN.")

        if _c_reading.size != _value_mgal.size:
            raise ValueError(
                "counter_reading and value_mgal arrays must be the same shape."
            )

        nrows: int = _c_reading.size

        if interval_factor is not None:
            _interval_factor = _np.atleast_1d(interval_factor).astype(float)
            if _interval_factor.ndim != 1 or _interval_factor.size == 0:
                raise ValueError(
                    "if specified, interval_factor must be a non-empty 1-dimensional array."
                )

            if _interval_factor.size == nrows:
                _interval_factor[-1] = _np.nan
            elif _interval_factor.size == nrows - 1:
                _interval_factor = _np.append(_interval_factor, _np.nan)
            else:
                raise ValueError(
                    f"invalid interval_factor: array size {_interval_factor.size} is not "
                    f"the same as or 1 less than counter_reading ({nrows})."
                )
            if _np.isnan(_interval_factor[:-1]).any():
                raise ValueError(
                    "interval_factor is specified, but contains NaN values."
                )
            recalc_value_mgal = True
        else:
            _interval_factor = _np.full_like(_c_reading, _np.nan)
            recalc_value_mgal = False

        self.table = _pd.DataFrame(
            data={
                "counter_reading": _c_reading,
                "value_mgal": _value_mgal,
                "interval_factor": _interval_factor,
                "value_mgal_from_ifactor": _np.nan,
            },
            dtype=float,
        ).set_index("counter_reading")

        if (
            not self.table.index.is_monotonic_increasing
            or not self.table.index.is_unique
        ):
            raise ValueError(
                "counter_reading values must be unique and in ascending order."
            )

        if recalc_value_mgal:
            ifac = self.table["interval_factor"].astype(float).to_numpy()

            # Use interval factor to re-calculate gravity values if possible
            # L&R tables typically are intended for 'human' use, so
            # - 'value_mgal' is listed to 2 dp (i.e. 10 ugal precision) for space/legibility
            # - 'interval_factor' is listed to 4 dp (. 0.1 ugal precision)
            # -> using 'value_mgal' as listed will result in a loss of precision of up to 7 ugal

            mgal_ifac = self.table["value_mgal_from_ifactor"].to_numpy(copy=True)

            mgal_ifac[0] = self.table["value_mgal"].iloc[0]
            mgal_ifac[1:] = _np.diff(counter_reading) * ifac[:-1]
            self.table["value_mgal_from_ifactor"] = _np.cumsum(mgal_ifac)

    @property
    def meter_id(self) -> str:
        """The ID/serial number of the meter."""
        return getattr(self, "_meter_id", "")

    @meter_id.setter
    def meter_id(self, val: str) -> None:
        if val is None or not str(val).strip():
            raise ValueError("meter_id must be specified.")
        self._meter_id = str(val).strip()

    def set_datetime_range(
        self,
        starttime: DatetimeScalar | None | NaTType,
        endtime: DatetimeScalar | None | NaTType,
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
        if starttime is _pd.NaT or starttime is None:
            st = None
        elif isinstance(starttime, DatetimeScalar):
            try:
                st = to_naive_utc_datetime(starttime, allow_nat=False)
            except ValueError as e:
                raise ValueError(f"Error setting starttime: {e}")
        else:
            raise TypeError(
                f"invalid starttime type {type(starttime)}. Should be datetimelike or None."
            )

        if endtime is _pd.NaT or endtime is None:
            et = None
        elif isinstance(endtime, DatetimeScalar):
            try:
                et = to_naive_utc_datetime(endtime, allow_nat=False)
            except ValueError as e:
                raise ValueError(f"Error setting endtime: {e}")
        else:
            raise TypeError(
                f"invalid endtime type {type(endtime)}. Should be datetimelike or None."
            )

        if st is not None and et is not None and st >= et:
            raise ValueError(f"invalid time combination: ({st}) is >= endtime ({et})")

        self._starttime = st
        self._endtime = et

    @property
    def starttime(self) -> _pd.Timestamp | None:
        """The date from which correction parameters are valid."""  # noqa: DOC501
        st = getattr(self, "_starttime", None)
        if st is not None and not isinstance(st, _pd.Timestamp):
            raise TypeError(
                f"invalid starttime type {type(st)}. Should be pandas.Timestamp or None."
            )
        return st

    @property
    def endtime(self) -> _pd.Timestamp | None:
        """The date up to which correction parameters are valid."""  # noqa: DOC501
        r = getattr(self, "_endtime", None)
        if r is not None and not isinstance(r, _pd.Timestamp):
            raise TypeError(
                f"invalid endtime type {type(r)}. Should be pandas.Timestamp or None."
            )
        return r

    def convert_readings(
        self,
        readings: _npt.ArrayLike,
        meter_id: _npt.ArrayLike | None = None,
        date_time: DatetimeScalar | DatetimeArray | None = None,
    ) -> _npt.NDArray[_np.float64]:
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
        interval_bounds: _npt.NDArray[_np.float64] = self.table.index.to_numpy(
            _np.float64
        )

        _readings = _np.atleast_1d(readings).astype(float)
        if (_readings < interval_bounds.min()).any() | (
            _readings > interval_bounds.max()
        ).any():
            raise ValueError(
                "1 or more readings are outside range of convertible values: "
                f"{interval_bounds.min()} - {interval_bounds.max()}."
            )

        if meter_id is not None:
            _m_meter_id = _np.atleast_1d(meter_id).astype(str) == self.meter_id

            if _m_meter_id.size == 0:
                raise ValueError("invalid meter_id arg: empty array.")
            if _m_meter_id.ndim != 1:
                raise ValueError(
                    "invalid meter_id arg: must be a scalar or 1-dimensional array."
                )

            if _m_meter_id.size == 1 and _readings.size > 1:
                _m_meter_id = _np.full(_readings.shape, _m_meter_id[0])
            elif _m_meter_id.size != _readings.size:
                raise ValueError(
                    "invalid meter_id arg: length must match readings array."
                )
        else:
            _m_meter_id = _np.full(_readings.shape, True)

        if date_time is not None:
            _dt = to_naive_utc_datetime(date_time)
            if isinstance(_dt, _pd.Timestamp):
                _date_time = _pd.DatetimeIndex([_dt] * _readings.size)
            elif isinstance(_dt, (_pd.Series, _pd.DatetimeIndex)):
                _date_time = _pd.DatetimeIndex(_dt)
            else:
                raise TypeError(
                    "date_time could not be converted to a Timestamp or DatetimeIndex."
                )

            if _date_time.size != _readings.size:
                raise ValueError(
                    "invalid date_time array: date_time values must be the same length as readings."
                )
            if any(_date_time.isna()):
                raise ValueError("date_time contains NaT values.")

            _m_datetime = _np.full_like(_readings, True, dtype=bool)
            if self.starttime is not None:
                _m_datetime &= _date_time >= self.starttime.asm8
            if self.endtime is not None:
                _m_datetime &= _date_time <= self.endtime.asm8
        else:
            _m_datetime = _np.full_like(_readings, True, dtype=bool)

        interval_mgal: _npt.NDArray[_np.float64]
        if self.table["value_mgal_from_ifactor"].notna().any():
            interval_mgal = self.table["value_mgal_from_ifactor"].to_numpy(_np.float64)
        else:
            interval_mgal: _npt.NDArray[_np.float64] = self.table[
                "value_mgal"
            ].to_numpy(_np.float64)
        converted: _npt.NDArray[_np.float64] = _np.interp(
            _readings, interval_bounds, interval_mgal
        )

        m = _np.logical_and(_m_meter_id, _m_datetime)
        converted[~m] = _np.nan

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
        table: _pd.DataFrame,
        starttime: DatetimeScalar = _pd.Timestamp.min,
        endtime: DatetimeScalar = _pd.Timestamp.max,
    ) -> "LaCosteRombergDialConverter":
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
        # meter_id: str,
        # counter_reading: _npt.ArrayLike,
        # value_mgal: _npt.ArrayLike,
        # interval_factor: float | None = None,
        # starttime: DatetimeScalar | NaTType = _pd.Timestamp.min,
        # endtime: DatetimeScalar | NaTType = _pd.Timestamp.max,

    @classmethod
    def from_csv(cls, fname: FilePath, **kwargs) -> "LaCosteRombergDialConverter":
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
            with open(fname, encoding="utf-8-sig", mode="r") as fh:
                data = fh.readlines()
        if not data:
            raise ValueError("empty file")
        hdr = {}
        while data[0].startswith("#"):
            hdr_line = data.pop(0).lstrip("#").strip()
            if not hdr_line:
                continue

            h = [v.strip() for v in hdr_line.split(",")]
            if len(h) == 1:
                raise ValueError(
                    "reading csv header: "
                    f"header key '{h[0]}' has no corresponding value"
                )
            if len(h) > 2 and "".join(h[2:]):
                raise ValueError(
                    "reading csv header: "
                    f"key '{h[0]}' has multiple corresponding values '{hdr_line}'"
                )
            if h[0] not in cls._table_header_keys:
                raise ValueError(
                    f"reading csv header: invalid header key name '{h[0]}'"
                )

            hdr[h[0]] = h[1]

        if "meter_id" not in hdr:
            raise ValueError(f"reading '{fname}': meter_id not specified in header")

        # if have column labels
        if data[0].strip() == ",".join(cls._table_column_labels):
            data.pop(0)

        with StringIO("\n".join(data)) as buffer:
            df = _pd.read_csv(
                buffer, dtype=float, names=cls._table_column_labels, **kwargs
            )

        return cls.from_dataframe(**hdr, table=df)
