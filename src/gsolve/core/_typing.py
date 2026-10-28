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


from __future__ import annotations

import datetime as dt
import numbers
from collections.abc import Callable, Hashable, Mapping, Sequence
from enum import IntEnum
from os import PathLike
from typing import Any, Literal, Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from pandas import DatetimeIndex, Index, Series

# from pandas.api.typing.aliases import TimedeltaConvertibleTypes

__all__ = [
    "AllowedTimestampResolution",
    "AllowedTimestampRoundingMethods",
    "ArrayOrCoords",
    "DatasetOrArray",
    "DatetimeArray",
    "DatetimeScalar",
    "DatetimeScalarOrArray",
    "FilePath",
    "GSolveSolverMethod",
    "GSolveSolverReturn",
    "IfSheetExists",
    "IfWorkbookExists",
    "Points2D",
    "Points3D",
    "Renamer",
    "TCorrDistanceMaskType",
]

# Aliases by gsolve for various functions arguments
type AllowedTimestampResolution = Literal[
    "year", "month", "day", "hour", "minute", "second", "microsecond", "nanosecond"
]
type AllowedTimestampRoundingMethods = Literal["round", "floor", "ceil"]

type IfWorkbookExists = Literal["error", "replace", "append"]
type IfSheetExists = Literal["error", "replace", "new"]


class GSolveSolverMethod(IntEnum):
    """The least-squares solution method used by the gsolve solver.

    Members are ``IntEnum`` so existing integer call sites (``method == 2``,
    ``method in {1, 2}``, dict-key lookups) and callers passing a plain ``int``
    continue to work unchanged.

    Members
    -------
    UNCONSTRAINED
        ``1`` -- Unconstrained least squares.
    PARTIALLY_CONSTRAINED
        ``2`` -- Partially constrained least squares (the default).
    CONSTRAINED
        ``3`` -- Constrained least squares.
    """

    UNCONSTRAINED = 1
    PARTIALLY_CONSTRAINED = 2
    CONSTRAINED = 3

    @classmethod
    def to_int(cls, method: GSolveSolverMethod | int) -> int:
        """Return a plain ``int`` for a member or an equivalent integer.

        Use this at serialization boundaries (e.g. writing to Excel) so the
        stored value is a bare integer rather than an enum object, while still
        validating that the input is a known method.

        Parameters
        ----------
        method : GSolveSolverMethod | int
            A member or its equivalent integer value.

        Returns
        -------
        int
            The method's integer value.

        Raises
        ------
        ValueError
            If ``method`` is not a valid :class:`GSolveSolverMethod` value.
        """
        return int(cls(method))

    @classmethod
    def from_value(cls, value: GSolveSolverMethod | float | str) -> Self:
        """Parse a value read from Excel/CSV/text back into a member.

        Round-trips the serialized forms this enum produces or that readers
        hand back, and returns the matching :class:`GSolveSolverMethod`.
        Accepted inputs:

        - a member (returned unchanged);
        - an integer, or an integral float (e.g. ``2`` or ``2.0``);
        - a numeric string (e.g. ``"2"`` -- CSV reads a mixed column as text);
        - a member name, case-insensitive (e.g. ``"PARTIALLY_CONSTRAINED"``).

        Parameters
        ----------
        value : GSolveSolverMethod | int | float | str
            The value to interpret.

        Returns
        -------
        GSolveSolverMethod
            The matching member.

        Raises
        ------
        ValueError
            If ``value`` cannot be interpreted as a known method.
        TypeError
            If ``value`` is not an int, float, str or member.
        """
        if isinstance(value, cls):
            return value

        if isinstance(value, bool):
            # bool is an int subclass but is never a valid method.
            msg = f"invalid method {value!r}: bool is not a valid GSolveSolverMethod"
            raise TypeError(msg)

        if isinstance(value, str):
            text = value.strip()
            try:  # numeric string, e.g. "2" or "2.0"
                return cls(int(float(text)))
            except ValueError:
                pass
            try:  # member name, case-insensitive
                return cls[text.upper()]
            except KeyError:
                valid = ", ".join(m.name for m in cls)
                msg = f"invalid method {value!r}, must be an int or one of: {valid}"
                raise ValueError(msg) from None

        if isinstance(value, numbers.Real):
            as_int = int(value)
            if as_int != value:  # reject non-integral values like 2.5
                msg = f"invalid method {value!r}: not an integer value"
                raise ValueError(msg)
            return cls(as_int)

        msg = f"cannot interpret {value!r} ({type(value).__name__}) as a method"
        raise TypeError(msg)


type GSolveSolverReturn = tuple[
    NDArray, NDArray, NDArray, NDArray, NDArray, float | np.float64 | None, NDArray
]

type FilePath = str | PathLike

# The following type aliases are copied/adapted from pandas to ensure
# function parameters are compatible with pandas methods they are passed to

type Renamer = Mapping[Any, Hashable] | Callable[[Any], Hashable]


type DateTimeConvertibleTypes = (
    str | int | float | dt.timedelta | list | tuple | ArrayLike | Index | Series
)
type DatetimeScalar = int | float | str | dt.date | np.datetime64 | pd.Timestamp

type DatetimeArray = list | tuple | Series | Index | DatetimeIndex | np.ndarray
type DatetimeScalarOrArray = DatetimeScalar | DatetimeArray

type TimedeltaScalar = str | int | float | pd.Timedelta | dt.timedelta

type SiteIDArray = Sequence[str] | Series | Index | NDArray[np.str_]
type FloatArray = Sequence[float] | Series | Index | NDArray[np.floating]
type StringArray = Sequence[str] | Series | Index | NDArray[np.str_]
type BoolArray = Sequence[bool] | Series | Index | NDArray[np.bool_]

# aliases used in terrain correction
type DatasetOrArray = xr.DataArray | xr.Dataset
type ArrayOrCoords = DatasetOrArray | Sequence[ArrayLike]
type Points2D = tuple[FloatArray, FloatArray]
type Points3D = tuple[FloatArray, FloatArray, FloatArray]
type Points3DTrue = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
type TCorrDistanceMaskType = Literal["radial", "rectangular"]


# protocols for select Gsolve classes
@runtime_checkable
class SitesLike(Protocol):
    data: pd.DataFrame

    def get_points(self, xcol: str, ycol: str, zcol: str = "") -> Points3DTrue: ...

    def get_site_ids(self) -> NDArray[np.str_]: ...
