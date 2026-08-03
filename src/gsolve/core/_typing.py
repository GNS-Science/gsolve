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

import datetime
from collections.abc import Callable, Hashable, Mapping, Sequence
from os import PathLike
from typing import Any, Literal, Protocol, TypeAlias, Union, runtime_checkable

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, DatetimeIndex, Index, Series, Timestamp
from pandas.api.typing import NaTType

# from pandas.api.typing.aliases import TimedeltaConvertibleTypes

__all__ = [
    "AllowedTimestampResolution",
    "AllowedTimestampRoundingMethods",
    "IfWorkbookExists",
    "IfSheetExists",
    "GSolveSolverMethod",
    "GSolveSolverReturn",
    "Renamer",
    "DatetimeScalar",
    "DatetimeArray",
    "DatetimeScalarOrArray",
    "FilePath",
    "DatasetOrArray",
    "ArrayOrCoords",
    "Points2D",
    "Points3D",
    "TCorrDistanceMaskType",
]

# Aliases by gsolve for various functions arguments
AllowedTimestampResolution: TypeAlias = Literal[
    "year", "month", "day", "hour", "minute", "second", "microsecond", "nanosecond"
]
AllowedTimestampRoundingMethods: TypeAlias = Literal["round", "floor", "ceil"]

IfWorkbookExists: TypeAlias = Literal["error", "replace", "append"]
IfSheetExists: TypeAlias = Literal["error", "replace", "new"]

GSolveSolverMethod: TypeAlias = Literal[1, 2, 3]

GSolveSolverReturn: TypeAlias = tuple[
    NDArray, NDArray, NDArray, NDArray, NDArray, float | np.float64 | None, NDArray
]

type FilePath = str | PathLike

# The following type aliases are copied/adapted from pandas to ensure
# function parameters are compatible with pandas methods they are passed to

Renamer: TypeAlias = Mapping[Any, Hashable] | Callable[[Any], Hashable]


DateTimeConvertibleTypes: TypeAlias = Union[
    str,
    int,
    float,
    datetime.timedelta,
    list,
    tuple,
    range,
    ArrayLike,
    Index,
    Series,
]
DatetimeScalar: TypeAlias = (
    int | float | str | datetime.date | np.datetime64 | pd.Timestamp
)

type DatetimeArray = list | tuple | Series | Index | DatetimeIndex | np.ndarray
DatetimeScalarOrArray: TypeAlias = DatetimeScalar | DatetimeArray

TimedeltaScalar: TypeAlias = str | int | float | pd.Timedelta | datetime.timedelta

SiteIDArray: TypeAlias = Sequence[str] | Series | Index | NDArray[np.str_]
FloatArray: TypeAlias = Sequence[float] | Series | Index | NDArray[np.floating]
StringArray: TypeAlias = Sequence[str] | Series | Index | NDArray[np.str_]
BoolArray: TypeAlias = Sequence[bool] | Series | Index | NDArray[np.bool_]

# aliases used in terrain correction
DatasetOrArray: TypeAlias = xr.DataArray | xr.Dataset
ArrayOrCoords: TypeAlias = DatasetOrArray | Sequence[ArrayLike]
Points2D: TypeAlias = tuple[FloatArray, FloatArray]
Points3D: TypeAlias = tuple[FloatArray, FloatArray, FloatArray]
type TCorrDistanceMaskType = Literal["radial", "rectangular"]


# protocols for select Gsolve classes
@runtime_checkable
class SitesLike(Protocol):
    data: pd.DataFrame

    def get_points(
        self, xcol: str, ycol: str, zcol: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
