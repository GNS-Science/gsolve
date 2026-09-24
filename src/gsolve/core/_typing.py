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
from typing import Any, Literal, Protocol, runtime_checkable

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

type GSolveSolverMethod = Literal[1, 2, 3]

type GSolveSolverReturn = tuple[
    NDArray, NDArray, NDArray, NDArray, NDArray, float | np.float64 | None, NDArray
]

type FilePath = str | PathLike

# The following type aliases are copied/adapted from pandas to ensure
# function parameters are compatible with pandas methods they are passed to

type Renamer = Mapping[Any, Hashable] | Callable[[Any], Hashable]


type DateTimeConvertibleTypes = (
    str | int | float | datetime.timedelta | list | tuple | ArrayLike | Index | Series
)
type DatetimeScalar = int | float | str | datetime.date | np.datetime64 | pd.Timestamp

type DatetimeArray = list | tuple | Series | Index | DatetimeIndex | np.ndarray
type DatetimeScalarOrArray = DatetimeScalar | DatetimeArray

type TimedeltaScalar = str | int | float | pd.Timedelta | datetime.timedelta

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
