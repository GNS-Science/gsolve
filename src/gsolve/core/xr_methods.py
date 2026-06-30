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

"""Methods for loading and manipulating XArray data structures."""

import warnings
from collections.abc import Sequence

import numpy as np
import rasterio
import xarray as xr
from numpy.typing import ArrayLike
from rasterio.transform import xy as rio_xy

from gsolve.core._typing import (
    DatasetOrArray,
    FilePath,
    Points2D,
    SitesLike,
    TCorrDistanceMaskType,
)
from gsolve.core.utils import GSolveDataWarning, round_coords
from gsolve.core.xr_accessor import TCorrMethods as _TCorrMethods

__all__ = [
    "load_dem",
    "prepare_dem",
    "check_dem",
    "create_empty_dataarray",
]


def load_dem(
    dem_file: FilePath,
    input_var_name: str | None = None,
    output_var_name: str | None = "elevation",
    output_x_dim: str | None = "easting",
    output_y_dim: str | None = "northing",
    fill_nan: float | None = 0.0,
    round_dp: int | None = 3,
    **kwargs,
) -> xr.DataArray:
    """
    Load DEM from a raster file and prepare it for use with gsolve.

    Parameters
    ----------
    dem_file : str or PathLike
        A raster grid readable with ``xarray.open_dataset``.
    input_var_name : str | None, optional
        Name of the data variable to read from ``dem_file``. Required if dataset
        contains multiple variables.
    output_var_name : str | None, default is "elevation"
        If not None, name of the data variable in the output DataArray.
    output_x_dim : str | None, default is "easting"
        If not None, rename x dimension to ``output_x_dim``.
    output_y_dim : str | None, default is "northing"
        If not None, rename y dimension to ``output_y_dim``.
    fill_nan : float | None, default is 0.0
        If not None, replace nan's with ``fill_nan``.
    round_dp : int | None, default is 3
        If not None, round data to ``round_dp`` decimal places.
    kwargs:
        Additional arguments to be passed to ``xarray.open_dataset()``.

    Returns
    -------
    xarray.DataArray

    See Also
    --------
    xarray.open_dataset : The function used to read the DEM file.
    """

    try:
        ds = xr.open_dataset(dem_file, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to read DEM file '{dem_file}': {e}") from e

    return prepare_dem(
        dem=ds,
        input_var_name=input_var_name,
        output_var_name=output_var_name,
        x_dim=output_x_dim,
        y_dim=output_y_dim,
        fill_nan=fill_nan,
        round_dp=round_dp,
    )


def prepare_dem(
    dem: DatasetOrArray,
    input_var_name: str | None = None,
    output_var_name: str | None = None,
    x_dim: str | None = "easting",
    y_dim: str | None = "northing",
    fill_nan: float | None = 0.0,
    round_dp: int | None = 3,
) -> xr.DataArray:
    """Prepare a DataArray or DataSet for use with gsolve terrain correction methods.

    Parameters
    ----------
    dem : DataArray or DataSet
        The Dataset or DataArray to tidy up.
    input_var_name : str or None, defaut is None
        Name of the data variable to extracted from ``dem``. Required if ``dem``
        is a DataSet containing multiple variables.
    output_var_name : str or None, default None
        Rename the DataArray to ``output_var_name`` if not None.
    x_dim : str, default is "easting"
        Rename the x-dimension if not ``None``.
    y_dim : str, default "northing"
        Rename the y-dimension if not ``None``.
    fill_nan : float or None, default = 0.0
        If not None, replace null values with ``fill_nan``.
    round_dp : int or None, default = 3
        If not None, round data to ``round_dp`` decimal places.

    Returns
    -------
    xr.DataArray
        The prepared DEM as a DataArray.

    """
    if isinstance(dem, xr.Dataset):
        if input_var_name is None:
            if len(dem) != 1:
                raise ValueError(
                    "Cannot convert multi-variable DataSet to a single variable"
                    "DataArray object. Use 'var_name' to specify the variable to "
                    f"convert. Variables in dem: {list(dem.data_vars)}"
                )
            input_var_name = str(list(dem.data_vars.keys())[0])
        dem = dem[input_var_name]
    if not isinstance(dem, xr.DataArray):
        raise TypeError(
            f"dem must be an xarray Dataset or DataArray, not {type(dem).__name__}"
        )

    dem = dem.squeeze()
    if dem.ndim != 2:
        raise ValueError(
            f"Dem must be a 2D array. Object is {dem.ndim}D with shape {dem.shape}"
        )

    # Drop singleton coordinate variables that are not dimensions (e.g. 'band', 'spatial_ref')
    # These can cause xarray/rioxarray broadcasting/indexing issues during operations
    # such as fillna/round when they remain as coordinate variables.
    for coord_name in list(dem.coords):
        if coord_name not in dem.dims:
            try:
                csize = getattr(dem.coords[coord_name], "size", 0)
                # drop singleton non-dimension coords or coords whose size doesn't match any dimension
                if csize == 1 or csize not in dem.shape:
                    dem = dem.drop_vars([coord_name])
            except Exception as e:
                raise RuntimeError(
                    f"prepare_dem(): dropping unused coordinate '{coord_name}' "
                    f"failed with error: {e}"
                ) from e

    # set dimension names
    if y_dim and dem.tcorr.ydim != y_dim:
        dem = dem.rename({dem.tcorr.ydim: y_dim})
    if x_dim and dem.tcorr.xdim != x_dim:
        dem = dem.rename({dem.tcorr.xdim: x_dim})
    if output_var_name and output_var_name != dem.name:
        dem = dem.rename(output_var_name)

    # flip grid so that northing values are increasing
    if dem.shape[0] > 1 and dem.tcorr.yc[0] > dem.tcorr.yc[-1]:
        dem = dem.sel({dem.tcorr.ydim: dem.tcorr.yc[::-1]})

    if fill_nan is not None:
        try:
            dem = dem.fillna(fill_nan)
        except Exception as e:
            raise RuntimeError(f"prepare_dem(): fillna failed with error: {e}") from e

    if round_dp is not None:
        try:
            dem = dem.round(round_dp)
        except Exception as e:
            raise RuntimeError(f"prepare_dem(): round failed with error: {e}") from e

    return dem


def check_dem(dem: xr.DataArray, show: bool = True) -> bool:
    """Check ``dem`` has increasing and evenly spaced coordinates.

    Parameters
    ----------
    dem : xr.DataArray
        The DEM to check.
    show : bool, default is True
        If True, print warning messages for any issues found.

    Returns
    -------
    bool
        True if no issues found, False otherwise.
    """
    error_collector = GSolveDataWarning(prefix="check_dem()", show=show)
    if dem.shape[0] > 1:
        dff = np.diff(dem.tcorr.yc)
        if not np.all(dff > 0):
            error_collector("y coordinates are not increasing")
        if not np.all(dff[0] == dff[1:]):
            error_collector("y coordinates are not evenly spaced")
    if dem.shape[1] > 1:
        dff = np.diff(dem.tcorr.xc)
        if not np.all(dff > 0):
            error_collector("x coordinates are not increasing")
        if not np.all(dff[0] == dff[1:]):
            error_collector("x coordinates are not evenly spaced")

    error_collector.final_msg()

    return error_collector.count == 0


def create_empty_dataarray(
    var_name: str = "empty",
    x_dim: str = "easting",
    y_dim: str = "northing",
) -> xr.DataArray:
    """Create an empty DataArray with specified dimensions.

    Parameters
    ----------
    var_name : str, default is 'elevation'
        The name of the variable in the output DataArray.
    x_dim : str, default is 'easting'
        The name of the x dimension.
    y_dim : str, default is 'northing'
        The name of the y dimension.

    Returns
    -------
    xarray.DataArray
    """
    da = xr.DataArray(
        [],
        dims=(y_dim,),
        coords={y_dim: []},
        name=var_name,
    )
    return da
