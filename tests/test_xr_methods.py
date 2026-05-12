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

import numpy as np
import pytest
import xarray as xr

# from gsolve.core.xr_accessor import TCorrMethods as _TCorrMethods
from gsolve.core.xr_methods import check_dem, prepare_dem


@pytest.fixture
def simple_dem():
    y = np.array([100, 200, 300])
    x = np.array([10, 20, 30])
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    da = xr.DataArray(
        data,
        coords={"northing": y, "easting": x},
        dims=("northing", "easting"),
        name="elevation",
    )
    return da


def test_prepare_dem_renames(simple_dem):
    da = prepare_dem(simple_dem, x_dim="x", y_dim="y", output_var_name="z")
    assert "x" in da.dims
    assert "y" in da.dims
    assert da.name == "z"


def test_prepare_dem_flip(simple_dem):
    flipped = simple_dem.isel(northing=slice(None, None, -1))
    da = prepare_dem(flipped)
    assert np.all(np.diff(da.tcorr.yc) > 0)


def test_prepare_dem_fillna(simple_dem):
    da = simple_dem.copy()
    da = da.where(da > 0)
    out = prepare_dem(da, fill_nan=42)
    assert np.all(out.values >= 1)


def test_prepare_dem_round(simple_dem):
    da = simple_dem * 1.123456
    out = prepare_dem(da, round_dp=2)
    assert np.allclose(out.values, np.round(out.values, 2))


def test_check_dem_true(simple_dem):
    assert check_dem(simple_dem)


def test_check_dem_false_spacing(simple_dem):
    da = simple_dem.copy()
    da = da.assign_coords(northing=[100, 200, 400])
    assert not check_dem(da)
