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

from gsolve.core.xr_accessor import TCorrMethods as _TCorrMethods


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


def test_tcorr_accessor_dims(simple_dem):
    assert simple_dem.tcorr.xdim == "easting"
    assert simple_dem.tcorr.ydim == "northing"


def test_tcorr_dx_dy(simple_dem):
    assert simple_dem.tcorr.dx == 10
    assert simple_dem.tcorr.dy == 100


def test_tcorr_xc_yc(simple_dem):
    np.testing.assert_array_equal(simple_dem.tcorr.xc, [10, 20, 30])
    np.testing.assert_array_equal(simple_dem.tcorr.yc, [100, 200, 300])


def test_tcorr_bounds(simple_dem):
    bounds = simple_dem.tcorr.bounds
    assert bounds.shape == (4,)


def test_tcorr_coords_to_indices_scalar(simple_dem):
    i, j = simple_dem.tcorr.coords_to_indices([20, 200])
    assert i == 1 and j == 1


def test_tcorr_coords_to_indices_array(simple_dem):
    i, j = simple_dem.tcorr.coords_to_indices([[10, 30], [100, 300]])
    np.testing.assert_array_equal(i, [0, 2])
    np.testing.assert_array_equal(j, [0, 2])


def test_tcorr_clip_to_points(simple_dem):
    clipped = simple_dem.tcorr.clip_to_points([[10, 30], [100, 300]])
    assert clipped.shape == (3, 3)


def test_tcorr_get_land_sea_mask(simple_dem):
    mask = simple_dem.tcorr.get_land_sea_mask(sea_level_elevation=5)
    assert mask.dtype == bool
    assert mask.shape == simple_dem.shape


def test_tcorr_get_bathymetry_elevation(simple_dem):
    mask = simple_dem.tcorr.get_land_sea_mask(sea_level_elevation=5)
    bathy = simple_dem.tcorr.get_bathymetry_elevation(
        land_sea_mask=mask, sea_level_elevation=0
    )
    assert np.all((bathy.values == 0) | (bathy.values == simple_dem.values))


def test_tcorr_generate_bathymetry_density(simple_dem):
    mask = simple_dem.tcorr.get_land_sea_mask(sea_level_elevation=5)
    dens = simple_dem.tcorr.generate_bathymetry_density(land_sea_mask=mask)
    assert dens.shape == simple_dem.shape


def test_tcorr_get_topography_elevation(simple_dem):
    mask = simple_dem.tcorr.get_land_sea_mask(sea_level_elevation=5)
    topo = simple_dem.tcorr.get_topography_elevation(land_sea_mask=mask)
    assert np.all((topo.values == simple_dem.values) | (topo.values == 0.0))


def test_tcorr_generate_topo_density(simple_dem):
    dens = simple_dem.tcorr.generate_topo_density(2670)
    assert np.all(dens.values == 2670)


def test_tcorr_clip_to_arr(simple_dem):
    arr2 = simple_dem.copy()
    clipped = simple_dem.tcorr.clip_to_arr(arr2)
    assert clipped.shape == simple_dem.shape


def test_tcorr_is_compatible(simple_dem):
    arr2 = simple_dem.copy()
    assert simple_dem.tcorr.is_compatible(arr2)
    arr3 = simple_dem.rename({"easting": "x"})
    assert not simple_dem.tcorr.is_compatible(arr3)


def test_tcorr_generate_distance_mask(simple_dem):
    mask = simple_dem.tcorr.generate_distance_mask(
        point=[20, 200], min_dist=0, max_dist=200
    )
    assert mask.shape == simple_dem.shape


def test_tcorr_apply_mask(simple_dem):
    mask = np.ones(simple_dem.shape, dtype=bool)
    masked = simple_dem.tcorr.apply_mask(mask, fill_value=-1)
    assert np.all(masked.values == simple_dem.values)


def test_tcorr_cell_edges(simple_dem):
    x_edges, y_edges = simple_dem.tcorr.cell_edges()
    assert len(x_edges) == simple_dem.shape[1] + 1
    assert len(y_edges) == simple_dem.shape[0] + 1
