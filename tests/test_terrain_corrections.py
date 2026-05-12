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

from typing import Any

import harmonica as hm
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gsolve.reductions.terrain_corrections import (
    TerrainCorrectionData,
    TerrainCorrectionParameters,
    TerrainCorrector,
    calculate_terrain_correction,
)


class DummyPrism:
    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        # expose prism_layer attribute as used in production code
        self.prism_layer = self

    def gravity(
        self,
        point,
        density_name=None,
        field=None,
        parallel=False,
        disable_checks=False,
        progressbar=False,
    ):
        # deterministic fake gravity value based on point for test assertions
        return float(point[0] + point[1] + point[2])


def make_simple_dem():
    # create a 3x3 DEM with named dims 'northing' (y) and 'easting' (x)
    y = np.array([100.0, 200.0, 300.0])
    x = np.array([10.0, 20.0, 30.0])
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    da = xr.DataArray(
        data,
        coords={"northing": y, "easting": x},
        dims=("northing", "easting"),
        name="elevation",
    )
    return da


@pytest.mark.skip
def test_calculate_terrain_correction_monkeypatched_prism(monkeypatch):
    dem = make_simple_dem()
    # patch harmonica.prism_layer to return DummyPrism
    monkeypatch.setattr(hm, "prism_layer", lambda *a, **k: DummyPrism())

    # single point (x,y,z)
    pts = ([20.0], [200.0], [50.0])
    topo, bathy = calculate_terrain_correction(
        points=pts,
        dem=dem,
        min_dist=0.0,
        max_dist=100.0,
        distance_mask_type="radial",
        compute_topography=True,
        compute_bathymetry=True,
        show_progress=False,
    )

    # gravity value returned by DummyPrism.gravity is x+y+z
    expected = 20.0 + 200.0 + 50.0
    assert np.allclose(topo, np.array([expected], dtype=float))
    assert np.allclose(bathy, np.array([expected], dtype=float))


@pytest.mark.skip
def test_terrain_corrector_compute_stores_results(monkeypatch):
    dem = make_simple_dem()
    # prepare params
    params = TerrainCorrectionParameters(name="zone1", min_dist=0.0, max_dist=100.0)

    # monkeypatch calculate_terrain_correction to return known arrays
    def fake_calc(points, dem, min_dist, max_dist, **kwargs):
        n = len(np.atleast_1d(points[0]))
        return np.full(n, 10.0), np.full(n, 5.0)

    monkeypatch.setattr(
        "gsolve.reductions.terrain_corrections.calculate_terrain_correction", fake_calc
    )

    # create corrector with the dem supplied for the single zone
    tc = TerrainCorrector(params=params, dem=dem)

    # two sample points
    x = np.array([10.0, 30.0])
    y = np.array([100.0, 300.0])
    z = np.array([0.0, 1.0])
    results = tc.compute(points=(x, y, z), show_progress=False)

    # check resulting dataframe has expected tcorr columns
    topo_col = "tcorr:zone1:topo"
    bathy_col = "tcorr:zone1:bath"
    total_col = "tcorr:total"

    assert topo_col in results.data.columns
    assert bathy_col in results.data.columns
    assert total_col in results.data.columns

    # check values
    np.testing.assert_allclose(
        results.data[topo_col].to_numpy(), np.array([10.0, 10.0])
    )
    np.testing.assert_allclose(results.data[bathy_col].to_numpy(), np.array([5.0, 5.0]))
    # total should be the sum of existing tcorr columns for each row
    np.testing.assert_allclose(
        results.data[total_col].to_numpy(), np.array([15.0, 15.0])
    )
