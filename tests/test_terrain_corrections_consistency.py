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
import pandas as pd
import pytest
import xarray as xr

from gsolve.reductions.terrain_corrections import (
    TerrainCorrectionData,
    TerrainCorrectionParameters,
    TerrainCorrector,
    calculate_terrain_correction,
)
from gsolve.sites import GravitySites

dem_max = 1000.0
tc_max_dist = 150.0
tc_min_dist = 7.0


@pytest.fixture
def ripple_dem() -> xr.DataArray:
    x = np.linspace(-dem_max, dem_max, 401)
    y = np.linspace(-dem_max, dem_max, 401)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt((X / 90) ** 2 + (Y / 90) ** 2)
    Z = 100 * np.sin(R) / (R + 1)
    da = xr.DataArray(
        Z,
        coords={"northing": y, "easting": x},
        dims=["northing", "easting"],
        name="elevation",
    )
    return da


@pytest.fixture
def sites(ripple_dem) -> GravitySites:

    rng = np.random.default_rng(seed=42)
    n_points = 10
    x = rng.uniform(-dem_max, dem_max, size=n_points)
    y = rng.uniform(-dem_max, dem_max, size=n_points)
    z = rng.uniform(0, 20, size=n_points)

    sites = GravitySites(
        site_id=[str(n) for n in range(1, n_points + 1)],
        latitude=rng.uniform(-90.0, 90.0, size=n_points),
        longitude=rng.uniform(-180.0, 180.0, size=n_points),
        height_ellipsoidal=z,
        easting=x,
        northing=y,
    )
    sites.sample_elevation(dem=ripple_dem, output_col="height_dem")
    return sites


@pytest.fixture
def pre_calced_tcorr_data(tc_params) -> TerrainCorrectionData:
    # Output from these test methods as at 09-04-2026
    # - These values are not certain to be correct
    # - Catch changes that alter the results
    # -> Investigate causes
    data = [
        "site_id,easting,northing,site_height_field,tcorr:topo:topo,tcorr:bath:bath,tcorr:both:topo,tcorr:both:bath,tcorr:total",
        "1,547.9120971119266,-258.4039515348376,5.909,0.046479,0.012204,0.046479,0.012204,0.117366",
        "2,-122.24312049589537,853.5299776972035,-1.582,NaN,NaN,NaN,NaN,NaN",
        "3,717.1958398227648,287.730240161329,7.86,0.030694,0.002449,0.030694,0.002449,0.06628600000000001",
        "4,394.7360581187279,645.52322654166,9.067,0.020475,0.001368,0.020475,0.001368,0.043686",
        "5,-811.6453042247009,-113.17160234533776,3.254,0.030521,0.007531,0.030521,0.007531,0.07610399999999999",
        "6,951.244703273512,-545.5225564304462,-2.937,NaN,NaN,NaN,NaN,NaN",
        "7,522.2794039807059,109.16957403166953,-5.338,0.445764,-0.208579,0.445764,-0.208579,0.47437000000000007",
        "8,572.1286105539077,-872.3654877916493,-6.743,NaN,NaN,NaN,NaN,NaN",
        "9,-743.7727346489082,655.2623439851641,-8.315,0.660062,-0.402143,0.660062,-0.402143,0.5158380000000002",
        "10,-99.22812420886578,263.32879824412976,-0.133,0.181248,0.053141,0.181248,0.053141,0.468778",
    ]
    data = [r.split(",") for r in data]
    idx = [r.pop(0) for r in data]
    idx_name = idx.pop(0)
    cols = data.pop(0)
    df = pd.DataFrame(data, columns=cols, index=idx).astype(float)
    df.index.name = idx_name
    return TerrainCorrectionData.from_dataframe(df, params=tc_params)


@pytest.fixture
def tc_params(ripple_dem) -> list[TerrainCorrectionParameters]:
    p1 = TerrainCorrectionParameters(
        name="topo",
        min_dist=tc_min_dist,
        max_dist=tc_max_dist,
        distance_mask_type="radial",
        compute_bathymetry=False,
        dem_source=ripple_dem,
        site_height_field="height_dem",
    )
    p2 = TerrainCorrectionParameters(
        name="bath",
        min_dist=tc_min_dist,
        max_dist=tc_max_dist,
        distance_mask_type="radial",
        compute_topography=False,
        dem_source=ripple_dem,
        site_height_field="height_dem",
    )
    p3 = TerrainCorrectionParameters(
        name="both",
        min_dist=tc_min_dist,
        max_dist=tc_max_dist,
        distance_mask_type="radial",
        dem_source=ripple_dem,
        site_height_field="height_dem",
    )
    return [p1, p2, p3]


def test_terrain_correction_consistency(
    ripple_dem, sites, tc_params, pre_calced_tcorr_data
):
    tc = TerrainCorrector(params=tc_params)
    results = tc.compute(sites, show_progress=False)

    to_close_dist = dem_max - tc_max_dist
    bad_points = results.data.easting.abs().gt(
        to_close_dist
    ) | results.data.northing.abs().gt(to_close_dist)
    tcorr_cols = [col for col in results.data.columns if col.startswith("tcorr:")]

    # test that points too close to edges (i. dem to small) produce NaN
    assert results.data.loc[~bad_points, tcorr_cols].notna().all(axis=None)
    assert results.data.loc[bad_points, tcorr_cols].isna().all(axis=None)

    # test that pre-calculated results are close to calculated results
    for col in tcorr_cols:
        assert np.allclose(
            results.data[col].values,
            pre_calced_tcorr_data.data[col].values,
            atol=1e-6,
            equal_nan=True,
        )
