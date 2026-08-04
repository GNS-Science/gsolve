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

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:54:24 2023.

@author: craigm
"""

# %%
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gsolve import (
    GravitySites,
    TerrainCorrectionParameters,
    TerrainCorrector,
)

data_path = pathlib.Path(__file__).parent.parent

grid_dir = data_path / "surveys" / "terrain_corrections"


data_file = "RIG_Terrain160_2610.xlsx"
dem_file1 = grid_dir / "Wellington8m_Land.tif"
dem_file2 = grid_dir / "Wellington200m_Land.tif"
# %%
# load file with station locations and elevations
sites = GravitySites.from_excel(
    grid_dir / data_file,
    sheet_name="Locations",
)

# Get the DEM elevation at station location to use instead of the original sites.height_ellipsoidal
sites.sample_elevation(
    dem_file1,
    output_col="dem_elevation",
)

# Set up the terrain correction parameters for each grid
inner_zone_params = TerrainCorrectionParameters(
    min_dist=160.0,  # in meters
    max_dist=2160.0,  # in meters
    terrain_density=2670.0,  # in kg/m3
    water_density=1030.0,  # in kg/m3
    distance_mask_type="radial",
    dem_source=grid_dir / dem_file1,
    compute_topography=True,
    compute_bathymetry=False,
    site_height_field="dem_elevation",
    site_easting_field="easting",
    site_northing_field="northing",
    name="8m_dem",
)

outer_zone_params = TerrainCorrectionParameters(
    min_dist=2160.0,  # in meters
    max_dist=21900.0,  # in meters
    terrain_density=2670.0,  # in kg/m3
    water_density=1030.0,  # in kg/m3
    distance_mask_type="radial",
    dem_source=grid_dir / dem_file2,
    compute_topography=True,
    compute_bathymetry=False,
    site_height_field="dem_elevation",
    # site_easting_field="easting",  default is "easting" so can be omitted
    # site_northing_field="northing", ditto
    name="200m_dem",
)
# Set up the Terrain corrector
corrector = TerrainCorrector(params=[inner_zone_params, outer_zone_params])

# Calculate the corrections
results = corrector.compute(points=sites, show_progress=True)

# Save results to file
results.to_excel(
    grid_dir / "terrain_correction_results.xlsx",
    sheet_name="Terrain_Correction",
    if_workbook_exists="replace",
)

# once terrain corrections have been calculated
# the results can be read back in during a standard work flow

# from gsolve.reductions.terrain_corrections import TerrainCorrectionOutput
# results_reread_from_disk = TerrainCorrectionData.from_excel(
#     grid_dir / "terrain_correction_results.xlsx"
# )
