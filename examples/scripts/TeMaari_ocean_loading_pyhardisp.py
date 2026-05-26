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
Created on Wed Feb 25 16:23:07 2026

@author: craigm
"""

import pathlib

from gsolve import (
    GravityObservations,
    GravitySites,
    GravitySurvey,
    LaCosteRombergDialConverter,
    ReferenceGravity,
)
from gsolve.reports import GSolveReport
from gsolve.tide.earth_tide import LongmanTidalCorrection
from gsolve.tide.ocean_load import HardispOceanLoadCorrector

# %%
data_path = pathlib.Path(__file__).parent.parent

obs_path = data_path / "surveys" / "TeMaari"

ocean_load_path = data_path / "ocean_load" / "hardisp" / "TeMaari_olmpp_mGal.dat"

survey_file = obs_path / "G106_temaari.xlsx"

corr_table_file = data_path / "correction_tables" / "G106.csv"

# the calibration factor for your meter (determined in calibration survey)
calibration_factor = 1 - -0.0019

# %%
# Read in observations
obs = GravityObservations.from_excel(
    survey_file, sheet_name="obs", parse_split_datetime=False
)

# Read in site location information
sites = GravitySites.from_excel(survey_file, sheet_name="Locations")

# set which reference stations are used in this survey (must be in sites). In
# this case i just set TGKB to be 0.0 mGal
ref_sites = ReferenceGravity(site_id="TGKB", gravity=0.0)
_ = sites.set_reference_gravity(ref_sites)

# plot a network map

# %%
"""Process the observed data.
As this is a manually read G meter we need to convert dial values to mgal via a
conversion table. First read in the conversion table"""
g106converter = LaCosteRombergDialConverter.from_csv(corr_table_file)

# apply dial conversion to convert values to mGal.
obs.apply_dial_to_mgal(g106converter)

# set the calibration factor
obs.set_calibration_factor(calibration_factor)

# calculate the earth tide correction which requires location information from sites
longman = LongmanTidalCorrection(amp_factor=1.16)
obs.apply_earth_tide_correction(sites, tide_corrector=longman)

# Ocean Load Corrections using pyhardisp
# read in the BLQ file from the online provider
ocean_load = HardispOceanLoadCorrector(ocean_load_path)
# apply the ocean loading correction
obs.apply_ocean_load_correction(corrector=ocean_load)

# calculate tide corrected gravity
obs.calculate_tide_corrected_gravity()

# create a survey object using observations and site objects
survey = GravitySurvey(obs, sites)

# %%
"""
Run the network adjustment.
Here we use solve method "2", see documentation.  We process each loop individually
and apply a 99 percentile cutoff filter to the residuals.  As this is not a calibration
survey we do not need to calculate calibration factor"""
results = survey.solve_lstsq(
    method=2, use_loops=True, calculate_calibration_factor=False, percentile_clipping=95
)

# results.site_solution contains the adjusted gravity per station
print(results.site_solution)

# results.obs_solution contains the residuals for each reading.
print(results.obs_solution)


report = GSolveReport(observations=obs, sites=sites, results=results)
