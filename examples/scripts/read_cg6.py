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

from pathlib import Path

from gsolve import GravitySurvey
from gsolve.scintrex import CG6Data
from gsolve.tide.earth_tide import LongmanTidalCorrection

data_path = Path(__file__).parent.parent
obs_path = data_path / "scintrex"
instrument_file = obs_path / "CG6_internal_format.dat"


# load and parse CG6 data
# loop information is read from the line column by setting loop_from_line=True
cg6data = CG6Data.from_file(instrument_file, loop_from_line=True)

# alternate method is to set loops based on gaps in observation times
# - use 12h gap to define a new loop
# - start loop numbering from 1
cg6data.set_loop(time_gap="12h", loop_start=1)

# prepare GSolve objects

# GravityObservations
obs = cg6data.to_gsolve_observations()

# set the (beta) calibration factor
obs.set_calibration_factor(1.0)

# plot the observed data for loop #2
fig, ax = obs.plot_observed_data(
    2,
    "datetime",
    "meter_reading_mgal",
    savefilename=str(obs_path) + "/cg6_observations.png",
)

# GravitySites
# this takes coordinates from CG-6 file, which may not be accurate enough for bouguer corrections
# if coords_source = 'gps', these should not be used for bouguer corrections
sites = cg6data.to_gsolve_sites(coords_source="user")  # gps or user

# Set reference gravity, activate ties
sites.set_reference_gravity({"WAIRAKEI_ABS": 0.0})
sites.activate_ties(["WAIRAKEI_ABS"])


# calculate the earth tide correction which requires location information from sites
longman = LongmanTidalCorrection(amp_factor=1.2)
obs.apply_earth_tide_correction(sites, tide_corrector=longman)

# calculate earth tide corrected gravity
obs.calculate_tide_corrected_gravity()

# make a survey from the observations and sites
survey = GravitySurvey(obs, sites)

# solve for gravity
results = survey.solve_lstsq(
    method=1, use_loops=True, calculate_calibration_factor=False, percentile_clipping=99
)

# results.site_solution contains the adjusted gravity per station
print(results.site_solution)

# results.obs_solution contains the residuals for each reading.
print(results.obs_solution)

# plot drift and residual curves.
results.plot_residual_drift(loop=1, filename=str(obs_path) + "/ cg6_residual_cdf.png")
results.plot_residual_cdf(loop=1, filename=str(obs_path) + "/ cg6_residual_drift.png")

# plot drift and residual curves.
results.plot_residual_drift(loop=2, filename=str(obs_path) + "/ cg6_residual_cdf.png")
results.plot_residual_cdf(loop=2, filename=str(obs_path) + "/ cg6_residual_drift.png")
