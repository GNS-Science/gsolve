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

# %%
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

# %%
data_path = pathlib.Path(__file__).parent.parent

cal_data_path = data_path / "surveys" / "calibration"

obs_file = cal_data_path / "G106-Data23Nov2018.xlsx"

ref_site_file = data_path / "absolute_gravity" / "base_stations_calibration.csv"

corr_table_file = data_path / "correction_tables" / "G106.csv"


# %%
# Read in observations
obs = GravityObservations.from_excel(
    obs_file, parse_split_datetime=True, sheet_name="Survey Data"
)

# Read in site location information
sites = GravitySites.from_excel(obs_file, sheet_name="Locations")

# Read in list reference (i.e. absolute) stations
ref_sites = ReferenceGravity.from_csv(ref_site_file)

# set which reference stations are used in this survey (must be in sites)
_ = sites.set_reference_gravity(ref_sites)

# %%
"""Process the observed data.
As this is a manually read G meter we need to convert dial values to mgal via a
conversion table. First read in the conversion table"""
g106converter = LaCosteRombergDialConverter.from_csv(corr_table_file)

# apply dial conversion to convert values to mGal.
obs.apply_dial_to_mgal(g106converter)

# we don't need to set a calibration factor, as this is what we are calculating

# calculate the earth tide correction which requires location information from sites
longman = LongmanTidalCorrection(amp_factor=1.2)
obs.apply_earth_tide_correction(sites, tide_corrector=longman)

# calculate earth tide corrected gravity
obs.calculate_tide_corrected_gravity()

# plot the observed data for loop #2
obs.plot_observed_data(1, "datetime", "meter_reading_mgal")

# %%
# create a survey object using observations and site objects
survey = GravitySurvey(obs, sites)

# %%
"""
Run the network adjustment. As this is a calibration survey we set calculate_beta=True.

Here we use solve method "2", as we have high confidence in our absolute stations.

We process each loop individually and apply a 99 percentile cutoff filter to the residuals.
"""

results = survey.solve_calibration_factor(
    method=2, use_loops=True, percentile_clipping=100
)

# results.site_solution contains the adjusted gravity per station
print(results.site_solution)

# results.obs_solution contains the residuals for each reading.
print(results.obs_solution)
# %%
# plot drift and residual curves.
print("Calibration factor", results.calibration_factor)

_ = results.plot_residual_drift(
    loop=1, filename=str(cal_data_path / "Calibration_residual_drift.png")
)
_ = results.plot_residual_cdf(
    loop=1, filename=str(cal_data_path / "Calibration_residual_cdf.png")
)


# %%
# save output files
# as individual csv files
results.site_solution.to_csv(cal_data_path / "calibration_site_solution.csv")
results.obs_solution.to_csv(cal_data_path / "calibration_observations_solution.csv")
print(results.params.to_dict())
# # or save as single excel
# report = GSolveReport(survey, results)
# report.to_excel(cal_data_path / "calibration_survey_results.xlsx")
