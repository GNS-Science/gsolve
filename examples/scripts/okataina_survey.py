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

try:
    import contextily as cx  # ty:ignore[unresolved-import]
except ImportError:
    has_contextily = False
else:
    has_contextily = True

from gsolve import (
    GravityAnomalies,
    GravityCorrectionParameters,
    GravityObservations,
    GravitySites,
    GravitySurvey,
    LaCosteRombergDialConverter,
    ReferenceGravity,
)
from gsolve.reports import GSolveReport
from gsolve.tide.earth_tide import LongmanTidalCorrection
from gsolve.tide.ocean_load import generate_qtp_input, qtp_to_corrector

# %%
data_path = pathlib.Path(__file__).parent.parent

obs_path = data_path / "surveys" / "Okataina"

ocean_load_path = data_path / "ocean_load" / "quicktide"

survey_file = obs_path / "Okataina_2020_all_4_gsolve.xlsx"

ref_site_file = data_path / "absolute_gravity" / "base_stations.csv"

corr_table_file = data_path / "correction_tables" / "G106.csv"

# the calibration factor for your meter (determined in calibration survey)
calibration_factor = 1 - -0.0019

# %%
# Read in observations
obs = GravityObservations.from_excel(
    survey_file, sheet_name="Survey Data", parse_split_datetime=True
)

# Read in site location information
sites = GravitySites.from_excel(survey_file, sheet_name="Locations")

# Read in list reference (i.e. absolute) stations
ref_sites = ReferenceGravity.from_csv(ref_site_file)

# set which reference stations are used in this survey (must be in sites)
_ = sites.set_reference_gravity(ref_sites)

# plot a network map
# %%
fig, ax = obs.plot_network_map(
    sites,
    savefilename=str(obs_path) + "/ Okataina_network_map.png",
    figsize=(10, 8),
    marker_scale_factor=25,
)

# add a basemap (optional)
if has_contextily:
    cx.add_basemap(ax, source=cx.providers.OpenTopoMap, crs="EPSG:4326")
    fig.savefig(str(obs_path) + "/ Okataina_network_map.png")

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
longman = LongmanTidalCorrection(amp_factor=1.2)
obs.apply_earth_tide_correction(sites, tide_corrector=longman)

# Ocean Load Corrections
# - these are generated externally using Quick Tide Pro or similar.

# Step 1: generate the input file for QTP using the site and observation datetimes.
# - this has been run, uncomment code below to generate a new file.

#generate_qtp_input(
s = obs.data.site_id.to_numpy()
generate_qtp_input(
    site_id=s,
    datetimes=obs.data.datetime,
    latitude=sites.data.loc[s, "latitude"].to_numpy(),
    longitude=sites.data.loc[s, "longitude"].to_numpy(),
    elevation=sites.data.loc[s, "height_ellipsoidal"].to_numpy(),
    output_file=ocean_load_path / "okataina_qtp_input.csv"
)

# step 2: run QTP externally to generate the output file (not shown here)
# Step 3: read in the QTP output file and convert to a corrector object.
#  - QTP output file
qtp_output_file = ocean_load_path / "okataina_qtp_input_Modified.csv"

if not qtp_output_file.exists():
    raise FileNotFoundError(f"You didn't run QuickTide Pro yet did you?")

qtp_ocean_load_corrector = qtp_to_corrector(
    qtp_output_file,
)
obs.apply_ocean_load_correction(corrector=qtp_ocean_load_corrector)


# Calculate the final correcity gravity value that will be passed to network adjustment
# - all previously applied corrections are included.
obs.calculate_tide_corrected_gravity()


# plot the observed data for loop #2
fig, ax = obs.plot_observed_data(
    2,
    "datetime",
    "meter_reading_mgal",
    savefilename=str(obs_path) + "/ Okataina_observations.png",
)

# %%
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

# %%
# plot drift and residual curves.
results.plot_residual_drift(
    loop=2, filename=str(obs_path) + "/ Okataina_residual_drift.png"
)
results.plot_residual_cdf(
    loop=2, filename=str(obs_path) + "/ Okataina_residual_cdf.png"
)

# %%
# Define parameters for calculating gravity corections and anomalies
correction_params = GravityCorrectionParameters(
    ellipsoid="GRS80",
    density_crust=2670.0,
    density_water=1030.0,
    spherical_cap_radius=166735.0,
    use_curvature_corrected=True,
    use_atmospheric_correction=True,
)


# %%
# calculate anomalies
# anomalies are calculated using normal_gravity_on_ellipsoid_surface.

anomalies = GravityAnomalies(
    absolute_gravity=results.site_solution,
    sites=survey.sites,
    corrections_parameters=correction_params,
    terrain_corrections=None,
)
# %%
anomalies.data


# %%
report = GSolveReport(
    observations=obs,
    results=results,
    sites=survey,
    anomalies=anomalies,
)

# %%
# save output files
# as individual csv

results.site_solution.to_csv(
    obs_path / "Okataina_site_solution.csv", float_format="%.5f"
)
results.obs_solution.to_csv(
    obs_path / "Okataina_observations_solution.csv", float_format="%.5f"
)

# %%
# or everything as a single excel from the report object.
report.to_excel(obs_path / "okataina_survey_results.xlsx", if_workbook_exists="replace")
