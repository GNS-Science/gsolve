# # %%
# %load_ext autoreload

# %autoreload 1

# %aimport gsolve.gsolve_algorithms
# %aimport gsolve.observations
# %aimport gsolve
# %matplotlib inline

# %%
import pathlib

from gsolve import (
    GravityObservations,
    GravitySites,
    GravitySurvey,
    LaCosteRombergDialConverter,
    ReferenceGravity,
)
from gsolve.tide.earth_tide import LongmanTidalCorrection
from gsolve.tide.ocean_load import generate_qtp_input, qtp_to_corrector

# %%
data_path = pathlib.Path("examples")

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

# generate_qtp_input(
s = obs.data.site_id.to_numpy()
generate_qtp_input(
    site_id=s,
    datetimes=obs.data.datetime,
    latitude=sites.data.loc[s, "latitude"].to_numpy(),
    longitude=sites.data.loc[s, "longitude"].to_numpy(),
    elevation=sites.data.loc[s, "height_ellipsoidal"].to_numpy(),
    output_file=ocean_load_path / "okataina_qtp_input.csv",
)

# step 2: run QTP externally to generate the output file (not shown here)
# Step 3: read in the QTP output file and convert to a corrector object.
#  - QTP output file
qtp_output_file = ocean_load_path / "okataina_qtp_input_Modified.csv"

if not qtp_output_file.exists():
    msg = "You didn't run QuickTide Pro yet did you?"
    raise FileNotFoundError(msg)

qtp_ocean_load_corrector = qtp_to_corrector(
    qtp_output_file,
)
obs.apply_ocean_load_correction(corrector=qtp_ocean_load_corrector)


# Calculate the final corrected gravity value that will be passed to network adjustment
# - all previously applied corrections are included.
obs.calculate_tide_corrected_gravity()
survey = GravitySurvey(obs, sites)

# %%
"""
Run the network adjustment.
Here we use solve method "2", see documentation.  We process each loop individually
and apply a 95 percentile cutoff filter to the residuals."""
results = survey.solve_lstsq(method=2, use_loops=True, percentile_clipping=90)
