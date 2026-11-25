# %%
import pathlib

import matplotlib.pyplot as plt

from gsolve import (
    GravityObservations,
    GravitySites,
    ReferenceGravity,
)

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
obs_orig = GravityObservations.from_excel(
    survey_file, sheet_name="Survey Data", parse_split_datetime=True
)
obs_test = obs_orig.copy()
# Read in site location information
sites_orig = GravitySites.from_excel(survey_file, sheet_name="Locations")
sites_test = sites_orig.copy()
# Read in list reference (i.e. absolute) stations
ref_sites = ReferenceGravity.from_csv(ref_site_file)

# set which reference stations are used in this survey (must be in sites)
_ = sites_orig.set_reference_gravity(ref_sites)
_ = sites_test.set_reference_gravity(ref_sites)
# plot a network map

# %%
_ = obs_orig.plot_observed_data(y_column="meter_reading", loop="1", show=False)
plt.show()
