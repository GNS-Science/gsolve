import pathlib

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

# from gsolve.tide.ocean_load import generate_qtp_input, qtp_to_corrector

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
# obs_test = obs_orig.copy()
# Read in site location information
sites = GravitySites.from_excel(survey_file, sheet_name="Locations")

# Read in list reference (i.e. absolute) stations
ref_sites = ReferenceGravity.from_csv(ref_site_file)

# set which reference stations are used in this survey (must be in sites)
_ = sites.set_reference_gravity(ref_sites)


# %%
"""Process the observed data.
As this is a manually read G meter we need to convert dial values to mgal via a
conversion table. First read in the conversion table"""
g106converter = LaCosteRombergDialConverter.from_csv(corr_table_file)

obs.apply_dial_to_mgal(g106converter)
obs.set_calibration_factor(calibration_factor)
longman = LongmanTidalCorrection(amp_factor=1.2)
obs.apply_earth_tide_correction(sites, tide_corrector=longman)
obs.calculate_tide_corrected_gravity()

results = GravitySurvey(obs, sites).solve_lstsq(
    method=2, use_loops=True, percentile_clipping=99.0
)


results.plot_residual_drift(
    loop=2, filename=str(obs_path) + "/ Okataina_residual_drift.png"
)
results.plot_residual_cdf(
    loop=2, filename=str(obs_path) + "/ Okataina_residual_cdf.png"
)

# Define parameters for calculating gravity corrections and anomalies
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
    sites=sites,
    corrections_parameters=correction_params,
    terrain_corrections=None,
)

# %%
anomalies.data  # ruff: ignore[useless-expression]


# %%
report = GSolveReport(
    observations=obs,
    results=results,
    sites=sites,
    anomalies=anomalies,
)

# or everything as a single excel from the report object.
report.to_excel(filename="test_report.xlsx", if_workbook_exists="replace")
