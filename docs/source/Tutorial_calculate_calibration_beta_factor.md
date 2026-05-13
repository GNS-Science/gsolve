# Calculate the calibration factor

```{tip}  Example script to process the calibration data in the examples folder.
```

Here is an example on how to use gSolve to calculate the ```calibration_factor```.  The ```calibration_factor``` scales the readings against a set of well constrained absolute stations on a calibration range.  Note that the calibration_factor is calculated using un-corrected data and is solved as part of the network adjustment.

Once the calibration_factor is calculated you can use the previous tutorial to apply it to your measurements.

```python
import pathlib

from gsolve import (
    GravityObservations,
    GravitySites,
    GravitySurvey,
    ReferenceGravity,
    DialToMgalConverter,
    GSolveReport
)
```

Set up the directory and files required.  We need a calibration survey dataset (in excel format here), the G meter dial correction table and the reference stations list.

```python
data_path = pathlib.Path.cwd() 
cal_data_path = data_path / "surveys/calibration/"

obs_file = cal_data_path + "G106-Data23Nov2018.xlsx"
ref_site_file = data_path / "absolute_gravity" / "base_stations_calibration.csv"
corr_table_file = data_path / "correction_tables" / "G106.csv"
```

## Read in observations

```{tip}
Here we are using the old gsovle spreadsheet format so you need to supply the ```sheet_name```

```python
obs = GravityObservations.from_excel(
    obs_file, parse_split_datetime=True, sheet_name="Survey Data"
)
```

### Read in site location information

```python
sites = GravitySites.from_excel(obs_file, sheet_name="Locations")
```

### Read in list reference (i.e. absolute) stations

```python
ref_sites = ReferenceGravity.from_csv(ref_site_file)
```

### set which reference stations are used in this survey (must be in sites)

```python
_ = sites.set_reference_gravity(ref_sites)
```

## Process the observed data

As this is a manually read G meter we need to convert dial values to mgal via a conversion table. First read in the conversion table

```python
g106converter = DialToMgalConverter.from_csv(corr_table_file)
```

### apply dial conversion to convert values to mGal

```python
obs.apply_dial_to_mgal(g106converter)
```

Note that as we want to calculate the calibration factor we do not need to apply an existing calibration.

### calculate the earth tide correction which requires location information from sites

```python
obs.apply_earth_tide_correction(sites)
```

### calculate earth tide corrected gravity

```python
obs.calculate_tide_corrected_gravity()
```

### plot the observed data for loop #1

```python
obs.plot_observed_data(1, "datetime", "meter_reading_mgal")
```

### create a survey object using observations and site objects

```python
surv = GravitySurvey(obs, sites)
```

## Run the network adjustment

As this is a calibration survey we set calculate_calibration=True.  Here we use solve method "2", as we have high confidence in our absolute stations.  We process each loop individually and use all data (confidence_interval=100).

```python
results = surv.solve_lstsq(
    method=2, use_loops=True, calculate_calibration=True, confidence_interval=100
)
```

```{tip}
```results.site_solution``` contains the adjusted gravity per station
```python
print(results.site_solution)
```

```{tip}
```results.obs_solution``` contains the residuals for each reading.
```python
print(results.obs_solution)
```

```{tip}
```results.calibration_factor``` contains the calibration_factor.
```python
print("calibration factor", results.calibration_factor)
Calibration factor = 1.00148517
```

### plot drift and residual curves

```python
results.plot_residual_drift(
    loop=1, savefilename=str(cal_data_path) + "Calibration_residual_drift.png"
)
results.plot_residual_cdf(
    loop=1, savefilename=str(cal_data_path) + "Calibration_residual_cdf.png"
)
```

### save output files

as individual csv files

```python
results.site_solution.to_csv(cal_data_path / "calibration_site_solution.csv")
results.obs_solution.to_csv(cal_data_path / "calibration_observations_solution.csv")
```

or as a single excel

```python
report = GSolveReport(survey, results)
report.to_excel(obs_path / "okataina_survey_results.xlsx")
```
