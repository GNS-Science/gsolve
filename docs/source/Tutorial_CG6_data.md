# CG6 data

GSolve can import CG6 data from the instrument format file (not the Lynx format file) using the `CG6Data` class from the `scintrex` module.

```{tip}  Example script to process CG6 data can be found in the examples folder.
```

Import what's needed.

```python
from pathlib import Path
from gsolve import GravitySurvey
from gsolve.scintrex import CG6Data
from gsolve.tide.earth_tide import LongmanTidalCorrection
```

Set up the directory and files required.

```python
data_path = Path(__file__).parent.parent
obs_path = data_path / "scintrex"
instrument_file = obs_path / "CG6_internal_format.dat"
```

## Load and parse CG6 data

```{tip} loop information is read from the line column by setting loop_from_line=True
```

```python
cg6data = CG6Data.from_file(instrument_file, loop_from_line=True)
```

```{tip} An alternate method for defining loop is to set them based on gaps in observation times.
```

 Use 12h gap to define a new loop and start loop numbering from 1.

```python
cg6data.set_loop(time_gap="12h", loop_start=1)
```

## Prepare GSolve objects

Both observations and sites objects can be created directly from the cg6data.

### Gravity Observations

```python
obs = cg6data.to_gsolve_observations()
```

#### Set the (beta) calibration factor

```python
obs.set_calibration_factor(1.0)
```

### Plot the observed data for loop #2

```python
fig, ax = obs.plot_observed_data(
    2,
    "datetime",
    "meter_reading_mgal",
    savefilename=str(obs_path) + "/cg6_observations.png",
)
```

### GravitySites

```{warning} This takes coordinates from CG-6 file, which may not be accurate enough for bouguer corrections
if coords_source = 'gps', these should not be used for bouguer corrections as it just uses the instrument GPS```

```python
sites = cg6data.to_gsolve_sites(coords_source='user') # gps or user
```

#### Set reference gravity, activate ties

Here we use a dictionary to set the value of "WAIRAKEI_ABS to 0.0".

```python
sites.set_reference_gravity({"WAIRAKEI_ABS": 0.0})
sites.activate_ties(["WAIRAKEI_ABS"])
```

### Calculate the earth tide correction which requires location information from sites

```python
longman = LongmanTidalCorrection(amp_factor=1.2)
obs.apply_earth_tide_correction(sites, tide_corrector=longman)
```

### Calculate earth tide corrected gravity

```python
obs.calculate_tide_corrected_gravity()
```

### Make a survey from the observations and sites

```python
survey = GravitySurvey(obs, sites)
```

## Solve for gravity

```python
results = survey.solve_lstsq(
    method=1, use_loops=True, calculate_calibration_factor=False, percentile_clipping=99
)
```

## Inspect the results

```python
print(results.site_solution)

print(results.obs_solution)

results.plot_residual_drift(loop=1, filename=str(obs_path) + "/ cg6_residual_cdf.png")
results.plot_residual_cdf(loop=1, filename=str(obs_path) + "/ cg6_residual_drift.png")
results.plot_residual_drift(loop=2, filename=str(obs_path) + "/ cg6_residual_cdf.png")
results.plot_residual_cdf(loop=2, filename=str(obs_path) + "/ cg6_residual_drift.png")
```
