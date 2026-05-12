# GSolve fundamentals

[Gsolve](index.rst) has four main data objects called [`GravityObservations`](#observations), [`GravitySites`](#sites), [`ReferenceGravity`](#reference-gravity) and [`GravitySurvey`](#surveys).

For processing beyond a network adjustment (i.e. to Complete Bouguer anomaly) there are [`GravityCorrections`](#gravity-corrections), [`TerrainCorrector`](#Terrain corrections) and [`GravityAnomalies`](#gravity-anomalies) and finally a [`GSolveReports`](#output-files) class for output results files.

A [command line interface](gsolve_cli_manual.md) is available for network adjustment processing.

(observations)=

## Observations

The `GravityObservations` object deals with gravity readings and time data only.  It does not take geographic information of reading locations.

At a minimum it requires a "site_id", "meter_id", "datetime" and "meter_reading" or "meter_reading_mgal" along with a survey "loop" number and a flag indicating if the reading is "active".

Tide corrections are calculated and applied to observations using Longman formula.

For manually read La Coste and Romberg meters, the dial to mGal conversion table can be read in and applied to the dial readings.

Several plotting functions exist to visualise the data as both time series or as a survey network map.

(sites)=

## Sites

The `GravitySites` object holds the coordinate information for each gravity station.

At a minimum it requires a "site_id" which matches those in the `GravityObservation` object, along with "latitude", "longitude" and "height_ellipsoidal".

If terrain corrections are to be calculated then cartesian coordinates, easting, northing are required.

(reference-gravity)=

## Reference Gravity

The `ReferenceGravity` object holds "reference" gravity information for tie sites, base stations and so on.  Reference gravity is usually taken to be a well constrained absolute gravity measurement, but can be set to any arbitary value for the survey purpose, such as time change gravity where absolute gravity values are not of interest.

If Bouguer anomalies are the goal, then absolute values are required.  Otherwise the reference station value can be set to zero for time change surveys.

(tides)=

## Earth tide and ocean loading

Gsolve uses the Longman or [pygtide](https://github.com/hydrogeoscience/pygtide) implementation of Eterna tide correction algorithms and can optionally undertake ocean loading corrections.  Ocean loading can be either precalculated from Quicktide Pro and reading in the file from QTP, or can be calculated within gsolve by using [pyhardisp](https://github.com/craigmillernz/pyhardisp).  Pyhardisp reads in a BLQ formatted file obtained from the Chalmers [ocean loading provider](https://barre.oso.chalmers.se/loading/l.php) and then computes ocean load corrections.  Gsolve provides an easy method to read in the blq file and calculate the ocean load.

(surveys)=

## Surveys

The `GravitySurvey` links together `GravityObservations` and `GravitySites` in to a combined object used for running a [network adjustment](Tutorial_network_adjustment_and_anomaly_calculation.md).

(network-adjustment)=

## Network adjustment

GSolve uses the Reilly 1970 [algorithm](gsolve_algorithms.md) with three solving options to suit different survey situations.

The [adjustment](Tutorial_network_adjustment_and_anomaly_calculation.md) calculates residuals for each observation, along with the loop drift rate.  It calculates the variance, standard error and stardard deviation of gravity values at each site.

The [meter calibration](Tutorial_calculate_calibration_beta_factor.md) (or beta) factor can also be calculated from a set of measurements at locations with absolute gravity measurements.

Plots of residuals distribution and drift rate can be made as well as outputing solution tables at the observations or sites level.

## Data reading

For manually read LaCoste and Romberg meters data can be read in using xlsx or csv files containing the relevant Observation and Site information.

For CG6 meters, data can be read in directly from the instrument file (Lynx tablet files not yet supported) and converted easily to `GravityObservations` and `GravitySites` objects.

(gravity-corrections)=

## Gravity corrections

The `GravityCorrections` class contains functions to compute the following:

1. Normal gravity on the ellipsoid

2. Normal gravity at the station elevation

3. Free air correction

4. Atmospheric correction

5. Bouguer slab correction

6. Spherical bouguer cap correction or Bouguer correction with the spherical cap correction applied

To initiate the `GravityCorrections` object requires, "latitude", "height_ellipsoidal", as well as choices of ellipsoid, unit choice (SI or "mGal"), crust and water density, cap extent and ellipsoid radius.

## Terrain corrections

Gsolve can compute [terrain corrections](terrain_corrections.md) using the prism technique implemented in [Harmonica](https://www.fatiando.org/harmonica/latest/).

It requires several DEM at appropriate resolutions and a data file with station locations (such as from `GravitySites`).

DEM and station locations must be in cartesian coordinates.

(gravity-anomalies)=

## Gravity anomalies

The `GravityAnomalies` class contains functions to compute the following:

1. Free air anomaly

2. Simple Bouguer anomaly

3. Complete Bouguer anomaly

More information on how  [gravity anomalies](Tutorial_gravity_corrections_anomalies.md) are calculated.

(output-files)=

## Output files

The `GSolveReport` class is used to create output excel or csv files of the input observations, gravity results per site after network adjustment, any corrections or anomalies calculated.

Additional meta data is included on the processing parameters used.
