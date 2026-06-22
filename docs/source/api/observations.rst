======================================
Gravity Observations
======================================


GravityObservations
===================

A class for handling gravity observation data, including methods for data manipulation,
correction.

.. currentmodule:: gsolve

Object Creation
---------------
.. autosummary::
   :toctree: api/

   GravityObservations
   GravityObservations.from_dataframe
   GravityObservations.from_excel
   GravityObservations.from_csv

Data Attributes
---------------
.. autosummary::
   :toctree: api/

   GravityObservations.known_fields
   GravityObservations.required_fields
   GravityObservations.set_column
   GravityObservations.set_obs_id
   GravityObservations.loop_ids
   GravityObservations.loop_summary
   GravityObservations.site_summary
   GravityObservations.params
   GravityObservations.activate
   GravityObservations.deactivate
   GravityObservations.check_data

Time Handing
------------
.. autosummary::
   :toctree: api/

   GravityObservations.starttime
   GravityObservations.endtime
   GravityObservations.timedelta_unit
   GravityObservations.set_timedelta_unit
   GravityObservations.fixed_time_datum
   GravityObservations.set_fixed_time_datum
   GravityObservations.set_tdelta

Corrections
-----------
.. autosummary::
   :toctree: api/

   GravityObservations.apply_dial_to_mgal
   GravityObservations.set_calibration_factor
   GravityObservations.apply_earth_tide_correction
   GravityObservations.apply_ocean_load_correction
   GravityObservations.calculate_tide_corrected_gravity

Data Export
-----------
.. autosummary::
   :toctree: api/

   GravityObservations.write_to_csv
   GravityObservations.to_excel

Plotting
--------
.. autosummary::
   :toctree: api/

   GravityObservations.plot_observed_data
   GravityObservations.plot_network_map
   GravityObservations.plot_site_visits

GravityObservationsParameters
=============================
A class for storing parameters and metadata related to reductions of data
in a GravityObservations object.

.. currentmodule:: gsolve.observations

.. autosummary::
   :toctree: api/

   GravityObservationsParameters
   GravityObservationsParameters.from_series
   GravityObservationsParameters.copy
   GravityObservationsParameters.default_values
   GravityObservationsParameters.non_default_values
   GravityObservationsParameters.summary
   GravityObservationsParameters.to_dict
   GravityObservationsParameters.to_excel
   GravityObservationsParameters.to_series

Functions
=========
.. currentmodule:: gsolve.observations

.. autosummary::
   :toctree: api/

   combine_gravity_observations