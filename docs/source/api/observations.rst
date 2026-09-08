======================================
Gravity Observations
======================================


GravityObservations
===================

A class for handling gravity observation data, including methods for data manipulation
and reduction.

.. currentmodule:: gsolve

Object Creation
---------------
.. autosummary::
   :toctree: api/

   GravityObservations
   GravityObservations.from_excel
   GravityObservations.from_dataframe
   GravityObservations.from_csv
   GravityObservations.merge
   GravityObservations.copy

Data Attributes and Methods
---------------------------
.. autosummary::
   :toctree: api/

   GravityObservations.set_column
   GravityObservations.set_obs_id
   GravityObservations.known_fields
   GravityObservations.required_fields
   GravityObservations.check_data
   GravityObservations.loop_ids
   GravityObservations.loop_summary
   GravityObservations.site_summary
   GravityObservations.params
   GravityObservations.activate
   GravityObservations.deactivate

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

   GravityObservations.to_excel
   GravityObservations.write_to_csv

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
