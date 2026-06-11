
===================
GravityObservations
===================
.. currentmodule:: gsolve

Object Creation
~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   GravityObservations
   GravityObservations.from_excel
   GravityObservations.from_csv
   GravityObservations.from_dataframe

Information
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   GravityObservations.loop_ids
   GravityObservations.loop_summary
   GravityObservations.site_summary
   GravityObservations.set_obs_id
   GravityObservations.params
   GravityObservations.activate
   GravityObservations.deactivate
   GravityObservations.check_data

Time Handing
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   GravityObservations.starttime
   GravityObservations.endtime
   GravityObservations.timedelta_unit
   GravityObservations.set_timedelta_unit
   GravityObservations.fixed_time_datum
   GravityObservations.set_fixed_time_datum
   GravityObservations.set_tdelta

Correction Methods
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   GravityObservations.apply_dial_to_mgal
   GravityObservations.apply_earth_tide_correction
   GravityObservations.apply_ocean_load_correction
   GravityObservations.set_calibration_factor
   GravityObservations.calculate_tide_corrected_gravity

Data Export
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   GravityObservations.write_to_csv
   GravityObservations.to_excel


Plotting
~~~~~~~
.. autosummary::
   :toctree: api/

   GravityObservations.plot_observed_data
   GravityObservations.plot_network_map
   GravityObservations.plot_site_visits
