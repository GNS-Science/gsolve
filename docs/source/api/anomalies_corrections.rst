==================================
Standard Corrections and Anomalies
==================================

GravityAnomalies
================
.. currentmodule:: gsolve

.. autosummary::
   :toctree: api/

   GravityAnomalies
   GravityAnomalies.copy
   GravityAnomalies.from_csv
   GravityAnomalies.from_dataframe
   GravityAnomalies.from_excel
   GravityAnomalies.known_fields
   GravityAnomalies.required_fields
   GravityAnomalies.set_column
   GravityAnomalies.write_to_csv


GravityCorrectionParameters
===========================
.. currentmodule:: gsolve

.. autosummary::
   :toctree: api/

   GravityCorrectionParameters
   GravityCorrectionParameters.from_series
   GravityCorrectionParameters.copy
   GravityCorrectionParameters.to_dict
   GravityCorrectionParameters.to_excel
   GravityCorrectionParameters.to_series
   GravityCorrectionParameters.summary
   GravityCorrectionParameters.bouguer_correction_fields
   GravityCorrectionParameters.bouguer_correction_type
   GravityCorrectionParameters.default_values
   GravityCorrectionParameters.non_default_values


Functions
=========
.. currentmodule:: gsolve.reductions.anomalies

.. autosummary::
   :toctree: api/

   compute_complete_bouguer_anomaly
   compute_free_air_anomaly
   compute_simple_bouguer_anomaly

GravityCorrectionProvider
=========================
.. currentmodule:: gsolve.reductions.corrections

.. autosummary::
   :toctree: api/

   GravityCorrectionProvider
   GravityCorrectionProvider.compute
   GravityCorrectionProvider.available_corrections
   GravityCorrectionProvider.bouguer_corrections
   GravityCorrectionProvider.free_air_corrections

GravityCorrections
==================
.. currentmodule:: gsolve.reductions.corrections

.. autosummary::
   :toctree: api/

   GravityCorrections
   GravityCorrections.copy
   GravityCorrections.from_csv
   GravityCorrections.from_dataframe
   GravityCorrections.from_excel
   GravityCorrections.known_fields
   GravityCorrections.required_fields
   GravityCorrections.set_column
   GravityCorrections.write_to_csv


Functions
=========
.. currentmodule:: gsolve.reductions.corrections

.. autosummary::
   :toctree: api/

   atmospheric_correction
   bouguer_slab_correction
   bouguer_slab_curvature_corrected
   free_air_correction
   normal_gravity_at_ellipsoid
   normal_gravity_at_stn_elevation
   spherical_bouguer_cap_correction
