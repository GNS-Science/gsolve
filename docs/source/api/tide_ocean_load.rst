=========================
Ocean Loading Corrections
=========================

.. currentmodule:: gsolve.tide.ocean_load

HardispOceanLoadCorrector
=========================

.. autosummary::
   :toctree: api/

   HardispOceanLoadCorrector
   HardispOceanLoadCorrector.identifier
   HardispOceanLoadCorrector.ocean_load_correction
   HardispOceanLoadCorrector.stations

QuickTidePro Functions
======================

.. currentmodule:: gsolve.tide.ocean_load

.. autosummary::
   :toctree: api/

   generate_qtp_input
   qtp_to_corrector
   read_qtp_multistation
   read_qtp_timeseries


OceanLoadTimeSeries
===================

.. currentmodule:: gsolve.tide.ocean_load

.. autosummary::
   :toctree: api/

   OceanLoadTimeSeries
   OceanLoadTimeSeries.starttime
   OceanLoadTimeSeries.sample_rate
   OceanLoadTimeSeries.endtime
   OceanLoadTimeSeries.identifier
   OceanLoadTimeSeries.ocean_load_correction


OceanLoadAtSiteTime
===================
.. currentmodule:: gsolve.tide.ocean_load

.. autosummary::
   :toctree: api/

   OceanLoadAtSiteTime
   OceanLoadAtSiteTime.identifier
   OceanLoadAtSiteTime.ocean_load_correction

OceanLoadCorrectionProvider
===========================
.. currentmodule:: gsolve.tide.ocean_load

.. autosummary::
   :toctree: api/

   OceanLoadCorrectionProvider
   OceanLoadCorrectionProvider.identifier
   OceanLoadCorrectionProvider.ocean_load_correction
