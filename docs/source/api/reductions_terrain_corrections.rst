===========================
TerrainCorrectionParameters
===========================
Set parameters for a terrain correction computation.

.. currentmodule:: gsolve

.. autosummary::
   :toctree: api/

   TerrainCorrectionParameters
   TerrainCorrectionParameters.from_dataframe
   TerrainCorrectionParameters.from_series

   TerrainCorrectionParameters.summary
   TerrainCorrectionParameters.default_values
   TerrainCorrectionParameters.non_default_values

   TerrainCorrectionParameters.copy
   TerrainCorrectionParameters.to_dict
   TerrainCorrectionParameters.to_excel
   TerrainCorrectionParameters.to_series

================
TerrainCorrector
================
Compute terrain coorections for a set of stations.

.. currentmodule:: gsolve

.. autosummary::
   :toctree: api/

   TerrainCorrector
   TerrainCorrector.zones
   TerrainCorrector.add_zone
   TerrainCorrector.compute

=====================
TerrainCorrectionData
=====================
Class to store and manage terrain corrections produced by a TerrainCorrector.

.. currentmodule:: gsolve

.. autosummary::
   :toctree: api/

   TerrainCorrectionData
   TerrainCorrectionData.create_empty
   TerrainCorrectionData.from_excel
   TerrainCorrectionData.from_csv
   TerrainCorrectionData.from_dataframe
   TerrainCorrectionData.copy
   TerrainCorrectionData.set_corrections
   TerrainCorrectionData.get_corrections
   TerrainCorrectionData.known_fields
   TerrainCorrectionData.required_fields
   TerrainCorrectionData.to_csv
   TerrainCorrectionData.to_excel
   TerrainCorrectionData.write_to_csv

=========
Functions
=========
.. currentmodule:: gsolve.reductions.terrain_corrections

.. autosummary::
   :toctree: api/

   calculate_terrain_correction
   tcorr_harmonica_bathymetry
   tcorr_harmonica_topography