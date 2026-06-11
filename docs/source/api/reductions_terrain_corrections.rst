===========================
TerrainCorrectionParameters
===========================
.. currentmodule:: gsolve

.. autosummary::
   :toctree: api/

   TerrainCorrectionParameters
   TerrainCorrectionParameters.from_dataframe
   TerrainCorrectionParameters.from_series

   TerrainCorrectionParameters.dem_source
   TerrainCorrectionParameters.density_dataset_source
   TerrainCorrectionParameters.distance_mask_type
   TerrainCorrectionParameters.non_default_values
   TerrainCorrectionParameters.sea_level_elevation
   TerrainCorrectionParameters.terrain_density
   TerrainCorrectionParameters.water_density
   TerrainCorrectionParameters.default_values
   TerrainCorrectionParameters.compute_bathymetry
   TerrainCorrectionParameters.compute_topography

   TerrainCorrectionParameters.summary
   TerrainCorrectionParameters.copy
   TerrainCorrectionParameters.to_dict
   TerrainCorrectionParameters.to_excel
   TerrainCorrectionParameters.to_series

================
TerrainCorrector
================
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
.. currentmodule:: gsolve

.. autosummary::
   :toctree: api/

   TerrainCorrectionData
   TerrainCorrectionData.from_csv
   TerrainCorrectionData.from_dataframe
   TerrainCorrectionData.copy
   TerrainCorrectionData.create_empty
   TerrainCorrectionData.from_excel
   TerrainCorrectionData.get_corrections
   TerrainCorrectionData.known_fields
   TerrainCorrectionData.required_fields
   TerrainCorrectionData.set_column
   TerrainCorrectionData.set_corrections
   TerrainCorrectionData.to_csv
   TerrainCorrectionData.to_excel
   TerrainCorrectionData.write_to_csv
