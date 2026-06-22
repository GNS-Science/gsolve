==============
XArray Methods
==============

DEM Functions
=============
.. currentmodule:: gsolve.core.xr_methods

.. autosummary::
   :toctree: api/

   load_dem
   prepare_dem
   check_dem
   create_empty_dataarray

Xarray Accessor Class and Methods
=================================
.. currentmodule:: gsolve.core.xr_accessor

.. autosummary::
   :toctree: api/

   TCorrMethods
   TCorrMethods.xc
   TCorrMethods.yc
   TCorrMethods.xdim
   TCorrMethods.ydim
   TCorrMethods.dx
   TCorrMethods.dy
   TCorrMethods.bounds
   TCorrMethods.cell_edges
   TCorrMethods.coords_to_indices

   TCorrMethods.is_compatible
   TCorrMethods.clip_to_arr
   TCorrMethods.clip_to_points

   TCorrMethods.get_land_sea_mask
   TCorrMethods.generate_distance_mask
   TCorrMethods.apply_mask

   TCorrMethods.generate_bathymetry_density
   TCorrMethods.generate_topo_density
   TCorrMethods.get_bathymetry_elevation
   TCorrMethods.get_topography_elevation

