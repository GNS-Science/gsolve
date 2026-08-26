==========================================
Gravity Site and Reference Gravity Classes
==========================================

GravitySites
============
.. currentmodule:: gsolve

Object Creation
---------------
.. autosummary::
   :toctree: api/

   GravitySites
   GravitySites.from_excel
   GravitySites.from_csv
   GravitySites.from_dataframe
   GravitySites.copy
   GravitySites.set_column


Information
-----------
.. autosummary::
   :toctree: api/

   GravitySites.known_fields
   GravitySites.required_fields
   GravitySites.check_data
   GravitySites.set_reference_gravity
   GravitySites.activate_ties
   GravitySites.deactivate_ties
   GravitySites.get_points
   GravitySites.get_ties
   GravitySites.sample_elevation

Data Export
-----------
.. autosummary::
   :toctree: api/

   GravitySites.to_excel
   GravitySites.write_to_csv

ReferenceGravity
================
.. currentmodule:: gsolve

.. autosummary::
   :toctree: api/

   ReferenceGravity
   ReferenceGravity.copy
   ReferenceGravity.from_csv
   ReferenceGravity.from_dataframe
   ReferenceGravity.from_dict
   ReferenceGravity.from_excel
   ReferenceGravity.known_fields
   ReferenceGravity.required_fields
   ReferenceGravity.set_column
   ReferenceGravity.to_excel
   ReferenceGravity.write_to_csv


Functions
=========
.. currentmodule:: gsolve.sites

.. autosummary::
   :toctree: api/

   combine_gravity_sites
   combine_reference_gravity

