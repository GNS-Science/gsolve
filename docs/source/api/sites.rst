===================
GravitySites
===================
.. currentmodule:: gsolve

Object Creation
~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   GravitySites
   GravitySites.from_excel
   GravitySites.from_csv
   GravitySites.from_dataframe
   GravitySites.copy
   GravitySites.set_column


Information
~~~~~~~~~~~
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
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   GravitySites.to_excel
   GravitySites.write_to_csv

Internal Methods
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   GravitySites._check_bad_site_ids
   GravitySites._data_ok
   GravitySites._default_excel_sheet_name
   GravitySites._get_writable_df
   GravitySites._index_field
   GravitySites._known_fields

