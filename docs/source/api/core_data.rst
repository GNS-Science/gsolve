==========================
Base Classes and Protocols
==========================

.. currentmodule:: gsolve.core.data

DataFieldSpecification
======================
.. autosummary::
   :toctree: api/

   DataFieldSpecification


GSolveParameters
================
.. currentmodule:: gsolve.core.data

.. autosummary::
   :toctree: api/

   GSolveParameters
   GSolveParameters.copy
   GSolveParameters.from_series
   GSolveParameters.default_values
   GSolveParameters.non_default_values
   GSolveParameters.summary
   GSolveParameters.to_dict
   GSolveParameters.to_excel
   GSolveParameters.to_series

GSolveTable
===========
.. currentmodule:: gsolve.core.data

.. autosummary::
   :toctree: api/

   GSolveTable
   GSolveTable.copy
   GSolveTable.from_csv
   GSolveTable.from_dataframe
   GSolveTable.from_excel
   GSolveTable.known_fields
   GSolveTable.required_fields
   GSolveTable.set_column
   GSolveTable.write_to_csv