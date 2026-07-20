Utility Functions and Classes
=============================

GSolveDataWarning
=================
.. currentmodule:: gsolve.core.utils

.. autosummary::
   :toctree: api/

   GSolveDataWarning
   GSolveDataWarning.count
   GSolveDataWarning.final_msg
   GSolveDataWarning.print_msgs

=========
Functions
=========
.. currentmodule:: gsolve.core.utils

Data Conversion
---------------
.. autosummary::
   :toctree: api/

   to_1d_ndarray
   to_1d_ndarray_or_float
   to_naive_utc_datetime
   normalize_field_names
   normalize_str
   check_duplicate_index

Excel Data Handling
---------------
.. autosummary::
   :toctree: api/

   columns_to_timestamp
   timestamp_to_columns
   merge_datetime_columns
   expand_datetime_column
   prepare_writable_df

Survey Loop Handling
--------------------
.. autosummary::
   :toctree: api/

   generate_loop_intervals
   generate_loop_names
   identify_loop_blocks
   loops_from_gaps

Other Utilities
-------------------
.. autosummary::
   :toctree: api/

   dms2rad
   round_coords
