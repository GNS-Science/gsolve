=============================
Utility Functions and Classes
=============================

Functions
=========
.. currentmodule:: gsolve.core.utils

Type Checking
-------------
.. autosummary::
   :toctree: api/

   is_filepath_like
   is_datetime_array
   is_points3d_like
   is_in_literal


Data Conversion
---------------
.. autosummary::
   :toctree: api/

   to_1d_ndarray
   to_1d_ndarray_or_float
   to_naive_utc_datetime
   to_points3D
   normalize_field_names
   normalize_str
   check_duplicate_index

Excel Data Handling
-------------------
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

Other Utility Functions
-----------------------
.. autosummary::
   :toctree: api/

   dms2rad
   round_coords

GSolveDataWarning
-----------------
.. currentmodule:: gsolve.core.utils

.. autosummary::
   :toctree: api/

   GSolveDataWarning
   GSolveDataWarning.count
   GSolveDataWarning.final_msg
   GSolveDataWarning.print_msgs