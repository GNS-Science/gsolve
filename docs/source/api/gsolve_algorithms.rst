=============================
gSolve Algorithms and Outputs
=============================

gSolve Algorithm
================

.. currentmodule:: gsolve.gsolve_algorithms

.. autosummary::
   :toctree: api/

   call_gsolve_lstsq
   g_solver_lstsq

GSolveResults
=============

Results of a gSolve network adjustment.

.. currentmodule:: gsolve.gsolve_outputs

.. autosummary::
   :toctree: api/

   GSolveResults
   GSolveResults.calibration_factor
   GSolveResults.plot_residual_cdf
   GSolveResults.plot_residual_drift
   GSolveResults.set_inputs
   GSolveResults.set_solutions


GSolveSolutionParameters
========================

Class for storing network adjustment parameters and solution metadata.

.. currentmodule:: gsolve.gsolve_outputs

.. autosummary::
   :toctree: api/

   GSolveSolutionParameters
   GSolveSolutionParameters.calculated_calibration_factor
   GSolveSolutionParameters.copy
   GSolveSolutionParameters.default_values
   GSolveSolutionParameters.from_series
   GSolveSolutionParameters.gsolve_run_datetime
   GSolveSolutionParameters.gsolve_version
   GSolveSolutionParameters.non_default_values
   GSolveSolutionParameters.summary
   GSolveSolutionParameters.to_dict
   GSolveSolutionParameters.to_excel
   GSolveSolutionParameters.to_series
