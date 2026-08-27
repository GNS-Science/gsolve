# GSolve - gravity processing software.
# Copyright (c) 2026 Earth Sciences New Zealand.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPLv3

# Copyright (c) 2025 Earth Sciences New Zealand.

from typing import Any, Literal, TypeAlias

import numpy as _np
import pandas as _pd

from gsolve.core._typing import GSolveSolverMethod, GSolveSolverReturn
from gsolve.gsolve_outputs import GSolveResults

__all__ = ["call_gsolve_lstsq", "GSolveSolverMethod"]

_GSOLVE_SOLVER_METHODS: dict[int, str] = {
    1: "Unconstrained least squares",
    2: "Partially constrained least squares",
    3: "Constrained least squares",
}


def call_gsolve_lstsq(
    obs: _pd.DataFrame,
    ref_sites: _pd.DataFrame,
    method: GSolveSolverMethod,
    percentile_clipping: float = 100,
    use_loops: bool = True,
) -> GSolveResults:
    """Calculate drift and adjust gravity observations.

    Parameters
    ----------
    obs : DataFrame
        The gravity observations to be corrected. The DataFrame must
        include columns labeled `['site_id', 'gravity', 'timedelta', 'loop']`.
        Other columns are ignored.
    ref_sites : DataFrame
        The reference sites that gravity will be 'tied' to after drift
    method : {1, 2, 3}
        The gsolve solution method to use. Available methods are:

            - ``1`` : Unconstrained least squares
            - ``2`` : Partially constrained least squares
            - ``3`` : Constrained least squares

    percentile_clipping: float, default=100.0
        Exclude observations with residuals outside percentile range from
        the final gsolve solution. Must be between a value between 0 and 100.
        The default is 100.0, which means no clipping. Clipping is symmetric,
        so if 99.0 is specified, the upper and lower 0.5% of residuals are excluded.
    use_loops : bool, default True
        Control how survey loops are treated in the solution.
        If ``True``, drift curves are fit to each loop.
        If ``False``, a single drift curve is fit to all observations.

    Returns
    -------
    GSolveResults
        An object containing the computed drift curves, observation and
        site residuals, and the model run parameters.
    """
    # index in obs where ties are located
    m_ties = ref_sites.index.intersection(obs["site_id"].to_list())
    if m_ties.empty:
        raise ValueError("no tie sites")
    else:
        ref_sites = ref_sites.loc[m_ties]

    # set up g_solver_lstsq input arguments
    kwargs: dict[str, Any] = {
        "obs_g": obs["gravity"].to_numpy(),
        "obs_site_id": obs["site_id"].to_numpy(),
        "obs_timedelta": obs["timedelta"].to_numpy(),
        "ties_site_id": ref_sites.index.to_numpy(),
        "ties_g": ref_sites.loc[:, "reference_gravity"].astype(float).to_numpy(),
        "use_loops": use_loops,
        "obs_loop": obs["loop"].to_numpy(),
        "method": method,
        "calculate_calibration_factor": False,
        "percentile_clipping": percentile_clipping,
        "obs_g_not_detided": None,
    }

    results = g_solver_lstsq(**kwargs)

    results_obj = GSolveResults(
        method=method,
        use_loops=use_loops,
        calculate_calibration_factor=False,
        percentile_clipping=percentile_clipping,
    )
    results_obj.set_inputs(obs, ref_sites)
    results_obj.set_solutions(results)
    return results_obj


def call_g_solver_calibration(
    obs: _pd.DataFrame,
    ref_sites: _pd.DataFrame,
    method: GSolveSolverMethod,
    percentile_clipping: float = 100,
    use_loops: bool = True,
) -> GSolveResults:
    """Calculate drift and adjust gravity observations.

    Parameters
    ----------
    obs : DataFrame
        The gravity observations to be corrected. The DataFrame must include
        columns labeled ``site_id``, ``gravity``, ``timedelta``, ``loop``, and
        ``gravity_not_detided``. Other columns are ignored.
    ref_sites : DataFrame
        The reference sites that gravity will be 'tied' to after drift
    method : {1, 2, 3}
        The gsolve solution method to use. Available methods are:

            - ``1`` : Unconstrained least squares
            - ``2`` : Partially constrained least squares
            - ``3`` : Constrained least squares

    percentile_clipping: float, default=100.0
        Exclude observations with residuals outside percentile range from
        the final gsolve solution. Must be between a value between 0 and 100.
        The default is 100.0, which means no clipping. Clipping is symmetric,
        so if 99.0 is specified, the upper and lower 0.5% of residuals are excluded.
    use_loops : bool, default True
        Control how survey loops are treated in the solution.
        If True, drift curves are fit to each loop.
        If False, a single drift curve is fit to all observations.

    Returns
    -------
    GSolveResults
        An object containing the computed drift curves, observation and
        site residuals, and the model run parameters.
    """
    # index in obs where ties are located
    m_ties = ref_sites.index.intersection(obs["site_id"].to_list())
    if m_ties.empty:
        raise ValueError("no tie sites")
    else:
        ref_sites = ref_sites.loc[m_ties]

    # set up g_solver_lstsq input arguments
    kwargs: dict[str, Any] = {
        "obs_g": obs["gravity"].to_numpy(),
        "obs_site_id": obs["site_id"].to_numpy(),
        "obs_timedelta": obs["timedelta"].to_numpy(),
        "ties_site_id": ref_sites.index.to_numpy(),
        "ties_g": ref_sites.loc[:, "reference_gravity"].astype(float).to_numpy(),
        "use_loops": use_loops,
        "obs_loop": obs["loop"].to_numpy(),
        "method": method,
        "calculate_calibration_factor": True,
        "percentile_clipping": percentile_clipping,
        "obs_g_not_detided": obs["meter_reading_mgal"].to_numpy(),
    }

    results = g_solver_lstsq(**kwargs)

    results_obj = GSolveResults(
        method=method,
        use_loops=use_loops,
        calculate_calibration_factor=True,
        percentile_clipping=percentile_clipping,
    )
    results_obj.set_inputs(obs, ref_sites)
    results_obj.set_solutions(results)
    return results_obj


def g_solver_lstsq(
    obs_g: _np.ndarray,
    obs_site_id: _np.ndarray,
    obs_timedelta: _np.ndarray,
    ties_site_id: _np.ndarray,
    ties_g: _np.ndarray,
    obs_loop: _np.ndarray,
    use_loops: bool,
    method: GSolveSolverMethod,
    calculate_calibration_factor: bool,
    obs_g_not_detided: _np.ndarray | None = None,
    percentile_clipping: float = 100.0,
) -> GSolveSolverReturn:
    """Least squares solution for gravity drift.

    Parameters
    ----------
    obs_g : ndarray (float)
        Gravity observations.
    obs_site_id : ndarray
        ``site_id`` of the gravity observations.
    obs_timedelta : ndarray (float)
        Time delta of the gravity observations.
    ties_site_id : ndarray
        ``site_id`` of the tie sites.
    ties_g : ndarray (float)
        Reference gravity at tie sites.
    obs_loop : ndarray (str)
        Loop id for each observation.
    use_loops : bool
        Control how survey loops are treated in the solution.
        If ``True``, drift curves are fit to each loop.
        If ``False``, a single drift curve is fit to all observations.
    method : {1, 2, 3}
        The gsolve solution method to use. Available methods are:

            - ``1`` : Unconstrained least squares
            - ``2`` : Partially constrained least squares
            - ``3`` : Constrained least squares
    calculate_calibration_factor : bool
        Calculate gravity meter calibration factor.
    obs_g_not_detided : ndarray
        The gravity observations specified in ``obs_g`` converted to milligals,
        but without any tidal corrections, calibration_factor or other corretions
        applied.
    percentile_clipping : float, default = 100.0
        Exclude observations with residuals outside percentile range from
        the gsolve solution. Must be between a value between 0 and 100.0 inclusive.
        If 100, no data is excluded. If 99.0, the upper and lower 0.5% of are excluded.

    Returns
    -------
    gravity : ndarray
        Corrected gravity value for each site.
    residuals : ndarray
        Residuals of each gravity observation.
    gravity_var : ndarray
        Variance of the estimated gravity values  for each site.
    drift : ndarray
        The drift gradient for each loop.
    baseline :
        The baseline shift for each loop.
    calibration_factor : float or None,
        The calulated ``calibration_factor``  if ``calculate_calibration_factor``
        is True, otherwise None.
    mask : ndarray
        Boolean array indicating whether an observation was included in the solution
        (True) or was excluded after ``percentile_clipping`` (False)

    """
    if method not in _GSOLVE_SOLVER_METHODS:
        valid_methods = tuple(_GSOLVE_SOLVER_METHODS.keys())
        raise ValueError(f"invalid method '{method}', must be one of {valid_methods}")

    percentile_clipping = float(percentile_clipping)
    if percentile_clipping < 0.0 or percentile_clipping > 100.0:
        raise ValueError(
            f"invalid percentile value {percentile_clipping}, "
            "must be between 0 and 100 inclusive"
        )

    n_obs = _np.size(obs_g)
    n_ties = _np.size(ties_site_id)
    site_ids = _np.unique(obs_site_id)
    n_sites = _np.size(site_ids)

    mask = _np.ones((n_obs,), dtype=bool)

    if use_loops:
        loop_ids = _np.unique(obs_loop)
    else:
        loop_ids = _np.asarray([1])
    n_loops = len(loop_ids)

    if calculate_calibration_factor:
        n_parameters = n_sites + (2 * n_loops) + 1
    else:
        n_parameters = n_sites + (2 * n_loops)

    if method == 1 or method == 2:
        n_constraints = n_sites
    else:  # method == 3
        n_constraints = n_ties

    # Predefine design and constraint matrices and right hand vectors
    A = _np.zeros((n_obs, n_parameters))
    b = _np.zeros((n_obs, 1))
    C = _np.zeros((n_constraints, n_parameters))
    d = _np.zeros((n_constraints, 1))

    # Gravity values
    for i in range(n_obs):
        for j in range(n_sites):
            if obs_site_id[i] == site_ids[j]:
                A[i, j] = 1
                b[i] = obs_g[i]
            if method == 2:
                # Loop to decouple Tie Stations from the rest of the equations
                for k in range(n_ties):
                    if obs_site_id[i] == ties_site_id[k]:
                        A[i, j] = 0
                        b[i] = obs_g[i] - ties_g[k]
                        break

        # Gravimeter zero-reading and drift
        for n in range(n_loops):
            if obs_loop[i] == loop_ids[n]:
                A[i, n_sites + (2 * n)] = -1
                A[i, n_sites + (2 * n) + 1] = -float(obs_timedelta[i])
                break

        # Calibration factor
        if calculate_calibration_factor:
            if obs_g_not_detided is None:
                raise ValueError(
                    "obs_g_not_detided must be provided when "
                    "calculate_calibration_factor is True"
                )
            A[i, n_sites + (2 * n_loops)] = float(obs_g_not_detided[i])

    # Ties to absolute sites
    for k in range(n_sites):
        for j in range(n_ties):
            if site_ids[k] == ties_site_id[j]:
                if method == 1 or method == 2:
                    C[k, k] = 1
                    d[k] = ties_g[j]
                    break
                elif method == 3:
                    C[j, k] = 1
                    d[j] = ties_g[j]

    ###############################################################
    # Combine design and constraint matrices and right hand vectors
    if method == 1 or method == 2:
        E = _np.vstack((A, C))
        f = _np.vstack((b, d))
    else:  # method 3
        E = _np.vstack(
            (
                _np.hstack((_np.dot(A.T, A), C.T)),
                _np.hstack((C, _np.zeros(shape=(n_constraints, n_constraints)))),
            )
        )
        f = _np.vstack((_np.dot(A.T, b), d))

    #####################################
    # Least-square solution of the system

    # Filter observations
    if percentile_clipping != 100.0:
        # Preliminary adjustment
        solution, _, _, _ = _np.linalg.lstsq(E, f, rcond=None)
        # Preliminary residuals
        residuals = b - _np.dot(A, solution[:n_parameters])

        # Define percentile clipping interval
        perc = (100.0 - percentile_clipping) / 2
        ci_l = _np.percentile(residuals[:n_obs], perc)
        ci_h = _np.percentile(residuals[:n_obs], 100.0 - perc)

        # Build mask of outliers
        mask = ((residuals[:n_obs] > ci_l) & (residuals[:n_obs] < ci_h)).flatten()

        # Mask outliers
        A[:n_obs, :][~mask] = 0
        b[:n_obs, :][~mask] = 0
        # Redefine number of observations
        n_obs = _np.sum(mask)

        ###############################################################
        # Re-combine design and constraint matrices and right hand vectors
        if method == 1 or method == 2:
            E = _np.vstack((A, C))
            f = _np.vstack((b, d))
        elif method == 3:
            E = _np.vstack(
                (
                    _np.hstack((_np.dot(A.T, A), C.T)),
                    _np.hstack((C, _np.zeros(shape=(n_constraints, n_constraints)))),
                )
            )
            f = _np.vstack((_np.dot(A.T, b), d))

    # Main adjustment
    solution, _, _, _ = _np.linalg.lstsq(E, f, rcond=None)
    # Compute residuals
    residuals = b - _np.dot(A, solution[:n_parameters])
    # Compute the residual sum of squares
    rss = _np.dot(residuals.T, residuals).squeeze()

    # Estimates standard errors of the parameters
    # Extracts diagonal elements of the inverted normal matrix
    Pv = _np.linalg.pinv(_np.dot(E.T, E)).diagonal()
    # Computes squared unit weight
    sigma_0_squared = rss / (n_obs + n_ties - n_parameters)
    # Variance of the estimated parameters
    var = sigma_0_squared * Pv
    # Round to zero tiny negative variance
    tol = 10**-16
    var[_np.abs(var) < tol] = 0

    # Extract values from the solution vector
    gravity = solution[:n_sites]
    gravity_var = var[:n_sites]

    indices = _np.arange(n_sites, n_sites + 2 * n_loops, 2)
    baseline = _np.atleast_1d(solution[indices].squeeze())
    drift = _np.atleast_1d(solution[indices + 1].squeeze())

    if calculate_calibration_factor:
        calibration_factor = float((1 - solution[n_sites + 2 * n_loops]).item())
    else:
        calibration_factor = None

    return gravity, residuals, gravity_var, drift, baseline, calibration_factor, mask
