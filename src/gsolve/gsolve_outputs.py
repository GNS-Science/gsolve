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

import dataclasses
import importlib.metadata
import pathlib
import warnings
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd

from gsolve.core._typing import FilePath, GSolveSolverMethod, GSolveSolverReturn
from gsolve.core.data import GSolveParameters
from gsolve.core.utils import is_list_like, to_naive_utc_datetime

__all__ = ["GSolveSolutionParameters", "GSolveResults"]

_PlotGravityUnit: TypeAlias = Literal["uGal", "mGal", "mgal", "ugal"]


@dataclasses.dataclass
class GSolveSolutionParameters(GSolveParameters):
    """Class to store input parameters and solution metadata for a Gsolve solution.

    This class is generated internally and returned as ``GsolveResults.params``
    attribute.  You probably do not need to create this class directly.

    Attributes
    ----------
    method : {1, 2, 3}
        The gsolve algorithm method used.
    use_loops: bool
        If loops were used in the solution.
    percentile_clipping: float
        The percentile clip applied.
    solve_for_calibration_factor : bool, default False
        If soluton solved for gravity meter calibration factor.
    calculated_calibration_factor : float, default NaN
        The output calibration factor if ``calculate_calibration_factor`` is True.
    gsolve_run_datetime : pandas.Timestamp, optional
        The solution runtime.  If not defined, will be set automatically at
        object initialisation.
    gsolve_version : str, optional
        If unset, attempt to set by calling ``importlib.metadata.version()``.

    """

    method: GSolveSolverMethod
    use_loops: bool
    percentile_clipping: float
    calculate_calibration_factor: bool
    calculated_calibration_factor: float = _np.nan
    gsolve_run_datetime: _pd.Timestamp | None = None
    gsolve_version: str | None = None

    def __post_init__(self) -> None:
        if self.gsolve_run_datetime is None:
            self.gsolve_run_datetime = to_naive_utc_datetime(
                _pd.Timestamp.now("UTC"), allow_nat=False
            )
        else:
            self.gsolve_run_datetime = to_naive_utc_datetime(
                self.gsolve_run_datetime, allow_nat=False
            )

        if self.gsolve_version is None:
            try:
                self.gsolve_version = importlib.metadata.version("gsolve")
            except importlib.metadata.PackageNotFoundError:
                self.gsolve_version = "unknown"


class GSolveResults:
    """Class to store the results from a gsolve network adjustment.

    This class is returned by ``gsolve_algorithms.gsolve_lstsq()`` and is not intended to be
    created directly by the user.

    Attributes
    ----------
    params : GSolveSolutionParameters
        An object containing the gsolve solution input parameters, runtime metadata, and
        calculated instrument calibration factor (if applicable).
    obs_solution : DataFrame
        The observation residuals following network adjustment. Observations where the
        'active' field is False were excluded from the solution after 'percentile_clipping'.
    loop_solution : DataFrame
        The drift rate and baseline solution for each loop. Columns 'drift' and 'baseline'
        are the coefficients of a linear equation to compute gravity adjustments
        as a function of time a given loop: . The adjustment is given by:

        .. math:: g_{adj}(t) = g_{drift} t + g_{baseline}

        where :math:`t` is time since the start of the  (``timedelta`` column in
        obj.obs_solution arribute), :math:`g_{drift}` is the drift rate and :math:`g_{baseline}`
        is a constant taken from the ``drift`` and ``baseline`` columns of loop_solutiom.
        Finally, the absolute gravity for some observed gravity :math:`g_{obs}` at
        time :math:`t` is given by:

        .. math:: g_{abs} = g_{obs} + g_{adj}

    site_solution : DataFrame
        The final 'absolute_gravity' solution for each site after adjustment, with
        solution statistics.

    Parameters
    ----------
    method : {1, 2, 3}
        The gsolve algorithm used.
    use_loops: bool
        If loops were used in the solution.
    calculate_calibration_factor : bool
        If soluton solved for gravity meter calibration factor.
    percentile_clipping: float
        The percentile clip applied.

    """  # noqa: D420

    def __init__(
        self,
        method: GSolveSolverMethod,
        use_loops: bool,
        calculate_calibration_factor: bool,
        percentile_clipping: float,
    ) -> None:
        self.params = GSolveSolutionParameters(
            method=method,
            use_loops=use_loops,
            percentile_clipping=percentile_clipping,
            calculate_calibration_factor=calculate_calibration_factor,
        )

        self.obs_solution: _pd.DataFrame
        self.site_solution: _pd.DataFrame
        self.loop_solution: _pd.DataFrame
        self.observations_input: _pd.DataFrame
        self.reference_sites_input: _pd.DataFrame

    def set_inputs(self, obs: _pd.DataFrame, ref_sites: _pd.DataFrame) -> None:
        """Add input data used in the gsolve run."""
        self.observations_input = obs.copy()
        self.reference_sites_input = ref_sites.copy()

    def set_solutions(self, results: GSolveSolverReturn) -> None:
        """Set the results from a gsolve run."""
        (
            site_grav,
            obs_residuals,
            site_var,
            drift,
            baseline,
            calibration_factor,
            mask,
        ) = results

        if self.params.calculate_calibration_factor:
            if calibration_factor is None:
                raise ValueError(
                    "calibration factor was not calculated but "
                    "calculate_calibration_factor is True."
                )
            # store the calculated calibration factor in the params object
            self.params.calculated_calibration_factor = calibration_factor

        # set up obs_solution dataframe
        obs = self.observations_input
        n_obs = obs.shape[0]
        self.obs_solution = _pd.DataFrame(
            index=_pd.Index(obs.index, name="obs_id"),
            data={
                "site_id": obs["site_id"].astype(str).to_numpy(),
                "loop": obs["loop"].astype(str).to_numpy(),
                "residual": _np.atleast_1d(obs_residuals[:n_obs].squeeze()),
                "timedelta": obs["timedelta"].to_numpy(),
                "active": _np.atleast_1d(mask),
            },
        )

        site_ids = obs["site_id"][mask].value_counts().sort_index()
        mask_gravity = ~_np.isin(
            _np.unique(obs["site_id"]),
            _np.setdiff1d(_np.unique(obs["site_id"]), site_ids.index),
        )

        self.site_solution = _pd.DataFrame(
            index=site_ids.index,
            data={
                "n_obs": site_ids.to_numpy(),
                "absolute_gravity": _np.atleast_1d(site_grav[mask_gravity].squeeze()),
                "variance": _np.atleast_1d(site_var[mask_gravity].squeeze()),
                "stdev": _np.sqrt(_np.atleast_1d(site_var[mask_gravity].squeeze())),
                "stderr": _np.divide(
                    _np.sqrt(_np.atleast_1d(site_var[mask_gravity])),
                    _np.sqrt(site_ids.to_numpy()),
                ).squeeze(),
            },
        ).sort_index()

        # TODO: R-squared value column
        loops = obs["loop"].value_counts().sort_index()
        self.loop_solution = _pd.DataFrame(
            index=loops.index,
            data={
                "n_obs": loops.to_numpy(),
                "drift": _np.atleast_1d(drift.squeeze()),
                "baseline": _np.atleast_1d(baseline.squeeze()),
            },
        ).sort_index()

    def plot_residual_cdf(
        self,
        loop: str | int | float | Sequence | None = None,
        unit: _PlotGravityUnit = "mGal",
        filename: FilePath | None = None,
        show: bool = True,
    ) -> _plt.Axes:
        """
        Plot the empirical cumulative density function of residuals.

        Parameters
        ----------
        loop: str, Sequence[str], or None, default None
            The loop(s) to plot. If ``loop`` is None, plot cdf of each loop.
            If ``loop`` = "all", then ignore loops and plot cdf of combined data.
        unit: {'uGal', 'mGal'}, default 'mGal'
            If 'uGal', plot residuals in microGal's. If 'mGal', plot residuals
            in milliGal's.
        filename: str, default None
            If not None, save plot to ``filename``. The specified loop id is appended
            to the end of the filename (before suffix).
        show: bool, default True
            Show the plot in a new window.

        Returns
        -------
        matoplotlibs.axes.Axes
            The plot axes instance.

        """
        m = self.obs_solution.active.eq(True) & self.obs_solution.residual.notna()
        df = self.obs_solution.loc[m].copy()

        unit_label: str = ""
        if isinstance(unit, str) and unit.lower() in ["ugal", "mgal"]:
            if unit.lower() == "ugal":
                df["residual"] = df["residual"] * 1000.0
                # drift *= 1000.0
                unit_label = "μGal"
                precision = ".01f"
            elif unit.lower() == "mgal":
                unit_label = "mGal"
                precision = ".04f"
        else:
            raise ValueError(f"unrecgnised unit '{unit}'. Must be 'mGal' or 'uGal'")

        df_loops: list[str] = [str(l) for l in df["loop"].unique()]
        # if len(df_loops) == 1:
        #     loop = df_loops[0]

        ax_title = "Distribution of residuals for"
        if loop is None:
            loops = df_loops
            ax_title = f"{ax_title} each loop"
        elif is_list_like(loop):
            loops: list[str] = [str(l) for l in loop]  # type: ignore[bad-assignment-type]
            ax_title = f"{ax_title} loops {', '.join(loops)}"
        else:
            if loop == "all":
                loops = ["all"]
                df["loop"] = "all"
                ax_title = f"{ax_title} all loops combined"
            else:
                loops = [str(loop)]
                ax_title = f"{ax_title} loop {loop}"

        for l in loops:
            if l not in df_loops:
                raise ValueError(f"loop '{loop}' not found in observation residuals.")

        df = df.loc[df["loop"].isin(loops)]
        fig = _plt.figure()
        ax = fig.add_subplot(111)

        for l, loop_df in df.groupby("loop"):
            residuals = loop_df["residual"].to_numpy()
            label = (
                f"{l}: n={len(residuals)}, x̄={residuals.mean():<{precision}}, "
                f"σ={residuals.std():{precision}}"
            )
            if hasattr(ax, "ecdf"):
                ax.ecdf(residuals, label=label)
            else:
                x = _np.sort(residuals)
                y = _np.array(_np.arange(len(x))) / len(x)
                ax.step(x, y, label=label)

        ax.set_title(ax_title)
        ax.set_xlabel(f"residual ({unit_label})")
        ax.set_ylabel("cumulative probability")
        ax.text(
            x=0.95,
            y=0.05,
            s=(
                f"method = {self.params.method}, "
                f"Percentile clipping = {self.params.percentile_clipping}"
            ),
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize="small",
        )
        ax.legend(loc="best", fontsize="small", title_fontsize="small")

        if show:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.show()

        if filename is not None:
            fpath = pathlib.Path(filename)
            if len(loops) == 1:
                fpath = fpath.parent / f"{fpath.stem}_loop_{loops[0]}{fpath.suffix}"
            else:
                fpath = (
                    fpath.parent / f"{fpath.stem}_loops_{'_'.join(loops)}{fpath.suffix}"
                )

            fig.savefig(fpath, dpi=300)

        return ax

    @property
    def calibration_factor(self) -> float:
        """Convenience property to access the calculated calibration factor."""
        return float(_np.asarray(self.params.calculated_calibration_factor).item())

    def plot_residual_drift(
        self,
        loop: str | int | float,
        plot_drift: bool = True,
        unit: _PlotGravityUnit = "mGal",
        filename: FilePath | None = None,
        show: bool = True,
    ) -> _plt.Axes:
        """
        Plot the residuals and drift curve.

        Parameters
        ----------
        loop : str | int | float
            The loop id to plot.
        plot_drift : bool, default True
            If True, plot the drift curve along with the residuals + drift.
            If False, plot only the residuals as stem plot.
        unit: {'uGal', 'mGal'}, default 'mGal'
            If 'uGal', plot residuals in microGal's. If 'mGal', plot residuals
            in milliGal's.
        filename: str, default None
            If not None, save plot to ``filename``. The specified loop id is appended
            to the end of the filename (before suffix).
        show: bool, default True
            Show the plot in a new window.

        Returns
        -------
        matoplotlibs.axes.Axes
            The plot axes instance.
        """
        loop = str(loop)
        x_col: str = "timedelta"
        y_col: str = "residual"

        drift = float(self.loop_solution.at[loop, "drift"])  # type: ignore[bad-argument-type]  # noqa: PD008
        m_loop = self.obs_solution["loop"].eq(loop)
        m_active = self.obs_solution["active"].eq(True)

        df = self.obs_solution.loc[m_loop & m_active].copy()

        unit_label = None
        if isinstance(unit, str):
            if unit.lower() == "ugal":
                df["residual"] = df["residual"] * 1000.0
                drift *= 1000.0
                unit_label = "μGal"
                precision = ".01f"
            elif unit.lower() == "mgal":
                unit_label = "mGal"
                precision = ".04f"
        if unit_label is None:
            raise ValueError(f"unrecgnised unit '{unit}'. Must be 'mGal' or 'uGal'")

        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        fig = _plt.figure()
        ax = fig.add_subplot(111)

        if plot_drift:
            drift_y = drift * x
            y = y + drift_y
            ax.scatter(x, y, marker=".", label="residuals")
            ax.plot(
                x,
                drift_y,
                c="orange",
                label=f"drift curve ({drift:{precision}} {unit_label}/hr)",
            )
            ax.set_ylabel(f"{y_col} + drift ({unit_label})")
            ax.set_title(
                f"Plot of the drift function and residuals for loop {loop}.\n"
                f"Drift = {drift:{precision}} {unit_label}/hour."
                f"Percentile clipping = {self.params.percentile_clipping:.1f}"
            )
        else:
            ax.stem(x, y, label="residuals", markerfmt=".")
            ax.set_ylabel(f"{y_col} ({unit_label})")

        ax.legend(loc="best")

        ax.set_title(f"Plot of the drift function and residuals for loop {loop}.")
        ax.set_xlabel("Time since start of loop (hours)")

        if show:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.show()

        if filename is not None:
            fpath = pathlib.Path(filename)
            fpath = fpath.parent / f"{fpath.stem}_loop_{loop}{fpath.suffix}"
            _plt.savefig(fpath, dpi=300)

        return ax
