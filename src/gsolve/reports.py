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

from copy import deepcopy

from pathlib import Path
from typing import Any, Self

import pandas as _pd

from gsolve.core._typing import FilePath, IfSheetExists, IfWorkbookExists
from gsolve.core.data import GSolveParameters
from gsolve.core.excel_io import get_excel_worksheets, write_excel_worksheet
from gsolve.core.utils import prepare_writable_df
from gsolve.gsolve_outputs import GSolveResults
from gsolve.observations import GravityObservations, GravitySurvey
from gsolve.reductions.anomalies import GravityAnomalies
from gsolve.reductions.terrain_corrections import TerrainCorrectionData
from gsolve.sites import GravitySites

__all__ = ["GSolveReport"]


class GSolveReport:
    """Class for summarising and reporting results of a gSolve network adjustment.

    This class provides a simple interface for collating the various inputs
    and outputs from a gSolve run. The report can be written to an Excel workbook.

    Parameters
    ----------
    observations : GravityObservations | GravitySurvey
        The observations used in the gSolve network adjustment.
    sites: GravitySites | GravitySurvey
        The sites associated with the observations.
    results : GSolveResults
        The results of a gSolve network adjustment.
    anomalies : GravityAnomalies, optional
        The gravity standard corrections for each site and gravity anomalies
        calculated from adjusted gravity results.
    terrain_corrections: TerrainCorrectionData, optional
        Terrain corrections for sites. Not required if terrain correction data
        are already included in the ``anomalies`` object.

    Attributes
    ----------
    obs_data : Dataframe
        Gravity observation input data and network adjusted outputs.
    site_data : Dataframe
        All site related data, including input data, site solutions, normal gravity
        corrections and anomalies.
    loop_data : Dataframe
        gSolve network adjustment solutions for each loop.
    terrain_correction_data : Dataframe
        Terrain corrections for each site, if provided.
    params : dict
        A dictionary of objects storing the various parameters used in
        reducing input data, performing network adjustment and calculating anomalies.
        The dictionary provides the following keys:

                        - 'observations' : GravityObservationsParameters.
                        - 'solution' : GSolveSolutionParameters.
                        - 'anomalies' : GravityCorrectionParameters, if ``anomalies`` argument provided.
            - 'terrain_corrections': dict of terrain correction parameters of the form
                            {"zone_id": TerrainCorrectionParameters}.

    """

    def __init__(
        self,
        observations: GravityObservations | GravitySurvey,
        sites: GravitySites | GravitySurvey,
        results: GSolveResults,
        anomalies: GravityAnomalies | None = None,
        terrain_corrections: TerrainCorrectionData | None = None,
    ) -> None:
        self.site_data: _pd.DataFrame
        self.obs_data: _pd.DataFrame
        self.loop_data: _pd.DataFrame
        self.terrain_correction_data: _pd.DataFrame | None = None
        self.params: dict[str, GSolveParameters] = {}
        self._tcorr_added_from_anomalies: bool = False

        if sites is None or results is None or observations is None:
            raise ValueError(
                "'observations', 'sites' and 'results' arguments must be provided."
            )

        if isinstance(observations, GravitySurvey):
            observations = observations.observations
        if isinstance(sites, GravitySurvey):
            sites = sites.sites

        self._set_params(
            observations=observations,
            sites=sites,
            results=results,
            anomalies=anomalies,
        )
        self._set_site_data(
            site_input=sites,
            obs_input=observations,
            results=results,
            anomalies=anomalies,
        )
        self._set_obs_data(obs_input=observations, results=results)
        self._set_loop_data(results=results, obs_input=observations)
        self._set_terrain_correction_data(terrain_corrections=terrain_corrections)

    def copy(self) -> Self:
        """Return a deep copy."""  # noqa: DOC201
        return self.__copy__()

    def __copy__(self) -> Self:
        """Return a deep copy."""  # noqa: DOC201
        return deepcopy(self)

    def _set_params(
        self,
        observations: GravityObservations,
        sites: GravitySites,
        results: GSolveResults,
        anomalies: GravityAnomalies | None = None,
    ) -> None:

        self.params["observations"] = observations.params()
        if hasattr(sites, "params"):
            self.params["sites"] = sites.params.copy()  # ty:ignore[unresolved-attribute]
        self.params["solution"] = results.params.copy()
        if anomalies is not None:
            self.params["anomalies"] = anomalies.params.copy()
            if anomalies.tcorr_params is not None:
                for zone_id, p in anomalies.tcorr_params.items():
                    self.params[zone_id] = p.copy()
                self._tcorr_added_from_anomalies = True

    def _set_site_data(
        self,
        site_input: GravitySites,
        obs_input: GravityObservations,
        results: GSolveResults,
        anomalies: GravityAnomalies | None,
    ) -> None:
        """
        Set site_data attribute from site, observation, network adjustment data.

        Parameters
        ----------
        site_input : GravitySites
            Gravity site data.
        obs_input : GravityObservations
            Gravity observation data.
        results : GSolveResults
            Results of gSolve network adjustment.
        anomalies : GravityAnomalies, optional
            Standard gravity anomalies and corrections for each site.
        """
        merge_kwargs = dict(
            left_index=True,
            right_index=True,
            how="left",
            copy=True,
            suffixes=[None, "_duplicate"],
        )

        # start with site input data, will be left side of table
        df = site_input.data.copy()

        # merge in site solution
        if isinstance(results.site_solution, _pd.DataFrame):
            df = df.merge(results.site_solution, **merge_kwargs)  # type: ignore[invalid-argument-type]

        # merge anomalies if present
        if anomalies is not None:
            df = df.merge(anomalies.data, **merge_kwargs)  # type: ignore[invalid-argument-type]

        # clean up duplicate columns
        df = df.drop(
            columns=[c for c in df.columns if c.endswith("_duplicate")]
        ).rename(columns={"n_obs": "n_obs_used"})

        # add count of obs in original data for each site
        n_obs_input = obs_input.data["site_id"].value_counts().loc[df.index]

        i_n_obs_used = df.columns.get_loc("n_obs_used")
        if not isinstance(i_n_obs_used, int):
            raise ValueError("unexpected non-integer column index")
        df.insert(i_n_obs_used, "n_obs_input", n_obs_input)

        # add a csv string of loops each site is included in
        loop_df = (
            obs_input.data.loc[:, ["site_id", "loop"]]
            .drop_duplicates(ignore_index=True)
            .sort_values(by=["site_id", "loop"])
            .groupby("site_id")
            .agg(lambda x: ",".join(x.astype(str).to_list()))
        ).iloc[:, 0]
        df.insert(i_n_obs_used + 1, "in_loops", loop_df)

        self.site_data = df

    def _set_obs_data(
        self, obs_input: GravityObservations, results: GSolveResults
    ) -> None:
        """Set self.obs_data by merging GravityObservations and GSolveResults objects."""
        self.params["observations"] = obs_input.params()

        # start with observation input data, will be left side of table
        df = obs_input.data.copy()

        # merge in observation solution
        df = df.merge(
            results.obs_solution,
            left_index=True,
            right_index=True,
            how="left",
            copy=True,
            suffixes=[None, "_duplicate"],
        )

        # after gsolve run, 'active' indicates if obs was used in solution
        # rename to 'included_in_solution' for clarity
        df = df.rename(columns={"active_duplicate": "included_in_solution"})
        # add flag for if site has a gsolve solution
        i_included_in_solution = df.columns.get_loc("included_in_solution")
        if not isinstance(i_included_in_solution, int):
            raise ValueError("unexpected non-integer column index")
        df.insert(
            loc=i_included_in_solution + 1,
            column="has_site_solution",
            value=df["site_id"].isin(results.site_solution.index),
        )

        df = df.drop(columns=[c for c in df.columns if c.endswith("_duplicate")])
        self.obs_data = df

    def _set_loop_data(
        self, results: GSolveResults, obs_input: GravityObservations | GravitySurvey
    ) -> None:
        """Set self.loop_data from GSolveResults and GravityObservations objects."""
        if isinstance(obs_input, GravitySurvey):
            obs_input = obs_input.observations
        # start with loop solution data
        df = results.loop_solution.copy().rename(columns={"n_obs": "n_obs_input"})

        # add count of obs used in solution for each loop
        # - from obs_solution 'active' flag
        n_obs_used = results.obs_solution.loc[
            results.obs_solution["active"], "loop"
        ].value_counts()
        i_n_obs_input = df.columns.get_loc("n_obs_input")
        if not isinstance(i_n_obs_input, int):
            raise ValueError("unexpected non-integer column index")
        df.insert(i_n_obs_input + 1, "n_obs_used", 0)
        df.loc[n_obs_used.index, "n_obs_used"] = n_obs_used

        self.loop_data = df

    def _set_terrain_correction_data(
        self, terrain_corrections: TerrainCorrectionData | None
    ) -> None:
        """Set self.terrain_correction_data from TerrainCorrectionData object."""
        if terrain_corrections is None:
            self.terrain_correction_data = None
        else:
            if self._tcorr_added_from_anomalies:
                # terrain corrections were already added from anomalies object, so
                # check for consistency in both parameters and data
                tcorr_errmsg = (
                    "terrain_corrections are inconsistent with "
                    " terrain corrections data added from anomalies"
                    "object"
                )
                # check all zones were added
                for zone_id, p in terrain_corrections.params.items():
                    if zone_id not in self.params or p != self.params[zone_id]:
                        raise ValueError(tcorr_errmsg)
                # check no zones were added that are not in terrain corrections``
                for k in self.params.keys():
                    if k.startswith("tcorr_") and k not in terrain_corrections.params:
                        raise ValueError(tcorr_errmsg)

            else:
                # anomalies object did not provide terrain corrections, so add them here
                self.terrain_correction_data = terrain_corrections.data.copy()
                for zone_id, p in terrain_corrections.params.items():
                    self.params[zone_id] = p.copy()

            self.terrain_correction_data = terrain_corrections.data.copy()

    def to_excel(
        self,
        filename: FilePath,
        if_workbook_exists: IfWorkbookExists = "error",
        if_sheet_exists: IfSheetExists = "error",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """
        Save the report data to an Excel file.

        Parameters
        ----------
        filename : str or PathLike
            Write the report to ``filename``.
        if_workbook_exists : {'error', 'append', 'replace'}, default 'error'
            Behaviour if ``filename`` already exists:

                - 'error' : raise an error if the workbook already exists.
                - 'append' : attempt append worksheets to an existing workbook.
                - 'replace' : overwrite the existing workbook.

        if sheet_exists : {'error', 'replace', 'new'}, default 'error'
            Behaviour if the worksheet already exists (Only applicable when
            ``if_workbook_exists='append'``)

                - 'error': raise a ValueError
                - 'replace' : overwrite workseet.
                - 'new' : create a new worksheet with a different name.
        **kwargs :
            Arguments to be passed to ``DataFrame.to_excel()`` method.
        """
        sheets_to_write = ["observations", "sites", "loops", "metadata"]

        filename = Path(filename)
        if filename.exists():
            if if_workbook_exists == "error":
                raise ValueError(
                    f"file {filename} already exists, and arg {if_workbook_exists=}"
                )

            if if_workbook_exists == "append" and if_sheet_exists == "error":
                existing_worksheets = [
                    ws for ws in sheets_to_write if ws in get_excel_worksheets(filename)
                ]
                if existing_worksheets:
                    raise ValueError(
                        f"worksheets '{sheets_to_write}' already exist in {filename}. "
                        "Use 'if_workbook_exists' and 'if_sheet_exists' parameters "
                        "to specify behaviour."
                    )

        # observations
        write_excel_worksheet(
            df=prepare_writable_df(
                df=self.obs_data,
                expand_datetime="datetime",
                drop_datetime=False,
                bool_to_int=True,
            ),
            filename=filename,
            sheet_name="observations",
            if_workbook_exists=if_workbook_exists,
            if_sheet_exists=if_sheet_exists,
            **kwargs,
        )

        # if_workbook_exists, set to "append" for subsequent sheets, as workbook will now exist

        # sites
        write_excel_worksheet(
            df=prepare_writable_df(
                df=self.site_data,
                expand_datetime=None,
                bool_to_int=True,
            ),
            filename=filename,
            sheet_name="sites",
            if_workbook_exists="append",
            if_sheet_exists=if_sheet_exists,
            **kwargs,
        )

        # loop
        write_excel_worksheet(
            df=prepare_writable_df(self.loop_data),
            filename=filename,
            sheet_name="loops",
            if_workbook_exists="append",
            if_sheet_exists=if_sheet_exists,
        )

        if self.terrain_correction_data is not None:
            write_excel_worksheet(
                df=prepare_writable_df(self.terrain_correction_data),
                filename=filename,
                sheet_name="terrain_corrections",
                if_workbook_exists="append",
                if_sheet_exists=if_sheet_exists,
                **kwargs,
            )

        # Write all parameters to a single spreadsheet
        all_params = []

        # This is a kludge - should create method on parameter objects to
        # to normalise parameter outputs for writing to excel.
        def _format_value(x: Any) -> str | float | int | bool:  # noqa: ANN401
            if isinstance(x, _pd.Timedelta):
                return x.total_seconds()
            elif isinstance(x, Path):
                return str(x)
            return x

        for section, param_obj in self.params.items():
            df: _pd.DataFrame = (
                param_obj.to_series(series_name="value", index_name="parameter")
                .to_frame()
                .reset_index()
            ).copy()
            df.iloc[:, 1] = df.iloc[:, 1].map(_format_value)
            df.insert(0, "section", section)
            all_params.append(df)

        if "index" in kwargs:
            _ = kwargs.pop("index")

        write_excel_worksheet(
            df=_pd.concat(all_params),
            filename=filename,
            sheet_name="metadata",
            if_workbook_exists="append",
            if_sheet_exists=if_sheet_exists,
            index=False,
            **kwargs,
        )
