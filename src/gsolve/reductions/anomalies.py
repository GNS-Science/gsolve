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

import numpy as _np
import pandas as _pd
from numpy.typing import ArrayLike

from gsolve.core.data import DataFieldSpecification, GSolveTable
from gsolve.core.utils import to_1d_ndarray, to_1d_ndarray_or_float
from gsolve.gsolve_outputs import GSolveResults
from gsolve.observations import GravityObservations, GravitySurvey
from gsolve.reductions.corrections import (
    GravityCorrectionParameters,
    GravityCorrectionProvider,
    GravityCorrections,
)
from gsolve.reductions.terrain_corrections import TerrainCorrectionData, TerrainCorrectionParameters
from gsolve.sites import GravitySites

__all__ = [
    "GravityAnomalies",
    "compute_complete_bouguer_anomaly",
    "compute_simple_bouguer_anomaly",
    "compute_free_air_anomaly",
]


def _args_contain_nulls(*args: ArrayLike) -> list[bool]:
    contains_nulls = []
    for a in args:
        contains_nulls.append(bool(_pd.isna(_np.asanyarray(a)).any(axis=None)))

    return contains_nulls


def compute_complete_bouguer_anomaly(
    absolute_gravity: ArrayLike,
    normal_gravity: ArrayLike,
    free_air_correction: ArrayLike,
    bouguer_correction: ArrayLike,
    terrain_correction: ArrayLike,
    atmospheric_correction: ArrayLike = 0.0,
    spherical_bouguer_cap_correction: ArrayLike = 0.0,
) -> _np.ndarray:
    """
    Calculate the Complete Bouguer anomaly.

    The complete Bouguer anomaly is calculated using the following formula:

    .. math::

        CBA = AG - (NG + FAC + AC + BSC + SBC - TC)

    Where:
        - CBA = Complete Bouguer Anomaly
        - AG = Absolute Gravity
        - NG = Normal Gravity on the ellipsoid surface
        - FAC = Free Air Correction
        - AC = Atmospheric Correction
        - BSC = Bouguer Slab Correction
        - SBC = Spherical Bouguer Cap Correction
        - TC = Terrain Correction

    Parameters
    ----------
    absolute_gravity : array-like
        Observed absolute gravity.
    normal_gravity : array-like
        Normal gravity value at the ellipsoid.
    free_air_correction : array-like
        The free air correction.
    bouguer_correction : array-like
        Bouguer correction for infinite planar slab or for curvature corrected.
        If curvature corrected, ensure ``spherical_bouguer_cap_correction`` = 0.0.
    terrain_correction : array-like
    atmospheric_correction : array-like, default = 0.0
    spherical_bouguer_cap_correction : array-like, default = 0.0

    Returns
    -------
    numpy.ndarray
        The complete Bouguer anomaly in mGal.

    See Also
    --------
    compute_simple_bouguer_anomaly : Calculate the Simple Bouguer anomaly.
    compute_free_air_anomaly : Calculate the Free Air anomaly.
    """
    if any(
        _args_contain_nulls(
            absolute_gravity,
            normal_gravity,
            free_air_correction,
            # atmospheric_correction,
            bouguer_correction,
            # terrain_correction,
            spherical_bouguer_cap_correction,
        )
    ):
        raise ValueError("inputs contain nan's")

    return _np.atleast_1d(
        to_1d_ndarray_or_float(absolute_gravity)
        - (
            to_1d_ndarray_or_float(normal_gravity)
            + to_1d_ndarray_or_float(free_air_correction)
            + to_1d_ndarray_or_float(atmospheric_correction)
            + to_1d_ndarray_or_float(bouguer_correction)
            + to_1d_ndarray_or_float(spherical_bouguer_cap_correction)
            - to_1d_ndarray_or_float(terrain_correction)
        )
    )


def compute_simple_bouguer_anomaly(
    absolute_gravity: ArrayLike,
    normal_gravity: ArrayLike,
    free_air_correction: ArrayLike,
    bouguer_correction: ArrayLike,
    atmospheric_correction: ArrayLike = 0.0,
    spherical_bouguer_cap_correction: ArrayLike = 0.0,
) -> _np.ndarray:
    """Calculate the Simple Bouguer anomaly from provided corrections.

    The Simple Bouguer anomaly differs from the Complete Bouguer anomaly
    in that terrain corrections are not included. It is calculated using
    the following formula:

    .. math::
        SBA = AG - (NG + FAC + AC + BSC + SBC)

    Where:
        - SBA = Simple Bouguer Anomaly
        - AG = Absolute Gravity
        - NG = Normal Gravity on the ellipsoid surface
        - FAC = Free Air Correction
        - AC = Atmospheric Correction
        - BSC = Bouguer Slab Correction
        - SBC = Spherical Bouguer Cap Correction

    Parameters
    ----------
    absolute_gravity : array-like
        Observed absolute gravity in mGal.
    normal_gravity : array-like
        Normal gravity value at the ellipsoid in mGal.
    free_air_correction : array-like
        The free air correction in mGal.
    bouguer_correction : array-like
        Bouguer correction for infinite planar slab or for curvature corrected in mGal.
        If curvature corrected, ensure ``spherical_bouguer_cap_correction`` = 0.0.
    atmospheric_correction : array-like, default = 0.0
        The atmospheric correction in mGal.
    spherical_bouguer_cap_correction : array-like, default = 0.0
        The spherical Bouguer cap correction in mGal. Should be zero if
        ``bouguer_correction`` is curvature corrected.

    Returns
    -------
    numpy.ndarray
        The simple Bouguer anomaly in mGal.

    See Also
    --------
    compute_complete_bouguer_anomaly : Calculate the Complete Bouguer anomaly.
    compute_free_air_anomaly : Calculate the Free Air anomaly.
    """
    if any(
        _args_contain_nulls(
            absolute_gravity,
            normal_gravity,
            free_air_correction,
            atmospheric_correction,
            bouguer_correction,
            spherical_bouguer_cap_correction,
        )
    ):
        raise ValueError("inputs contain nan's")

    return compute_complete_bouguer_anomaly(
        absolute_gravity,
        normal_gravity,
        free_air_correction=free_air_correction,
        bouguer_correction=bouguer_correction,
        spherical_bouguer_cap_correction=spherical_bouguer_cap_correction,
        atmospheric_correction=atmospheric_correction,
        terrain_correction=0.0,
    )


def compute_free_air_anomaly(
    absolute_gravity: ArrayLike,
    normal_gravity: ArrayLike,
    free_air_correction: ArrayLike,
) -> _np.ndarray:
    """
    Calculate the free air anomaly.

    The free air anomaly is calculated using the formula:

    .. math::
        FAA = AG - (NG + FAC)

    Where:
        - FAA = Free Air Anomaly
        - AG = Absolute Gravity
        - NG = Normal Gravity on the ellipsoid surface
        - FAC = Free Air Correction

    Parameters
    ----------
    absolute_gravity : ArrayLike
        Absolute gravity in mGal, typically from the gsolve network adjustment.
    normal_gravity : ArrayLike
        Gravity at the ellipsoid surface in mGal.
    free_air_correction : ArrayLike
        Free Air Correction in mGal at the station elevation.

    Returns
    -------
    free_air_anomaly : ndarray
        The free air anomaly in mGal.
    """
    if any(_args_contain_nulls(absolute_gravity, normal_gravity, free_air_correction)):
        raise ValueError("inputs contain nan's")
    return _np.atleast_1d(
        to_1d_ndarray_or_float(absolute_gravity)
        - (
            to_1d_ndarray_or_float(normal_gravity)
            + to_1d_ndarray_or_float(free_air_correction)
        )
    )


class GravityAnomalies(GSolveTable):
    """Compute and store gravity anomalies for a set of sites.

    This class provide a simple mechanism to compute free-air and Bouguer
    anomalies from the outputs of a gsolve network adjustment.

    Parameters
    ----------
    absolute_gravity : GSolveResults, DataFrame or Series
        An object providing site_id's and associated absolute gravity values for which
        anomalies will be computed. Can be any of the following:

            - GSolveResults : the output of a gsolve network adjustment.
            - DataFrame : must contain an 'absolute_gravity' column and be indexed by
                            'site_id'
            - Series : absolute gravity values indexed by 'site_id'.

    sites : GravitySites, GravitySurvey or DataFrame
        An object providing the geographic coordiates and ellipsoidal height for each
        site. Can be any of the following:

            - GravitySites or GravitySurvey : A gsolve object providing site metadata.
            - DataFrame : must contain columns 'latitude', 'longitude' and
              'height_ellipsoidal' and be indexed by 'site_id'.

    corrections_parameters : GravityCorrectionParameters, GravityCorrectionProvider or GravityCorrections
        An object providing either the parameters used to compute the various gravity
        corrections and/or a set of pre-computed gravity corrections. Can be any
        of the following:

            - GravityCorrectionParameters : a parameter object defining
              how to compute gravity corrections. The parameters object will be copied
              to self.params attribute.
            - GravityCorrectionProvider : a class for computing gravity corrections
              as specified in a GravityCorrectionParameters object. This will be used
              directly to compute the necessary gravity corrections, and
              its ``params`` copied to self.params.
            - GravityCorrections : pre-computed gravity corrections for a set of sites
              according to parameters in a GravityCorrectionParameters object. The
              corrections used dircetly and , and its ``params`` copied to self.params

    terrain_corrections : TerrainCorrectionData, optional
        An object providing terrain corrections at each site. These are required to
        compute the complete Bouguer anomaly. If provided, georgraphic coordinates and
        terrain corrections will be copied to self.data and the associated
        TerrainCorrectionParameter objects copied to self.tcorr_params. If None,
        then a terrain correction column 'tcorr:total' will be added and set to NaN.

    Attributes
    ----------
    data : pandas.DataFrame
        Table of computed gravity corrections and anomalies indexed by ``site_id``.
        The primary columns are:

            - absolute_gravity : the input absolute gravity values.
            - normal_gravity_at_ellipsoid : normal gravity at surface of the ellipsoid
                    self.params.ellipsoid
            - free_air_correction : the free-air correction.
            - atmospheric_correction : the atmospheric corrections due to elevation.
              Only inclued if ``self.params.use_atmospheric_correction`` is True.
            - bouguer_slab_correction or bouguer_slab_curvature_corrected : the
              Bouguer correction, with form determined by ``self.params.use_curvature_corrected``.
            - tcorr:* : terrain correction for various zones, if terrain corrections
              were provided. Note that only the ``tcorr:total`` column is used in anomaly
              calculations.
            - tcorr:total : sum of contributions from each terrain correction zone. Will
              be NaN if no terrain corrections were provided.
            - free_air_anomaly : the free-air anomaly in mGal.
            - bouguer_anomaly_simple : the Bouguer anomaly without terrain corrections.
            - bouguer_anomaly_complete : the Bouguer anomaly including terrain corrections.
              Will be NaN if no terrain corrections were provided.

    params : GravityCorrectionParameters
        A copy of the parameters used to compute corrections and anomalies:

            - params.ellipsoid : the ellipsoid used to compute normal gravity.
            - params.density_crust : the crustal density used in Bouguer corrections.
            - params.density_water : the water density used in Bouguer corrections.
            - params.spherical_cap_radius : the radius of spherical cap used in
              computing curvature-corrected form of the Bouguer correction.
            - params.use_curvature_corrected : The type of Bouguer correction used.
              If True, the Bouger correction was the curvature-corrected form, otherwise
              the infinite planar slab form was used.
            - params.use_atmospheric_correction : If True, atmospheric corrections
              were included in anomaly calculations.

    tcorr_params : dict[str, TerrainCorrectionParameters]
        A dictionary of cpoies of the TerrainCorrectionParameters objects associated
        with terrain corrections. The keys are the terrain correction zone ID's, and will
        partially correspond to columns in the ``self.data`` attribute.
        Will be an empty dict if no terrain corrections were provided.

    """

    _known_fields = {
        "site_id": DataFieldSpecification("site_id", str, required=True),
        "height_ellipsoidal": DataFieldSpecification(
            "height_ellipsoidal", float, required=True, legacy_name="height"
        ),
        "normal_gravity_at_stn_elevation": DataFieldSpecification(
            "normal_gravity_at_stn_elevation", float, required=False, default=_np.nan
        ),
        "normal_gravity_at_ellipsoid": DataFieldSpecification(
            "normal_gravity_at_ellipsoid", float, required=False, default=_np.nan
        ),
        "free_air_correction": DataFieldSpecification(
            "free_air_correction", float, required=False, default=_np.nan
        ),
        "bouguer_slab_correction": DataFieldSpecification(
            "bouguer_slab_correction", float, required=False, default=_np.nan
        ),
        "bouguer_slab_curvature_corrected": DataFieldSpecification(
            "bouguer_slab_curvature_corrected", float, required=False, default=_np.nan
        ),
        "atmospheric_correction": DataFieldSpecification(
            "atmospheric_correction", float, required=False, default=0.0
        ),
        "spherical_bouguer_cap_correction": DataFieldSpecification(
            "spherical_bouguer_cap_correction", float, required=False, default=_np.nan
        ),
        "tcorr:total": DataFieldSpecification(
            "terrain_correction", float, required=False, default=0.0
        ),
    }

    def __init__(
        self,
        absolute_gravity: GSolveResults | _pd.DataFrame | _pd.Series,
        sites: GravitySites | GravitySurvey | _pd.DataFrame,
        corrections_parameters: (
            GravityCorrectionParameters | GravityCorrectionProvider | GravityCorrections
        ),
        terrain_corrections: TerrainCorrectionData | None = None,
    ) -> None:
        self.params: GravityCorrectionParameters
        self.tcorr_params: dict[str, TerrainCorrectionParameters] = {}
        self.data: _pd.DataFrame

        precomputed_corrections: GravityCorrections | None = None

        abs_grav_df: _pd.DataFrame
        if isinstance(absolute_gravity, GSolveResults):
            if not absolute_gravity.site_solution:
                raise ValueError("absolute_gravity has no site_solution data")
            abs_grav_df = absolute_gravity.site_solution
        elif isinstance(absolute_gravity, _pd.DataFrame):
            abs_grav_df = absolute_gravity
        elif isinstance(absolute_gravity, _pd.Series):
            abs_grav_df = absolute_gravity.to_frame(name="absolute_gravity")
        else:
            raise TypeError(
                f"invalid type for arg 'absolute_gravity': {type(absolute_gravity)}"
            )
        if abs_grav_df is None:
            raise ValueError("absolute_gravity has no site_solution data")

        sites_df: _pd.DataFrame
        if isinstance(sites, GravitySurvey):
            sites_df = sites.sites.data
        elif isinstance(sites, GravitySites):
            sites_df = sites.data
        elif isinstance(sites, _pd.DataFrame):
            sites_df = sites
        else:
            raise TypeError(f"invalid type for arg 'sites': {type(sites)}")

        if isinstance(corrections_parameters, GravityCorrectionParameters):
            self.params = corrections_parameters.copy()
            corr_provider = GravityCorrectionProvider(params=self.params)
            # self.params = corrections_parameters.copy()

            corrections_parameters = GravityCorrectionProvider(
                params=corrections_parameters
            )
        elif isinstance(corrections_parameters, GravityCorrections):
            self.params = corrections_parameters.params.copy()
            corr_provider = GravityCorrectionProvider(params=self.params)
            precomputed_corrections = corrections_parameters.copy()
            # self.params = corrections_parameters.params.copy()

        elif isinstance(corrections_parameters, GravityCorrectionProvider):
            self.params = corrections_parameters.params.copy()
            corr_provider = corrections_parameters

        else:
            raise TypeError(
                "invalid type for corrections_provider argument: "
                f"{type(corrections_parameters).__name__}"
            )

        # ensure we have entry in `sites` for all absolute gravity data sites
        if not abs_grav_df.index.isin(sites_df.index).all():
            raise ValueError(
                "absolute_gravity has sites with no corresponding site info in sites"
            )

        sites_df = sites_df.loc[abs_grav_df.index]

        self.data = _pd.DataFrame(
            index=_pd.Index(abs_grav_df.index.to_numpy(), name="site_id"),
            data=abs_grav_df["absolute_gravity"],
        )

        if precomputed_corrections is None:
            corrs = corr_provider.compute(sites=sites_df)
        else:
            if not self.data.index.isin(precomputed_corrections.data.index).all():
                raise ValueError(
                    "precomputed corrections do not provide corrections for all sites"
                )
            precomputed_corrections.data = precomputed_corrections.data.loc[
                self.data.index
            ]
            corrs = precomputed_corrections

        if terrain_corrections is None:
            tc = _pd.DataFrame(index=self.data.index, data={"tcorr:total": _np.nan})
            self.tcorr_params = {}
        else:
            if not isinstance(terrain_corrections, TerrainCorrectionData):
                raise TypeError(
                    "invalid type for terrain_corrections argument: "
                    "required TerrainCorrectionData: "
                    f"got {type(terrain_corrections).__name__}"
                )
            tc = terrain_corrections.get_corrections(
                self.data.index, if_missing="fill", fill_value=_np.nan
            )
            self.tcorr_params = {
                k: v.copy() for k, v in terrain_corrections.params.items()
            }

        self.data = _pd.merge(self.data, corrs.data, how="inner", on="site_id")
        for c in tc.columns:
            self.set_column(c, tc[c])

        self._compute_free_air_anomaly()
        self._compute_bouguer_anomaly()

    def _compute_free_air_anomaly(self) -> None:
        """Compute free-air anomaly and update the instance's ``data`` attribute.

        This method is called internally during initialization and is not intended to
        be used directly.
        """
        cols = ["normal_gravity_at_ellipsoid", "free_air_correction"]
        self.set_column(
            "free_air_anomaly",
            self.data["absolute_gravity"]
            - (self.data.loc[:, cols].sum(axis=1, skipna=False)),
        )

    def _compute_bouguer_anomaly(self) -> None:
        """Compute Bouguer anomalies and update the instance's ``data`` attribute.

        This method is called internally during initialization and is not intended to
        be used directly.
        """
        cols = self.params.bouguer_correction_fields()
        tcorr_total_col = "tcorr:total"
        simple_anom_col = "bouguer_anomaly_simple"
        complete_anom_col = "bouguer_anomaly_complete"

        corrs_sum = self.data.loc[:, cols].sum(axis=1, skipna=False)

        self.set_column(simple_anom_col, self.data["absolute_gravity"].sub(corrs_sum))

        if tcorr_total_col not in self.data.columns:
            self.set_column(complete_anom_col, _np.nan)
        else:
            self.set_column(
                complete_anom_col,
                self.data.loc[:, [simple_anom_col, tcorr_total_col]].sum(
                    axis=1, skipna=False
                ),
            )

    def __repr__(self) -> str:
        rval = [f"n_sites={len(self.data)}"]
        faa_field = "free_air_anomaly"
        if faa_field in self.data:
            rval.append(
                "free_air_anomaly="
                f"{self.data[faa_field].min():0.03f}"
                f"<->{self.data[faa_field].max():0.03f}"
            )

        ba_field = "bouguer_anomaly_complete"
        if ba_field not in self.data or self.data[ba_field].isna().all():
            ba_field = "bouguer_anomaly_simple"
        if ba_field in self.data:
            rval.append(
                "bouguer_anomaly="
                f"{self.data[ba_field].min():0.03f}<->{self.data[ba_field].max():0.03f}"
            )

        return f"{type(self).__name__}(" + ", ".join(rval) + ")"
