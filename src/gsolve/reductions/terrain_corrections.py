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
import logging
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import harmonica as hm
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from tqdm import tqdm as _tqdm

from gsolve.core._typing import (
    DatasetOrArray,
    FilePath,
    FloatArray,
    IfSheetExists,
    IfWorkbookExists,
    Points2D,
    Points3D,
    SitesLike,
    TCorrDistanceMaskType,
)
from gsolve.core.data import DataFieldSpecification, GSolveParameters, GSolveTable
from gsolve.core.excel_io import read_excel_worksheet, write_excel_worksheet
from gsolve.core.utils import is_list_like, prepare_writable_df, to_1d_ndarray
from gsolve.core.xr_accessor import TCorrMethods as _TCorrMethods
from gsolve.core.xr_methods import *

__all__ = [
    "calculate_terrain_correction",
    "TerrainCorrectionData",
    "TerrainCorrectionParameters",
    "TerrainCorrector",
]


def calculate_terrain_correction(
    points: Points3D,
    dem: xr.DataArray,
    min_dist: float,
    max_dist: float,
    density_dataset: xr.DataArray | None = None,
    terrain_density: float = 2670.0,
    water_density: float = 1030.0,
    sea_level_elevation: float = 0.0,
    distance_mask_type: TCorrDistanceMaskType = "radial",
    show_progress: bool = True,
    compute_topography: bool = True,
    compute_bathymetry: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate terrain corrections for a set of points based on a digital elevation model (DEM).

    Parameters
    ----------
    points : array_like of shape (3, n)
        An list-like object containing the x, y, and z coordinates of the points.
    dem : xarray.DataArray
        The digital elevation model (DEM) in meters. Should be in the same coordinate system
        and vertical datum as the ``points``.
    min_dist : float
        The minimum distance for the terrain correction mask. Must be >= 0.0.
    max_dist : float
        The maximum distance for the terrain correction mask. Must be > ``min_dist``.
    density_dataset : xarray.DataArray, optional
        An array containing density values in kg/m^3. If not specified, a density model
        will be generated from the ``dem`` using ``terrain_density``, ``water_density``
        and ``sea_level_elevation``.
    terrain_density : float, optional
        The density of the terrain in kg/m^3, by default 2670.0.
    water_density : float, optional
        The density of water in kg/m^3, by default 1030.0.
    sea_level_elevation : float, optional
        The elevation of sea level in m, by default 0.0. Should be in the same vertical
        datum as ``dem`` and ``points``. Used to separate land/topographic and sea/bathymetric
        portions of the DEM.
    distance_mask_type : {"radial", "rectangular"}, default="radial"
        The type of distance mask to use.
    show_progress : bool, default is True.
        Report progress, including a progress bar if the tqdm package is installed.
    compute_topography : bool, default is True
        If Tue, compute terrain corrections for topography (land).
    compute_bathymetry : bool, default is True
        If True, compute terrain corrections for bathymetry.

    Returns
    -------
    ndarray
        The terrain corrections at each point.

    """
    compute_bathymetry_by_arg = bool(compute_bathymetry)
    compute_topography_by_arg = bool(compute_topography)

    # ensure supplied density is a DataArray (if a Dataset was provided)
    use_supplied_density = density_dataset is not None
    if use_supplied_density:
        if not isinstance(density_dataset, xr.DataArray):
            raise TypeError(
                "density_dataset must be an xarray.DataArray not "
                f"'{type(density_dataset).__name__}'"
            )

        if not dem.tcorr.is_compatible(density_dataset):
            raise ValueError(
                "Specified density_dataset is incompatible with dem. "
                "Check that the DataArrays have the same shape and coordinates."
            )

    # format the points
    if len(points) != 3:
        raise ValueError("invalid points arg: must be a list_like of form (x, y, z)")

    try:
        pts_x = to_1d_ndarray(points[0]).astype(np.float64)
        pts_y = to_1d_ndarray(points[1], expected_size=pts_x.size).astype(np.float64)
        pts_z = to_1d_ndarray(points[2], expected_size=pts_x.size).astype(np.float64)
    except Exception as e:
        raise ValueError(f"Points must contain 1d x,y,z arrays of equal size: {e}")

    # check the correction distances
    use_distance_mask = True
    if min_dist < 0:
        raise ValueError(f"min_dist must be >= 0.0, not {min_dist}")
    if max_dist <= min_dist:
        raise ValueError(
            f"Incompatible distance args, {max_dist=} not greater than {min_dist=}"
        )

    # get land sea mask
    land_sea_mask: xr.DataArray = dem.tcorr.get_land_sea_mask(sea_level_elevation)

    if compute_topography:
        if not bool(land_sea_mask.any()):
            # if all false, there is no land/topography
            compute_topography = False
        else:
            topo_elev: xr.DataArray = dem.tcorr.get_topography_elevation(
                land_sea_mask, sea_level_elevation
            )
            if use_supplied_density:
                topo_density: xr.DataArray = density_dataset  # ty:ignore[invalid-assignment]
            else:
                topo_density: xr.DataArray = dem.tcorr.generate_topo_density(
                    terrain_density
                )
    if not compute_topography:
        topo_elev = create_empty_dataarray()
        topo_density = create_empty_dataarray()

    if compute_bathymetry:
        if bool(land_sea_mask.all()):
            # if all true, there is no sea/bathymetry
            compute_bathymetry = False
        else:
            bathy_dem: xr.DataArray = dem.tcorr.get_bathymetry_elevation(
                land_sea_mask, sea_level_elevation
            )
            bathy_density: xr.DataArray = dem.tcorr.generate_bathymetry_density(
                land_sea_mask, terrain_density, water_density, sea_level_elevation
            )
    if not compute_bathymetry:
        bathy_dem = create_empty_dataarray()
        bathy_density = create_empty_dataarray()

    progress_bar = _tqdm(
        total=pts_x.shape[0],
        disable=not show_progress,
        desc="    Progress",
    )

    tcorr_topo = np.zeros_like(pts_x)
    tcorr_bathy = np.zeros_like(pts_x)
    distance_mask = np.empty(shape=(0, 0))
    distance_mask_generated = False
    bad_points = []

    for i, pt in enumerate(np.column_stack((pts_x, pts_y, pts_z))):
        if np.isnan(pt).any():
            bad_points.append(i)
            progress_bar.update(1)
            continue

        (px, py, pz) = pt

        # set up land sea mask it is used by both topography and bathymetry corrections
        if use_distance_mask:
            pt_land_sea_mask = land_sea_mask.tcorr.clip_to_points(
                (px, py), max_dist=max_dist
            )
            if pt_land_sea_mask is None:
                # mismatch between actual DEM extent and requested extent
                # - cannot ctreate point dem etc
                # - return nan - this is an error
                tcorr_bathy[i] = np.nan
                tcorr_topo[i] = np.nan
                bad_points.append(i)
                continue

            if not distance_mask_generated:
                # do this on first loop
                distance_mask: np.ndarray = (
                    pt_land_sea_mask.tcorr.generate_distance_mask(
                        min_dist=min_dist,
                        max_dist=max_dist,
                        mask_type=distance_mask_type,
                    )
                )
            distance_mask_generated = True
        else:
            pt_land_sea_mask = land_sea_mask

        if compute_topography:
            if bool(pt_land_sea_mask.any()):
                if use_distance_mask:
                    pt_topo_elev: xr.DataArray = topo_elev.tcorr.clip_to_arr(
                        pt_land_sea_mask, clip_other=False
                    )
                    pt_topo_density: xr.DataArray = topo_density.tcorr.clip_to_arr(
                        pt_land_sea_mask, clip_other=False
                    ).tcorr.apply_mask(distance_mask)

                else:
                    pt_topo_elev = topo_elev
                    pt_topo_density = topo_density

                tcorr_topo[i] = tcorr_harmonica_topography(
                    (px, py, pz),
                    topography=pt_topo_elev,
                    topography_density=pt_topo_density,
                )

        if compute_bathymetry:
            if bool(pt_land_sea_mask.any()):
                if use_distance_mask:
                    pt_bathy_depth = bathy_dem.tcorr.clip_to_arr(
                        pt_land_sea_mask, clip_other=False
                    )
                    pt_bathy_density: xr.DataArray = bathy_density.tcorr.clip_to_arr(
                        pt_land_sea_mask, clip_other=False
                    ).tcorr.apply_mask(distance_mask)
                else:
                    pt_bathy_depth = bathy_dem
                    pt_bathy_density = bathy_density

                tcorr_bathy[i] = tcorr_harmonica_bathymetry(
                    (px, py, pz),
                    bathymetry=pt_bathy_depth,
                    bathymetry_density=pt_bathy_density,
                    sea_level_elevation=sea_level_elevation,
                )

        progress_bar.update(1)

    progress_bar.close()

    return tcorr_topo, tcorr_bathy


def tcorr_harmonica_topography(
    point: tuple[float, float, float],
    topography: xr.DataArray,
    topography_density: xr.DataArray,
    parallel: bool = False,
    disable_checks: bool = False,
) -> float:
    """Calculate topographic terrain correction using harmonica.

    This is called by ``calculate_terrain_correction()`` to compute terrain
    corrections for the topography portions of a DEM. You should not need to call this
    function directly, however it is exposed for advanced use cases.

    Parameters
    ----------
    point : tuple of floats (x, y, z)
        The (x, y, z) coordinates of the observation point.
    topography : xarray.DataArray
        The topographic surface elevations in meters. The DataArray should have
        no values below 'sea level'
    topography_density : xarray.DataArray
        The density of the topographic surface in kg/m^3, with the same coordinates as ``topography``. Must be compatible with
        ``topography``.
    parallel : bool, optional
        Whether to use parallel processing, by default False.
    disable_checks : bool, optional
        Whether to disable input checks, by default False

    Returns
    -------
    float
        The calculated topographic terrain correction at ``point``

    See Also
    --------
    harmonica.prism_layer.gravity : the underkying harmonica method used to compute
        the terrain correction.
    """
    site_z = point[2]

    # flip density for cells where elevation above station
    topography_density = xr.where(
        topography <= site_z, topography_density, -1 * topography_density
    )

    topo_prims = hm.prism_layer(
        coordinates=(topography.tcorr.xc, topography.tcorr.yc),
        surface=topography.to_numpy(),
        reference=site_z,
        properties={"density": topography_density.to_numpy()},
    )

    return topo_prims.prism_layer.gravity(
        point,
        density_name="density",
        field="g_z",
        parallel=parallel,
        disable_checks=disable_checks,
        progressbar=False,
    )


def tcorr_harmonica_bathymetry(
    point: tuple[float, float, float],
    bathymetry: xr.DataArray,
    bathymetry_density: xr.DataArray,
    sea_level_elevation: float = 0.0,
    parallel: bool = False,
    disable_checks: bool = False,
) -> float:
    """Calculate bathymetric terrain correction using harmonica.

    This function is called by ``calculate_terrain_correction()`` to compute terrain
    corrections for the bathymetry portions of a DEM. You should not need to call this
    function directly, however it is exposed for advanced use cases.

    Parameters
    ----------
    point : tuple of floats (x, y, z)
        The (x, y, z) coordinates of the observation point.

    bathymetry : xarray.DataArray
        Bathymetric surface/base depths in meters. The DataArray should have
        no values above 'sea_level_elevation'.
    bathymetry_density : xarray.DataArray
        An array of 'densities' the material 'filling' bathymetry in kg/m^3, with the
        same coordinates as ``bathymetry``. For the typical case of oceans, where sea water
        (1030.0 kg/m^3) is replacing crustal rocks (2670.0 kg/m^3),
        ``bathymetry_density`` will be the density contrast 1640.0 kg/m^3. Cells where
        ``bathymetry`` is above ``sea_level_elevation`` must have density set to 0.0,
    sea_level_elevation : float, optional
        The elevation of sea level in m, by default 0.0. Should be in the same vertical
        datum as ``dem`` and ``points``. Used to separate land/topographic and sea/bathymetric
        portions of the DEM.
    parallel : bool, optional
        Whether to use parallel processing, by default False.
    disable_checks : bool, optional
        Whether to disable input checks, by default False

    Returns
    -------
    float
        The calculated bathymetric terrain correction at ``point``
    """
    bathy_prims = hm.prism_layer(
        coordinates=(bathymetry.tcorr.xc, bathymetry.tcorr.yc),
        surface=bathymetry.to_numpy(),
        reference=sea_level_elevation,
        properties={"density": bathymetry_density.to_numpy()},
    )

    return bathy_prims.prism_layer.gravity(
        point,
        density_name="density",
        progressbar=False,
        field="g_z",
        parallel=parallel,
        disable_checks=disable_checks,
    )


# @dataclasses.dataclass(frozen=True)
@dataclasses.dataclass()
class TerrainCorrectionParameters(GSolveParameters):
    """Class to store parameters for computing terrain corrections for a single "zone".

    A "zone" here is analagous to classic Hammer zones; ia symmetric region
    surrounding a point over which terrain corrections arecomputed.
    It is, defined by its extent (``min_dist`` and ``max_dist``), material densities,
    and topography data sources.

    A full terrain correction would typically include several zones, covering different
    distance ranges.

    Attributes
    ----------
    name : str
        The name for this "zone". This name will be used as the key for this parameter
        object when it is added to the ``TerrainCorrector`` and
        ``TerrainCorrectionData`` objects. It will be used to label output columns
        in the ``TerrainCorrectionData.data`` DataFrame.
    min_dist : float
        Minimum distance or inner radius for this terrain correction zone. Data within
        this radius are excluded.
    max_dist : float
        Maximum distance or outer radius for this terrain correction zone. Data beyond
        this radius are excluded.
    terrain_density : float, default=2670.0
        Density of the terrain in kg/m^3.  This is and ``water_density`` are used to
        generate a simple density model from the DEM.
    water_density : float, default=1030.0
        Density of water in kg/m^3. This is and ``terrain_density`` are used to
        generate a simple density model from the ``dem``.
    sea_level_elevation : float, default=0.0
        Elevation of sea level in meters using the same vertical datum as ``dem`` and
        ``points``.  Defines the boundary between topography and bathymetry when
        generating the density model.
    distance_mask_type : {"radial", "rectangular"}, default="radial"
        Type of distance mask to use. A "radial" mask creates an approximately circular
        zone, while a "rectangular" mask creates a rectangular mask.
    dem_source : str, PathLike, default=""
        Path to a terrain dataset file. This will be loaded during terrain correction
        computation. If an empty string, then DEM data must be supplied directly to a
        ``TerrainCorrector`` instance. Note that ``dem_source`` inputs are converted to and
        stored as a string.
    density_dataset_source : str, PathLike, default=""
        Path to a density model file, which will be loaded during terrain correction
        computation. If an empty string, then a simple density model will be generated
        from the DEMusing ``terrain_density``, ``water_density`` and
        ``sea_level_elevation``. Note that ``density_dataset_source`` inputs are converted
        to and stored as a string.
    compute_topography : bool, default is True
        Compute gravity corrections due to topographic masses.
    compute_bathymetry : bool, default is True
        Compute gravity corrections due to water bodies such as the ocean.
    site_height_field : str, default is "height_ellipsoidal"
        Column in a ``GravitySites.data`` object containing
        site elevations/z coordinates.
    site_easting_field : str, default is "easting"
        Column in a ``GravitySites.data`` object containing
        site easting/x coordinates.
    site_northing_field : str, default is "northing"
        Column in a ``GravitySites.data`` object containing
        site northing/y coordinates.
    method : {"harmonica"}, default is "harmonica"
        Method to use for computing terrain corrections. Currently only "harmonica" is
        supported.
    """

    name: str
    min_dist: float
    max_dist: float
    terrain_density: float = 2670.0
    water_density: float = 1030.0
    sea_level_elevation: float = 0.0
    distance_mask_type: TCorrDistanceMaskType = "radial"
    dem_source: FilePath | xr.DataArray = ""
    density_dataset_source: FilePath | xr.DataArray = ""
    compute_topography: bool = True
    compute_bathymetry: bool = True
    site_height_field: str = "height_ellipsoidal"
    site_easting_field: str = "easting"
    site_northing_field: str = "northing"
    method: Literal["harmonica"] = "harmonica"

    def __post_init__(self) -> None:
        self._normalize_fields()
        self._sanity_check()

    def _normalize_fields(self) -> None:
        if self.name is None:
            raise ValueError("'name' attribute must be a non-zero length string")

        name = str(self.name)
        if not name:
            raise ValueError("'name' attribute must be a non-zero length string")
        object.__setattr__(self, "name", name)

        object.__setattr__(self, "min_dist", float(self.min_dist))
        object.__setattr__(self, "max_dist", float(self.max_dist))
        object.__setattr__(self, "terrain_density", float(self.terrain_density))
        object.__setattr__(self, "water_density", float(self.water_density))
        object.__setattr__(self, "sea_level_elevation", float(self.sea_level_elevation))
        object.__setattr__(self, "distance_mask_type", str(self.distance_mask_type))
        object.__setattr__(self, "site_height_field", str(self.site_height_field))
        object.__setattr__(self, "site_easting_field", str(self.site_easting_field))
        object.__setattr__(self, "site_northing_field", str(self.site_northing_field))
        object.__setattr__(self, "compute_topography", bool(self.compute_topography))
        object.__setattr__(self, "compute_bathymetry", bool(self.compute_bathymetry))

        for field_name in ("dem_source", "density_dataset_source"):
            value = getattr(self, field_name)
            if _is_dataarray(value):
                continue
            if is_filepath_like(value):
                object.__setattr__(self, field_name, str(value))
                continue
            if not value:
                object.__setattr__(self, field_name, "")
                continue
            raise TypeError(
                f"{field_name} attribute must be a DataArray, str, or "
                f"Path-like object, not a {type(value).__name__}"
            )

    def _sanity_check(self) -> None:
        if (
            np.isnan(self.min_dist)
            or np.isnan(self.max_dist)
            or self.min_dist < 0.0
            or self.max_dist <= self.min_dist
        ):
            raise ValueError(
                "invalid 'min_dist' and 'max_dist' parameters. Must be real values where"
                "0.0 <= min_dist < max_dist: "
                f"got min_dist={self.min_dist}, max_dist={self.max_dist}"
            )
        # check distance msk type is valid
        if not is_in_literal(self.distance_mask_type, TCorrDistanceMaskType):
            raise ValueError(
                f"invalid 'distance_mask_type': {self.distance_mask_type}. "
                f"Expected one of: {get_args(TCorrDistanceMaskType.__value__)}"
            )

        if _is_dataarray(self.dem_source):
            if not self.dem_source.tcorr.is_valid_dem:
                raise ValueError(
                    "invalid dem_source DataArray: must be a 2D array of floats"
                )
        elif not self.dem_source:
            raise TypeError(
                "invalid dem_source: must be an xarray.DataArray or file path"
            )
        if _is_dataarray(self.density_dataset_source):
            if not self.density_dataset_source.tcorr.is_valid_dem():
                raise ValueError(
                    "density_dataset_source is an xr.DataArray object but is not a valid DEM. "
                    "Check that it has the correct dimensions and coordinates."
                )

    def to_series(
        self,
        series_name: str | None = "value",
        index_name: str | None = "parameter",
        index_prefix: str | Sequence[str] | None = None,
        include_data_array: bool = True,
    ) -> pd.Series:
        """Return parameters as a pandas Series.

        Series values will be indexed by parameter_name.

        Parameters
        ----------
        series_name : str | None, optional
            Name for the resulting Series. If None, the Series will be unnamed.
        index_name : str | None, optional
            Name for the Series index. If None, the index will be unnamed.
        index_prefix : str | None, optional
            If specified, the returned Series will have a MultiIndex where the
            first level is ``index_prefix``. E.g. if ``index_prefix="zone1"``, then
            the Series index will be of the form: ("zone1", parameter_name,...).
            This is useful to avoid index collisions when combining parameter Series.
        include_data_array: bool, default True
            Control how fields 'dem_source` and `density_dataset_source` are treated if
            they contain an ``xarray.DataArray`` object.  If True,  include the full
            array. If False, replace the array with a string defining output.

        Returns
        -------
        pd.Series
            A pandas Series containing the parameters.
        """
        pars_dict = self.to_dict()
        if not include_data_array:
            for f in ("density_dataset_source", "dem_source"):
                if _is_dataarray(pars_dict[f]):
                    pars_dict[f] = "xarray.DataArray"
        ds = pd.Series(data=self.to_dict(), name=series_name).rename_axis(index_name)

        if index_prefix is not None:
            ds = ds.to_frame().reset_index()

            if isinstance(index_prefix, str):
                idx_val, idx_name = [index_prefix, "zone"]
            elif is_list_like(index_prefix):
                if len(index_prefix) != 2:
                    raise ValueError(
                        "if index_prefix is list-like, it must have length 2"
                    )
                idx_val, idx_name = index_prefix
            else:
                raise ValueError(
                    "index_prefix must be a string or list-like of length 2"
                )

            ds[idx_name] = idx_val
            ds = ds.set_index([idx_name, "parameter"])[series_name]

        return ds

    def to_dict(self, path2str: bool = False) -> dict[str, Any]:
        """Return parameters as a dictionary.

        The dictionary will be of the form ``{parameter_name: parameter_value, ...}``.

        Parameters
        ----------
        path2str : bool, default False
            If True, convert any Path-like objects to their string representations.

        Returns
        -------
        dict[str, Any]
            A dictionary of parameter names and their values.
        """
        rval = dataclasses.asdict(self)
        if path2str:
            to_str = lambda x: str(x) if x is not None else None
            rval["dem_source"] = to_str(rval["dem_source"])
            rval["density_dataset_source"] = to_str(rval["density_dataset_source"])

        return rval

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> dict[str, Self]:
        """Generate a dict of TerrainCorrectionParameters objects from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing terrain correction parameters. Each row corresponds
            to a single TerrainCorrectionParameters object. Column names must match
            the attribute names of the TerrainCorrectionParameters class.

        Returns
        -------
        dict[str, TerrainCorrectionParameters]
            A dictionary of TerrainCorrectionParameters objects created from the DataFrame.
            The keys are of the form "tcorr:{name}", where {name} is the ``name`` attribute
            of each TerrainCorrectionParameters object.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, not {type(df)}")

        df = df.fillna(value="")
        params = []

        if df.shape[1] == 3:
            df = df.set_index(df.columns[0:2].to_list()).sort_index()
            for idx in df.index.unique(level=0):
                ds = df.loc[idx].squeeze()
                params.append(ds)

        elif df.shape[1] == 2:
            ds = df.set_index(df.columns[0]).squeeze()
            params.append(ds)
        elif df.shape[1] == 1:
            ds = df.squeeze()
            params.append(ds)
        else:
            raise ValueError(
                f"params dataframe must have 2 or 3 columns: found {df.shape[1]}"
            )
        if not all([isinstance(ds, pd.Series) for ds in params]):
            raise ValueError("error converting dataframe to series")
        params = [cls.from_series(p) for p in params]
        return {f"tcorr:{p.name}": p for p in params}


class TerrainCorrector:
    """
    A class for computing gravity terrain corrections from elevation models (DEMs).

    It supports multiple calculation "zones", each with its own parameters and data sources.

    A typical workflow using this class would be:

        - Define one or more TerrainCorrectionParameters objects for the desired zones.
        - Instantiate a TerrainCorrector with these parameters and optional DEM/density models.
        - Add additional zones as needed using ``add_calculation_zone``.
        - Call ``compute()`` on a set of points.

    Attributes
    ----------
    params : dict
        A dictionary of TerrainCorrectionParmeter objects defining the "zones" to be computed.

    Parameters
    ----------
    params: TerrainCorrectionParameters or list-like of TerrainCorrectionParameters
        The TerrainCorrectionParameters object(s) defining the "zones" to be computed.


    """

    def __init__(
        self,
        params: TerrainCorrectionParameters | Sequence[TerrainCorrectionParameters],
    ) -> None:
        self.params: dict[str, TerrainCorrectionParameters] = {}

        if isinstance(params, TerrainCorrectionParameters):
            params = [params]
        elif isinstance(params, Iterable):
            if not all(isinstance(p, TerrainCorrectionParameters) for p in params):
                raise TypeError(
                    "if params is a list-like, all items must be TerrainCorrectionParameters objects"
                )
            params = list(params)
        else:
            raise TypeError(
                "params arg must be a TerrainCorrectionParameters object or a list-like"
                f" of TerrainCorrectionParameters objects, not '{type(params)}'"
            )

        for p in params:
            self.add_zone(params=p)

    def add_zone(self, params: TerrainCorrectionParameters) -> None:
        """Add a terrain correction calculation zone to the TerrainCorrector.

        Parameters
        ----------
        params : TerrainCorrectionParameters
            The parameters defining the terrain correction zone.
        """
        if not isinstance(params, TerrainCorrectionParameters):
            raise TypeError(
                "params must be a TerrainCorrectionParameters object, "
                f"not {type(params)}"
            )

        self.params[params.name] = params

    @property
    def zones(self) -> list[str]:
        """Return list of defined zone names sorted by min_dist."""
        zones = [(k, v.min_dist) for k, v in self.params.items()]
        return [str(z[0]) for z in sorted(zones, key=lambda x: x[1])]

    def compute(
        self,
        points: SitesLike | Points3D,
        site_id: npt.ArrayLike | None = None,
        show_progress: bool = True,
    ) -> "TerrainCorrectionData":
        """Compute terrain corrections for a set of points.

        Parameters
        ----------
        points : GravitySites or sequence of array_likes (x, y, z)
            The observation points where terrain corrections are to be computed. Must
            be in the same coordinate reference system as the dem.
            If ``points`` is a ``GravitySites`` object, then the site coordinate fields
            must be defined on the ``TerrainCorrectionParameters`` objects used by
            this ``TerrainCorrector``. If points is a sequence of array_likes, then it
            must be of the form (x, y, z), where x, y, and z are arrays of equal
            length.
        site_id : array_like, optional
            An array of site IDs corresponding to each point if ``points`` is a Points3D
            instance (see above).   If None, then a RangeIndex will be generated.
            Ignored if points is a `SitesLike` object.
        show_progress : bool, default is True.
            Report progress, including a progress bar if the tqdm package is installed.

        Returns
        -------
        TerrainCorrectionData
            An object containing the computed terrain corrections and the
            TerrainCorrectionParameters used.

        """
        if not isinstance(points, SitesLike) and not is_list_like(points):
            raise TypeError(
                "points must be a sequence of arrays of form (x, y, z) "
                f"or a GravitySites object, not {type(points)}"
            )
        # Establish if we need to get points for each zone.
        # - If the points is a tuple, get 1 set of x,y,z now
        # - If the points is a GravitySites object, check if the site coordinate fields
        #   are the same for all zones.
        #     - If True, we will get 1 set of x,y,z now
        #     - If False, we will get x,y,z for each zone in the loop

        get_points_per_zone = False
        if is_list_like(points):
            x = to_1d_ndarray(points[0]).astype(np.float64)
            y = to_1d_ndarray(points[1], expected_size=x.size).astype(np.float64)
            z = to_1d_ndarray(points[2], expected_size=x.size).astype(np.float64)
            if site_id is None:
                site_id = np.arange(len(x), dtype=int).astype(str)
            else:
                site_id = to_1d_ndarray(site_id, expected_size=x.size).astype(str)

        else:
            site_id = points.data.index.astype(str).to_numpy()
            # test if site coordinate labels in attached TerrainCorrectionParameters objects
            if (
                len({p.site_easting_field for p in self.params.values()}) > 1
                or len({p.site_northing_field for p in self.params.values()}) > 1
                or len({p.site_height_field for p in self.params.values()}) > 1
            ):
                x, y, z = None, None, None
                get_points_per_zone = True

            else:
                p = self.params[self.zones[0]]
                x, y, z = points.get_points(
                    xcol=p.site_easting_field,
                    ycol=p.site_northing_field,
                    zcol=p.site_height_field,
                )

        # empty object to store results
        results = TerrainCorrectionData(
            site_id=site_id,
            easting=x,
            northing=y,
            params=None,
        )

        full_warning_displayed = False

        for zone in self.zones:
            pars = self.params[zone].copy()
            if show_progress:
                print(f"\nCalculating terrain corrections for zone: {zone}")

            # get points if necessary
            if get_points_per_zone:
                try:
                    x, y, z = points.get_points(
                        xcol=pars.site_easting_field,
                        ycol=pars.site_northing_field,
                        zcol=pars.site_height_field,
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error extracting site coordinates from GravitySites object: {e}"
                    )

            # get the dem for this zone
            if _is_dataarray(pars.dem_source):
                dem = pars.dem_source.copy()
            elif pars.dem_source:
                # defined as a source file to be loaded
                dem = load_dem(pars.dem_source)
            else:
                raise ValueError(
                    f"DEM not specified or zone='{zone}': TerrainCorrectionParameter "
                    "object must provide source file or an xarray.DataArray object."
                )

            # get the density model if defined
            if _is_dataarray(pars.density_dataset_source):
                # defined as an argument
                density_model = pars.density_dataset_source.copy()
            elif pars.density_dataset_source:
                # defined as a source file to be loaded
                density_model = load_dem(pars.density_dataset_source)
            else:
                # not specified, will auto generate from DEM
                density_model = None

            tc = calculate_terrain_correction(
                dem=dem,
                points=(x, y, z),
                min_dist=pars.min_dist,
                max_dist=pars.max_dist,
                density_dataset=density_model,
                terrain_density=pars.terrain_density,
                water_density=pars.water_density,
                distance_mask_type=pars.distance_mask_type,
                show_progress=show_progress,
                method=pars.method,
                compute_topography=pars.compute_topography,
                compute_bathymetry=pars.compute_bathymetry,
            )

            # print a summary if something went wrong
            n_missing_tc = 0
            if pars.compute_topography:
                n_missing_tc = np.sum(np.isnan(tc[0]))
            elif pars.compute_bathymetry:
                n_missing_tc = np.sum(np.isnan(tc[1]))
            if n_missing_tc > 0:
                indent = "    " if show_progress else ""
                print(
                    f"{indent}Warning: zone '{zone}': terrain corrections not calculated for "
                    f"{n_missing_tc} of {len(x)} sites."
                )
                if not full_warning_displayed:
                    full_warning_displayed = True
                    print(
                        f"{indent}    This is probably due to:"
                        f"\n{indent}    (1) insufficient DEM coverage and/or"
                        f"\n{indent}    (2) errant or incomplete site coordinates and elevation."
                    )

            results.set_corrections(
                params=pars, topography_corrections=tc[0], bathymetry_corrections=tc[1]
            )
        return results


class TerrainCorrectionData(GSolveTable):
    """Class to store terrain correction outputs and parameters.

    In general, a user should not need to instantiate a TerrainCorrectionData object
    directly. Instances will be generated from a ``TerrainCorrector`` object
    via the ``compute()`` method. A TerrainCorrectionData object can be written to a
    file and then reloaded and re-instantiated, supporting a workflow where terrain
    corrections need only be computed once.

    Attributes
    ----------
    params : dict
        Dictionary of containing copies of the ``TerrainCorrectionParameters`` objects
        used in computing terrain corrections for each zone. For each parameter object,
        the dictionary key will be ``'tcorr:{obj.name}'``. This naming pattern is also
        used to label the corresponding terrain correction columns in the
        ``TerrainCorrectionData.data`` DataFrame.
    data : DataFrame
        A DataFrame containing terrain correction data and indexed by ``'site_id'``.
        The DataFrame will contain columns for site locations, terrain corrections
        for each zone, and the total terrain correction. For a given zone defined by
        TerrainCorrectionParameters object = ``obj``, the output columns will be:

                    - 'tcorr:{obj.name}:topo' : the topography only component of the terrain
            correction.  Ommited if ``compute_topography`` is False.
                    - 'tcorr:{obj.name}:bath' : the bathymetry only component of the terrain
            correction.  Ommited if ``compute_bathymetry`` is False.

        The total terrain correction column will be labeled ``'tcorr:total'``. This is
        computed at initialisation and whenever new corrections are added via the
        ``set_corrections()`` method.

        Columns are ordered by minimum distance of the corresponding zone, with the
        total terrain correction column last.

    Parameters
    ----------
    site_id : array_like of str
        The unique site identifiers as a sequence. All elements are converted to str. Will be
        used to index the ``obj.data`` DataFrame.
    params : TerrainCorrectionParameters or sequence of TerrainCorrectionParameters, optional
        The parameters for the various terrain correction zones.
    terrain_corrections : array_like or list of array_like
        The terrain correction values for each zone.
    **kwargs : dict
        Additional columns to be included in ``obj.data`` DataFrame. This could include
        site location information such as easting, northing, latitude, longitude etc.
    """

    _known_fields = {
        "site_id": DataFieldSpecification(
            "site_id", str, required=True, legacy_name="station"
        ),
        "easting": DataFieldSpecification("easting", float, required=False),
        "northing": DataFieldSpecification("northing", float, required=False),
        "latitude": DataFieldSpecification("latitude", float, required=False),
        "longitude": DataFieldSpecification("longitude", float, required=False),
        "height_ellipsoidal": DataFieldSpecification(
            "height_ellipsoidal", float, required=False
        ),
        "height_orthometric": DataFieldSpecification(
            "height_orthometric", float, required=False
        ),
    }
    _index_field: str | None = "site_id"
    _default_excel_sheet_name = "terrain_corrections"

    def __init__(
        self,
        site_id: npt.ArrayLike,
        params: TerrainCorrectionParameters
        | list[TerrainCorrectionParameters]
        | tuple[TerrainCorrectionParameters, ...]
        | None = None,
        terrain_corrections: npt.ArrayLike
        | list[FloatArray]
        | tuple[FloatArray, ...]
        | None = None,
        **kwargs,
    ) -> None:
        self.params = {}

        if params is not None and terrain_corrections is None:
            raise ValueError(
                "params is specified but terrain_corrections is None: "
                "must specify both or neither"
            )
        elif params is None and terrain_corrections is not None:
            raise ValueError(
                "terrain_corrections is specified but params is None: "
                "must specify both or neither"
            )

        # initialise data frame with site_id's as index
        sids = to_1d_ndarray(site_id).astype(str)
        if sids.ndim != 1:
            raise ValueError(f"site_id must be a 1D array, not {sids.ndim}D")
        n_tcorr = len(sids)
        if n_tcorr == 0:
            raise ValueError("terrain_correction arg must not be empty")

        self.data = pd.DataFrame(index=pd.RangeIndex(n_tcorr), data=None)
        self.set_column("site_id", site_id)
        self.data = self.data.set_index("site_id")

        for k, v in self._known_fields.items():
            if k not in self.data.columns and k in kwargs:
                self.set_column(k, data=kwargs.pop(k))

        for k, v in kwargs.items():
            self.set_column(k, v)
        self.set_column("tcorr:total", 0.0)

        if params is None:
            return

        if isinstance(params, TerrainCorrectionParameters):
            _params_list = [params]
        elif isinstance(params, (list, tuple)):
            _params_list = list(params)
            if not all(
                isinstance(p, TerrainCorrectionParameters) for p in _params_list
            ):
                raise TypeError(
                    "if params is a list or tuple, all items must be "
                    "TerrainCorrectionParameters objects"
                )
        else:
            raise TypeError(
                "params arg must be a TerrainCorrectionParameters object, None or a "
                "list or tuple of TerrainCorrectionParameters objects, "
                f"not '{type(params)}'"
            )

        if terrain_corrections is None:
            _tc_list = [None] * len(_params_list)
        elif isinstance(terrain_corrections, FloatArray):
            _tc_list = [terrain_corrections]
        elif isinstance(terrain_corrections, (list, tuple)):
            _tc_list = list(terrain_corrections)
            if not all(isinstance(tc, FloatArray) for tc in _tc_list):
                raise TypeError(
                    "if terrain_corrections is a list or tuple, all items must be "
                    "array-like (e.g. numpy arrays or pandas Series)"
                )
        else:
            raise TypeError(
                "terrain_corrections arg must be an array-like, None or a list or tuple "
                f"of array-likes, not '{type(terrain_corrections)}'"
            )

        if len(_params_list) != len(_tc_list):
            raise ValueError(
                "inconsistent params and terrain_corrections arg lengths: "
                f"{len(_params_list)} params and {len(_tc_list)} "
                "terrain_corrections specified"
            )

        for p, tc in zip(_params_list, _tc_list):
            self.set_corrections(p, tc)

    @classmethod
    def create_empty(
        cls,
        site_id: npt.ArrayLike,
    ) -> Self:
        """Create an empty TerrainCorrectionData object.

        The objects ``data`` attribute will be initialized as an empty DataFrame with
        "site_id" as index. The object can then be "loaded" with terrain correction data
        using the ``set_corrections()`` method.

        Parameters
        ----------
        site_id : array_like of str
            The unique site identifiers. All elements are converted to str.

        Returns
        -------
        TerrainCorrectionData
            An empty TerrainCorrectionData object.
        """
        return cls(site_id=site_id, params=None, terrain_corrections=None)

    def set_corrections(
        self,
        params: TerrainCorrectionParameters,
        topography_corrections: npt.ArrayLike | None = None,
        bathymetry_corrections: npt.ArrayLike | None = None,
    ) -> None:
        """Add a set of terrain correction parameters and values.

        This will add a new column to the ``obj.data`` DataFrame, and recalculate the
        total terrain correction column.

        Parameters
        ----------
        params : TerrainCorrectionParameters
            The parameters for the terrain correction calculations.
        topography_corrections : array-like
            The terrain correction values.
        bathymetry_corrections : array-like
            Corrections for bathymetry
        """
        if not isinstance(params, TerrainCorrectionParameters):
            raise TypeError(
                "params must be a TerrainCorrectionParameters object, "
                f"not {type(params)}"
            )
        if bathymetry_corrections is None and topography_corrections is None:
            raise ValueError(
                "Must specify at least one of topography_corrections or "
                "bathymetry_corrections"
            )

        tcor_prefix = "tcorr:"
        tcor_base_name = f"{tcor_prefix}{params.name}"
        self.params[tcor_base_name] = params

        corrs_dict = {
            "topography_corrections": (
                topography_corrections if params.compute_topography else None
            ),
            "bathymetry_corrections": (
                bathymetry_corrections if params.compute_bathymetry else None
            ),
        }

        for corr_type, corrs in corrs_dict.items():
            if corrs is None:
                continue
            corrs = np.atleast_1d(corrs).astype(float)
            if len(corrs.shape) != 1:
                raise ValueError(
                    f"{corr_type} must be a 1D array, not {len(corrs.shape)}D"
                )

            if len(corrs) != len(self.data):
                raise ValueError(
                    f"{corr_type} must have same length as site_id "
                    f"({self.data.shape[0]}), not {len(corrs)}"
                )
            topo_col_name = f"{tcor_base_name}:{corr_type[:4]}"
            self.set_column(topo_col_name, corrs, dtype=float)

        # now set the total column
        tcorr_total_col_name = f"{tcor_prefix}total"

        if tcorr_total_col_name in self.data.columns:
            self.data = self.data.drop(columns=[tcorr_total_col_name])
        existing_tcorr_cols = [
            c
            for c in self.data.columns
            if (c.startswith(tcor_prefix) and c != tcorr_total_col_name)
        ]
        if len(existing_tcorr_cols) == 0:
            self.set_column(tcorr_total_col_name, 0.0, dtype=float)
        else:
            df = self.data[existing_tcorr_cols].round(decimals=6)
            self.data = self.data.drop(columns=existing_tcorr_cols)
            df[tcorr_total_col_name] = df.sum(axis=1, skipna=False)

            self.data = pd.concat([self.data, df], axis=1)

    def __repr__(self) -> str:
        zones = ", ".join(
            [
                f"{v.name}(min_dist={v.min_dist}, max_dist={v.max_dist})"
                for v in self.params.values()
            ]
        )
        return f"{type(self).__name__}(n_sites={self.data.shape[0]}, tcorr=[{zones}])"

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        params: TerrainCorrectionParameters
        | Sequence[TerrainCorrectionParameters]
        | pd.DataFrame
        | pd.Series,
        include_extra_cols: bool = True,
    ) -> Self:
        """Create a TerrainCorrectionData object from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing terrain correction data.
        params : TerrainCorrectionParameters, array-like, DataFrame, or Series
            Parmeters for terrain correction calculations.
            If a DataFrame, it must have a column named 'parameters'.
        include_extra_cols : bool, default is True
            If True, include any extra columns in the DataFrame that are not
            terrain correction values.

        Returns
        -------
        TerrainCorrectionOutput
            TerrainCorrectionOutput object.

        """
        # fist get the parameters
        # this informs whet to expect from the dataframe
        if isinstance(params, pd.DataFrame):
            params_dict = TerrainCorrectionParameters.from_dataframe(params)
        elif isinstance(params, pd.Series):
            p = TerrainCorrectionParameters.from_series(params)
            p_name = f"tcorr:{p.name}"
            params_dict = {p_name: p}
        elif isinstance(params, TerrainCorrectionParameters):
            params_dict = {f"tcorr:{params.name}": params}
        elif is_list_like(params):
            params_dict = {f"tcorr:{p.name}": p for p in params}
        else:
            raise TypeError(
                "params must be a Dataframe, Series or TerrainCorrectionParameters "
                f"object, not {type(params)}"
            )

        consumed_columns = ["tcorr:total"]

        if "site_id" in df.columns:
            df = df.set_index("site_id")

        extra_cols = [c for c in df.columns if not c.startswith("tcorr:")]
        consumed_columns.extend(extra_cols)
        extra_data: dict[str, Any] = {}
        if include_extra_cols:
            extra_data: dict[str, Any] = {
                str(k): v for k, v in df[extra_cols].to_dict(orient="list").items()
            }

        # create 'empty' terrain correction data object
        # with site ids and extra columns
        obj = cls(site_id=df.index, **extra_data)

        # now add data fro each parameter set
        for k, v in params_dict.items():
            tc_args: dict[str, Any] = {}
            tc_args["params"] = v
            if v.compute_topography:
                kk = f"{k}:topo"
                if kk not in df.columns:
                    raise KeyError(
                        f"terrain correction data missing for zone '{k}': '{kk}'"
                    )
                tc_args["topography_corrections"] = df[kk].to_numpy()
                consumed_columns.append(kk)

            if v.compute_bathymetry:
                kk = f"{k}:bath"
                if kk not in df.columns:
                    raise KeyError(
                        f"terrain correction data missing for zone '{k}': '{kk}'"
                    )
                tc_args["bathymetry_corrections"] = df[kk].to_numpy()
                consumed_columns.append(kk)

            obj.set_corrections(**tc_args)

        uncomsumed_columns = [
            c
            for c in df.columns
            if c.startswith("tcorr:") and c not in consumed_columns
        ]
        if uncomsumed_columns:
            raise ValueError(
                "terrain corrections parameters and values are inconsistent: "
                f"no parameters for terrain_correction data {uncomsumed_columns}"
            )

        return obj

    def _params_to_dataframe(self, paths_to_str: bool = True) -> pd.DataFrame:
        """Convert the parameters to a DataFrame."""  # noqa: DOC201
        params_df = self._params_to_series().to_frame()
        return params_df.reset_index()

    def _params_to_series(self) -> pd.Series:
        """Convert the parameters to a Series."""  # noqa: DOC201
        pars = []
        for k, v in self.params.items():
            # TerrainCorrectionParameters.to_series()
            s = v.to_series(series_name="parameter_value", index_prefix=[k, "zone"])
            pars.append(s)
        params_sr = pd.concat(pars)
        return params_sr

    def to_excel(
        self,
        fname: FilePath,
        sheet_name: str | None = None,
        params_sheet_name: str | None = None,
        if_workbook_exists: IfWorkbookExists = "error",
        if_sheet_exists: IfSheetExists = "error",
        **kwargs,
    ) -> None:
        """Write the terrain correction data to an Excel file.

        Parameters
        ----------
        fname : str or PathLike
            The path to the output Excel file.
        sheet_name : str, optional
            The name of the excel worksheet to write terrain corrections. If not
            specified then ``'terrain_corrections'`` will be used.
        params_sheet_name : str, optional
            The name of the excel worksheet to write terrain correction parameters.
            If None, then params_sheet_name will be set to ``'{sheet_name}_params'``.
        if_workbook_exists : {'error', 'append', 'replace'}, default='error'
            Action to take if the workbook already exists. Options are:
            'error', 'append', or 'replace'.
        if_sheet_exists : IfSheetExists, default='error'
            Action to take if the sheet already exists. Options are:
            'error', 'replace', or 'overlay'.
        kwargs : dict
            Additional keyword arguments passed to ``pandas.DataFrame.to_excel``.

        """
        kwargs["header"] = kwargs.get("header", True)
        kwargs["index"] = kwargs.get("index", True)

        if sheet_name is None:
            if isinstance(self._default_excel_sheet_name, str):
                sheet_name = self._default_excel_sheet_name
            else:
                sheet_name = self._default_excel_sheet_name[0]

        if params_sheet_name is None:
            params_sheet_name = f"{sheet_name}_params"

        df_tcorr = prepare_writable_df(self.data, normalize_column_names=True)
        write_excel_worksheet(
            df_tcorr,
            fname,
            sheet_name=sheet_name,
            if_workbook_exists=if_workbook_exists,
            if_sheet_exists=if_sheet_exists,
            **kwargs,
        )

        df_params = prepare_writable_df(
            self._params_to_series().to_frame().reset_index(level=1),
            normalize_column_names=True,
        )
        write_excel_worksheet(
            df_params,
            fname,
            sheet_name=params_sheet_name,
            if_workbook_exists="append",
            if_sheet_exists=if_sheet_exists,
            **kwargs,
        )

    @classmethod
    def from_excel(
        cls,
        fname: FilePath,
        sheet_name: str | None = None,
        params_sheet_name: str | None = None,
        **kwargs,
    ) -> "TerrainCorrectionData":
        """Read terrain corrections from an Excel file.

        Parameters
        ----------
        fname : str or PathLike
            The Excel file to read from.
        sheet_name : str, optional
            The name of the excel worksheet from which to read terrain corrections. If
            not specified then 'Terrain Corrections' will be used.
        params_sheet_name : str, optional
            The name of the excel worksheet from which to read terrain correction parameters.
            If not specified then  ``'{sheet_name} Params'`` will be used.
        kwargs : dict
            Additional keyword arguments passed to ``pandas.read_excel``.

        Returns
        -------
        TerrainCorrectionOutput

        """
        if sheet_name is None:
            if isinstance(cls._default_excel_sheet_name, str):
                sheet_name = cls._default_excel_sheet_name
            else:
                sheet_name = cls._default_excel_sheet_name[0]

        if params_sheet_name is None:
            params_sheet_name = f"{sheet_name}_params"

        tcorr_df = read_excel_worksheet(fname, sheet_name=sheet_name, **kwargs)
        params_df = read_excel_worksheet(fname, sheet_name=params_sheet_name, **kwargs)
        return cls.from_dataframe(df=tcorr_df, params=params_df)

    def to_csv(self, fname: FilePath | None = None, **kwargs) -> str | None:
        """Write the terrain correction data to a CSV file.

        The output CSV file includes Terrain correction parameters as headers
        lines prefixed with '#' prefix.

        Parameters
        ----------
        fname : str or PathLike | None
            The path to the output CSV file. If None, then the CSV string is returned.
        kwargs : dict
            Additional keyword arguments passed to ``pandas.DataFrame.to_csv``.

        Returns
        -------
        None | str
            None if CSV written to file, otherwise the CSV string.

        """
        kwargs["header"] = kwargs.get("header", True)
        kwargs["index"] = kwargs.get("index", True)
        csv = []

        params_csv_str = self._params_to_dataframe().to_csv(**kwargs)
        csv.extend([f"# {c}" for c in params_csv_str.splitlines()])

        if not self.data.index.name:
            kwargs["index"] = False
        csv.extend(self.data.to_csv(**kwargs).splitlines())
        csv = "\n".join(csv)
        if fname is None:
            return csv
        with open(fname, "w") as f:
            f.write(csv)

    @classmethod
    def from_csv(
        cls,
        fname: FilePath,
        **kwargs,
    ) -> "TerrainCorrectionData":
        """Read terrain corrections from a CSV file.

        Parameters
        ----------
        fname : str or PathLike
            The input CSV file.
        kwargs :
            Additional keyword arguments passed to ``pandas.read_csv``.

        Returns
        -------
        TerrainCorrectionOutput

        """
        with open(fname, "r") as f:
            lines = f.readlines()
        params = [l.lstrip("#").strip().split(",") for l in lines if l.startswith("#")]
        if len(params) == 0:
            raise ValueError("No terrain correction parameters found in CSV file.")
        params = pd.DataFrame(params[1:], columns=params[0])
        params = params.set_index(params.columns[0])

        tcorrs = pd.read_csv(fname, header=0, comment="#", **kwargs)

        return cls.from_dataframe(
            df=tcorrs,
            params=params,
        )

    def get_corrections(
        self,
        site_id: npt.ArrayLike,
        total_only: bool = False,
        if_missing: Literal["drop", "raise", "fill"] = "drop",
        fill_value: float = np.nan,
    ) -> pd.DataFrame:
        """Get terrain correction values for the specified sites.

        Parameters
        ----------
        site_id : str or array-like of str
            The unique site identifiers for which to get the terrain corrections.
        total_only : bool, default False
            If :const:`True`, return only the total terrain correction values,
            otherwise return all terrain correction values.
        if_missing : {'drop', 'raise', 'fill'}, default 'drop'
            How to handle any ``site_id`` arguments for which there are no
            terrain correction data:

            - If :const:`'drop'`, then missing sites will be dropped from
              the output. A warning will be issued.
            - If :const:`'fill'`, then missing site will be have all fields set to
              ``fill_value``. A warning will be issued.
            - If:const:`'raise'`, then raise an exception.

        fill_value : float, default np.nan
            Value used to fill data for missing sites when ``if_missing='fill'``.

        Returns
        -------
        Series or DataFrame
            The terrain correction values. Will be a Series if ``total_only=True``,
            otherwise a DataFrame is returned.

        """
        if if_missing not in ["drop", "raise", "fill"]:
            raise ValueError(
                f"invalid if_missing arg '{if_missing}'"
            )  # fixed typo in message

        site_id_idx = pd.Index(np.atleast_1d(site_id).astype(str).tolist())
        tcorrs = self.data.reset_index().set_index("site_id")
        if total_only:
            cols = ["tcorr:total"]
        else:
            cols = [c for c in tcorrs.columns if c.startswith("tcorr:")]

        site_id_found = site_id_idx.intersection(tcorrs.index)
        site_id_missing = site_id_idx.difference(tcorrs.index)

        if site_id_missing.empty:
            rval = tcorrs.loc[site_id_found, cols]
        else:
            if site_id_found.empty:
                msg = (
                    "no terrain corrrection data found for any of the "
                    "specified site_id's"
                )
            else:
                msg = (
                    "no terrain corrrection data found for site_id's "
                    f"{site_id_missing.to_list()}"
                )
            if if_missing == "raise":
                raise ValueError(msg)
            if if_missing == "drop":
                warnings.warn(f"{msg}, dropping from ouput")
                rval = tcorrs.loc[site_id_found, cols]
            else:  # must be 'fill'
                warnings.warn(f"{msg}, filling with {fill_value}")
                rval = tcorrs.loc[site_id_found, cols].copy()
                rval_fill = pd.DataFrame(
                    index=site_id_missing, columns=cols, data=fill_value
                )
                rval = pd.concat([rval, rval_fill]).loc[site_id_idx]

        return rval
