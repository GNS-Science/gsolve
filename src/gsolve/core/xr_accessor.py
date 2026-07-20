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

"""An accessor class for ``xarray.DataArray`` objects."""

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from gsolve.core._typing import (
    Points2D,
    SitesLike,
    TCorrDistanceMaskType,
)
from gsolve.core.utils import GSolveDataWarning, round_coords


# Todo: rename accessor key to 'gsolve'?
@xr.register_dataarray_accessor("tcorr")
class TCorrMethods:
    """
    A class to providing accessor methods for ``xarray.DataArray`` objects.

    This effectively extends the ``xarray.DataArray`` class by adding the ``tcorr``
    accessor, which provides various properties and methods.  The accessor
    is primarily used in the context of terrain correction calculations,
    but could be generally useful for working with 2D DataArrays.
    """

    def __init__(self, xarray_dataarray: xr.DataArray) -> None:
        self._obj = xarray_dataarray

    @property
    def ydim(self) -> str:
        """y dimension name."""  # noqa: D403
        return str(self._obj.dims[0])

    @property
    def xdim(self) -> str:
        """x dimension name."""  # noqa: D403
        return str(self._obj.dims[1])

    @property
    def dx(self) -> float:
        """Node separation in x direction."""
        x = self.xc
        if len(x) < 2:
            return 1.0  # Avoid IndexError for single column
        return float(x[1] - x[0])

    @property
    def dy(self) -> float:
        """Node separation in y direction."""
        y = self.yc
        if len(y) < 2:
            return 1.0  # Avoid IndexError for single row
        return float(y[1] - y[0])

    @property
    def xc(self) -> np.ndarray:
        """X coordinates as a numpy array."""
        return self._obj[self.xdim].values

    @property
    def yc(self) -> np.ndarray:
        """Y coordinates as a numpy array."""
        return self._obj[self.ydim].values

    @property
    def bounds(self) -> np.ndarray:
        """Return array extent as an ndarray of form (xmin, ymin, xmax, ymax).

        Returns
        -------
        np.ndarray: [(xmin, ymin, xmax, ymax)]
        """
        x, y = self.cell_edges()
        return np.asarray([x[0], y[0], x[-1], y[-1]])

    def coords_to_indices(
        self, point: tuple[ArrayLike, ArrayLike]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert cartesian coordinates ``(x, y)`` to array indices ``(i, j)``.

        Note that arrays are row-major, so ``i`` is the ``y`` dimension.
        Accepts scalars or arrays for x and y.

        Parameters
        ----------
        point : Sequence[ArrayLike, ArrayLike]
            Cartesian coordinates (x, y) to convert to array indices.

        Returns
        -------
        i, j : tuple[np.ndarray, np.ndarray]
            Array indices (i, j) corresponding to the input coordinates.
        """
        y = np.atleast_1d(point[1])
        x = np.atleast_1d(point[0])
        i = round_coords((y - self.yc.min()) / self.dy).astype(int)
        j = round_coords((x - self.xc.min()) / self.dx).astype(int)
        return i, j

    def clip_to_points(
        self,
        points: SitesLike | Points2D | xr.DataArray,
        max_dist: float = 0.0,
        error_if_outside_bounds: bool = True,
    ) -> xr.DataArray | None:
        """Return a copy of this array clipped to the extent of points + max_dist.

        Parameters
        ----------
        points : SitesLike | Points2D | xr.DataArray
            The points defining the clipping region. Can be:

              - SitesLike object providing xdim/ydim columns
              - 2-tuple/sequence of (x, y) arrays or scalars
              - DataArray : use coordinate arrays as x & y.
        max_dist : float, default is 0.0
            Extend clip region defined by points by ``max_dist`` in all directions.
        error_if_outside_bounds : bool, default is True
            If True, raise ValueError if requested clip region lies outside

        Returns
        -------
        xarray.DataArray
            The clipped dataarray.
        """
        if isinstance(points, SitesLike):
            if (
                self.xdim not in points.data.columns
                or self.ydim not in points.data.columns
            ):
                raise TypeError(
                    "GravitySites object missing required point columns: "
                    f"{self.xdim}, {self.ydim}"
                )
            x = points.data[self.xdim].to_numpy()
            y = points.data[self.ydim].to_numpy()

        elif isinstance(points, xr.DataArray):
            if points.dims != self._obj.dims:
                raise ValueError("DataArray's have incompatible dimensions")
            x = points.tcorr.xc
            y = points.tcorr.yc
        else:
            x = np.atleast_1d(points[0])
            y = np.atleast_1d(points[1])

        ii, jj = self.coords_to_indices((x, y))
        i_dist = int(np.ceil(max_dist / self.dy))
        j_dist = int(np.ceil(max_dist / self.dx))
        i0 = int(ii.min()) - i_dist
        i1 = int(ii.max()) + i_dist + 1
        j0 = int(jj.min()) - j_dist
        j1 = int(jj.max()) + j_dist + 1
        if i0 < 0 or i1 > self._obj.shape[0] or j0 < 0 or j1 > self._obj.shape[1]:
            if error_if_outside_bounds:
                return None
            else:
                i0 = max(i0, 0)
                i1 = min(i1, self._obj.shape[0])
                j0 = max(j0, 0)
                j1 = min(j1, self._obj.shape[1])

        return self._obj.isel(
            {self.ydim: slice(i0, i1), self.xdim: slice(j0, j1)}
        ).copy()

    def get_land_sea_mask(
        self, sea_level_elevation: float = 0.0, name: str = "mask"
    ) -> xr.DataArray:
        """Generate a mask array where "land" areas are True and "sea" areas are False.

        Parameters
        ----------
        sea_level_elevation : float, default 0.0
            The theshold elevation.
        name : str, defaut = 'mask'
            The name of the output DataArray.

        Returns
        -------
        xarray.DataArray
        """
        da = (
            xr.where(self._obj >= sea_level_elevation, 1, 0)
            .astype(bool)
            .rename(name)
            .drop_attrs()
        )
        return da

    def get_bathymetry_elevation(
        self,
        land_sea_mask: xr.DataArray | None = None,
        sea_level_elevation: float = 0.0,
        name: str = "depth",
    ) -> xr.DataArray:
        """Return an array with all elevations > sea level set to sea level.

        Parameters
        ----------
        land_sea_mask : xr.DataArray | None, default is None
            a boolean mask where land is True and sea is False. If None, the mask
            will be computed from sea_level_elevation.
        sea_level_elevation : float, default is 0.0
            The threshold elevation.
        name : str, default is 'depth'
            The name of the output DataArray.

        Returns
        -------
        xr.DataArray
        """
        land_sea_mask = (
            land_sea_mask
            if land_sea_mask is not None
            else self.get_land_sea_mask(sea_level_elevation)
        )
        da = (
            xr.where(~land_sea_mask, self._obj, sea_level_elevation)
            .rename(name)
            .drop_attrs()
        )
        return da

    def generate_bathymetry_density(
        self,
        land_sea_mask: xr.DataArray | None = None,
        terrain_density: float = 2670.0,
        water_density: float = 1030.0,
        sea_level_elevation: float = 0.0,
        name: str = "density",
    ) -> xr.DataArray:
        """
        Generate a bathymetry density grid from a boolean mask grid or DEM.

        The output density DataArray can be used in calculating bathymetric terrain
        corrections.  Bathymety cells are assigned a density of
        terrain_density - water_density. Density in topograpjy cells is set to 0.0.

        Parameters
        ----------
        land_sea_mask : xarray.DataArray, None, optional
            A boolean mask where land/topgraphy is True and sea/bathymetry is
            False. If None, a mask will be generated from the calling DataArray
            using ``sea_level_elevation`` as the threshold.
        terrain_density : float, default is 2670.0
            The density of terrain (kg/m^3).
        water_density : float, default is 1030.0
            The density of water (kg/m^3).
        sea_level_elevation : float, default is 0.0
            The threshold elevation (m). Ignored if ``land_sea_mask`` is provided.
        name : str, default is 'density'
            The name of the output DataArray.

        Returns
        -------
        xarray.DataArray
            Density grid with same dimensions as the calling DataArray.
        """
        land_sea_mask = (
            land_sea_mask
            if land_sea_mask is not None
            else self.get_land_sea_mask(sea_level_elevation)
        )
        da = (
            xr.where(~land_sea_mask, terrain_density - water_density, 0.0)
            .rename(name)
            .drop_attrs()
        )
        return da

    def get_topography_elevation(
        self,
        land_sea_mask: xr.DataArray | None = None,
        sea_level_elevation: float = 0.0,
        name: str = "elevation",
    ) -> xr.DataArray:
        """Return an array with all elevations < sea level set to sea level.

        This will be used in computing terrain corrections for topography only

        Parameters
        ----------
        land_sea_mask : xr.DataArray | None, default is None
            a boolean mask where land is True and sea is False. If None, the mask
            will be computed using ``sea_level_elevation``.
        sea_level_elevation : float, default is 0.0
            The threshold elevation in 'm'.
        name : str, default is 'elevation'
            The name of the output DataArray.

        Returns
        -------
        xr.DataArray
        """
        land_sea_mask = (
            land_sea_mask
            if land_sea_mask is not None
            else self.get_land_sea_mask(sea_level_elevation)
        )
        da = (
            xr.where(land_sea_mask, self._obj, sea_level_elevation)
            .rename(name)
            .drop_attrs()
        )

        return da

    def generate_topo_density(
        self, terrain_density: float = 2670.0, name: str = "density"
    ) -> xr.DataArray:
        """Generate a terrain density grid.

        Output is effectively a copy of the calling DataArray with all values
        set to ``terrain_density``.

        Parameters
        ----------
        terrain_density : float, default is 2670.0
            The density of the terrain (kg/m^3).
        name : str, default is 'density'
            The name of the output DataArray.

        Returns
        -------
        xarray.DataArray
            A grid with same dimensions as calling DataArray.
        """
        da = xr.full_like(self._obj, terrain_density).drop_attrs().rename(name)
        return da

    def clip_to_arr(self, other: xr.DataArray, clip_other: bool = True) -> xr.DataArray:
        """Clip calling DataArray to the dimensions of ``other``, or vice versa.

        Parameters
        ----------
        other : xarray.DataArray
            The DataArray to be clipped or used as the source for clipping.
        clip_other : bool, default True
            If True, ``other`` is clipped to the dimensions of the calling
            DataArray.  If False, then the calling DataArray is clipped to
            the dimensions of ``other``.

        Returns
        -------
        xarray.DataArray
            The clipped DataArray.
        """
        if clip_other:
            src = self._obj
            target = other
        else:
            target = self._obj
            src = other

        return target.sel(
            {
                self.ydim: src.coords[self.ydim],
                self.xdim: src.coords[self.xdim],
            }
        ).copy()

    def is_compatible(self, other: xr.DataArray) -> bool:
        """
        Test that caller and ``other`` have same dimensions and size.

        Parameters
        ----------
        other : xarray.DataArray
            The DataArray to compare with the caller.

        Returns
        -------
        bool
            True if DataArrays are compatible, False otherwise.
        """
        if (
            self._obj.ndim != other.ndim
            or self.xdim != other.tcorr.xdim
            or self.ydim != other.tcorr.ydim
            or any(a != b for a, b in zip(self._obj.shape, other.shape))
        ):
            return False
        return bool(
            np.all(self.xc == other.tcorr.xc) and np.all(self.yc == other.tcorr.yc)
        )

    def generate_distance_mask(
        self,
        point: tuple[float, float] | None = None,
        min_dist: float = 0.0,
        max_dist: float | None = None,
        mask_type: TCorrDistanceMaskType = "radial",
    ) -> np.ndarray:
        """Generate a boolean distance mask from a specified point.

        Parameters
        ----------
        point : (float, float), optional
            The (x, y) coordinates of the point to measure distance from.
            If not specified, the centerpoint of the calling DataArray is used.
        min_dist : float, default is 0.0
            Cells within ``min_dist`` of ``point`` are set to False.
            The default 0.0 means no minimum distance is applied.
        max_dist : float, optional
            If specified, cells beyond ``max_dist`` from ``point`` are
            set to False
        mask_type : {'radial', 'rectangular'}, default is 'radial'
            The type of distance mask to generate. If 'radial', create a
            circular mask centered on ``point``. If 'rectangular', create
            a square mask centered on ``point``.

        Returns
        -------
        np.ndarray
            A boolean array of same dimensions as the calling DataArray.
        """
        if mask_type not in ("radial", "rectangular"):
            raise ValueError(
                f"mask_type must be 'radial' or 'rectangular', not '{mask_type}'"
            )
        if max_dist is not None and max_dist <= min_dist:
            raise ValueError(f"invalid {max_dist=}, must be > {min_dist=}")
        if min_dist < 0.0:
            raise ValueError(f"invalid {min_dist=}, must be >= 0.0")

        if point is None:
            point = (
                float((self.xc.max() - self.xc.min()) / 2 + self.xc.min()),
                float((self.yc.max() - self.yc.min()) / 2 + self.yc.min()),
            )
        px, py = point  # px is x, py is y
        mask = np.ones(self._obj.shape, dtype=bool)
        if min_dist == 0.0 and max_dist is None:
            return mask

        min_dist_incr = max(min_dist - 0.5 * self.dx, 0.0) ** 2
        max_dist_incr = (
            (max_dist + 0.5 * self.dx) ** 2 if max_dist is not None else None
        )

        a, b = np.meshgrid((self.yc - py) ** 2, (self.xc - px) ** 2, indexing="ij")
        if mask_type == "radial":
            d = a + b
            if min_dist_incr > 0.0:
                mask[d < min_dist_incr] = False
            if max_dist_incr is not None:
                mask[d > max_dist_incr] = False

        if mask_type == "rectangular":
            if min_dist_incr > 0.0:
                mask[(a < min_dist_incr) & (b < min_dist_incr)] = False
            if max_dist_incr is not None:
                mask[(a > max_dist_incr) | (b > max_dist_incr)] = False

        return mask

    def apply_mask(self, mask: np.ndarray, fill_value: float = 0.0) -> xr.DataArray:
        """Mask calling DataArray and set masked elements to ``fill_value``.

        Primarily used to apply a radial or rectangular distance mask to a density
        grid before computing terrain corrections.

        Parameters
        ----------
        mask : numpy.ndarray
            A boolean array of the same shape as calling DataArray.
        fill_value : float, default is 0.0
            Set masked elements to this value.

        Returns
        -------
        xr.DataArray
            A copy of calling DataArray with masked elements set to ``fill_value``.
        """
        return self._obj.where(mask, fill_value)

    def cell_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Return coordinates of cell edges.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            The x and y coordinates of cell edges.
        """
        x = self.xc
        y = self.yc
        dx = self.dx
        dy = self.dy
        if len(x) == 1:
            x_edges = np.array([x[0] - 0.5, x[0] + 0.5])
        else:
            x_edges = np.linspace(x[0] - dx / 2, x[-1] + dx / 2, len(x) + 1)
        if len(y) == 1:
            y_edges = np.array([y[0] - 0.5, y[0] + 0.5])
        else:
            y_edges = np.linspace(y[0] - dy / 2, y[-1] + dy / 2, len(y) + 1)
        return x_edges, y_edges
