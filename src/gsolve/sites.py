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

from __future__ import annotations

import typing as _typing
from collections.abc import Iterable, Mapping, Sequence
from typing import Self

import numpy as _np
import numpy.typing as _npt
import pandas as _pd

from gsolve.core._typing import (
    DatasetOrArray,
    FilePath,
    IfSheetExists,
    IfWorkbookExists,
    Points2D,
    Points3D,
    Renamer,
)
from gsolve.core.data import COMMON_FIELDS, DataFieldSpecification, GSolveTable
from gsolve.core.excel_io import read_excel_worksheet, write_excel_worksheet
from gsolve.core.utils import (
    GSolveDataWarning,
    is_list_like,
    normalize_field_names,
    prepare_writable_df,
    to_1d_ndarray,
)
from gsolve.core.xr_methods import load_dem, prepare_dem

__all__ = [
    "GravitySites",
    "combine_gravity_sites",
    "ReferenceGravity",
    "combine_reference_gravity",
]


class GravitySites(GSolveTable):
    """Class to store gravity site/station data and metadata.

    Gravity site/station information stored as a pandas DataFrame.
    The data fields (i.e. columns) required by GSolve have explicitly
    defined names, dtypes and default values. The preferred method for setting
    fields is to use the :meth:`obj.set_column` method.
    Other fields may be added to ``obj.data`` as required, but will
    be ignored by gsolve.

    The defined fields are:

    **Mandatory Fields** - Must be defined at object creation.

    ==================== ======= =======================================================
    Name                 Type    Description
    ==================== ======= =======================================================
    site_id              *str*   unique site identifier, set as obj.data.index
    latitude             *float* site latitude in decimal degrees.
    longitude            *float* site longitude in decimal degrees.
    height_ellipsoidal   *float* site elevation relative to the ellipsoid in meters.
    ==================== ======= =======================================================

    **Required Fields** - Will be created with default values if not specified. Must be
    at least partially set for computing drift.

    ======================= ======= ========= ===============================================
    Name                    Type    Default   Description
    ======================= ======= ========= ===============================================
    reference_gravity       *float* ``NaN``   Reference gravity value at that site. Typically
                                              absolute gravity, but could be set to some
                                              arbitrary value if no reference gravity data
                                              are available. At least one `reference_gravity`
                                              value must be set for solve for drift.
    gsolve_tie              *bool*  ``False`` Indicates whether a site is to be used as "tie"
                                              when solving for drift. At least one site with
                                              a `reference_gravity` value must be set as a
                                              `gsolve_tie`.
    ======================= ======= ========= ===============================================


    **Cartesian coordinates** - Not required for network adjustment, but are required
    for calculating terrain corrections.

    ==================== ======= =======================================================
    Name                 Type    Description
    ==================== ======= =======================================================
    `easting`            *float* site locations in some cartesian coordinate system
    `northing`           *float* site locations in some cartesian coordinate system
    `height_orthometric` *float* height of site above some datum
    ==================== ======= =======================================================

    Parameters
    ----------
    site_id : ArrayLike of str
        The unique identifier for each site.
    latitude : ArrayLike of float
        The latitude of the site in decimal degrees.
    longitude : ArrayLike of float
        The longitude of the site in decimal degrees.
    height_ellipsoidal : ArrayLike of float
        The height of the site above the ellipsoid in meters.
    reference_gravity : ArrayLike or None, default nan
        The reference gravity value at the site in mGal.
    gsolve_tie : ArrayLike or None, default False
        An array of booleans indicating whether the site will be used as
        a fixed tie when solving for drift.  A valid tie must have a
        non-null reference_gravity value.
    **kwargs : dict[str, ArrayLike]
        Additional fields to be added to the site data.

    """

    _known_fields: dict[str, DataFieldSpecification] = {
        "site_id": COMMON_FIELDS["site_id"],
        "longitude": DataFieldSpecification("longitude", float, _np.nan, True),
        "latitude": DataFieldSpecification("latitude", float, _np.nan, True),
        "height_ellipsoidal": DataFieldSpecification(
            "height_ellipsoidal",
            float,
            _np.nan,
            True,
            legacy_name="elevation",
        ),
        "reference_gravity": DataFieldSpecification(
            "reference_gravity", float, _np.nan, False, legacy_name="gravity"
        ),
        "gsolve_tie": DataFieldSpecification("gsolve_tie", bool, False, False),
        "easting": DataFieldSpecification("easting", float, _np.nan, False),
        "northing": DataFieldSpecification("northing", float, _np.nan, False),
        "height_orthometric": DataFieldSpecification("height_orthometric", float, 0.0),
    }

    _index_field: str = "site_id"
    _default_excel_sheet_name: str | tuple[str, ...] = ("sites", "Locations")

    def __init__(
        self,
        site_id: _npt.ArrayLike,
        latitude: _npt.ArrayLike,
        longitude: _npt.ArrayLike,
        height_ellipsoidal: _npt.ArrayLike,
        reference_gravity: _npt.ArrayLike | float | None = _np.nan,
        gsolve_tie: _npt.ArrayLike | bool | None = False,
        **kwargs: _npt.ArrayLike,
    ) -> None:

        idx = _pd.Index(
            data=to_1d_ndarray(site_id).astype(str),
            name=self._index_field,
            dtype=self._known_fields[self._index_field].dtype,
        )
        if idx.duplicated().any():
            duplicates = idx[idx.duplicated().tolist()].unique().tolist()
            raise ValueError(f"site_id contains duplicated values: {duplicates}")

        self.data = _pd.DataFrame(index=idx, data=None)
        self.set_column("latitude", latitude)
        self.set_column("longitude", longitude)
        self.set_column("height_ellipsoidal", height_ellipsoidal)
        self.set_column("reference_gravity", reference_gravity)
        self.set_column("gsolve_tie", gsolve_tie)

        for k, v in kwargs.items():
            self.set_column(k, v)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_sites={self.data.shape[0]}, "
            f"n_reference_gravity={self.data.reference_gravity.notna().sum()}"
            ")"
        )

    @classmethod
    def from_excel(
        cls,
        excel_file: FilePath,
        sheet_name: str | int | list[int | str] | None = None,
        ignore_unknown_fields: bool = True,
        parse_split_datetime: bool = True,
        mapper: Renamer | None = None,
        **kwargs,
    ) -> "GravitySites":
        """Read a GravitySites object from an excel workbook.

        Parameters
        ----------
        excel_file : str or PathLike
            The excel workbook to read from.
        sheet_name : str | int, optional
            The worksheet name or location within ``excel_file``. If not
            specified. attempt to read from the standard sheet name 'sites'
            and then from the legacy sheet name 'Locations'
        ignore_unknown_fields : bool, default True
            If True, columns that have no defined specification are dropped.
            Use ``GravitySites.known_fields()`` to return a list of the
            defined fields
        mapper : dict or function, optional
            Dict-like or function transformations to apply to column names before
            creating object. See ``DataFrame.rename`` method for full documentation
        kwargs :
            Arguments passed to ``pandas.read_excel`` method.

        Returns
        -------
        Sites
            The GravitySites object created from the excel worksheet.

        """
        if sheet_name is None:
            sheet_name = list(cls._default_excel_sheet_name)

        df = read_excel_worksheet(excel_file, sheet_name, **kwargs)
        df = normalize_field_names(df)

        for f in cls.known_fields():
            if f not in df.columns:
                legacy_name = cls._known_fields[f].legacy_name
                if legacy_name is not None and legacy_name in df.columns:
                    df = df.rename(columns={legacy_name: f})

        return cls.from_dataframe(
            df,
            use_index=False,
            ignore_unknown_fields=ignore_unknown_fields,
            parse_split_datetime=False,
            mapper=mapper,
        )

    def get_ties(
        self, active_only: bool = True, gravity_only: bool = True
    ) -> _pd.DataFrame:
        """Return rows sites that will be used as gsolve ties.

        Parameters
        ----------
        active_only : bool, optional, default is True
            Only , by default True
        gravity_only : bool, optional, default is True
            If `True` only return reference_gravity values, otherwise
            return all fields.

        Returns
        -------
        DataFrame

        """
        if gravity_only:
            cols = ["reference_gravity"]
        else:
            cols = self.data.columns
        if active_only:
            return self.data.loc[self.data["gsolve_tie"], cols]
        else:
            return self.data.loc[self.data["reference_gravity"].notna(), cols]

    def activate_ties(self, site_id: str | _npt.ArrayLike | None = None) -> None:
        """Set one or more "tie" sites as active, i.e. to be used in gsolve.

        Parameters
        ----------
        site_id : str or ArrayLike, default None
            The site_id(s) to be activated. If None, then all sites reference
            gravity are activated.

        """
        if site_id is None:
            _site_id = self.get_ties(False).index.tolist()
        elif isinstance(site_id, str):
            _site_id = [str(site_id)]
        elif is_list_like(site_id) and isinstance(site_id, Iterable):
            _site_id = [str(s) for s in site_id]
        else:
            raise TypeError(
                "site_id must be None, a string, or an array-like of strings"
            )

        self._check_bad_site_ids(_site_id)

        m = self.data.index.isin(_site_id).tolist()
        if self.data.loc[m, "reference_gravity"].isna().any():
            raise ValueError("Cannot activate sites without reference gravity.")
        self.data.loc[m, "gsolve_tie"] = True

    def deactivate_ties(self, site_id: str | _npt.ArrayLike | None = None) -> None:
        """Set one or more "tie" sites as inactive, i.e. not used in gsolve.

        Parameters
        ----------
        site_id : str or ArrayLike, default None.
            The site_id(s) to be deactivated. If None, the all sites
            with reference gravity are  deactivated.

        """
        if site_id is None:
            _site_id = self.get_ties(False).index.tolist()
        elif isinstance(site_id, str):
            _site_id = [str(site_id)]
        else:
            _site_id = to_1d_ndarray(site_id).astype(str).tolist()

        self._check_bad_site_ids(_site_id)
        m = self.data.index.isin(_site_id)
        self.data.loc[m, "gsolve_tie"] = False

    def set_reference_gravity(
        self,
        ref_sites: _typing.Union["ReferenceGravity", _pd.DataFrame, dict],
        reset: bool = False,
    ) -> None:
        """Load reference gravity values into the sites table.

        Parameters
        ----------
        ref_sites : ReferenceGravity | DataFrame | dict
            The reference gravity values to be loaded.
        reset : bool, optional
            Blank any existing reference gravity values, by default False.

        """
        ref_gravity_field = "reference_gravity"
        if reset:
            ref_gravity_default = self._known_fields[ref_gravity_field].default
            self.data.loc[:, ref_gravity_field] = ref_gravity_default
            self.data.loc[:, "gsolve_tie"] = self._known_fields["gsolve_tie"].default

        if isinstance(ref_sites, _pd.DataFrame):
            ref_sites = ReferenceGravity.from_dataframe(ref_sites)
        elif isinstance(ref_sites, Mapping):
            ref_sites = ReferenceGravity.from_dict(ref_sites)

        has_ref_grav = self.data.index.intersection(ref_sites.data.index)
        self.data.loc[has_ref_grav, ref_gravity_field] = ref_sites.data.loc[
            has_ref_grav, "gravity"
        ]
        self.data.loc[has_ref_grav, "gsolve_tie"] = True

    def _check_bad_site_ids(self, site_id: str | _npt.ArrayLike) -> None:
        if isinstance(site_id, (str, bytes)):
            _site_id = [str(site_id)]
        elif is_list_like(site_id):
            _site_id = [str(s) for s in site_id]  # type: ignore[not-iterable, ty:not-iterable]
        else:
            raise TypeError("site_id must be a string or an array-like of strings")

        bad_site_names = [s for s in _site_id if s not in self.data.index]
        if bad_site_names:
            raise ValueError(f"site_id(s) not in existing sites: {bad_site_names}")

    def check_data(self, warn: bool = True) -> bool:
        """Check the data errors.

        Parameters
        ----------
        warn : bool, default=True
            If True, print warnings for common errors.

        Returns
        -------
        bool
            True if data is OK, False otherwise.

        """
        warner = GSolveDataWarning(prefix=f"{type(self).__name__} error", show=warn)

        for c in ["latitude", "longitude", "height_ellipsoidal"]:
            if c not in self.data.columns:
                warner(f"missing required column: '{c}'")
            elif self.data[c].isna().any():
                warner(f"null values in column: '{c}'")

        if (
            "reference_gravity" not in self.data.columns
            or self.data["reference_gravity"].isna().all()
        ):
            warner("no reference gravity values have been set.")

        if "gsolve_tie" not in self.data.columns:
            warner("mising require  column: 'gsolve_tie'")
        else:
            m = self.data["gsolve_tie"].eq(True)
            if not m.any():
                warner(f"no sites are set as ties.")
            elif (
                "reference_gravity" not in self.data.columns
                or self.data.loc[m, "reference_gravity"].isna().any()
            ):
                warner(
                    "'gsolve_tie' is True for sites with no/null "
                    "'reference_gravity' values"
                )

        warner.final_msg()
        return warner.count == 0

    def _get_writable_df(
        self,
        normalize_column_names: bool = True,
        bool_to_int: bool = True,
        include_unknown_fields: bool = False,
    ) -> _pd.DataFrame:
        """Return a DataFrame suitable for writing to a file."""
        cols: list[str] = [
            str(c) for c in self.known_fields() if c in self.data.columns
        ]
        if include_unknown_fields:
            cols.extend(c for c in self.data.columns if c not in cols)

        return prepare_writable_df(
            self.data.loc[:, cols],
            normalize_column_names=normalize_column_names,
            bool_to_int=bool_to_int,
        )

    def write_to_csv(
        self,
        fname: FilePath,
        normalize_column_names: bool = True,
        expand_datetime: str | None = None,
        drop_datetime: bool = False,
        bool_to_int: bool = True,
        include_unknown_fields: bool = False,
        **kwargs,
    ) -> None:
        """Write `data` DataFrame to csv file.

        Parameters
        ----------
        csv_file : str or PathLike
            The path to the excel file.
        normalize_column_names : bool, default True
            Convert columns name to snake case.
        bool_to_int : bool, default True
            Convert boolean True/False to 1,0.
        include_unknown_fields : bool, default False
            Include fields not in the known fields.
        **kwargs
            Additional keyword arguments passed to `pandas.DataFrame.to_csv`.

        See Also
        --------
        pandas.DataFrame.to_csv
            The underlying function used to write the DataFrame to a csv.

        """
        kwargs["header"] = kwargs.get("header", True)
        kwargs["index"] = kwargs.get("index", True)

        self._get_writable_df(
            normalize_column_names=normalize_column_names,
            bool_to_int=bool_to_int,
            include_unknown_fields=include_unknown_fields,
        ).to_csv(fname, **kwargs)

    def to_excel(
        self,
        excel_file: FilePath,
        sheet_name: str | None = None,
        normalize_column_names: bool = True,
        bool_to_int: bool = True,
        include_unknown_fields: bool = False,
        if_workbook_exists: IfWorkbookExists = "error",
        if_sheet_exists: IfSheetExists = "error",
        **kwargs,
    ) -> None:
        """Write `data` DataFrame to an excel file.

        Parameters
        ----------
        excel_file : str or PathLike
            The excel workbook to write to.
        sheet_name : str, default None
            The name of the worksheet to write to.
        normalize_column_names : bool, default True
            Convert columns name to snake case.
        bool_to_int : bool, default True
            Convert boolean True/False to 1,0.
        include_unknown_fields : bool, default False
            Include fields not in the known fields.
        if_workbook_exists : {"error", "replace", "append"}, default "error"
            Behaviour if the excel file already exists.
        if_sheet_exists : {"error", "replace", "new"}, default "error"
            Behaviour if the worksheet already exists.
        **kwargs
            Additional keyword arguments passed to `pandas.DataFrame.to_excel`.

        See Also
        --------
        gsolve.core.excel_io._core_excel_io.write_excel_worksheet
            For complete explanation of parameters `if_workbook_exists`
            and `if_sheet_exists`.
        pandas.DataFrame.to_excel
            The underlying function used to write the DataFrame to the
            excel file.

        """
        if sheet_name is None:
            if isinstance(self._default_excel_sheet_name, str):
                sheet_name = self._default_excel_sheet_name
            else:
                sheet_name = self._default_excel_sheet_name[0]

        kwargs["header"] = kwargs.get("header", True)
        kwargs["index"] = kwargs.get("index", True)

        write_excel_worksheet(
            self._get_writable_df(
                normalize_column_names=normalize_column_names,
                bool_to_int=bool_to_int,
                include_unknown_fields=include_unknown_fields,
            ),
            excel_file=excel_file,
            sheet_name=sheet_name,
            if_workbook_exists=if_workbook_exists,
            if_sheet_exists=if_sheet_exists,
            **kwargs,
        )

    def sample_elevation(
        self,
        dem: DatasetOrArray | FilePath,
        output_col: str | None = None,
        xcol: str = "easting",
        ycol: str = "northing",
        method: str = "nearest",
    ) -> None | _pd.Series:
        """Get elevations at site locations from an DEM/xarray grid.

        Parameters
        ----------
        dem : xarray.DataArray, xarray.Dataset, str or PathLike
            The array of values to sample. If `dem` is not a
        output_col : str or None, default None
            If `output_col` is defined, write sampled values to `obj.data[output_col]`.
            If `output_col` is None, return a Series of sampled values.
        xcol : str, optional
            The column holding x coordinates, by default "easting"
        ycol : str, optional
            The column holding y coordinates, by default "northing"
        method : str, default "nearest"
            The interpolation method used.  See ``xarray.DataArray.interp``
            for available options.

        Returns
        -------
        elevations : Series or None
            None if `output_col` is defined, otherwise a Series of sampled elevations.

        """
        if isinstance(dem, FilePath):
            _dem = load_dem(dem)
        elif isinstance(dem, DatasetOrArray):
            _dem = prepare_dem(dem)
        else:
            raise TypeError("dem must be file path or an xarray Dataset/DataArray")

        z = (
            _dem.interp(
                {_dem.dims[0]: self.data[ycol], _dem.dims[1]: self.data[xcol]},
                method=method,  # type: ignore[invalid-argument-type, ty:invalid-argument-type]
            )
            .to_numpy()
            .diagonal()
        )
        if output_col is not None:
            self.set_column(output_col, z)
        else:
            return _pd.Series(index=self.data.index, data=z)

    def get_points(
        self, xcol: str, ycol: str, zcol: str | None = None
    ) -> tuple[
        _npt.NDArray[_np.float64], _npt.NDArray[_np.float64], _npt.NDArray[_np.float64]
    ]:
        x = self.data.loc[:, xcol].to_numpy().copy().astype(_np.float64)
        y = self.data.loc[:, ycol].to_numpy().copy().astype(_np.float64)
        if zcol is None:
            return x, y, _np.full_like(x, _np.nan, dtype=_np.float64)
        else:
            return x, y, self.data.loc[:, zcol].to_numpy().copy().astype(_np.float64)


def _siteid_exists(site_id: str, other: _pd.DataFrame | GSolveTable) -> bool:
    if isinstance(other, GSolveTable):
        other = other.data
    return site_id in other.index


class ReferenceGravity(GSolveTable):
    """Class providing a simple mechanism for merging reference gravity data.

    Attributes
    ----------
    data : DataFrame
        The reference gravity data indexed by `site_id`. The defined fields
        are:

            - ``'gravity'`` : (float) The reference gravity value for the site.
            - ``'active'`` : (bool) Indicates whether the site should be used
                as an active "gsolve_tie" when merged into a ``GravitySites``.

        Other fields may be added to ``obj.data`` as required, but will
        be ignored by gsolve.

    Parameters
    ----------
    site_id : array_like
        The unique site identifier.  Will be converted to str.
    gravity : array_like
        The reference gravity value for each site.
    active, array_like or bool, default True
        An array indicating whether a site should be set as an active
        "gsolve_tie" when merged into a ``GravitySites``.
    **kwargs : dict[str, array_like]
        Additional fields to be added to the site data.

    """

    _known_fields: dict[str, DataFieldSpecification] = {
        "site_id": DataFieldSpecification("site_id", str, "", True),
        "gravity": DataFieldSpecification("gravity", float, _np.nan, True),
        "active": DataFieldSpecification("active", bool, True, False),
    }
    _index_field: str = "site_id"
    _default_excel_sheet_name: str | tuple[str, ...] = ("reference_sites", "Tie_Data")

    def __init__(
        self,
        site_id: _npt.ArrayLike,
        gravity: _npt.ArrayLike,
        active: _npt.ArrayLike | bool = True,
        **kwargs: dict[str, _npt.ArrayLike],
    ) -> None:
        _site_id = to_1d_ndarray(site_id).astype(str)

        idx = _pd.Index(
            data=_site_id,
            name=self._index_field,
            dtype=self._known_fields[self._index_field].dtype,
        )

        # catch duplicate site_id
        if idx.duplicated().any():
            duplicates = idx[idx.duplicated().tolist()].unique().tolist()
            raise ValueError(
                "creating ReferenceGravity object: "
                f"site_id field contains duplicated values: {duplicates}"
            )

        # catch empty site_id
        if idx.isna().any() or (idx == "").any():  # type: ignore[unresolved-attribute, ty:unresolved-attribute]
            m = idx.isna() | (idx == "")
            empty = _pd.Series(m)
            empty = empty.loc[m.tolist()].index.to_list()
            raise ValueError(
                "creating ReferenceGravity object: "
                f"site_id field contains empty values at rows: {empty}"
            )

        self.data = _pd.DataFrame(index=idx, data=None)
        self.set_column("gravity", gravity)
        self.set_column("active", active)

        if self.data["gravity"].isna().any():
            nodata = self.data.loc[self.data["gravity"].isna()].index.to_list()
            raise ValueError(
                "creating ReferenceGravity object: "
                f"gravity field contains null values for sites: {nodata}"
            )

        for k, v in kwargs.items():
            self.set_column(k, v)

    def _get_writable_df(
        self,
        normalize_column_names: bool = True,
        include_unknown_fields: bool = False,
        bool_to_int: bool = True,
    ) -> _pd.DataFrame:
        """Return a DataFrame suitable for writing to a file."""
        cols = [c for c in self.known_fields() if c in self.data.columns]
        if include_unknown_fields:
            cols.extend(c for c in self.data.columns if c not in cols)

        return prepare_writable_df(
            self.data.loc[:, cols],
            normalize_column_names=normalize_column_names,
            bool_to_int=bool_to_int,
        )

    def write_to_csv(
        self,
        fname: FilePath,
        normalize_column_names: bool = True,
        expand_datetime: str | None = None,
        drop_datetime: bool = False,
        bool_to_int: bool = True,
        include_unknown_fields: bool = False,
        **kwargs,
    ) -> None:
        kwargs["header"] = kwargs.get("header", True)
        kwargs["index"] = kwargs.get("index", True)
        self._get_writable_df(
            normalize_column_names=normalize_column_names,
            bool_to_int=bool_to_int,
            include_unknown_fields=include_unknown_fields,
        ).to_csv(fname, **kwargs)

    def to_excel(
        self,
        excel_file: FilePath,
        sheet_name: str | None = None,
        normalize_column_names: bool = True,
        bool_to_int: bool = True,
        include_unknown_fields: bool = False,
        if_workbook_exists: IfWorkbookExists = "error",
        if_sheet_exists: IfSheetExists = "error",
        **kwargs,
    ) -> None:
        """Write data to an excel file.

        Parameters
        ----------
        fname : str or PathLike
            The path to the excel file.
        sheet_name : str, default None
            The name of the worksheet to write to.
        normalize_column_names : bool, default True
            Convert columns name to snake case.
        bool_to_int : bool, default True
            Convert boolean True/False to 1,0.
        include_unknown_fields : bool, default False
            Include fields not in the known fields.
        if_workbook_exists : {"error", "replace", "append"}, default "error"
            Behaviour if the excel file already exists.
        if_sheet_exists : {"error", "replace", "new"}, default "error"
            Behaviour if the worksheet already exists.
        **kwargs
            Additional keyword arguments passed to `pandas.DataFrame.to_excel`.

        See Also
        --------
        gsolve.core.excel_io.write_excel_worksheet
            For complete explanation of parameters `if_workbook_exists`
            and `if_sheet_exists`.
        pandas.DataFrame.to_excel
            The underlying function used to write the DataFrame to the
            excel file.

        """
        if sheet_name is None:
            if isinstance(self._default_excel_sheet_name, str):
                sheet_name = self._default_excel_sheet_name
            else:
                sheet_name = self._default_excel_sheet_name[0]

        kwargs["header"] = kwargs.get("header", True)
        kwargs["index"] = kwargs.get("index", True)

        write_excel_worksheet(
            self._get_writable_df(
                normalize_column_names=normalize_column_names,
                bool_to_int=bool_to_int,
                include_unknown_fields=include_unknown_fields,
            ),
            excel_file=excel_file,
            sheet_name=sheet_name,
            if_workbook_exists=if_workbook_exists,
            if_sheet_exists=if_sheet_exists,
            **kwargs,
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping,
        set_active: bool = True,
    ) -> Self:
        """Create a ReferenceGravity object from a dictionary.

        This method provides a simple mechanism for users to add reference
        gravity data to a GravitySites object.

        Parameters
        ----------
        data : dict of float or dict of (float, bool)
            Reference site data as a dictionary where keys are ``'site_id'``
            and values are either the reference gravity (float) or a sequence
            of (reference gravity, active) where active is a boolean

        Returns
        -------
        ReferenceGravity
            The created object.

        """
        site_ids = []
        ref_grav = []
        active = []

        set_active = bool(set_active)
        for k, v in data.items():
            site_ids.append(str(k))
            if is_list_like(v):
                ref_grav.append(float(v[0]))
                if len(v) >= 2:
                    active.append(bool(v[1]))
                else:
                    active.append(set_active)
            else:
                ref_grav.append(float(v))
                active.append(set_active)

        return cls(site_id=site_ids, gravity=ref_grav, active=active)


def combine_gravity_sites(
    sites: Sequence[GravitySites],
    duplicates: _typing.Literal["drop", "error"] = "drop",
) -> GravitySites:
    """Combine two or more GravitySites objects into a single object.

    Parameters
    ----------
    sites : GravitySites
        The sites to be combined.
    duplicates : {'drop', 'error'}, default is "drop"
        How to behave if any site_id's are duplicated.
        - "drop": drop the duplicates.
        - "error": raise a ValueError.

    Returns
    -------
    GravitySites
        The new GravitySites sites object

    """
    if not is_list_like(sites) or len(sites) < 2:
        raise ValueError("Must specify at least 2 GravitySites objects.")

    valid_duplicates_args = {"drop", "error"}
    if duplicates not in valid_duplicates_args:
        raise ValueError(
            f"duplicates must be one of {valid_duplicates_args}, not '{duplicates}'"
        )

    for s in sites:
        if not isinstance(s, GravitySites):
            raise TypeError(
                f"All arguments must be {GravitySites.__name__} objects, "
                f"not '{type(s)}'"
            )

    final_df = _pd.concat([o.data for o in sites])

    if final_df.index.duplicated().any():
        if duplicates == "drop":
            final_df = final_df[~final_df.index.duplicated(keep="first")]
        elif duplicates == "error":
            dupe_idx = final_df.index[final_df.index.duplicated().tolist()].unique()
            dupe_idx = [str(d) for d in dupe_idx]
            raise ValueError(
                f"Duplicate 'site_id' values found in merged data: {dupe_idx}"
            )

    return GravitySites.from_dataframe(final_df)


def combine_reference_gravity(
    ref_sites: Sequence[ReferenceGravity],
    duplicates: _typing.Literal["drop", "error"] = "drop",
) -> ReferenceGravity:
    """Combine two or more ReferenceSite objects into a single object.

    Parameters
    ----------
    ref_sites : list-like
        The Reference gravity objects to be combined.
    duplicates : {'drop', 'error'}, default is 'drop'
        How to handle duplicate site_id's.  If ``duplicates='drop'``, then
        keep the first occurrence and drop the rest.  If ``duplicates='error'``, then
        raise a ValueError.

    Returns
    -------
    ReferenceGravity
        The new ReferenceGravity object.

    """
    if not is_list_like(ref_sites) or len(ref_sites) < 2:
        raise ValueError("Must specify at least 2 ReferenceGravity objects.")

    valid_duplicates_args = {"drop", "error"}
    if duplicates not in valid_duplicates_args:
        raise ValueError(
            f"duplicates must be one of {valid_duplicates_args}, not '{duplicates}'"
        )

    for s in ref_sites:
        if not isinstance(s, ReferenceGravity):
            raise TypeError(
                f"All arguments must be ReferenceGravity objects, not '{type(s)}'"
            )

    final_df = _pd.concat([o.data for o in ref_sites])

    if final_df.index.duplicated().any():
        if duplicates == "drop":
            final_df = final_df[~final_df.index.duplicated(keep="first")]
        elif duplicates == "error":
            dupe_idx = final_df.index[final_df.index.duplicated().tolist()].unique()
            dupe_idx = [str(d) for d in dupe_idx]
            raise ValueError(
                f"Duplicate 'site_id' values found in merged data: {dupe_idx}"
            )

    return ReferenceGravity.from_dataframe(final_df)
