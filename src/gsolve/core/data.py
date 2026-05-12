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

"""Base class and function definitions for Gsolve data structures."""

import dataclasses
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any, Self

import numpy.typing as _npt
import pandas as _pd

from gsolve.core._typing import (
    FilePath,
    IfSheetExists,
    IfWorkbookExists,
    Renamer,
)
from gsolve.core.excel_io import (
    parse_sheet_name_arg,
    read_excel_worksheet,
    write_excel_worksheet,
)
from gsolve.core.utils import (
    is_list_like,
    merge_datetime_columns,
    normalize_field_names,
    prepare_writable_df,
    to_naive_utc_datetime,
)

__all__ = ["GSolveTable", "GSolveParameters", "DataFieldSpecification", "COMMON_FIELDS"]

## constants
TERRAIN_DENSITY: float = 2670.0
WATER_DENSITY: float = 1030.0


@dataclasses.dataclass
class DataFieldSpecification:
    """Class to define attributes for a data "field".

    Attributes
    ----------
    name : str
        The name of the data field.
    dtype : Any, default None
        The data type of the field. If None, no type conversion will be applied.
    default : Any, default None.
        The default value of the field.
    required : bool, default False.
        Indicates if the field is required.
    legacy_name : str | None, default None
        Legacy name of the field, used in parsing data prepared for
        older versions of gsolve. May disappear in future versions.
    converter : Callable | None, default None
        A function to convert input data to the specified dtype.
    """

    name: str
    dtype: Any
    default: Any = None
    required: bool = False
    legacy_name: str | None = None
    converter: Callable | None = None


_COMMON_FIELDS: list[DataFieldSpecification] = [
    DataFieldSpecification("site_id", str, required=True, legacy_name="station"),
    DataFieldSpecification("obs_id", str, required=True),
    DataFieldSpecification("loop", str, required=True),
    DataFieldSpecification(
        name="datetime",
        dtype="datetime",
        default=_pd.NaT,
        required=True,
        converter=to_naive_utc_datetime,
    ),
    DataFieldSpecification("active", bool, default=True, required=False),
]
# TODO: make this a class?
COMMON_FIELDS: dict[str, DataFieldSpecification] = {f.name: f for f in _COMMON_FIELDS}


class GSolveTable:
    """
    Base class for various GSolve tabular data structures.

    Attributes
    ----------
    _known_fields :  dict[str, DataFieldSpecification]
        Dictionary of known fields with specifications.
    _default_excel_sheet_name : str
        Default sheet name for Excel I/O operations.
    data : _pd.DataFrame
        The primary data storage object.


    """

    _known_fields: dict[str, DataFieldSpecification]
    _default_excel_sheet_name: str | tuple[str, ...] = ""

    def __init__(self) -> None:
        self.data: _pd.DataFrame
        pass

    def __repr__(self) -> str:
        rval = []
        if hasattr(self, "data"):
            rval.append(f"data:shape={self.data.shape}")
        if hasattr(self, "params"):
            if isinstance(self.params, GSolveParameters):
                rval.append(self.params.__param_str__())

        rval = ", ".join(rval)

        return f"{type(self).__name__}({rval})"

    def __bool__(self) -> bool:
        return (
            hasattr(self, "data")
            and isinstance(self.data, _pd.DataFrame)
            and not self.data.empty
        )

    def __len__(self) -> int:
        return len(self.data) if self else 0

    def __copy__(self) -> Self:
        """Ensure all copies are deep copies."""
        return deepcopy(self)

    def copy(self) -> Self:
        """Return a deep copy of object."""
        return self.__copy__()

    @classmethod
    def known_fields(cls) -> list[str]:
        """Return a list of known fields in a GSolveTable or subclass."""
        fields = [str(k) for k in getattr(cls, "_known_fields", {}).keys()]
        return fields

    @classmethod
    def required_fields(cls) -> list[str]:
        """Return a list of required fields in a GSolveTable or subclass."""
        if cls.known_fields():
            return [k for k, v in getattr(cls, "_known_fields").items() if v.required]
        return []

    def set_column(
        self,
        label: str,
        data: Any | None = None,  # noqa: ANN401
        default: Any | None = None,  # noqa: ANN401
        dtype: str | type | None = None,
    ) -> None:
        """
        Set a column in the ``obj.data`` attribute.

        This is the preferred method to set data columns on ``obj.data``. If ``label`` is
        a "known field" with a defined ``DataFieldSpecification``, then default values,
        type conversion will be applied correctly.

        Parameters
        ----------
        label : str
            The column name to be set.
        data : ArrayLike or None
            The data to be set in the column. If None, then use the ``default`` value.
        default : Any, default None
            The default value for missing entries in the column.
            If ``default`` is None, then:

              - if ``label`` is a "known field" with a defined  default value, use that
                as the default value.
              - if ``label`` is not a "known field", then raise a ValueError.
        dtype : Any, default None
            The data type of the column. Defaults to None, do
        """

        for attrname in ("_custom_fields", "_known_fields"):
            fields = getattr(self, attrname, {})
            if label in fields:
                fs: DataFieldSpecification = fields[label]
                dtype = dtype if dtype is not None else fs.dtype
                default = default if default is not None else fs.default
                break

        if dtype == "datetime":
            _data = to_naive_utc_datetime(data)  # ty:ignore[no-matching-overload] # pyrefly:ignore
            dtype = None
        elif dtype == "timedelta":
            _data = _pd.to_timedelta(data)  # ty:ignore[no-matching-overload] # pyrefly:ignore
            dtype = None
        elif data is None:
            if default is not None:
                _data = default
            else:
                raise ValueError(
                    f"data is None, but no default value provided or defined for column '{label}'"
                )
        else:
            _data = data

        self.data[label] = _pd.Series(data=_data, index=self.data.index, dtype=dtype)

    def _data_ok(self, warn: bool = True) -> bool:
        """Test whether data are complete according to specifications in ``obj._known_fields``."""
        rval = True
        for f in self.required_fields():
            if f not in self.data.columns:
                if warn:
                    warnings.warn(f"Missing required field: '{f}'")
                rval = False
            if self.data[f].isna().any() or self.data[f].eq("").any():
                if warn:
                    warnings.warn(f"Required field has empty records: '{f}'")
                rval = False

        return rval

    @classmethod
    def from_dataframe(
        cls,
        df: _pd.DataFrame,
        use_index: bool = True,
        ignore_unknown_fields: bool = False,
        parse_split_datetime: bool = False,
        mapper: Renamer | None = None,
    ) -> Self:
        """
        Create object from a pandas DataFrame.

        Parameters
        ----------
        df : _pd.DataFrame
            Data to be loaded.
        use_index : bool, default True
            Load dataframe index as a data column. Drop index if False.
        ignore_unknown_fields : bool, default False
            Ignore fields that are not in the `known_fields` attribute.
        parse_split_datetime: bool, default False
            If True, parse discrete year, month, day columns into a single
            datetime column and drop the original columns. Expected columns
            are [year, month, day, hour, minute, second, microsecond, nanosecond],
            with at least year, month, and day being required.
        mapper : dict-like or function, default None
            Rename input fields/columns prior to loading. See
            `pandas.Dataframe.rename` for details.

        Returns
        -------
        GSolveTable
        """
        if use_index:
            df = df.reset_index()
        else:
            df = df.copy()

        if mapper is not None:
            df = df.rename(columns=mapper)
        df.columns = [str(c).lower() for c in df.columns]

        for c in cls.known_fields():
            if c not in df.columns:
                lf = cls._known_fields[c].legacy_name
                if lf is not None and lf in df.columns:
                    df = df.rename(columns={lf: c})

        if parse_split_datetime:
            df = merge_datetime_columns(df, drop=True)

        missing_fields = [c for c in cls.required_fields() if c not in df.columns]
        if missing_fields:
            raise ValueError(f"DataFrame missing required columns {missing_fields}")

        if ignore_unknown_fields:
            df = df.loc[:, df.columns.intersection(cls.known_fields())]

        data = {str(k): v for k, v in df.to_dict("list").items()}
        return cls(**data)

    @classmethod
    def from_csv(
        cls,
        csv_file: FilePath,
        ignore_unknown_fields: bool = True,
        parse_split_datetime: bool = False,
        mapper: Renamer | None = None,
        **kwargs,
    ) -> Self:
        """
        Create object from a CSV file.

        Parameters
        ----------
        csv_file : str or PathLike
            The path to the CSV file.
        ignore_unknown_fields : bool, default True
            Only include `known fields` in the resulting object.
        parse_split_datetime: bool, default False
            If True, parse discrete year, month, day columns into a single
            datetime column and drop the original columns. Expected columns
            are [year, month, day, hour, minute, second, microsecond, nanosecond],
            with at least year, month, and day being required.
        mapper : dict-like or function, default None
            Rename input fields/columns prior to loading. See
            `pandas.Dataframe.rename` for details.
        **kwargs
            Additional keyword arguments to be passed to `pandas.read_csv`.

        Returns
        -------
        GSolveTable
        """
        return cls.from_dataframe(
            _pd.read_csv(csv_file, **kwargs),
            use_index=False,
            ignore_unknown_fields=ignore_unknown_fields,
            parse_split_datetime=parse_split_datetime,
            mapper=mapper,
        )

    @classmethod
    def from_excel(
        cls,
        excel_file: FilePath,
        sheet_name: str | int | list[str | int] | None = None,
        ignore_unknown_fields: bool = True,
        parse_split_datetime: bool = True,
        mapper: Renamer | None = None,
        **kwargs,
    ) -> Self:
        """
        Create a object from an Excel file.

        Parameters
        ----------
        excel_file : str or PathLike
            The path to the Excel file.
        sheet_name : str | int | list-like, optional
            The name or index of the worksheet to read. If None, then
            try to use the default sheet name(s) defined in the class.
        ignore_unknown_fields : bool, default True
            Only include `known fields` in the resulting object.
        parse_split_datetime: bool, default False
            If True, parse discrete year, month, day columns into a single
            datetime column and drop the original columns. Expected columns
            are [year, month, day, hour, minute, second, microsecond, nanosecond],
            with at least year, month, and day being required.
        mapper : dict-like or function, default None
            Rename fields/columns after loading. See
            `pandas.Dataframe.rename` for details.
        **kwargs
            Additional keyword arguments to be passed to `pandas.read_excel`.

        Returns
        -------
        GSolveTable
        """
        _sheet_name: str | int | list[str | int]
        if sheet_name is None:
            try:
                _sheet_name = getattr(cls, "_default_excel_sheet_name")
            except AttributeError:
                raise ValueError(
                    f"sheet_name is None, but {type(cls).__name__} class "
                    "does not define a default sheet name."
                )
        else:
            _sheet_name = sheet_name
        df = read_excel_worksheet(excel_file, sheet_name=_sheet_name, **kwargs)
        df = normalize_field_names(df)

        for f in cls.known_fields():
            if f not in df.columns:
                legacy_name = cls._known_fields[f].legacy_name
                if legacy_name is not None and legacy_name in df.columns:
                    cols = {legacy_name: f}
                    df = df.rename(columns=cols)

        return cls.from_dataframe(
            df,
            use_index=False,
            ignore_unknown_fields=ignore_unknown_fields,
            parse_split_datetime=parse_split_datetime,
            mapper=mapper,
        )

    def write_to_csv(
        self,
        fname: FilePath,
        normalize_column_names: bool = True,
        expand_datetime: str | None = None,
        drop_datetime: bool = False,
        bool_to_int: bool = False,
        include_unknown_fields: bool = False,
        **kwargs,
    ) -> None:
        """Write data to a csv file.

        Parameters
        ----------
        fname : str or PathLike
            Output file name.
        normalize_columns_names : bool, default True
            Make column names lowercase with no spaces.
        expand_datetime : str or None, default None
            Expand datetime fields to
        bool_to_int : bool, default False
            Convert True, False to 1, 0.
        **kwargs
            Optional arguments to be passed to `pandas.to_csv`.

        See Also
        --------
        pandas.to_csv

        """
        cols = [c for c in self.known_fields() if c in self.data.columns]
        cols += [c for c in self.data.columns if c not in self.known_fields()]

        df = prepare_writable_df(
            self.data[cols].copy(),
            normalize_column_names=normalize_column_names,
            expand_datetime=expand_datetime,
            drop_datetime=drop_datetime,
            bool_to_int=bool_to_int,
        )

        df.to_csv(fname, header=True, index=True, **kwargs)

    # TODO: incomplete
    # def to_excel(
    #     self,
    #     sheet_name: str | None = None,
    #     normalize_column_names: bool = True,
    #     expand_datetime: str | None = "datetime",
    #     drop_datetime: bool = False,
    #     bool_to_int: bool = True,
    #     include_unknown_fields: bool | Sequence[str] = False,
    #     active_only: bool = False,
    #     if_workbook_exists: _IfWorkbookExists = "error",
    #     if_sheet_exists: _IfSheetExists = "error",
    #     **kwargs,
    # ) -> None:
    #     if not isinstance(sheet_name, (str, bytes)):
    #         k = "_default_excel_sheet_name"
    #         sheet_name = getattr(self, k, None)
    #         if _is_list_like(sheet_name):
    #             sheet_name = sheet_name[0]
    #     if not isinstance(sheet_name, str) or sheet_name == "":
    #         raise TypeError(
    #             "'sheet_name' not defined and object has no valid "
    #             "_default_excel_sheet_name attribute"
    #         )


@dataclasses.dataclass
class GSolveParameters:
    """Base class to store parameters related to GSolveTable derived classes."""

    def __copy__(self) -> Self:
        """Ensure all copies are deep copies."""
        return deepcopy(self)

    def __param_str__(self) -> str:
        return repr(self).partition("(")[2].rpartition(")")[0]

    def copy(self) -> Self:
        """Return a deep copy of object."""
        return self.__copy__()

    def to_dict(self) -> dict:
        """Return parameters as a dict."""
        return dataclasses.asdict(self)

    def to_series(
        self,
        series_name: str | None = None,
        index_name: str | None = None,
        index_prefix: str | None = None,
    ) -> _pd.Series:
        """Return parameters as a Series with parameter names as the index.

        Parameters
        ----------
        series_name : str | None, default is None
            The series data field name.
        index_name : str | None, default is None
            The series index name.
        index_prefix : str | None, optional
            Create a multiinidex where level 0 is 'index_prefix` and level 1 are
            the parameter names.

        Returns
        -------
        Series

        """
        ds = _pd.Series(data=self.to_dict(), name=series_name).rename_axis(index_name)
        if index_prefix:
            ds.index = ds.index = _pd.MultiIndex.from_arrays(
                arrays=([index_prefix] * ds.shape[0], ds.index),
            )
            if index_name is not None:
                ds = ds.rename_axis(["", index_name])

        return ds

    @classmethod
    def from_series(
        cls,
        ds: _pd.Series,
        skip_missing: bool = False,
        skip_unknown_parameters: bool = False,
    ) -> Self:
        """Generate a GsolveParameters object from a pandas.Series.

        Parameters
        ----------
        ds : _pd.Series
            The input Series is parsed in a dict-like manner with indicies as parameter
            names and series data as values.
        skip_missing: bool, default False:
            How to handle cases where ``ds`` does not provide values for all parameters.
            If False, raise a TypeError exception. If True and the missing parameters
            have default values, create the object with default values. Parameters
            without a default value must always be defined in the input series.
        skip_unknown_parameters : bool, default False
            If False, raise a TypeError if ``ds`` contains indices that do not match
            known parameters. If True, silently ignore unknown parameters

        Returns
        -------
        GSolveParameters

        """
        _ds = ds.copy()
        if _ds.index.nlevels > 1:
            raise ValueError("MultiIndex series not supported.")
        args: dict[str, Any] = {
            str(k): v for k, v in ds.items() if k in cls.__dataclass_fields__
        }
        missing_args = [k for k in cls.__dataclass_fields__ if k not in args]

        if not skip_missing:
            if missing_args:
                raise TypeError(
                    f"skip_missing=False: missing required parameters: {missing_args}"
                )
        extra_args = [k for k in _ds.index if k not in cls.__dataclass_fields__]
        if extra_args and not skip_unknown_parameters:
            raise TypeError(
                f"series contains unknown parameters: {_ds.index[extra_args].to_list()}"
            )

        return cls(**args)

    @classmethod
    def default_values(cls) -> dict:
        """Return dict of default parameter values."""
        return {
            k: cls.__dataclass_fields__[k].default for k in cls.__dataclass_fields__
        }

    def non_default_values(self) -> dict:
        """Return dict of non-default parameter values."""
        defaults = self.default_values()
        return {k: v for k, v in self.to_dict().items() if defaults.get(k, None) != v}

    def to_excel(
        self,
        fname: FilePath,
        sheet_name: str | None = None,
        if_workbook_exists: IfWorkbookExists = "error",
        if_sheet_exists: IfSheetExists = "error",
        parameter_name_label: str = "parameter",
        parameter_value_label: str = "value",
        **kwargs,
    ) -> None:
        """Write parameters to an Excel worksheet.

        Parameters
        ----------
        fname : str or PathLike
            The path to the output Excel file.
        sheet_name : str
            The name of the excel worksheet to write terrain corrections.
        if_workbook_exists : {'error', 'append', 'replace'}, default 'error'
            Action to take if the workbook already exists. Options are:
            'error', 'append', or 'replace'.
        if_sheet_exists : {'error', 'replace', 'new'}, default 'error'
            Action to take if the sheet already exists. Options are:
            'error', 'replace', or 'new'.
        parameters_label : str, default is 'parameter'
            Set the header label for parameter names column in output worksheet.
        values_label : str, default is 'value'
            Set the header label for parameter names column in output worksheet.
        **kwargs : dict
            Additional keyword arguments passed to `pandas.DataFrame.to_excel`.

        See Also
        --------
        write_excel_worksheet
        pandas.Dataframe.to_excel

        """
        params_ds = self.to_series(
            index_name=parameter_name_label, series_name=parameter_value_label
        )
        if sheet_name is None:
            sheet_name = getattr(self, "_default_excel_sheet_name", None)
            if sheet_name is None:
                raise ValueError(
                    "sheet_name is None and object has no "
                    "_default_excel_sheet_name attribute."
                )

        write_excel_worksheet(
            prepare_writable_df(params_ds.to_frame(), normalize_column_names=True),
            fname,
            sheet_name=sheet_name,
            if_workbook_exists=if_workbook_exists,
            if_sheet_exists=if_sheet_exists,
            **kwargs,
        )

    def summary(
        self, include_name: bool = True, as_list: bool = True
    ) -> list[str] | str:
        """Return a list of params as strings in the form 'param: value'."""
        txt = []
        if include_name:
            txt.append(f"{type(self).__name__}")
        txt.extend([f"{k}: {v}" for k, v in self.to_dict().items()])
        if as_list:
            return txt
        return "\n".join(txt)
