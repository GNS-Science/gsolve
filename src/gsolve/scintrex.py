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

import abc
import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Literal, Self, Type, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from gsolve.core._typing import (
    DatetimeArray,
    DatetimeScalar,
    FilePath,
    TimedeltaScalar,
)
from gsolve.core.utils import (
    generate_loop_intervals,
    generate_loop_names,
    is_list_like,
    loops_from_gaps,
    to_naive_utc_datetime,
)
from gsolve.observations import GravityObservations
from gsolve.sites import GravitySites

__all__ = ["ScintrexData", "CG6Data"]

_ScintrexMetadataDataTypes: TypeAlias = Union[str, float, int, bool, pd.Timestamp]


class ScintrexData(abc.ABC):
    """Base class for Scintrex data files."""

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: dict[str, _ScintrexMetadataDataTypes],
        metadata_units: dict[str, str] | None = None,
        on_error: Literal["raise", "warn", "ignore"] = "raise",
    ) -> None:
        """
        Base class for reading and prcoessing Scintrex data files.
        """
        self.data: pd.DataFrame
        self.metadata: dict[str, _ScintrexMetadataDataTypes]
        self.metadata_units: dict[str, str]

        self._set_metadata(metadata, metadata_units)
        self._set_data(data, on_error)

    @abc.abstractmethod
    def to_gsolve_observations(self) -> GravityObservations:
        pass

    @abc.abstractmethod
    def set_loop(self) -> None:
        pass

    @abc.abstractmethod
    def _set_metadata(
        self,
        metadata: dict[str, _ScintrexMetadataDataTypes],
        metadata_units: dict[str, str] | None,
    ) -> None:
        pass

    @abc.abstractmethod
    def _set_data(
        self, data: pd.DataFrame, on_error: Literal["raise", "warn", "ignore"]
    ) -> None:
        pass

    def __bool__(self) -> bool:
        return (
            hasattr(self, "data")
            and isinstance(self.data, pd.DataFrame)
            and not self.data.empty
        )

    @property
    def meter_id(self) -> str:
        """Return instrument identifier - the last for digits of full serial number."""
        m = str(self.metadata.get("instrument_serial_number", None))
        if m is None:
            return ""
        elif len(m) < 4:
            return m
        else:
            return m[-4:]

    @property
    def stations(self) -> list[str]:
        """Return a list of unique station names in the data."""
        if not self:
            return []
        return self.data["station"].unique().tolist()

    def __copy__(self) -> Self:
        return deepcopy(self)

    def copy(self) -> Self:
        return self.__copy__()


class CG6Data(ScintrexData):
    """
    An object to read and store gravity observations recorded on a Scintrex CG-6.

    This class handles tsv data files written to internal storage of a CG-6.
    These files are typically named ``CG-6_####_Survey_Name.dat.``

    .. Note::

        The preferred method for initialising a CG6Data object is to use the
        ``from_file`` class method.

    Parameters
    ----------
    data : DataFrame
        The observation data.
    metadata : dict
        Metadata from file headers.
    metadata_units : dict, optional
        The measurement units of metadata fields.

    Attributes
    ----------
    data : pd.DataFrame
        CG6 gravity readings as a dataframe and converted to appropriate dtypes:

          - Column names are normalized to lowercase.
          - The 'date' and 'time' fields are combined to a single 'datetime' column.
          - The corrections flag field 'Corrections[drift-temp-na-tide-tilt]' is
            split into individual boolean columns.

    metadata : dict
        Metadata from file headers converted to approiate dtypes, with field names
        normalized to lowercase. Measurement units stored as a suffix to the field
        name (e.g. "fieldname [unit]") are removed and stored in the `'metadata_units'`
        attribute.
    metadata_units : dict
        The measurement units for metadata fields.
    loop_from_line : bool, optional
        If True, use the 'line' field as the loop identifier. The 'line' field is a
        user settable field on the CG-6.
    on_error : {'raise', 'warn', 'ignore'}, optional
        How to handle errors arising from null values in some output fields:
    """

    _file_id_header: str = "cg-6_calibration"
    _metadata_fields: MappingProxyType[str, Type] = MappingProxyType(
        {
            "survey_name": str,
            "instrument_serial_number": str,
            "created": pd.Timestamp,
            "operator": str,
            "gcal1": float,
            "goff": float,
            "gref": float,
            "x_scale": float,
            "y_scale": float,
            "x_offset": float,
            "y_offset": float,
            "temperature_coefficient": float,
            "temperature_scale": float,
            "drift_rate": float,
            "drift_zero_time": pd.Timestamp,
            "firmware_version": str,
        }
    )
    _data_fields: MappingProxyType[str, Type] = MappingProxyType(
        {
            "station": str,
            "date": str,
            "time": str,
            "corrgrav": float,
            "line": str,
            "stddev": float,
            "stderr": float,
            "rawgrav": float,
            "x": float,
            "y": float,
            "sensortemp": float,
            "tidecorr": float,
            "tiltcorr": float,
            "tempcorr": float,
            "driftcorr": float,
            "measurdur": float,
            "instrheight": float,
            "latuser": float,
            "lonuser": float,
            "elevuser": float,
            "latgps": float,
            "longps": float,
            "elevgps": float,
            "corrections[drift-temp-na-tide-tilt]": str,
        }
    )
    _extra_data_fields: MappingProxyType[str, Type] = MappingProxyType(
        {
            "datetime": pd.Timestamp,
            "meter_id": str,
            "correction_drift": bool,
            "correction_temp": bool,
            "correction_na": bool,
            "correction_tide": bool,
            "correction_tilt": bool,
            "loop": str,
        }
    )

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: dict[str, _ScintrexMetadataDataTypes],
        metadata_units: dict[str, str] | None = None,
        loop_from_line: bool = False,
        on_error: Literal["raise", "warn", "ignore"] = "warn",
    ) -> None:
        super().__init__(data, metadata, metadata_units, on_error)

        if loop_from_line:
            self.set_loop(field="line")

    def _set_metadata(
        self,
        metadata: dict[str, _ScintrexMetadataDataTypes],
        metadata_units: dict[str, str] | None = None,
    ) -> None:
        """Set metadata and metadata_units attributes"""
        self.metadata: dict[str, _ScintrexMetadataDataTypes] = {}
        self.metadata_units: dict[str, str] = {}

        for k_orig, v_orig in metadata.items():
            k = _normalize_keyword(k_orig)
            if k not in self._metadata_fields:
                raise ValueError(f"Unknown metadata field '{k}'.")
            v = _scintrex_header_type_conversion(v_orig, self._metadata_fields[k])
            self.metadata[k] = v if v is not None else ""
            if metadata_units is not None:
                self.metadata_units[k] = metadata_units.get(k_orig, "")
            else:
                self.metadata_units[k] = ""

    def _set_data(self, data: pd.DataFrame, on_error: str = "raise") -> None:
        """Set data attribute."""
        df = data.copy().rename(
            columns={c: _normalize_keyword(c) for c in data.columns}
        )

        for c in df.columns.intersection(list(self._data_fields.keys())):
            this_dtype = self._data_fields[c]
            # if this_dtype is float:
            try:
                df[c] = df[c].astype(this_dtype)
            except Exception:
                if on_error in ["warn", "ignore"]:
                    if on_error == "warn":
                        warnings.warn(
                            f"bad data encountered in column '{c}', setting to nan"
                        )
                    try:
                        df[c] = (
                            df[c]
                            .replace(to_replace=["--", "****", "******"], value=np.nan)
                            .astype(this_dtype)
                        )
                    except Exception as err:
                        raise TypeError(
                            f"unfixable error converting data in column '{c}' to {this_dtype}"
                        ) from err
                else:
                    raise TypeError(
                        f"error converting data in column '{c}' to {this_dtype}"
                    )

        if (
            "datetime" not in df.columns
            and "date" in df.columns
            and "time" in df.columns
        ):
            dt = to_naive_utc_datetime(
                pd.to_datetime(df["date"].str.cat(df["time"], sep="T"))
            )
            i_date_col = df.columns.get_loc("date")
            if not isinstance(i_date_col, int):
                raise TypeError("Unexpected error finding 'date' column index.")
            df.insert(i_date_col, "datetime", dt)  # ty:ignore[invalid-argument-type]
            df = df.drop(columns=["date", "time"])

        corr_flag_colname = "corrections[drift-temp-na-tide-tilt]"
        if corr_flag_colname in df.columns:
            print("hello", corr_flag_colname)

            flag_labels = corr_flag_colname.rstrip("]").rpartition("[")[-1].split("-")
            flag_labels = [f"correction_{f}" for f in flag_labels]

            flags = df.pop(corr_flag_colname).str.split("").str[1:-1].to_list()
            flags = [[bool(int(c)) for c in row] for row in flags]
            flags_df = pd.DataFrame(
                data=[[bool(int(c)) for c in row] for row in flags], columns=flag_labels
            ).drop(columns=["correction_na"], errors="ignore")
            df = pd.concat([df, flags_df], axis=1)

        self.data = df

    def _strip_corrections(self) -> pd.Series:
        """Return corrgrav values with all corrections removed."""
        return (
            self.data["corrgrav"]
            - (self.data["driftcorr"] * self.data["correction_drift"])
            - (self.data["tempcorr"] * self.data["correction_temp"])
            - (self.data["tidecorr"] * self.data["correction_tide"])
            - (self.data["tiltcorr"] * self.data["correction_tilt"])
        )

    @classmethod
    def from_file(
        cls,
        cg6_file: FilePath,
        loop_from_line: bool = False,
        on_error: Literal["raise", "warn", "ignore"] = "warn",
    ) -> Self:
        """
        Load and parse a Scintrex CG-6 data file.

        Parameters
        ----------
        cg6_file : FilePath
            The CG-6 data file to load.
        loop_from_line : bool, optional
            If True, use the 'line' field as the loop identifier, by default False.
        on_error : {"except", "warn", "ignore"}, deafault "warn"
            How to handle errors arising from null values in some output fields:

                - except: raise an Exception if bad data encountered
                - warn: issuse a wraning and fix errors
                - ignore: fix errrors silently

        Returns
        -------
        CG6Data
            A CG6Data object.
        """
        file_data = _slurp_scintrex_text_file(cg6_file)
        if not file_data:
            raise ValueError(f"No data read from {cg6_file}")

        idx_column_names = 0
        for idx_column_names, line in enumerate(file_data):
            if not line.startswith(r"/"):
                idx_column_names -= 1
                break

        file_id_found = False
        metadata_lines = file_data[:idx_column_names]
        column_headers = file_data[idx_column_names].lstrip("/").split()
        column_data = file_data[(idx_column_names + 1) :]

        metadata: dict[str, _ScintrexMetadataDataTypes] = {}
        metadata_units: dict[str, str] = {}

        for line in metadata_lines:
            k, v, u = _split_header_key_val_unit(
                line, normalize_key=True, extract_units=True
            )
            if k == "" or k is None:
                continue
            if v == "":
                if k == cls._file_id_header:
                    file_id_found = True
                continue
            metadata[k] = v
            metadata_units[k] = u

        if not file_id_found:
            warnings.warn(
                f"Expected file type identifier '{cls._file_id_header}'"
                f"not found in file {cg6_file}."
            )

        # load up the data
        df = pd.DataFrame(
            data=[c.split() for c in column_data if not c.startswith("#")],
            columns=column_headers,
            dtype=str,
        )

        obj = cls(df, metadata, metadata_units, on_error=on_error)
        if loop_from_line:
            obj.set_loop(field="line")
        return obj

    def set_loop(
        self,
        field: str | None = None,
        array: npt.ArrayLike | None = None,
        datetimes: Mapping[str, str] | DatetimeArray | None = None,
        time_gap: TimedeltaScalar | None = None,
        loop_start: int = 1,
        loop_step: int = 1,
        loop_format: str = "{LOOP}",
        output_column: str = "loop",
    ) -> None:
        """Set loop identifiers using one of several methods.

        Parameters
        ----------
        field : str, default None
            Set ``loop`` values from existing data ``field``. For CG6 data this
            would typically be the user set "line" field.
        array : array-like, default None
            Set ``loop`` values from an array-like object. Length must match
            the number of observations.
        datetimes : dict, Series or array-like, default None
            Use time intervals defined by ``datetimes`` and assign observations to those
            intervals based on observation times. If ``datetimes`` is dict-like or Series,
            then construct intervals from the keys/index and assign loop id's from the
            corrresponding values.  If ``datetimes`` is an array-like, then loop
            identifiers will be generated automatically.
        time_gap: timedelta-like, str int, default None
            Set ``loop`` values based on time gaps in the data. Loop intervals are
            defined where time gaps between observations exceed ``time_gap``.
        loop_start : int, default 1
            Loop identifier start value.
        loop_step : int, default 1
            Increment loop identifier by ``loop_step``.
        loop_format : str, default '{LOOP}'
            Format string for loop identifiers. Use 'LOOP' as a placeholder
            for the loop number. The default `"{LOOP}"` is effectively no
            formatting. Using, for example, ``loop_format="x_{'LOOP':02d}_y"`` would
            produce loop id's ``'x_01_y', 'x_02_y', ...``.
        output_column : str, default 'loop'
            Name of the output column.
        """
        if loop_format:
            if "LOOP" not in loop_format:
                raise ValueError("format_str must contain 'LOOP'.")

        # ensure only one method is used
        args = (field, array, datetimes, time_gap)
        if all(a is None for a in args):
            raise ValueError(
                "At least one of 'field', 'array', 'datetimes' or 'time_gap' must be set."
            )
        if sum(a is not None for a in args) > 1:
            raise ValueError("Only one of 'field', 'array', or 'datetimes' can be set.")

        if field is not None:
            if field not in self.data.columns:
                raise KeyError(
                    f"arg {field=}, but not column name '{field}' found in obj.data."
                )
            self.data[output_column] = self.data[field].astype(str)
            return

        if array is not None:
            array = np.atleast_1d(array)
            if len(array) != len(self.data):
                raise ValueError(
                    "Length of 'array' must match the number of observations."
                )
            self.data[output_column] = array.astype(str).tolist()
            return

        if datetimes is not None:
            if isinstance(datetimes, pd.Series):
                dates = to_naive_utc_datetime(datetimes.index)
                loop_ids = datetimes.astype(str).values
            elif isinstance(datetimes, Mapping):
                dates = to_naive_utc_datetime(list(datetimes.keys()))
                loop_ids = [str(l) for l in datetimes.values()]
            elif isinstance(datetimes, DatetimeArray):
                dates = to_naive_utc_datetime(datetimes)
                loop_ids = generate_loop_names(
                    len(dates), start=loop_start, step=loop_step
                )
            else:
                raise TypeError(
                    "datetimes must be a dictionary, Series or array-like object."
                )

            if not dates.is_monotonic_increasing:
                raise ValueError("datetimes must be sorted in increasing order.")
            if dates[0] > self.data["datetime"].min():
                raise ValueError(
                    f"First datetime in 'datetimes' ({dates[0]}) must be <= "
                    f"earliest observation time ({self.data.datetime.min()})"
                )
            if dates[-1] < self.data["datetime"].max():
                tmax = self.data["datetime"].max() + pd.Timedelta(seconds=1)
                dates = pd.DatetimeIndex(dates.to_list() + [tmax])

            loop_intervals = generate_loop_intervals(dates)
            loop_namer = pd.Series(loop_ids, index=loop_intervals)
            self.data[output_column] = loop_namer[self.data["datetime"]].to_list()
            return

        if time_gap is not None:
            self.data[output_column] = loops_from_gaps(
                self.data["datetime"],
                time_gap,
                loop_start=loop_start,
                loop_step=loop_step,
                loop_format=loop_format,
            )

    def to_gsolve_observations(
        self,
        tilt_corr: bool = True,
        temp_corr: bool = True,
        drift_corr: bool = True,
        tide_corr: bool = False,
        include_non_standard_fields: bool | Sequence = False,
    ) -> GravityObservations:
        """
        Export CG6 data to a GravityObservations object.

        Relevant data fields are renamed to match the GravityObservations schema.

            - Values from ``corrgrav`` field are not exported directly.
            - Output ``meter_reading`` is derived from ``corrgrav`` with all internally
              applied corrections removed.
            - ``meter_reading_mgal`` will contain ``meter_reading`` + the specified corrections

        Parameters
        ----------
        tilt_corr : bool, default is True
            Apply tilt correction ``tiltcorr`` to output field
            ``meter_reading_mgal``.
        temp_corr : bool, default is True
            Apply temperature correction  ``tempcorr`` to output field
            ``meter_reading_mgal``.
        drift_corr : bool, default is False
            Apply drift correction ``driftcorr`` to output field
            ``meter_reading_mgal``.
        tide_corr : bool, default False
            Include earth tide correction by copying ``tidecorr`` to
            output field ``earth_tide_corr``.
        include_non_standard_fields : bool or sequence, default is False
            Include non-standard fields in the output _GravityObservations
            object. If a sequence is provided, only specified
            non-standard fields will be included.

        Returns
        -------
        _GravityObservations

        """
        if "loop" not in self.data.columns:
            raise ValueError("Loop identifiers must be set before exporting to gsolve")
        df = self.data.copy()

        # corrgrav includes
        filt_rawgrav = self._strip_corrections()

        df["meter_id"] = f"CG6:{self.meter_id}"
        df["meter_reading"] = 0.0
        df["meter_reading_mgal"] = (
            filt_rawgrav
            + (tilt_corr * df["tiltcorr"])
            + (temp_corr * df["tempcorr"])
            + (drift_corr * df["driftcorr"])
        )
        df["site_id"] = df["station"].copy()
        if tide_corr:
            df["earth_tide_corr"] = df["tidecorr"].copy()

        convert_id_string = [
            "tilt" * tilt_corr,
            "temp," * temp_corr,
            "drift" * drift_corr,
        ]
        convert_id_string = ":".join(s for s in convert_id_string if s).rstrip(",")
        df["pre_import_corrections"] = convert_id_string

        if not include_non_standard_fields:
            to_drop = set(df.columns) - set(GravityObservations.known_fields())
            df = df.drop(columns=to_drop)
        else:
            if isinstance(include_non_standard_fields, Sequence):
                include_non_standard_fields = [
                    str(f) for f in include_non_standard_fields
                ]
                missing = [
                    f for f in include_non_standard_fields if f not in df.columns
                ]
                if missing:
                    raise KeyError(
                        f"Requested non-standard fields not found in data: {missing}"
                    )

                to_drop = set(df.columns) - set(
                    GravityObservations.known_fields() + include_non_standard_fields
                )
                df = df.drop(columns=to_drop)

        return GravityObservations.from_dataframe(
            df, ignore_unknown_fields=not include_non_standard_fields
        )

    def to_gsolve_sites(
        self, coords_source: Literal["user", "gps"] = "user"
    ) -> GravitySites:
        """
        Export CG6 data to a _GravitySites object.

        This returned ``_GravitySites`` object contains site locations only. The user
        will need to set "reference_gravity" and "gsolve_tie" fields.

        .. warning:: **Low accuracy location data**

                Coordinates may be sourced from the GC-6's onboard GPS receiver.
                These will have accuracy equivalent to a typical hand-held GPS unit,
                should not be used for computing gravity reductions such as free air or
                Bouguer corrections.

        Parameters
        ----------
        coords_source : {'user', 'gps'}, default 'user'
            Specify the source of lat, lon and elev data:

             - ``'gps'`` : the mean of 'latgps', 'longps' and 'elevgps'
               for each site. These positions are derived from the internal GPS
               reciever and are of low accuracy, but are almost certainly correct
               to within a few 10's of metres.
             - ``'user'`` : take values from 'latuser', 'lonuser' and 'elevuser'
               for each site. The 'user' coords are sourced from the instrument file
               ``stations.dat``. This file can be pre-populated with accurate
               station coordinates prior to field data collection, however there is no
               guarantee that these values are correct. Also, for sites that were not
               pre-defined in ``stations.dat``, the CG-6 will create a site and use the
               set it's coordinates from the initial gps fix. In this case, the 'user'
               coords will be less reliable than 'gps' coords, which are averaged over
               all readings at a site.

        Returns
        -------
        GravitySites

        """
        if coords_source not in ("user", "gps"):
            raise ValueError(
                f"coords_source must be 'user' or 'gps', not {coords_source}."
            )

        coord_cols = [f"{c}{coords_source}" for c in ("lat", "lon", "elev")]
        agg_method = "mean" if coords_source == "gps" else "first"
        agg_dict = dict(zip(coord_cols, [agg_method] * 3))
        # height_ellipsoidal
        df = (
            self.data.copy()
            .rename(columns={"station": "site_id"})
            .groupby("site_id")
            .agg(agg_dict)
            .rename(
                columns=dict(
                    zip(coord_cols, ("latitude", "longitude", "height_ellipsoidal"))
                )
            )
        )

        return GravitySites.from_dataframe(df)

    def set_drift_correction(
        self, drift_rate: float, drift_zero_time: DatetimeScalar
    ) -> None:
        """Apply linear drift correction to CG6 data.

        CG-6 data files include an internally applied drift correction based on
        rates calculated during a previous calibration sun.  This may be problematic
        because:

            - The drift rate estimation may be out of date and therefore not accurate
              for the meter when these data were collected.
            - The internal drift rate may not be accurate because the method
              used to determine drift function is simplistic and uses data
              that may not have had all time-dependent corrections applied.

        This method allows the user to specify a new drift function and apply it to the
        observations. For example, a calibration run could be performed after the survey
        data or a user could fit their own drift curve to calibration data.

        Parameters
        ----------
        drift_rate : float
            Drift rate in mGal per day.
        drift_zero_time : datetime-like
            Zero time for drift correction.

        """
        current_drift_rate = self.metadata.get("drift_rate", 0.0)
        current_drift_zero_time = self.metadata.get("drift_zero_time", pd.NaT)
        current_drift_corr = self.data.get(
            "driftcorr", pd.Series(0.0, index=self.data.index)
        )

        _drift_zero_time = to_naive_utc_datetime(drift_zero_time)
        _drift_rate = float(drift_rate)

        if not isinstance(_drift_zero_time, pd.Timestamp):
            raise ValueError(
                "drift_zero_time could not be converted to a valid Timestamp."
            )

        _drift_corr = (
            (self.data["datetime"] - _drift_zero_time)
            .dt.total_seconds()
            .div(pd.Timedelta(86400).total_seconds())
            .mul(-1 * _drift_rate)
        )

        self.metadata["drift_rate"] = _drift_rate
        self.metadata["drift_zero_time"] = _drift_zero_time
        self.data["driftcorr"] = _drift_corr
        self.data["corrgrav"] = self.data["corrgrav"] - current_drift_corr + _drift_corr


def _slurp_scintrex_text_file(filepath: FilePath) -> list[str]:
    """Read a Scintrex text file, fix encoding and return lines as a list."""
    with open(filepath, "r", encoding="utf-8-sig") as fh:
        return [l.strip() for l in fh.readlines()]


def _split_header_key_val_unit(
    header: str,
    normalize_key: bool = True,
    extract_units: bool = True,
) -> tuple[str, str, str]:
    """Split headers into key, value and units."""
    header = header.strip("/ ")
    if not header:
        return ("", "", "")
    if ":" not in header:
        if normalize_key:
            header = _normalize_keyword(header)
        return (header, "", "")

    k, _, v = [w.strip() for w in header.partition(":")]

    if extract_units:
        k, u = _extract_unit_from_keyword(k)
    else:
        u = ""
    if normalize_key:
        k = _normalize_keyword(k)

    return k, v, u


def _scintrex_header_type_conversion(
    header_val: _ScintrexMetadataDataTypes,
    data_type: Type | Callable | None = None,
) -> _ScintrexMetadataDataTypes | None:  # noqa: ANN401
    if header_val == "" or data_type is None:
        return ""
    elif data_type is pd.Timestamp:
        rval = to_naive_utc_datetime(header_val)
        if pd.isna(rval):
            raise ValueError(f"Could not convert '{header_val}' to a Timestamp.")

    elif data_type is bool:
        if isinstance(header_val, str):
            if header_val.lower() in ("true", "yes", "1", "enabled"):
                rval = True
            elif header_val.lower() in ("false", "no", "0", "disabled"):
                rval = False
            else:
                raise ValueError(
                    f"Could not convert '{header_val}' to a boolean value."
                )
        else:
            rval = bool(header_val)
    else:
        rval = data_type(header_val)

    return rval


def _extract_unit_from_keyword(header: str) -> tuple[str, str]:
    """Get header and unit form a header string."""
    if header.endswith(")"):
        sep = "("
    elif header.endswith("]"):
        sep = "["
    else:
        return header, ""

    h, _, u = [w.strip() for w in header.rpartition(sep)]
    u = u.rstrip("])")

    return h, u


def _normalize_keyword(keyword: str) -> str:
    return keyword.strip().lower().replace(" ", "_")
