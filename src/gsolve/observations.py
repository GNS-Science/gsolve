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
import pathlib
import warnings as _warnings
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Self

import matplotlib.dates as _mdates
import matplotlib.pyplot as _plt
import numpy as _np
import numpy.typing as npt
import pandas as _pd

from gsolve.core._typing import (
    DatetimeScalar,
    FilePath,
    IfSheetExists,
    IfWorkbookExists,
    TimedeltaScalar,
)
from gsolve.core.data import (
    COMMON_FIELDS,
    DataFieldSpecification,
    GSolveParameters,
    GSolveTable,
)
from gsolve.core.excel_io import write_excel_worksheet
from gsolve.core.utils import (
    GSolveDataWarning,
    is_list_like,
    prepare_writable_df,
    to_naive_utc_datetime,
)
from gsolve.gsolve_algorithms import GSolveSolverMethod, call_gsolve_lstsq
from gsolve.gsolve_outputs import GSolveResults
from gsolve.meter_conversion import MeterReadingConverter
from gsolve.sites import GravitySites, ReferenceGravity, combine_gravity_sites
from gsolve.tide.earth_tide import (
    EarthTideCorrectionProvider,
    LongmanTidalCorrection,
)
from gsolve.tide.ocean_load import OceanLoadCorrectionProvider

__all__ = [
    "GravityObservations",
    "GravitySurvey",
    "GravityObservationsParameters",
    "combine_gravity_observations",
    "combine_gravity_sites",
]


@dataclasses.dataclass
class GravityObservationsParameters(GSolveParameters):
    """Dataclass to hold parameters for pre-processing gravity observations.

    This class is not intended to be instantiated directly, but rather exists
    to store parameters in a consistent way.

    Attributes
    ----------
    timedelta_unit : _pd.Timedelta, default "1h"
        The time interval unit used in calculating survey time deltas.
    fixed_time_datum : _pd.Timestamp, default is pd.NaT
        The fixed time datum used for calculating survey time deltas.
    earthtide_correction_method: str = ""
        The method used for earth tide correction, e.g. "Longman".
    ocean_load_correction_method: str = ""
        The method used for ocean load correction, e.g. "Longman".
    """

    timedelta_unit: TimedeltaScalar = "1h"
    fixed_time_datum: _pd.Timestamp | None = None
    earthtide_correction_method: str = ""
    ocean_load_correction_method: str = ""

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        if name == "timedelta_unit":
            value = _pd.Timedelta(value)
        elif name == "fixed_time_datum":
            if value is None or _pd.isnull(value):
                value = _pd.NaT
            else:
                value = to_naive_utc_datetime(value)
        super().__setattr__(name, value)

    def to_excel(
        self,
        fname: FilePath,
        sheet_name: str | None = "observation_parameters",
        if_workbook_exists: IfWorkbookExists = "append",
        if_sheet_exists: IfSheetExists = "replace",
        parameter_name_label: str = "parameter",
        parameter_value_label: str = "value",
        **kwargs,
    ) -> None:
        params_ds = self.to_series(
            index_name=parameter_name_label, series_name=parameter_value_label
        )

        if isinstance(params_ds["timedelta_unit"], _pd.Timedelta):
            params_ds["timedelta_unit"] = params_ds["timedelta_unit"].isoformat()

        if params_ds["fixed_time_datum"] is not None and not _pd.isna(
            params_ds["fixed_time_datum"]
        ):
            params_ds["fixed_time_datum"] = "first"

        if sheet_name is None:
            sheet_name = getattr(self, "_default_excel_sheet_name", None)
            if sheet_name is None:
                raise ValueError(
                    "sheet_name is None and object has no "
                    "_default_excel_sheet_name attribute."
                )

        write_excel_worksheet(
            df=prepare_writable_df(
                df=params_ds.to_frame(), normalize_column_names=True
            ),
            excel_file=fname,
            sheet_name=sheet_name,
            if_workbook_exists=if_workbook_exists,
            if_sheet_exists=if_sheet_exists,
            **kwargs,
        )


class GravityObservations(GSolveTable):
    """
    Class to store and process gravity observations.

    Parameters
    ----------
    site_id : ArrayLike
        Observation site identifier.
    datetime : ArrayLike
        The observation datetime in a format parseable by ``pandas.to_datetime()``
        method. All datetimes will be converted to UTC with timezone information
        removed.
    meter_id : ArrayLike
        Gravity meter identifier.
    meter_reading : ArrayLike, optional
        Observed meter reading in meter units. At least one of ``meter_reading`` or
        ``meter_reading_mgal`` must be specified.
    meter_reading_mgal : ArrayLike, optional
        Observed meter readings in mGal.  At least one of ``meter_reading`` or
        ``meter_reading_mgal`` must be specified.
    obs_id : ArrayLike, optional
        Array-like object containing unique observation identifiers. If omitted,
        unique identifiers will be generated from the ``site_id`` and
        ``datetime`` fields.
    loop : ArrayLike, optional
        Array-like object containing survey loop identifiers. If omitted,
        all observations will be assigned to loop '1'.
    active : ArrayLike, optional
        Array-like object indicating whether an observation is 'active' (True) or
        inactive (False). Only 'active' observations will be included as a datapoints
        in network adjustment. All observations are considered active by default.
    timedelta_unit : TimedeltaConvertibleTypes, default "1h"
        Time interval unit for timedelta calculations. The default is '1h' (i.e. 1 hour),
        meaning 'survey time' is in decimal hours.
    fixed_time_datum : DatetimeScalar, optional
        The time datum used to compute survey time deltas. If None, then the datetime
        of the earliest observation will be used.
    **kwargs
        Additional keyword arguments can be used to specify additional fields to be
        included in the ``data`` DataFrame attribute.
        .

    Attributes
    ----------
    data : pandas.DataFrame
        DataFrame containing the gravity observations, gravity reductions and other
        derived information.

    params : GravityObservationsParameters
        A container class to store parameters.
    _known_fields : dict[str, DataFieldSpecification]
        A dictionary of 'known' field name and their associated DataFieldSpecification,
        which defines the expected data type, default value, and other metadata for
        that field. If data are added using the ``set_column(name, value,...)``,
        and 'name' is in ``_known_fields``, the associated DataFieldSpecification
        will be used to validate and coerce the data before it is added to the
        ``obj.data`` dataframe.

    """

    _known_fields: dict[str, DataFieldSpecification] = {
        "site_id": COMMON_FIELDS["site_id"],
        "datetime": COMMON_FIELDS["datetime"],
        "meter_id": DataFieldSpecification(
            "meter_id", str, "", True, legacy_name="meter"
        ),
        "loop": COMMON_FIELDS["loop"],
        "active": COMMON_FIELDS["active"],
        "meter_reading": DataFieldSpecification(
            "meter_reading", float, _np.nan, False, legacy_name="reading"
        ),
        "meter_reading_mgal": DataFieldSpecification(
            "meter_reading_mgal", float, _np.nan, False
        ),
        "loop_tdelta": DataFieldSpecification("loop_tdelta", float, _np.nan, False),
        "survey_tdelta": DataFieldSpecification("survey_tdelta", float, _np.nan, False),
        "calibration_factor": DataFieldSpecification(
            "calibration_factor", float, 1.0, False
        ),
        "earth_tide_corr": DataFieldSpecification("earth_tide_corr", float, 0.0, False),
        "ocean_load_corr": DataFieldSpecification("ocean_load_corr", float, 0.0, False),
        "custom_corr": DataFieldSpecification("custom_corr", float, 0.0, False),
        "gravity_corr": DataFieldSpecification("gravity_corr", float, _np.nan, False),
        "meter_reading_converter_id": DataFieldSpecification(
            "meter_reading_converter_id", str, "NA", False
        ),
    }
    _index_field: str = "obs_id"
    _default_excel_sheet_name: str | tuple[str, ...] = ("observations", "Survey Data")

    def __init__(
        self,
        site_id: npt.ArrayLike,
        datetime: npt.ArrayLike,
        meter_id: npt.ArrayLike,
        meter_reading: npt.ArrayLike | None = None,
        meter_reading_mgal: npt.ArrayLike | None = None,
        obs_id: npt.ArrayLike | None = None,
        loop: npt.ArrayLike | None = None,
        active: npt.ArrayLike | None = None,
        timedelta_unit: TimedeltaScalar = "1h",
        fixed_time_datum: DatetimeScalar | None = None,
        **kwargs,
    ) -> None:
        self.data: _pd.DataFrame
        self._timedelta_unit: _pd.Timedelta
        self._fixed_time_datum: _pd.Timestamp | None

        self._earthtide_correction_method: str = ""
        self._ocean_load_correction_method: str = ""

        n_readings = -1
        if meter_reading is not None:
            n_readings = _np.atleast_1d(_np.asarray(meter_reading, dtype=float)).size
        elif meter_reading_mgal is not None:
            n_readings = _np.atleast_1d(
                _np.asarray(meter_reading_mgal, dtype=float)
            ).size
        else:
            raise ValueError("meter_reading or meter_reading_mgal must be specified")

        # Set initial empty dataframe with n rows
        # Ensures that subsequent set_column() calls will fail if attempt to add
        # array of different length

        self.data = _pd.DataFrame(index=_pd.RangeIndex(n_readings), data=None)
        self.set_column("site_id", site_id)
        self.set_column("datetime", datetime)
        self.set_column("meter_id", meter_id)
        self.set_column("loop", loop)
        self.set_column("active", active)
        self.set_column("meter_reading", meter_reading)
        self.set_column("meter_reading_mgal", meter_reading_mgal)

        for k in self.known_fields():
            if k not in self.data.columns:
                self.set_column(k, data=kwargs.pop(k, self._known_fields[k].default))

        # set any additional data passed via kwargs
        for k, v in kwargs.items():
            self.set_column(k, v)

        if obs_id is not None:
            if isinstance(obs_id, str):
                # assume it is a column name
                self.set_obs_id(obs_id, drop=obs_id == "obs_id")
            if not isinstance(obs_id, str):
                # assume it is a sequence of obs_id values``
                self.set_column("obs_id", obs_id, dtype=str)
                self.set_obs_id("obs_id")
        else:
            # set auto-generated obs_id
            self.set_obs_id()

        # check that either meter_reading or meter_reading_raw is present
        # cannot both be NaN
        has_nan_readings = (
            self.data.loc[:, ["meter_reading", "meter_reading_mgal"]].isna().all(axis=1)
        )
        if has_nan_readings.any():
            _warnings.warn(
                "neither 'meter_reading' or 'meter_reading_mgal' present for "
                "1 or more records"
            )
        _ = self._data_ok(warn=True)
        self.set_timedelta_unit(timedelta_unit, set_tdelta=False)
        self.set_fixed_time_datum(fixed_time_datum, set_tdelta=False)
        self.set_tdelta()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_observations={self.data.shape[0]}, "
            f"n_loops={len(self.loop_ids)}, "
            f"obs_datetimes={self.starttime.isoformat()}_to_{self.endtime.isoformat()})"
        )

    def _default_index_generator(self) -> _pd.Index:
        """Generate a default obs_id index from site_id and datetime.

        Returns
        -------
        pandas.Index
            Index of obs_id strings in the form '<site_id>.<timestamp>' with
            duplicates disambiguated by appending a 3-digit sequence.
        """
        tstamps = self.data["datetime"].apply(lambda t: int(t.timestamp())).astype(str)
        siteid_tstamp_labels = (
            self.data["site_id"].astype(str).str.cat(tstamps, sep=".")
        )
        new_idx = _pd.Index(siteid_tstamp_labels, name=self._index_field, dtype=str)
        return self._index_deduplicator(new_idx)

    @staticmethod
    def _index_deduplicator(idx: _pd.Index) -> _pd.Index:
        """Deduplicate index by appending a 3-digit sequence number.

        Parameters
        ----------
        idx : Index
            The index to deduplicate.

        Returns
        -------
        Index
            A copy of `idx` with 3 digit sequence numbers appended.
        """
        suffix = (
            idx.to_series().groupby(level=0).cumcount().add(1).astype(str).str.zfill(3)
        ).to_list()
        new_idx = _pd.Index(idx.astype(str).str.cat(suffix, sep="."))
        if idx.name is not None:
            new_idx.name = idx.name
        return new_idx

    def set_obs_id(
        self,
        idx: npt.ArrayLike | str | None = None,
        duplicated_obs_id: Literal["error", "keep", "rename"] = "rename",
        drop: bool = True,
    ) -> None:
        """Set ``obs_id`` as the index of the ``data`` DataFrame attribute.

        Warning:: This method will overwrite the existing index of ``obj.data``.

        Parameters
        ----------
        idx : ArrayLike, str or None, default is None
            The obs_id values to set as the index of ``obj.data``. Behaiviour
            depends on the dtype of `idx`. If `idx` is:

              - ``None`` : a default obs_id will be auto-generated
                using the ``_default_index_generator()`` method.
              - ``str`` : `idx` is assumed to be the name of a column
                in ``obj.data`` to be used to set as the index. Equivalent to
                ``obj.data.set_index(idx)``. Note that the index will be renamed
                to `obs_id`.
              - ``array-like`` : `idx` is assumed to be a sequence of obs_id
                values to set as the index.
        duplicated_obs_id : {'error', 'keep', 'rename'}, default 'rename'
            The behaviour when duplicate obs_id values are found:

              - ``'error'`` : raise a ValueError.
              - ``'keep'`` : issue a warning and keep duplicate obs_id's as-is.
              - ``'rename'`` : issue a warning and rename duplicate obs_id's by appending
                  a 3-digit sequence number.

        drop : bool, default True
            If ``obs_id`` is to be set from an existing column (i.e. where
            ``idx`` is a string),
            this flag indicates whether to drop that column from ``obj.data`` after.

        Raises
        ------
        ValueError

        """

        if idx is None:
            # use autogenerated obs_id
            new_idx = self._default_index_generator()

        elif isinstance(idx, str):
            # assume it idx a column name, that will become obs_id
            new_idx = _pd.Index(
                self.data[idx].astype(str).to_list(), name=self._index_field
            )
            if drop:
                self.data = self.data.drop(columns=idx)

        elif is_list_like(idx) and isinstance(idx, Iterable):
            _idx = [str(i) for i in idx]
            # a sequence will converted to index
            new_idx = _pd.Index(_idx, name=self._index_field, dtype=str)

        else:
            raise TypeError(f"invalid idx arg of type '{type(idx).__name__}'")

        if self._index_field in self.data.columns:
            raise ValueError(
                "refusing to create new index named 'obs_id' index when 'obs_id' "
                "already exists as a column in obj.data "
            )

        if new_idx.has_duplicates:
            dupes = new_idx[new_idx.duplicated().tolist()]
            msg = f"{len(dupes)} duplicated obs_id's : {dupes.unique().to_list()}"

            if duplicated_obs_id == "error":
                raise ValueError(f"{msg}")
            elif duplicated_obs_id == "keep":
                _warnings.warn(f"keeping {msg}")
            elif duplicated_obs_id == "rename":
                _warnings.warn(f"renaming {msg}")
                new_idx = self._index_deduplicator(new_idx)

        self.data = self.data.set_index(new_idx)

    @property
    def loop_ids(self) -> list[str]:
        """Return unique survey loop id's sorted by loop start time."""
        if "loop" not in self.data.columns or "datetime" not in self.data.columns:
            raise ValueError("'loop' and/or 'datetime' columns are missing")
        loops = self.data["loop"].unique().tolist()
        return sorted(
            loops,
            key=lambda x: self.data["datetime"].loc[self.data["loop"].eq(x)].min(),
        )

    @property
    def starttime(self) -> _pd.Timestamp:
        """Earlest observation datetime"""
        return self.data["datetime"].min()

    @property
    def endtime(self) -> _pd.Timestamp:
        """Latest observation datetime."""
        return self.data["datetime"].max()

    def timedelta_unit(self) -> _pd.Timedelta:
        """Time interval unit used for calculating survey timedelta.

        Can be any valid argument for `pandas.Timedelta()`. The default
        is '1h' (i.e. 1 hour), meaning survey time is in decimal hours.

        ..  warning::
            Setting a ``time_delta`` unit that is very small in conjunction
            with setting a 'distant' ``fixed_time_datum`` will lead to very
            large timedelta values being used in gsolve drift calculations.
            Results may then be incorrect due to floating point rounding
            errors.

        """
        return self._timedelta_unit

    def set_timedelta_unit(
        self, unit: TimedeltaScalar, set_tdelta: bool = True
    ) -> None:
        self._timedelta_unit = _pd.Timedelta(unit)
        if set_tdelta:
            self.set_tdelta()

    def fixed_time_datum(self) -> None | _pd.Timestamp:
        """Return time datum used for calculating timedelta.

        Returns
        -------
        Timestamp or None
            None if no fixed time datum has been set.
        """
        return self._fixed_time_datum

    def set_fixed_time_datum(
        self, t: DatetimeScalar | None, set_tdelta: bool = True
    ) -> None:
        """Set time datum used for calculating timedelta.

        By default, gsolve will use the earliest survey and/or loop
        observation as the time datum. The `fixed_time_datum` feature is
        provided for reproducibility purposes. Legacy gsolve versions used
        the J1900.00 epoch as the datum.

        ..  warning::
            Setting a `fixed_time_datum` that is far from the survey
            time range in conjuction with setting a small `timedelta_unit`
            will lead to very large time_delta values being used in
            gsolve drift calculations. Results may then be incorrect due
            to floating point rounding errors.

        Parameters
        ----------
        t : _pd.Timestamp or None
            The time datum to use.  If None then fixed time datum is
            removed.

        """
        if t is None or _pd.isnull(t):
            self._fixed_time_datum = None
        else:
            _t = to_naive_utc_datetime(t)
            if isinstance(_t, _pd.Timestamp):
                self._fixed_time_datum = _t
            elif _t is _pd.NaT:
                self._fixed_time_datum = None
            else:
                raise TypeError(
                    f"invalid fixed_time_datum of type '{type(t).__name__}'"
                )
        if set_tdelta:
            self.set_tdelta()

    def params(self) -> GravityObservationsParameters:
        """Return parameters as a `GravityObservationsParameters` object."""
        return GravityObservationsParameters(
            timedelta_unit=self._timedelta_unit,
            fixed_time_datum=self._fixed_time_datum,
            earthtide_correction_method=self._earthtide_correction_method,
            ocean_load_correction_method=self._ocean_load_correction_method,
        )

    def apply_dial_to_mgal(
        self,
        converter: MeterReadingConverter,
        check_meter_id: bool = True,
        check_datetime: bool = True,
        set_converter_id_column: bool = True,
        input_column_name: str = "meter_reading",
        output_column_name: str = "meter_reading_mgal",
    ) -> None:
        """Apply dial to mgal conversion to observed "meter_reading".

        Parameters
        ----------
        converter : MeterReadingConverter
            The meter reading converter object.
        check_meter_id : bool, default=True
            Only convert readings where "meter_id" matches the converter
            "meter_id".  If False, ignore "meter_id"  and convert all
            observations.
        check_datetime : bool, default=True
            Only convert readings where observation datetime falls within
            the valid the converter date range.  If False, ignore
            "datetime" and convert all readings.
        set_converter_id_column: bool, default=True
            If True set the "meter_reading_converter_id" column to the
            `converter_id` from the `MeterReadingConverter` object.
        input_column_name : str, default='meter_reading'
            The columns holding gravity readings to convert.
        output_column_name : str, default='meter_reading_mgal'
            The column to store the converted readings.
        """
        vals = converter.convert_readings(
            readings=self.data[input_column_name],
            meter_id=self.data["meter_id"].to_list() if check_meter_id else None,
            date_time=self.data["datetime"].to_list() if check_datetime else None,
        )
        m = _np.isnan(vals)

        if set_converter_id_column:
            self.data.loc[~m, "meter_reading_converter_id"] = converter.converter_id()
        self.data.loc[~m, output_column_name] = vals[~m]

    def apply_earth_tide_correction(
        self,
        sites: GravitySites,
        tide_corrector: None | EarthTideCorrectionProvider = None,
        column_name: str = "earth_tide_corr",
        **kwargs,
    ) -> None:
        """
        Compute earth tide correction and store in column ``column_name``.

        Parameters
        ----------
        sites : GravitySites
            GravitySites object providing latitude, longitude and
            height_ellipsoidal for each site.
        tide_corrector : TideCorrectionProvider, optional
            A ``TideCorrectionProvider`` object to use. If not specified, then a
            ``LongmanTidalCorrection`` object with default parameters will be used.
        column_name : str, default='earth_tide_corr'
            The column name to store the earth tide correction.
        kwargs : dict
            Additional keyword arguments passed to the provider's ``tidal_correction()``
            method.

        See Also
        --------
        gsolve.tide.earth_tide.LongmanTidalCorrection : Longman tidal correction class.
        gsolve.tide.earth_tide.gravimetric_factor : Calculate amplification factor
            from Love numbers
        """
        ids = self.data["site_id"].to_list()
        if tide_corrector is None:
            tide_corrector = LongmanTidalCorrection()

        tcorr = tide_corrector.tidal_correction(
            lat=sites.data.loc[ids, "latitude"].to_numpy(),
            lon=sites.data.loc[ids, "longitude"].to_numpy(),
            elev=sites.data.loc[ids, "height_ellipsoidal"].to_numpy(),
            date_time=self.data["datetime"],
            site_id=self.data["site_id"],
            **kwargs,
        )
        self.set_column(column_name, tcorr)
        identifier = tide_corrector.identifier(**kwargs)
        self._earthtide_correction_method = identifier
        # self.set_column(f"{column_name}_method", identifier)

    def apply_ocean_load_correction(
        self,
        corrector: OceanLoadCorrectionProvider,
        column_name: str = "ocean_load_corr",
        if_not_matched: Literal["error", "warn"] = "error",
        **kwargs,
    ) -> None:
        """Get ocean loading corrections and store in column `column_name`.

        This method calls the ``ocean_load_correction()`` method of the provided
        `ocean_load_corrector` object to retrieve ocean loading corrections for
        each observation. Ocean load corrections will typically have been pre-computed
        in some Third Party software such as Quick Tide Pro.

        Parameters
        ----------
        ocean_load_corrector : OceanLoadCorrectionProvider
            An object providing ocean loading corrections.
        sites : GravitySites, optional
            GravitySites object providing latitude, longitude and height. Only required
            if the `ocean_load_corrector` requires site location parameters.
        column_name : str, default='ocean_load_corr'
            The column name to store the ocean loading correction.
        if_not_matched : {'error', 'warn'}, default 'error'
            Behaviour when an observation cannot be matched with the corrections
            provided by the `ocean_load_corrector`. E.G. for the timeseries based
            corrector `QuickTideTimeSeries`, if datetimes are outside the range of
            the time series. Options are:

                - 'error' : raise a ValueError.
                - 'warn' : issue a warning, and return nan for unmatched observations.

        kwargs : dict
            Additional keyword arguments passed to the provider's
            ``ocean_load_correction()`` method.

        """
        if not isinstance(corrector, OceanLoadCorrectionProvider):
            raise TypeError(
                f"ocean_load_corrector must implement OceanLoadCorrectionProvider protocol"
            )

        corrections = corrector.ocean_load_correction(
            site_id=self.data["site_id"],
            date_time=self.data["datetime"],
            if_not_matched=if_not_matched,
            **kwargs,
        )
        self.set_column(column_name, corrections)
        self._ocean_load_correction_method = corrector.identifier()

    def set_calibration_factor(
        self,
        calibration_factor: float = 1.0,
        meter_id: str | None = None,
    ) -> None:
        """
        Set gravity meter calibration factor.

        Parameters
        ----------
        calibration_factor : float or array_like
            The gravity meter `calculate_calibration`. Default is 1.0
        meter_id : str, default None
            Set `calibration_factor` for specifed `meter_id` only. If data
            contains multiple gravity meters, `meter_id` must be specified.

        Raises
        ------
        ValueError
            If data contains multiple gravity meter_id's and `meter_id` is None.
            When specified `meter_id` is not in data.

        """
        c_label: str = "calibration_factor"

        if self.data["meter_id"].nunique() > 1 and meter_id is None:
            raise ValueError(
                "Multiple gravity meters found in data, must specify `meter_id`"
            )
        if meter_id is None:
            self.set_column(c_label, float(calibration_factor))
        else:
            if meter_id not in self.data["meter_id"].to_list():
                raise ValueError(f"meter_id '{meter_id}' not found in data")
            if c_label not in self.data.columns:
                self.set_column(
                    c_label,
                    self._known_fields["calibration_factor"].default,
                )
            this_meter: list[bool] = self.data["meter_id"].eq(meter_id).to_list()
            self.data.loc[this_meter, c_label] = float(calibration_factor)

    def calculate_tide_corrected_gravity(self) -> None:
        """Calculate corrected gravity values and assign to column 'gravity_corr'"""
        reading = self.data["meter_reading_mgal"]
        calibration_factor = self.data.get("calibration_factor", 1.0)
        etide_corr = self.data.get("earth_tide_corr", 0.0)
        otide_corr = self.data.get("ocean_load_corr", 0.0)
        custom_corr = self.data.get("custom_corr", 0.0)

        gcorr = reading * calibration_factor + etide_corr + otide_corr + custom_corr
        self.set_column("gravity_corr", gcorr)

    def set_tdelta(self) -> None:
        """Calculate time delta for survey and loop observations and
        assign to columns "survey_tdelta","loop_tdelta".

        The datum for survey_tdelta is the earliest observation (i.e self.starttime),
        while the datum for loop_tdelta is the earliest observation in each loop.

        The default datum(s) can be overridden by calling `set_fixed_time_datum()`

        """
        datetime_col = "datetime"
        surv_tdelta_col = "survey_tdelta"
        loop_tdelta_col = "loop_tdelta"

        # set survey tdelta's
        has_fixed_time_datum = self.fixed_time_datum() is not None
        t0 = self.fixed_time_datum()
        if t0 is None:
            t0 = self.starttime
        td_unit_seconds = self.timedelta_unit().total_seconds()

        td: _pd.Series = (
            (self.data[datetime_col] - t0).dt.total_seconds().div(td_unit_seconds)
        )
        self.set_column(surv_tdelta_col, td)

        # set loop tdelta's
        if has_fixed_time_datum:
            self.set_column(loop_tdelta_col, td)
        else:
            temp_col = self.data[loop_tdelta_col].copy()
            for l in self.loop_ids:
                this_loop = self.data["loop"].eq(l)
                date_times = self.data.loc[this_loop, datetime_col]
                td = (
                    (date_times - date_times.min())
                    .dt.total_seconds()
                    .div(td_unit_seconds)
                )
                temp_col[this_loop] = td.to_numpy()
            self.set_column(loop_tdelta_col, temp_col)

    def activate(
        self,
        obs_id: str | Iterable[str] | None = None,
        site_id: str | Iterable[str] | None = None,
        loop: str | Iterable[str] | None = None,
        add_metadata: bool = False,
    ) -> None:
        """Activate observations.

        Only 'active' observations are included in gsolve solutions.
        By default, all observations are considered 'active'. This method
        allows for the reactivation of observations that were  specified
        as inactive in the input data or by calling the `deactivate`
        method.

        Parameters
        ----------
        obs_id : str or array_like, optional
            The `obs_id` of the observations to activate.
        site_id : str or array_like, optional
            The `site_id` of the observations to activate.
        loop : str or array_like, optional
            The `loop` of the observations to activate.
        add_metadata : bool, default=False
            Not implemented.

        Raises
        ------
        Value Error :
            If any of the specified `obs_id`, `site_id` or `loop` are not
            in the data.

        See Also
        --------
        deactivate : equivalent function to deactivate observations.
        """
        if add_metadata:
            raise NotImplementedError("add_metadata not implemented")

        self._activate_deactivate(True, obs_id, site_id, loop)

    def deactivate(
        self,
        obs_id: str | Iterable[str] | None = None,
        site_id: str | Iterable[str] | None = None,
        loop: str | Iterable[str] | None = None,
        add_metadata: bool = False,
    ) -> None:
        """Deactivate observations.

        Deactivated observations are not included in gsolve solutions.

        Parameters
        ----------
        obs_id : str or array_like, optional
            The `obs_id` of the observations to deactivate.
        site_id : str or array_like, optional
            The `site_id` of the observations to deactivate.
        loop : str or array_like, optional
            The `loop` of the observations to deactivate.
        add_metadata : bool, default=False
            Not implemented.

        See Also
        --------
        activate : Activate observations
        """
        if add_metadata:
            raise NotImplementedError("add_metadata not implemented")
        self._activate_deactivate(False, obs_id, site_id, loop)

    def _activate_deactivate(
        self,
        flag: bool,
        obs_id: str | Iterable[str] | None = None,
        site_id: str | Iterable[str] | None = None,
        loop: str | Iterable[str] | None = None,
    ) -> None:
        """Activate or deactivate observations.

        Use the activate and deactivate methods rather than calling
        this method directly.

        See Also
        --------
        activate : Activate observations
        deactivate: Deactivate observations
        """

        def _parse_inputs(o: str | Iterable[str] | None) -> list[str]:
            if o is None:
                return []
            elif isinstance(o, str):
                return [o]
            elif isinstance(o, Iterable):
                return [str(oi) for oi in o]
            else:
                raise TypeError(f"invalid input of type '{type(o).__name__}'")

        _obs_id = _parse_inputs(obs_id)
        _site_id = _parse_inputs(site_id)
        _loop = _parse_inputs(loop)
        if _obs_id:
            missing = [oi for oi in _obs_id if oi not in self.data.index.to_list()]
            if missing:
                raise ValueError(f"obs_id's '{missing}' not found in data")

        if _site_id:
            missing = [s for s in _site_id if s not in self.data["site_id"].to_list()]
            if missing:
                raise ValueError(f"site_id's '{missing}' not found in data")

            _obs_id += self.data.loc[
                self.data["site_id"].isin(_site_id)
            ].index.to_list()

        if _loop:
            missing = [li for li in _loop if li not in self.loop_ids]
            if missing:
                raise ValueError(f"loop's '{missing}' not found in data")

            _obs_id += self.data.loc[self.data["loop"].isin(_loop)].index.to_list()

        if _obs_id:
            self.data.loc[list(set(_obs_id)), "active"] = flag

    def _get_writable_df(
        self,
        normalize_column_names: bool = True,
        expand_datetime: str | None = "datetime",
        drop_datetime: bool = False,
        bool_to_int: bool = True,
        include_unknown_fields: bool | Sequence[str] = False,
        active_only: bool = False,
    ) -> _pd.DataFrame:
        """Return a DataFrame suitable for writing to a file."""
        cols = [c for c in self.known_fields() if c in self.data.columns]
        if include_unknown_fields is not False:
            if include_unknown_fields is True:
                cols.extend(c for c in self.data.columns if c not in cols)
            elif is_list_like(include_unknown_fields):
                bad_fields = [
                    c for c in include_unknown_fields if c not in self.data.columns
                ]
                if bad_fields:
                    raise ValueError(
                        "invalid 'include_unknown_fields' arg: "
                        f"specified fields {bad_fields} not found in data"
                    )
                cols.extend(c for c in include_unknown_fields if c not in cols)

        if active_only:
            idx = self.data.loc[self.data["active"]].index
        else:
            idx = self.data.index

        return prepare_writable_df(
            self.data.loc[idx, cols],
            normalize_column_names=normalize_column_names,
            expand_datetime=expand_datetime,
            drop_datetime=drop_datetime,
            bool_to_int=bool_to_int,
        )

    def write_to_csv(
        self,
        fname: FilePath,
        normalize_column_names: bool = True,
        expand_datetime: str | None = "datetime",
        drop_datetime: bool = False,
        bool_to_int: bool = True,
        include_unknown_fields: bool | Sequence[str] = False,
        active_only: bool = False,
        **kwargs,
    ) -> None:
        self._get_writable_df(
            normalize_column_names=normalize_column_names,
            expand_datetime=expand_datetime,
            drop_datetime=drop_datetime,
            bool_to_int=bool_to_int,
            include_unknown_fields=include_unknown_fields,
            active_only=active_only,
        ).to_csv(fname, header=True, **kwargs)

    def to_excel(
        self,
        fname: FilePath,
        sheet_name: str | None = None,
        params_sheet_name: str | None = None,
        normalize_column_names: bool = True,
        expand_datetime: str | None = "datetime",
        drop_datetime: bool = False,
        bool_to_int: bool = True,
        include_unknown_fields: bool | Sequence[str] = False,
        active_only: bool = False,
        if_workbook_exists: IfWorkbookExists = "error",
        if_sheet_exists: IfSheetExists = "error",
        **kwargs,
    ) -> None:
        """Write data to an excel file."""
        if sheet_name is None:
            if isinstance(self._default_excel_sheet_name, str):
                sheet_name = self._default_excel_sheet_name
            else:
                sheet_name = self._default_excel_sheet_name[0]
        if params_sheet_name is None:
            params_sheet_name = f"{sheet_name}_parameters"

        kwargs["header"] = kwargs.get("header", True)
        kwargs["index"] = kwargs.get("index", True)

        write_excel_worksheet(
            self._get_writable_df(
                normalize_column_names=normalize_column_names,
                expand_datetime=expand_datetime,
                drop_datetime=drop_datetime,
                bool_to_int=bool_to_int,
                include_unknown_fields=include_unknown_fields,
                active_only=active_only,
            ),
            excel_file=fname,
            sheet_name=sheet_name,
            if_workbook_exists=if_workbook_exists,
            if_sheet_exists=if_sheet_exists,
            **kwargs,
        )
        self.params().to_excel(
            fname,
            sheet_name=params_sheet_name,
            if_workbook_exists="append",
            if_sheet_exists=if_sheet_exists,
        )

    def plot_observed_data(
        self,
        loop: str | int,
        x_column: str = "datetime",
        y_column: str = "meter_reading_mgal",
        savefilename: FilePath | None = None,
        figsize: tuple[float, float] = (12, 8),
        ax=None,  # noqa: ANN001
        **kwargs,
    ) -> tuple[_plt.Figure, _plt.Axes]:
        """
        Plot observed data.

        Parameters
        ----------
        loop : str or int
            The loop to plot. Use `loop='all'` to plot all data, ignoring loops.
        x_column : str, default='datetime'
            The 'x' data column to plot.
        y_column : str, default='meter_reading_mgal'
            The 'y' data column to plot.
        savefilename : FilePath, default=None
            If not None, save the plot to `savefilename`.
        figsize : tuple, default=(12, 8)
            The pyplot figure size.
        ax = None
        **kwargs : dict
            Optional keyword arguments passed directly to  `matplotlib.pyplot.plot()`.


        Returns
        -------
            Plot of the observed data.
        """
        if loop == "all":
            y_data = self.data.loc[:, y_column]
            x_data = self.data.loc[:, x_column]
        else:
            loop = self._known_fields["loop"].dtype(loop)
            if loop not in self.loop_ids:
                raise ValueError(f"loop '{loop}' not found in data")
            this_loop = self.data["loop"].eq(loop)
            y_data = self.data.loc[this_loop, y_column]
            x_data = self.data.loc[this_loop, x_column]

        if "marker" not in kwargs:
            kwargs["marker"] = "x"

        fig = _plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.plot(x_data, y_data, **kwargs)

        if x_column == "datetime":
            ax.xaxis.set_major_formatter(_mdates.DateFormatter("%d-%m-%Y %H:%M:%S"))
            ax.set_xlabel("UTC Datetime")
            for label in ax.get_xticklabels(which="major"):
                label.set(rotation=20, horizontalalignment="right")
        else:
            ax.set_xlabel(x_column)

        ax.set_title(f"Plot of the observed data for loop {loop}.")
        ax.set_ylabel("mGal")
        _plt.tight_layout()

        if savefilename is not None:
            fout = pathlib.Path(savefilename)
            fout = fout.parent / f"{fout.stem}_loop_{loop}{fout.suffix}"
            _plt.savefig(fout, dpi=300)

        fig.show()
        return fig, ax

    def _make_network(self, sites: GravitySites) -> _pd.DataFrame:
        self.data["group"] = (
            self.data["site_id"] != self.data["site_id"].shift()
        ).cumsum()
        result = self.data.groupby("group").first().reset_index(drop=True)
        station_order = result[["site_id", "loop"]]
        # merge with 'site' object to get location information
        merged_df = _pd.merge(station_order, sites.data, on="site_id", how="inner")
        network_df = merged_df[["site_id", "loop", "latitude", "longitude"]]
        sites.data["station_occupations"] = network_df.site_id.value_counts()
        self.data.drop(["group"], axis=1, inplace=True)
        return network_df

    def plot_network_map(
        self,
        sites: GravitySites,
        savefilename: FilePath | None = None,
        figsize: tuple[float, float] = (10, 10),
        marker_scale_factor: float = 25,
        plot_stn_labels: bool = False,
        ax: _plt.Axes | None = None,
        **kwargs,
    ) -> tuple[_plt.Figure, _plt.Axes]:
        """
        Plot network map showing connections between stations.
        Station markers are scaled according to the number of occupations

        Parameters
        ----------
        sites : GravitySites
            DESCRIPTION.
        savefilename : str or PathLike, optional
            DESCRIPTION. The default is None.
        figsize : tuple, default=(10, 10)
            The pyplot figure size.
        marker_scale_factor: float, default=25
            Scale marker size by this value.
        plot_stn_labels: bool, default=False
            Plot station name next to station points.
        ax=None
        **kwargs : dict
            Optional keyword arguments passed directly to  `matplotlib.pyplot.plot()`.

        Returns
        -------
        A figure showing connections between sites as well as a tuple containing the
        figure and axis objects (fig, ax).

        """
        if "marker" not in kwargs:
            kwargs["marker"] = "o"

        if ax is None:
            fig, ax = _plt.subplots(figsize=figsize)
        elif isinstance(ax, _plt.Axes):
            fig = ax.figure.get_figure(root=True)
        else:
            raise TypeError(f"invalid ax arg of type '{type(ax).__name__}'")

        network_df = self._make_network(sites)
        ax.plot(network_df.longitude, network_df.latitude, **kwargs)
        ax.scatter(
            x=sites.data.longitude.to_numpy(),
            y=sites.data.latitude.to_numpy(),
            s=sites.data.station_occupations * marker_scale_factor,
            label="n_occupations",
            **kwargs,
        )
        if plot_stn_labels:
            for long, lat, site in zip(
                sites.data.longitude.values,
                sites.data.latitude.values,
                sites.data.index.values,
            ):
                ax.text(long, lat, site, ha="left", va="bottom")
        else:
            None

        ax.set_aspect(1)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title("Survey network")
        _plt.legend(loc="best")
        if savefilename is not None:
            fout = pathlib.Path(savefilename)
            fout = fout.parent / f"{fout.stem}{fout.suffix}"
            _plt.savefig(fout, dpi=300)

        _ = fig.show()

        return fig, ax

    def plot_site_visits(self, loop: str) -> None:
        """
        Plot the order of station visits for each loop.

        Parameters
        ----------
        loop : str
            Loop number to plot.

        Returns
        -------
        A figure showing the station occupation order for each loop.

        """
        df = self.data.loc[self.data["loop"].eq(loop)]

        site_id_to_int = {v: k for k, v in enumerate(df["site_id"].unique())}
        ax = df.assign(site_id_n=df["site_id"].map(site_id_to_int)).plot(
            y="site_id_n", x="loop_tdelta", style=".-", fontsize=10
        )
        _ = ax.set_yticks(
            ticks=list(site_id_to_int.values()), labels=list(site_id_to_int.keys())
        )

    def loop_summary(self) -> _pd.DataFrame:
        """Return a summary of the observations by loop."""
        from gsolve.core._summary_functions import (
            duration_hr,
            endtime_utc,
            n_sites,
            starttime_utc,
        )

        agg_dict = {
            "site_id": n_sites,
            "datetime": [starttime_utc, endtime_utc, duration_hr],
        }
        return self.data.groupby("loop").agg(agg_dict).droplevel(0, axis=1)

    def site_summary(self, data_col: str | None = None) -> _pd.DataFrame:
        """Return a summary of the observations by site."""
        from gsolve.core._summary_functions import (
            in_loops,
            n,
            n_inactive,
            range_ugal,
            stdev_ugal,
        )

        agg_dict: dict[str, Any] = {"active": [n, n_inactive]}

        if data_col is None:
            if self.data["gravity_corr"].notna().all():
                data_col = "gravity_corr"
            elif self.data["meter_reading_mgal"].notna().all():
                data_col = "meter_reading_mgal"
            else:
                data_col = "meter_reading"

        if data_col not in self.data.columns:
            raise ValueError(f"Data column '{data_col}' not found in data")

        agg_dict[data_col] = ["median", "mean", stdev_ugal, range_ugal]
        agg_dict["loop"] = in_loops

        return (
            self.data.groupby("site_id")
            .agg(agg_dict)
            .droplevel(0, axis=1)
            .rename_axis(data_col, axis=1)
        )

    def check_data(self, warn: bool = True) -> bool:
        """Check the data for errors.

        Parameters
        ----------
        warn : bool, default=True
            If True, print warnings.

        Returns
        -------
        bool
            True if data is OK, False otherwise.
        """
        warner = GSolveDataWarning(prefix=f"{type(self).__name__} error", show=warn)
        for c in self.required_fields():
            if c not in self.data.columns:
                warner(f"missing required input column '{c}'")
            elif self.data[c].isna().any() or self.data[c].eq("").any():
                warner(f"null data found in input column '{c}'")
            # elif self._known_fields[c]:
            #     if self.data[c].eq(0).any():
            #         warner(f"zero values found in input column '{c}'")
        if "active" not in self.data.columns:
            warner("'active' column not found in data")
        elif not self.data["active"].any():
            warner("all observations are inactive")

        calculated_cols = [
            "loop_tdelta",
            "survey_tdelta",
            "meter_reading_mgal",
            "calibration_factor",
            "gravity_corr",
            "earth_tide_corr",
        ]

        for c in calculated_cols:
            if c not in self.data.columns:
                warner(f"missing required derived/calculated field '{c}'")
            elif self.data[c].isna().any() or self.data[c].eq("").any():
                warner(f"null data found required derived/calculated field  '{c}'")

        other_cols = ["ocean_load_corr", "custom_corr"]
        for c in other_cols:
            if c in self.data.columns and (
                self.data[c].isna().any() or self.data[c].eq("").any()
            ):
                warner(f"null data found in optional column '{c}'")

        if self.data.index.duplicated().any():
            warner(f"{self.data.index.name} has duplicates")

        if "loop" in self.data.columns and "meter_id" in self.data.columns:
            for l in self.loop_ids:
                m = self.data["loop"].eq(l)
                if self.data.loc[m, "meter_id"].nunique() > 1:
                    warner(f"Multiple gravity meters found in loop '{l}'")

        warner.final_msg()
        return warner.count == 0


class GravitySurvey:
    """Class to store gravity observations and sites and facilitate running gsolve

    Parameters
    ----------
    obs : GravityObservations
        The gravity observations object.
    sites : GravitySites
        The gravity sites object.
    .
    """

    def __init__(self, obs: GravityObservations, sites: GravitySites) -> None:
        self.observations: GravityObservations = obs
        self.sites: GravitySites = sites
        # self.results = []
        self.observations.check_data()
        self.sites.check_data()

    @classmethod
    def from_excel(
        cls,
        fname: FilePath,
        ignore_unknown_fields: bool = True,
        parse_split_datetime: bool = True,
    ) -> Self:
        """Read gravity observations and sites from an excel file.

        Parameters
        ----------
        fname : str or PathLike
            The path to the excel file.
        ignore_unknown_fields : bool, default is True
            Ignore unknown fields in the excel file.
        parse_split_datetime : bool, default is True
            Parse split datetime fields into a single datetime column.
        """
        obs = GravityObservations.from_excel(
            fname,
            ignore_unknown_fields=ignore_unknown_fields,
            parse_split_datetime=parse_split_datetime,
        )
        sites = GravitySites.from_excel(
            fname,
            ignore_unknown_fields=ignore_unknown_fields,
        )
        return cls(obs, sites)

    # def set_reference_gravity(
    #     self, ref_grav: ReferenceGravity | _pd.DataFrame, reset: bool = False
    # ) -> _pd.DataFrame | _pd.Series:
    #     return self.sites.set_reference_gravity(ref_grav, reset)

    def apply_dial_to_mgal(
        self,
        converter: MeterReadingConverter,
        input_column_name: str = "meter_reading",
        output_column_name: str = "meter_reading_mgal",
    ) -> None:
        """Apply dial to mgal conversion to observations.

        Parameters
        ----------
        converter : MeterReadingConverter, optional
            The dial to mgal converter object. If None, `meter_reading`
            data are assumed to be in mgal and no conversion is applied.
        input_column_name : str, default='meter_reading'
            The input column name to convert.
        output_column_name : str, default='meter_reading_mgal'
            The output column name to store the converted data.
        """
        self.observations.apply_dial_to_mgal(
            converter=converter,
            input_column_name=input_column_name,
            output_column_name=output_column_name,
        )

    def apply_earth_tide_correction(
        self, tide_corrector: EarthTideCorrectionProvider, **kwargs
    ) -> None:
        self.observations.apply_earth_tide_correction(
            self.sites, tide_corrector, **kwargs
        )

    def set_calibration_factor(
        self, calibration_factor: float = 1.0, meter_id: str | None = None
    ) -> None:
        self.observations.set_calibration_factor(calibration_factor, meter_id=meter_id)

    def calculate_tide_corrected_gravity(self) -> None:
        self.observations.calculate_tide_corrected_gravity()

    def set_reference_gravity(
        self,
        ref_grav: ReferenceGravity | _pd.DataFrame,
        reset: bool = False,
    ) -> None:
        self.sites.set_reference_gravity(ref_grav, reset)

    # def summary(self, fmt: str = "dict") -> dict | _pd.DataFrame:
    #     raise NotImplementedError("summary not implemented")
    #     return self.observations.summary()

    def pre_flight_check(self, warn: bool = True) -> bool:
        """Check data are valid before running gsolve.

        Parameters
        ----------
        warn : bool, default=True
            If True, print warnings.
        """
        rval = True
        if not self.observations.check_data(warn=warn):
            rval = False
        if not self.sites.check_data(warn=warn):
            rval = False

        warner = GSolveDataWarning(prefix="Pre-flight checks", show=warn)

        # check that all sites in observations have a corresponding site in sites
        m = self.observations.data["site_id"].isin(self.sites.data.index)
        if not m.all():
            missing = self.observations.data.loc[~m, "site_id"].unique().tolist()
            warner(f"Observations without corresponding site data: {missing}")
            rval = False
        warner.final_msg()

        return rval

    def solve_lstsq(
        self,
        method: GSolveSolverMethod,
        percentile_clipping: float = 100,
        use_loops: bool = True,
        calculate_calibration_factor: bool = False,
    ) -> GSolveResults:
        self.observations.set_tdelta()

        td_column = "loop_tdelta" if use_loops else "survey_tdelta"

        isactive = self.observations.data["active"]
        cols = ["site_id", td_column, "loop", "gravity_corr", "meter_reading_mgal"]
        obs = (
            self.observations.data.loc[isactive, cols]
            .copy()
            .rename(columns={td_column: "timedelta", "gravity_corr": "gravity"})
        )
        ties = self.sites.get_ties()
        results = call_gsolve_lstsq(
            obs=obs,
            ref_sites=ties,
            method=method,
            percentile_clipping=percentile_clipping,
            use_loops=use_loops,
            calculate_calibration_factor=calculate_calibration_factor,
        )
        return results


def combine_gravity_observations(
    obs: list[GravityObservations],
    duplicated_loops: Literal["error", "keep", "drop", "rename"] = "error",
    duplicated_obs_ids: Literal["error", "drop", "rename", "regenerate"] = "error",
) -> GravityObservations:
    """Merge 2 or more GravityObservations objects.

    The returned object is formed by concatenating the `data` DataFrames attributes
    of each `obs` object, and then instantiating a new GravityObservations object.
    Non `data` attributes of the new object (e.g. `timedelta_unit`) are set
    from the the first `obs` specified.

    Parameters
    ----------
    obs : GravityObservations
        Sequence of two or more `GravityObservations` objects to be combined.
    duplicated_loops : {'error', 'keep', 'drop', 'rename'}, defaut is 'error'
        How to handle situations where ``loop`` identifiers are duplicated between
        GravityObservations objects:

            - ``'error'`` : raise a ValueError
            - ``'keep'`` : duplicates are unchanged
            - ``'drop'`` : drop all data with duplicated loop_id
            - ``'rename'`` : rename the duplicate loops by adding suffix
              ``_merged_{int}`` where ``{int}`` refers to the position in
              the input ``obs`` array.

    duplicated_obs_id : {'error', 'drop', 'rename'}, default is 'error'
        How to handle situations where ``obs_id`` identifiers  are duplicated between
        GravityObservations objects:

            - ``'error'`` : raise a ValueError
            - ``'drop'`` : drop data with duplicated ``obs_id``.
            - ``'rename'`` : rename the duplicate obs_id's by adding suffix
              ``_merged_{int}``` where ``{int}`` refers to the position in
              the input ``obs`` array.
            - ``'regenerate'`` : generate new obs_id's for all data in the
              merged object.

    Returns
    -------
    GravityObservations
        The new GravityObservations object.

    """
    if not is_list_like(obs) or len(obs) < 2:
        raise ValueError("Must specify at least 2 GravityObservations objects.")

    if not all([isinstance(o, GravityObservations) for o in obs]):
        raise TypeError(f"invalid type for elements in obs")

    if duplicated_loops not in ("error", "keep", "drop", "rename"):
        raise ValueError(
            f"invalid duplicated_loops arg '{duplicated_loops}', "
            "must be one of 'error', 'keep', 'drop', 'rename'"
        )
    if duplicated_obs_ids not in ("error", "drop", "rename"):
        raise ValueError(
            f"invalid duplicated_obs_id arg {duplicated_obs_ids},"
            "must be one of 'error', 'drop', 'rename'"
        )

    target = obs.pop(0)
    target_df = target.data
    regen_obs_ids = False

    # cumulatively concat observations to obs[0]
    for i, o in enumerate(obs, 1):
        other_df = o.data.copy()

        # check that loops are unique
        is_duplic_loop = other_df["loop"].isin(target_df["loop"].unique())
        is_duplic_loop_count = is_duplic_loop.sum()
        rename_suffix = f"_merged_{i}"

        if is_duplic_loop_count > 0:
            duplic_loops_unique = other_df.loc[is_duplic_loop, "loop"].unique().tolist()
            msg = (
                f"{is_duplic_loop_count} duplicate loop ids found in merge target "
                f"{i}: {duplic_loops_unique}"
            )

            if duplicated_loops == "error":
                raise ValueError(msg)

            elif duplicated_loops == "keep":
                _warnings.warn(f"keeping {msg}")

            elif duplicated_loops == "drop":
                _warnings.warn(f"dropping {msg}")
                other_df = other_df.loc[~is_duplic_loop]

            elif duplicated_loops == "rename":
                other_df.loc[is_duplic_loop, "loop"] = (
                    other_df.loc[is_duplic_loop, "loop"]
                    .astype(str)
                    .str.cat([rename_suffix] * is_duplic_loop_count)
                )
                _warnings.warn(f"adding suffix '{rename_suffix}' to {msg}")

        # check that obs_id are unique
        is_dupe_obsid = other_df.index.isin(target_df.index)
        not_dupe_obsid = (~is_dupe_obsid).tolist()
        is_duplic_loop_count = is_dupe_obsid.sum()
        is_dupe_obsid = is_dupe_obsid.tolist()

        if is_duplic_loop_count > 0:
            msg = f"{is_duplic_loop_count} duplicate obs_id's found in merge target {i}"

            if duplicated_obs_ids == "error":
                raise ValueError(msg)

            if duplicated_obs_ids == "drop":
                _warnings.warn(f"dropping {msg}")
                other_df = other_df.loc[not_dupe_obsid]

            elif duplicated_obs_ids == "rename":
                _warnings.warn(f"adding suffix '{rename_suffix}' to {msg}")
                ds = other_df.index.to_series()
                ds.loc[is_dupe_obsid] = (
                    ds.loc[is_dupe_obsid]
                    .astype(str)
                    .str.cat([rename_suffix] * is_duplic_loop_count)
                )
                other_df = other_df.set_index(_pd.Index(ds.to_list(), name="obs_id"))

            if duplicated_obs_ids == "regenerate":
                _warnings.warn(
                    f"duplicate 'obs_id' found, will regenerate 'obs_id"
                    "for regenerating obs_id's for all data: {msg}"
                )
                regen_obs_ids = True

        target_df = _pd.concat([target_df, other_df], axis=0)

    obj = GravityObservations.from_dataframe(target_df, use_index=True)
    obj.set_fixed_time_datum(target.fixed_time_datum())
    obj.set_timedelta_unit(target.timedelta_unit())
    if regen_obs_ids:
        obj.set_obs_id()

    return obj


def combine_gravity_surveys(
    surveys: Sequence[GravitySurvey],
    duplicated_loops: Literal["error", "keep", "drop", "rename"] = "error",
    duplicated_obs_ids: Literal["error", "drop", "rename", "regenerate"] = "error",
    duplicated_sites: Literal["error", "drop"] = "error",
) -> GravitySurvey:
    """Merge 2 or more GravitySurveys objects.

    The returned object is formed by concatenating the `observations` DataFrames
    attributes of each object in `survey`, and then instantiating a new GravitySurvey
    object. Non `observations` attributes of the new object (e.g. `timedelta_unit`) are
    set from the the first `survey` specified.

    Parameters
    ----------
    surveys : GravitySurveys
        A list of two or more `GravitySurveys` objects to be combined.
    ignore_duplicates : bool, default is False
        Specify how data with duplicate index and/or loop values are to be handled.
        If True, duplicates will be merged unchanged. If False, raise a ValueError.

    Returns
    -------
    GravitySurvey
        The new GravitySurvey object.

    Raises
    ------
    ValueError
        If less than 2 `GravitySurveys` are specified.
        If duplicate `obs_id` or `loop` values are found and ignore_duplicates=False.
    TypeError
        If any of `*surveys` is not a GravitySurvey object.

    """
    if len(surveys) < 2:
        raise ValueError("Must specify at least 2 GravitySurveys objects.")

    for i, s in enumerate(surveys):
        if not isinstance(s, GravitySurvey):
            raise TypeError(
                f"All arguments must be GravitySurvey objects, arg {i} type='{type(s)}'"
            )

    final_obs = combine_gravity_observations(
        [s.observations for s in surveys],
        duplicated_loops=duplicated_loops,
        duplicated_obs_ids=duplicated_obs_ids,
    )
    final_sites = combine_gravity_sites(
        [s.sites for s in surveys], duplicates=duplicated_sites
    )

    return GravitySurvey(final_obs, final_sites)
