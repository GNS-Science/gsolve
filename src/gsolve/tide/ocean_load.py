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

import os
import warnings
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pyhardisp
from numpy.typing import NDArray
from pandas.api.typing import NaTType

from gsolve.core._typing import (
    DatetimeArray,
    DatetimeScalar,
    FilePath,
    FloatArray,
    SiteIDArray,
)
from gsolve.core.utils import to_1d_ndarray, to_naive_utc_datetime


def _read_csv_with_fallback(file_path: FilePath, **kwargs) -> pd.DataFrame:
    """Read a CSV trying UTF-8 (with BOM) first, then fall back to iso-8859-1.

    Some QuickTide Pro outputs may be UTF-8 with a BOM; others may be
    iso-8859-1. Try the safe 'utf-8-sig' which strips a leading BOM, and
    fall back to latin1 if that fails.
    """
    try:
        return pd.read_csv(file_path, encoding="utf-8-sig", **kwargs)
    except (UnicodeDecodeError, ValueError):
        return pd.read_csv(file_path, encoding="iso-8859-1", **kwargs)


__all__ = [
    "OceanLoadTimeSeries",
    "OceanLoadAtSiteTime",
    "qtp_to_corrector",
    "generate_qtp_input",
]


@runtime_checkable
class OceanLoadCorrectionProvider(Protocol):
    """Protocol defining interface for classes that provide ocean loading corrections."""

    def ocean_load_correction(
        self,
        site_id: SiteIDArray,
        date_time: DatetimeArray,
        if_not_matched: Literal["error", "warn"] = "error",
        **kwargs,
    ) -> NDArray[np.float64]: ...

    def identifier(self, **kwargs) -> str: ...


class OceanLoadAtSiteTime(OceanLoadCorrectionProvider):
    """
    A class to provide ocean load corrections at discrete locations and times.

    The class is effctively a lookup table populated with precaclculated ocean load
    correction values for multiple at arbitrary times. Corrections are
    retrieved by matching a site identifier and datetime.

    Parameters
    ----------
    site_id : array-like[str]
        Site identifiers corresponding to each correction value.
    datetimes : DatetimeArray
        Sequence of datetime values corresponding to each correction value.
        Must have the same length as `site_id`.
    corrections : Sequence[float]
        Ocean load correction values in mGal. Must have the same length as
        `site_id` and `datetimes`.
    **metadata : dict[str, Any]
        Additional metadata to be stored in the `obj.metadata` dictionary.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame containing correction values, indexed by (site_id, datetime).
    metadata : dict[str, Any]
        Dictionary containing metadata about the corrections.

    Examples
    --------
    >>> # Create a generic ocean load correction provider
    >>> site_ids = ['SITE_A', 'SITE_A', 'SITE_B']
    >>> datetimes = pd.to_datetime(['2023-01-01 12:00', '2023-01-01 13:00', '2023-01-01 12:00'])
    >>> corrections = [0.025, 0.030, 0.015]  # mGal
    >>> provider = OceanLoadMultiStationGeneric(site_ids, datetimes, corrections)
    >>>
    >>> # Get corrections for specific site/datetime pairs
    >>> corr = provider.ocean_load_correction(
    ...     site_ids=['SITE_A'],
    ...     datetimes=pd.to_datetime(['2023-01-01 12:00'])
    ... )
    """

    def __init__(
        self,
        site_id: SiteIDArray,
        date_time: DatetimeArray,
        corrections: FloatArray,
        **metadata,
    ) -> None:
        if isinstance(site_id, str):
            _site_id = np.array([site_id] * len(date_time))
        _site_id = np.atleast_1d(site_id).astype(str)
        if _site_id.ndim != 1:
            raise ValueError("site_id argument must be 1-dimensional.")

        datetimes = _datetimes_to_np_datetime64(date_time)
        self.data = (
            pd.DataFrame(
                data={"correction": corrections},
                index=pd.MultiIndex.from_arrays([_site_id, datetimes]),
            )
            .sort_index()
            .drop_duplicates(ignore_index=False)
        )
        self.metadata: dict[str, Any] = metadata

    def identifier(self, **kwargs) -> str:
        """Corrector identifier string."""
        return f"{type(self).__name__}()"

    def ocean_load_correction(
        self,
        site_id: SiteIDArray,
        date_time: DatetimeScalar | DatetimeArray,
        if_not_matched: Literal["error", "warn"] = "error",
        **kwargs,
    ) -> NDArray[np.float64]:
        """
        Get ocean load corrections for specified site-datetime pairs.

        Parameters
        ----------
        site_id : array-like[str]
            Site identifiers where corrections are requested.
        datetime : datetime-like or array-like
            Datetime values for which to get corrections. Must have the same
            length as `site_id`.
        if_not_matched : {"error", "warn"}, optional
            Action to take when site_id/datetime pairs are not found in the data.
            If "error" (default), raises ValueError. If "warn", issues a warning
            and returns NaN for missing values.
        **kwargs : dict[str, Any]
            Additional keyword arguments. (Not used).

        Returns
        -------
        np.ndarray
            Array of ocean load corrections in mGal. Missing values are set to NaN.

        """
        dt = np.atleast_1d(_datetimes_to_np_datetime64(date_time))

        if isinstance(site_id, str):
            _site_id = np.array([site_id] * len(dt), dtype=str)
        else:
            _site_id = np.atleast_1d(site_id).astype(str)

        if len(_site_id) != len(dt):
            raise ValueError(
                "site_id and datetime arguments must have the same length."
            )

        rval = pd.Series(
            index=pd.MultiIndex.from_arrays([_site_id, dt]), data=np.nan, dtype=float
        )

        present_mask = rval.index.isin(self.data.index)
        missing_mask = (~present_mask).tolist()
        present_mask = present_mask.tolist()

        if any(missing_mask):
            missing = rval.index[missing_mask]

            msg = (
                f"{len(missing)} of {len(rval)} site_id/datetime pairs not found: "
                f"{missing.tolist()}"
            )
            if if_not_matched == "error":
                raise ValueError(msg)
            else:
                warnings.warn(msg, UserWarning)

        if any(present_mask):
            rval.loc[present_mask] = self.data.loc[
                rval.index[present_mask], "correction"
            ]

        return rval.to_numpy().astype(np.float64)


class OceanLoadTimeSeries(OceanLoadCorrectionProvider):
    """
    A class to provide ocean load corrections at discrete times for a single location/station,
    by interpolation.

    Parameters
    ----------
    datetimes : DatetimeArray
        Sequence of datetime values corresponding to each correction value.
    corrections : Sequence[float]
        Ocean load correction values in mGal. Must have the same length as
        `datetimes`.
    **metadata : dict[str, Any]
        Additional metadata to be stored in the `obj.metadata` dictionary.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame containing correction values, indexed by datetime.
    metadata : dict[str, Any]
        Dictionary containing metadata about the corrections.

    Examples
    --------
    >>> # Create a generic ocean load correction provider for a single station
    >>> datetimes = pd.to_datetime(['2023-01-01 12:00', '2023-01-01 13:00'])
    >>> corrections = [0.025, 0.030]  # mGal
    >>> provider = OceanLoadSingleStationGeneric(datetimes, corrections)
    >>>
    >>> # Get corrections for specific datetimes
    >>> corr = provider.ocean_load_correction(
    ...     datetime=pd.to_datetime(['2023-01-01 12:00'])
    ... )
    """

    def __init__(
        self,
        date_time: DatetimeArray,
        corrections: FloatArray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        datetimes = _datetimes_to_np_datetime64(date_time)
        self.data = (
            pd.DataFrame(
                data={"correction": corrections},
                index=pd.Index(datetimes, name="datetime"),
            )
            .sort_index()
            .drop_duplicates(ignore_index=False)
        )

        _validate_timeseries_data(self.data)

        self.metadata: dict[str, Any] = {}
        if metadata is not None:
            self.metadata.update(metadata)

    def __repr__(self) -> str:
        cname = self.__class__.__name__
        md = ",".join([f"{v}={k}" for v, k in self.metadata.items()])
        return f"{cname}({md})"

    def identifier(self, **kwargs) -> str:
        """Corrector identifier string."""
        return f"{self.__class__.__name__}()"

    @property
    def sample_rate(self) -> float:
        """The mean sampling interval in decimal seconds."""
        return (self.endtime - self.starttime).total_seconds() / (len(self.data) - 1)

    @property
    def starttime(self) -> pd.Timestamp:
        """The start time of the timeseries data."""
        return self.data.index[0]

    @property
    def endtime(self) -> pd.Timestamp:
        """The end time of the timeseries data."""
        return self.data.index[-1]

    def ocean_load_correction(
        self,
        site_id: SiteIDArray,
        date_time: DatetimeScalar | DatetimeArray,
        if_not_matched: Literal["error", "warn"] = "error",
        **kwargs,
    ) -> NDArray[np.float64]:
        # reformat datimes to be numpy datetime64[s] ie seconds precision
        t = _datetimes_to_np_datetime64(date_time, dtype="datetime64[s]")
        df_t = self.data.index.to_numpy(dtype="datetime64[s]")

        outside_timeseries_span = (t < self.starttime) | (t > self.endtime)

        if np.any(outside_timeseries_span):
            msg = (
                f"Warning: {sum(outside_timeseries_span)} of {len(t)} datetime "
                "args are outside time series range: "
                f"({self.starttime} <-> {self.endtime})"
            )
            if if_not_matched == "warn":
                warnings.warn(msg, UserWarning)
            else:
                raise ValueError(msg)

        ocean_load_corrs = np.interp(
            t.astype(np.int64),
            df_t.astype(np.int64),
            self.data["BergerLoadCorrection"].to_numpy(),
            **kwargs,
        )
        if any(outside_timeseries_span):
            ocean_load_corrs[outside_timeseries_span] = np.nan
        return ocean_load_corrs


def _datetimes_to_np_datetime64(
    dt: DatetimeScalar | DatetimeArray, dtype: str = "datetime64"
) -> np.ndarray:
    """Convert datetimes to numpy datetime64 array."""
    _dt = to_naive_utc_datetime(dt, allow_nat=False)
    if isinstance(_dt, pd.Timestamp):
        return np.array([_dt], dtype=dtype)
    if isinstance(_dt, (pd.DatetimeIndex, pd.Series)):
        return np.atleast_1d(_dt).astype(dtype)
    else:
        raise TypeError(
            "datetimes must be a pandas Timestamp, DatetimeIndex, or Series, not "
            f"{type(dt).__name__}."
        )


def _validate_timeseries_data(df: pd.DataFrame) -> None:
    """Test that timeseries data is valid, raise ValueError or TypeError if not.

    Data must be:

        - a pandas DataFrame with at least two rows
        - indexed by datetime
        - sorted in increasing order
        - contain a "correction" or "signal" timeseries
        - have uniform sampling interval/rate (warn if not)
    """
    # test size and
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, not {type(df).__name__}.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("timeseries not indexed by datetime.")
    if df.shape[0] < 2:
        raise ValueError("timeseries must contain at least two rows.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("timeseries not sorted in increasing order.")

    # warn if non-uniform sampling interval/rate
    sample_intervals = (df.index[1:] - df.index[:-1]).total_seconds()
    if not np.allclose(sample_intervals[1:], sample_intervals[1]):
        warnings.warn("timeseries has non-uniform sampling interval/rate.", UserWarning)


def qtp_to_corrector(
    file_path: FilePath,
    corr_type: Literal["auto", "timeseries", "site-datetime"] = "auto",
    metadata: dict[str, Any] | None = None,
) -> OceanLoadTimeSeries | OceanLoadAtSiteTime:
    if metadata is None:
        metadata = {}
    if "file_path" not in metadata:
        try:
            metadata["file_path"] = str(file_path)
        except Exception:
            metadata["file_path"] = repr(file_path)

    if corr_type == "auto":
        # determine file type by reading first line
        with open(file_path, "r", encoding="iso-8859-1") as f:
            first_line = f.readline()
        if first_line.strip().startswith("Year DOY  Time"):
            corr_type = "timeseries"
        else:
            corr_type = "site-datetime"

    if corr_type == "timeseries":
        df = read_qtp_timeseries(file_path)
        corr = OceanLoadTimeSeries(
            date_time=df.index,
            corrections=df["BergerLoadCorrection"].to_numpy(),
            metadata=metadata,
        )

    elif corr_type == "site-datetime":
        df = read_qtp_multistation(file_path)
        corr = OceanLoadAtSiteTime(
            site_id=df.index.get_level_values("site_id"),
            date_time=df.index.get_level_values("datetime"),
            corrections=df["BergerLoadCorrection"].to_numpy().astype(float),
            **metadata,
        )
    else:
        raise ValueError(
            f"invalid corr_type '{corr_type}', must be one of {'auto', 'timeseries', 'site-datetime'}."
        )

    return corr


def read_qtp_timeseries(file_path: FilePath) -> pd.DataFrame:
    """Read an ocean load timeseries generated by QuickTide Pro.

    Parameters
    ----------
    file_path : str or PathLike
        Path to the QuickTide Pro output file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the ocean load timeseries, indexed by datetime.

    """

    df = _read_csv_with_fallback(file_path, sep=r"\s+")
    # construct datetimes from Year, DOY, Time columns
    expected_columns = ["Year", "DOY", "Time"]
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(
            f"Format error reading '{file_path}': expected columns {expected_columns} not found, not QTP timeseries format?"
        )
    if df.isna().any(axis=None):
        raise ValueError(
            f"Missing values detected while reading '{file_path}': not QTP timeseries format?"
        )

    dt_strings: pd.Series = (
        df["Year"]
        .astype(str)
        .str.zfill(4)
        .str.cat(df["DOY"].astype(str).str.zfill(3), sep="-")
        .str.cat(df["Time"].astype(str), sep=" ")
    )

    datetimes = to_naive_utc_datetime(dt_strings, format="%Y-%j %H:%M:%S")

    df = (
        df.assign(datetime=datetimes)
        .drop(columns=["Year", "DOY", "Time"])
        .set_index("datetime")
        .multiply(1e-3)  # convert from uGal to mGal
        .rename(columns={col: col.rsplit(r"(")[0].strip() for col in df.columns})
    )

    # convert signal to correction if needed
    if any(col.endswith("Signal") for col in df.columns):
        df = df.rename(
            columns={c: f"{c.rstrip('Signal')}Correction" for c in df.columns}
        ).multiply(-1.0)

    _validate_timeseries_data(df)

    return df


def read_qtp_multistation(file_path: FilePath) -> pd.DataFrame:
    """Read a ocean load from a CSV file."""

    column_definitions: dict[str, type] = {
        "site_id": str,
        "datetime": str,
        "latitude": float,
        "longitude": float,
        "elevation": float,
        "BergerCorrection": float,
        "BergerLoadCorrection": float,
        "BergerCombinedCorrection": float,
        "ETGTABCorrection": float,
        "ETGTABLoadCorrection": float,
        "ETGTABCombined": float,
        "CombinedDiff": float,
    }
    df = _read_csv_with_fallback(
        file_path,
        header=None,
        dtype=column_definitions,
    )
    if df.shape[1] != len(column_definitions):
        raise ValueError(
            f"Format error reading '{file_path}': not QTP multiistation ocean load format?"
        )
    if df.isna().any(axis=None):
        raise ValueError(
            f"Missing values detected while reading '{file_path}': missing values detected, not QTP multi-station ocean load format?"
        )
    df.columns = list(column_definitions.keys())

    # Remove UTF-8 BOM if present in site_id (some QTP files include a BOM)
    # Ensure we operate on a plain Python string dtype, then strip any BOM
    # df['site_id'] = df['site_id'].astype(str).str.replace(r'^(?:\ufeff|ï»¿)', '', regex=True)
    # convert to datetime round to nearest minute, convert to mGal
    df["datetime"] = to_naive_utc_datetime(df["datetime"], format=r"%m/%d/%Y %H:%M")
    df["datetime"] = df["datetime"].dt.round("60s")
    df.iloc[:, 5:] *= 1e-3
    return df.set_index(["site_id", "datetime"]).sort_index()


def generate_qtp_input(
    site_id: SiteIDArray,
    datetimes: DatetimeArray,
    latitude: FloatArray,
    longitude: FloatArray,
    elevation: FloatArray | float | int | np.floating,
    output_file: FilePath,
) -> None:
    """
    Generate a QuickTide Pro site-time input CSV file from gravity observations and site data.

    The resultant CSV file can be used as an input to QuickTide Pro for generating
    ocean load corrections for multiple gravity stations.

    Parameters
    ----------
    site_id : SiteIDArray
        The unique site identifier for each site/datetime pair.
    datetimes : DatetimeArray
        Datetime values for each site/datetime pair.
        `site_id`.
    latitude : FloatArray
        The latitde for each site/datetime pair.
    longitude : FloatArray
        The longitude for each site/datetime pair.
    elevation : FloatArray | float
        The elevation for each site/datetime pair or a single elevation value.
    output_file : str or PathLike
        Path to the output CSV file to be created.

    Returns
    -------
    None
    """

    _site_id = np.atleast_1d(site_id).astype(str)
    _datetimes = _datetimes_to_np_datetime64(datetimes)
    _lat = np.atleast_1d(latitude).astype(float)
    _lon = np.atleast_1d(longitude).astype(float)

    if isinstance(elevation, (int, float, np.floating)):
        _elevation = np.full(_lat.shape, float(elevation), dtype=np.float64)
    else:
        _elevation = np.atleast_1d(elevation).astype(float)

    if not (
        _site_id.size == _datetimes.size == _lat.size == _lon.size == _elevation.size
    ):
        raise ValueError(
            "site_id, datetimes, latitude, longitude, and elevation arguments must all have the same shape."
        )

    # initial data frame with station IDs and datetimes
    qtp_df = pd.DataFrame(
        data={
            "Station ID": _site_id,
            "DateTime": _datetimes,
            "Latitude": _lat,
            "Longitude": _lon,
            "Elevation": _elevation,
        },
    )
    # wrap to [-180, 180]
    qtp_df["Longitude"] = qtp_df["Longitude"].apply(lambda x: (x + 180) % 360 - 180)

    if qtp_df["Elevation"].isna().any():
        raise ValueError(
            "Some site elevations are missing and no 'fill_elevation' was specified."
        )

    # remove duplicates
    qtp_df = qtp_df.drop_duplicates(
        subset=["Station ID", "DateTime", "Latitude", "Longitude", "Elevation"]
    )

    qtp_df["DateTime"] = qtp_df["DateTime"].dt.strftime("%m/%d/%Y %H:%M")

    # Todo: test output from linux
    # - may need to restore encoding="iso-8859-1" and newline="\r\n" for windows compatibility with QuickTide Pro
    np.savetxt(
        output_file,
        qtp_df.to_numpy(),
        # encoding="iso-8859-1", # maybe restored if
        fmt=["%s", "%s", "%.6f", "%.6f", "%.3f"],
        delimiter=",",
        # newline="\r\n",
    )


class HardispOceanLoadCorrector(OceanLoadCorrectionProvider):
    """Compute ocean load corrections using HARDISP.

    HARDISP is part of the International Earth Rotation and Reference Systems Service
    (IERS) Conventions software collection.

    Parameters
    ----------
    f : str or PathLike
        File containing ocean loading coefficients for a set of stations.
    **metadata : dict[str, Any]
        Additional metadata to be stored in the `obj.metadata` dictionary.

    Attributes
    ----------
    ocean_loading_model : dict
        The loaded ocean loading model coefficients.
    metadata : dict
        Additional metadata associated with the model.

    """

    def __init__(self, f: FilePath, **metadata) -> None:
        self.ocean_loading_model = pyhardisp.load_ocean_loading_coefficients(str(f))
        self.metadata = metadata
        self._get_model_parameters(f)

    def __repr__(self) -> str:
        cname = self.__class__.__name__
        md = ",".join([f"{v}={k}" for v, k in self.metadata.items()])
        return f"{cname}({md})"

    def identifier(self, **kwargs) -> str:
        return repr(self)

    def _get_model_parameters(self, f: FilePath) -> None:
        """Extract model parameters from the file and store in metadata."""
        # try to extract model name and parameters from file name
        matadata = {
            "Greens_function": "",
            "ocean_tide_model": "",
            "center_mass_correction": False,
        }
        with open(f) as fh:
            model_txt = [l.strip() for l in fh.readlines() if l.startswith("$$")]
            for l in model_txt:
                # print(l)
                if l.startswith("$$ Greens function:"):
                    matadata["Greens_function"] = l.split(":", 1)[1].strip()
                elif l.startswith("$$ Ocean tide model:"):
                    matadata["ocean_tide_model"] = l.split(":", 1)[1].strip()
                elif l.startswith("$$ CMC"):
                    v = l.split(":", 1)[1].strip().split()[0]
                    matadata["center_mass_correction"] = False if v == "NO" else True
                elif l.startswith("$$ END HEADER:"):
                    break
        self.metadata.update(matadata)

    @property
    def stations(self) -> list[str]:
        """Return the station identifiers for which this provider can provide corrections."""
        return list(self.ocean_loading_model.keys())

    def ocean_load_correction(
        self,
        site_id: SiteIDArray,
        date_time: DatetimeArray,
        if_not_matched: Literal["error", "warn"] = "error",
        **kwargs,
    ) -> NDArray[np.float64]:

        _datetime = pd.DatetimeIndex(to_naive_utc_datetime(date_time, allow_nat=False))
        _site_id = to_1d_ndarray(site_id).astype(str)
        i_idx = np.arange(np.size(_site_id))
        scale_factor: float = 1e-4  # convert from nm/s2 to mGal
        uniq_site_id = np.unique(_site_id)

        bad_site_ids = [s for s in uniq_site_id if s not in self.stations]
        if bad_site_ids:
            msg = (
                f"site_id(s) {bad_site_ids} not found in station loading model. "
                f"Available stations: {self.stations}"
            )
            if if_not_matched == "error":
                raise ValueError(msg)
            else:
                warnings.warn(msg, UserWarning)
                uniq_site_id = [s for s in uniq_site_id if s not in bad_site_ids]

        # set up the computers for each station
        site_computers: dict[str, pyhardisp.HardispComputer] = {}
        for s in uniq_site_id:
            site_computers[s] = pyhardisp.HardispComputer()
            site_computers[s].read_blq_format(
                self.ocean_loading_model[s][0],
                self.ocean_loading_model[s][1],
                units="nm/s2",
            )

        # simple first implementation non parallel
        corrs = np.zeros_like(_site_id, dtype=np.float64)
        for i, s, d in zip(i_idx, _site_id, _datetime):
            if s not in site_computers:
                corrs[i] = np.nan
                continue

            v = site_computers[s].compute_ocean_loading(
                year=d.year,
                month=d.month,
                day=d.day,
                hour=d.hour,
                minute=d.minute,
                second=d.second,
                num_epochs=1,
                sample_interval=1,
            )
            corrs[i] = v[0][0]

        return corrs * scale_factor  # convert from nm/s2 to mGal
