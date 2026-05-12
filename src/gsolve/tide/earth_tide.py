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
import warnings
from typing import Literal, Protocol, Self, overload, runtime_checkable

import numpy as _np
import pandas as _pd
import pygtide
from matplotlib.pyplot import axis
from numpy import arccos, arcsin, arctan, cos, deg2rad, pi, sin, sqrt
from numpy.typing import ArrayLike, NDArray
from pandas.api.interchange import from_dataframe
from pandas.api.typing import NaTType
from PIL.Image import new

from gsolve.core._typing import (
    DatetimeArray,
    DatetimeScalar,
    FilePath,
    FloatArray,
    SiteIDArray,
    StringArray,
    TimedeltaScalar,
)
from gsolve.core.utils import (
    GSolveDataWarning,
    dms2rad,
    to_1d_ndarray,
    to_naive_utc_datetime,
)

__all__ = [
    "EarthTideCorrectionProvider",
    "LongmanTidalCorrection",
    "EternaPredictTidalCorrection",
    "LongmanConstants",
    "gravimetric_factor",
]


@runtime_checkable
class EarthTideCorrectionProvider(Protocol):
    """Protocol defining interface for classes that provide earth tide corrections."""

    def tidal_correction(
        self,
        lat: FloatArray,
        lon: FloatArray,
        elev: FloatArray,
        date_time: DatetimeArray,
        site_id: SiteIDArray | None = None,
        **kwargs,
    ) -> NDArray[_np.float64]: ...

    def identifier(self, **kwargs) -> str: ...


def gravimetric_factor(
    k2: float = 0.2980, h2: float = 0.6032
) -> _np.float64 | NDArray[_np.float64]:
    """Compute gravimetric factor from Love numbers ``k2`` and ``h2``.

    The gravimetric factor accounts for the non-rigidity of the Earth.
    Default values for standard modern earth model (PREM) are from [1]_Agnew (2007) .

    Parameters
    ----------
    k2, h2 : float or array-like, optional
        Love numbers for earth response to semi-diurnal tides.

    Returns
    -------
    delta2: ndarray or float
        The gravimetric factor, float if both `k2` & `h2` are
        floats.

    References
    ----------
    .. [1] Agnew, D. C. (2007). 3.06 Earth Tides. In Treatise on Geophysics (pp. 163-195).
       Elsevier. https://doi.org/10.1016/B978-044452748-6.00056-0
    """
    _h2 = to_1d_ndarray(h2).astype(float)
    _k2 = to_1d_ndarray(k2, expected_size=_h2.size).astype(float)
    gfactor = 1 + _h2 - 1.5 * _k2
    return gfactor[0] if gfactor.size == 1 else gfactor


@dataclasses.dataclass
class LongmanConstants:
    """Constants used in Longman gravitational potential calculations.

    Parameters
    ----------
    a : float, default 6.3781366e08
        Earth's equatorial radius in cm (after UNSO 2011).
    c : float, default 3.84399e10
        Mean distance between the centers earth-moon (cm).
        c1 : float, default 1.495983e13
        Mean distance between centers earth-sun (cm).
    e : float, default 0.054900489
        Eccentricity of the moon's orbit.
    i : float, default 0.08979719
        Inclination of moon's orbit to the ecliptic, 5.145 degrees.
    m : float, default 0.074804
        Ratio of mean motion of the sun to that of the moon.
    mu : float, default 6.67428e-08
        Newton's gravitational constant, 6.670e-8 in orig.
    M : float, default 7.3477e25
        Mass of the moon in grams.
    omega : float, default 0.409314616
        Inclination of Earth's equator to ecliptic 23.452 degrees.
    S : float, default 1.98840987e33
        Mass of the sun in grams https://aa.usno.navy.mil/downloads/publications/Constants_2021.pdf.
    """

    a: float = 6.3781366e08  # Earth's equatorial radius in cm (after UNSO 2011)
    c: float = 3.84399e10  # Mean distance between the centers earth-moon (cm)
    c1: float = 1.495983e13  # Mean distance between centers earth-sun (cm)
    e: float = 0.054900489  # Eccentricity of the moon's orbit
    i: float = round(
        deg2rad(5.145), 9
    )  # = 0.08979719  Inclination of moon's orbit to the ecliptic
    m: float = 0.074804  # Ratio of mean motion of the sun to that of the moon
    mu: float = 6.67428e-08  # Newton's gravitational constant, 6.670e-8 in orig.
    M: float = 7.3477e25  # Mass of the moon in grams
    omega: float = round(
        deg2rad(23.452), 9
    )  # = 0.409315 Incl. of Earth's equator to ecliptic
    S: float = 1.98840987e33  # Mass of the sun in grams
    # https://aa.usno.navy.mil/downloads/publications/Constants_2021.pdf


class LongmanTidalCorrection(EarthTideCorrectionProvider):
    """Class to compute lunar and solar gravitational effects using Longman's method.

    Parameters
    ----------
    amp_factor : float, default 1.2
        The default amplification factor.
    kwargs : optional
        Additional keyword arguments are used when instantiating a
        ``LongmanConstants`` object, allowing the user to override the
        default constants.

    Attributes
    ----------
    amp_factor : float
        The amplification factor applied when calculating gravity corrections
        using :meth:`tidal_correction`.
    constants : LongmanConstants
        The physical constants used in the Longman method.

    See Also
    --------
    gsolve.tide.earth_tide.LongmanConstants : Physical constants used in Longman's method.
    gsolve.tide.earth_tide.EarthTideCorrectionProvider : Protocol defining interface for
        classes implementing earth tide correction.

    """

    def __init__(self, amp_factor: float = 1.2, **kwargs) -> None:
        self.amp_factor = float(amp_factor)
        self.constants = LongmanConstants(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(amp_factor={self.amp_factor})"

    def identifier(self, **kwargs) -> str:
        if not kwargs:
            return self.__repr__()

        afactor = kwargs.pop("amp_factor", self.amp_factor)
        _kwargs = dict(amp_factor=afactor, **kwargs)
        suffix = ",".join(f"{k}={v}" for k, v in _kwargs.items())
        return f"{self.__class__.__name__}({suffix})"

    def gravity_accelerations(
        self,
        lat: FloatArray,
        lon: FloatArray,
        elev: FloatArray,
        dt: DatetimeScalar | DatetimeArray,
    ) -> tuple[NDArray[_np.float64], NDArray[_np.float64]]:
        """Compute lunar and solar gravitational accelerations.

        Parameters
        ----------
        lat : scalar or array_like of shape(M,)
            Latitude in decimal degrees.
        lon : scalar or array_like of shape(M,)
            Longitude in decimal degrees.
        elev : scalar or array_like of shape(M,)
            Elevation in meters (datum independent)
        dt : str, datetime, or array_like of shape(M,)
            The date-times to be at which to calculate corrections. Can
            be any format parsable by Pandas.to_datetime() method.
        amp_factor : float, or array_like of shape(M,), optional
            Factor to apply to tidal corrections to account for earth's
            deformation response to tidal forces. Tyypically in range
            1.14-1.2 for semi-diurnal tides (default 1.2). To calculate this factor
            using h2 and k2 parameters the gravimetric_factor function can be used.

        Returns
        -------
        g_lunar, g_solar :
            The tidal acceleration in milligals due to the moon and sun.
            Scalar if all input args are scalar.

        """
        try:
            _lon = to_1d_ndarray(lon).astype(float)
            _lat = to_1d_ndarray(lat, expected_size=_lon.size).astype(float)
            _elev = to_1d_ndarray(elev, expected_size=_lon.size).astype(float)
        except ValueError:
            raise ValueError(
                "Invalid lat, lon, or elev: must be equal sized 1d arrays of floats"
            )

        T = _decimal_julian_century(dt)
        t0 = _decimal_hour_of_day(dt)

        # original Longman notation for lat, lon and elevation
        lmbda = deg2rad(_lat)
        L = -1 * _lon
        H = _elev * 100.0  # in cm

        T = _decimal_julian_century(dt)
        T_2 = T**2
        T_3 = T**3
        t0 = _decimal_hour_of_day(dt)

        a = self.constants.a
        mu = self.constants.mu
        M = self.constants.M
        S = self.constants.S
        e = self.constants.e
        m = self.constants.m
        c = self.constants.c
        c1 = self.constants.c1
        i = self.constants.i
        omega = self.constants.omega

        rev = 2 * pi

        # Lunar Calculations
        # Longman eq 10' (s) Mean longitude of moon in its orbit reckoned
        # from the referred equinox. This is Bartels(1957) formula.
        s = (
            dms2rad(270, 26, 11.72)
            + (1336 * rev + dms2rad(0, 0, 1108406.05)) * T
            + dms2rad(0, 0, 7.128) * T_2
            + dms2rad(0, 0, 0.0072) * T_3
        )
        # Longman eq 11' (p) Mean longitude of lunar perigee. This is Bartels(1957) formula.
        p = (
            dms2rad(334, 19, 46.42)
            + (11 * rev + dms2rad(0, 0, 392522.51)) * T
            + dms2rad(0, 0, 37.15) * T_2
            + dms2rad(0, 0, 0.036) * T_3
        )
        # Longman eq 12' (h) Mean longitude of the sun. This is Bartels(1957) formula.
        h = (
            dms2rad(279, 41, 48.05)
            + dms2rad(0, 0, 129602768.11) * T
            + dms2rad(0, 0, 1.080) * T_2
        )
        # Longman eq 19 (N) Longitude of the moon's ascending node in its orbit. From Schureman(1941)
        N = (
            dms2rad(259.0, 10.0, 57.12)
            - (5 * rev + dms2rad(0, 0, 482912.63)) * T
            + dms2rad(0, 0, 7.480) * T_2
            + dms2rad(0, 0, 0.007) * T_3
        )

        # eq. 20 (I) Inclination of the moon's orbit to the equator
        I = arccos(cos(omega) * cos(i) - sin(omega) * sin(i) * cos(N))

        # eq. 21 (nu) Longitude in the celestial equator of its intersection A with the moon's orbit
        nu = arcsin(sin(i) * sin(N) / sin(I))

        # eq. 24 (t) Hour angle of mean sun measured west-ward from
        # the place of observations
        t = deg2rad(15.0 * (t0 - 12) - L)

        # eq. 23 (chi) right ascension of meridian of place of observations reckoned from A
        chi = t + h - nu

        # eq. 15, 16, 18
        cos_alpha = cos(N) * cos(nu) + sin(N) * sin(nu) * cos(omega)
        sin_alpha = sin(omega) * sin(N) / sin(I)
        alpha = 2 * arctan(sin_alpha / (1 + cos_alpha))

        # eq. 14 (xi) Longitude in the moon's orbit of its ascending
        # intersection with the celestial equator
        xi = N - alpha

        # eq. 13 (sigma) Mean longitude of moon in radians in its orbit
        # reckoned from A
        sigma = s - xi

        # eq. 9 (l) Longitude of moon in its orbit reckoned from its ascending
        # intersection with the equator
        l = (  # noqa: E741
            sigma
            + 2 * e * sin(s - p)
            + 5.0 / 4 * e**2 * sin(2 * (s - p))
            + 15.0 / 4 * m * e * sin(s - 2 * h + p)
            + 11.0 / 8 * m**2 * sin(2 * (s - h))
        )

        # Solar Calculations
        # eq. 26 (p1) Mean longitude of solar perigee
        p1 = (
            dms2rad(281, 13, 15.00)
            + dms2rad(0, 0, 6189.03) * T
            + dms2rad(0, 0, 1.63) * T**2
            + dms2rad(0, 0, 0.012) * T**3
        )

        # eq. 27 (e1) Eccentricity of the Earth's orbit
        e1 = 0.01675104 - 0.00004180 * T - 0.000000126 * T**2

        # eq. 28 (chi1) right ascension of meridian of place of observations
        # reckoned from the vernal equinox
        chi1 = t + h

        # eq. 25 (l1) Longitude of sun in the ecliptic reckoned from the
        # vernal equinox
        l1 = h + 2 * e1 * sin(h - p1)

        # eq. 7 cosine(theta) Theta represents the zenith angle of the moon
        cos_theta = sin(lmbda) * sin(I) * sin(l) + cos(lmbda) * (
            cos(I / 2) ** 2 * cos(l - chi) + sin(I / 2) ** 2 * cos(l + chi)
        )

        # eq. 8 cosine(phi) Phi represents the zenith angle of the run
        cos_phi = sin(lmbda) * sin(omega) * sin(l1) + cos(lmbda) * (
            cos(omega / 2) ** 2 * cos(l1 - chi1) + sin(omega / 2) ** 2 * cos(l1 + chi1)
        )

        # Distance Calculations
        # eq. 34 (C) Distance parameter
        C = sqrt(1.0 / (1 + 0.006738 * sin(lmbda) ** 2))

        # eq. 33 (r) Distance from point P to the center of the Earth
        r = C * a + H

        # eq. 31 & 32 (a') Distance parameter, equation 31
        a_prime = 1.0 / (c * (1 - e**2))
        a1_prime = 1.0 / (c1 * (1 - e1**2))

        # eq. 29 & 30 (d) Distance between centers of the Earth and the moon
        d = (
            1 / c
            + a_prime * e * cos(s - p)
            + a_prime * e**2 * cos(2 * (s - p))
            + (15 / 8) * a_prime * m * e * cos(s - 2 * h + p)
            + a_prime * m**2 * cos(2 * (s - h))
        ) ** -1

        # (D) Distance between centers of the Earth and the sun
        D = (1 / c1 + a1_prime * e1 * cos(h - p1)) ** -1

        # eq. 1 (g_lunar) Vertical componet of tidal acceleration due to the moon
        g_lunar = (
            mu * M * r * d**-3 * (3 * cos_theta**2 - 1)
            + (3 / 2) * mu * M * r**2 / d**4 * (5 * cos_theta**3 - 3 * cos_theta)
        ) * 1e03

        # eq. 3 (g_solar) Vertical componet of tidal acceleration due to the sun
        g_solar = mu * S * r * D**-3 * (3 * cos_phi**2 - 1) * 1e03

        return g_lunar, g_solar

    def tidal_correction(
        self,
        lat: FloatArray,
        lon: FloatArray,
        elev: FloatArray,
        date_time: DatetimeArray,
        site_id: SiteIDArray | None = None,
        **kwargs,
    ) -> NDArray[_np.float64]:
        """Compute tidal corrections at specified locations and times.

        Parameters
        ----------
        site_id : array_like
            REMOVE
        lat : float or array_like
            Latitude in decimal degrees.
        lon : float or array_like
            Longitude in decimal degrees.
        elev : float or array_like
            Elevation in meters
        date_time : array_like
            The datetimes at which to calculate corrections.

        Returns
        -------
        corrections : NDArray
            The tidal corrections in milligals.

        """
        g = self.gravity_accelerations(lat, lon, elev, date_time)
        return (g[0] + g[1]) * self.amp_factor

    def time_series(
        self,
        starttime: DatetimeScalar,
        endtime: DatetimeScalar,
        step: TimedeltaScalar,
        lat: float,
        lon: float,
        elev: float = 0.0,
        method: Literal["correction", "acceleration"] = "correction",
    ) -> _pd.Series:
        """Compute a time series of tidal corrections at some location.

        Parameters
        ----------
        starttime : str, datetime
            The start time of the time series.
        endtime : str, datetime
            The end time of the time series.
        step : int, float, str, Timedelta
            The sampling interval of the time series.  Can be a timedelta-like object,
            a timedelta string (e.g. "1S", "1H", "1D"), or a number of seconds.
        lat : float
            Latitude in decimal degrees.
        lon : float
            Longitude in decimal degrees.
        elev : float
            Elevation in meters (datum independent)
        method: {'correction', 'acceleration'}, default 'correction'
            Whether to return gravity corrections or accelerations.

        Returns
        -------
        time_series : Series
            The time series of tidal corrections or accelerations.

        """
        valid_methods = ("correction", "acceleration")
        if method not in valid_methods:
            raise ValueError(
                f"method parameter must be one of {valid_methods}, not '{method}'."
            )

        try:
            step = _pd.Timedelta(step)
            if not isinstance(step, _pd.Timedelta):
                raise ValueError("step must be a valid timedelta or timedelta string.")
        except ValueError as e:
            raise ValueError(f"error parsing step: {e}") from e

        try:
            t0 = to_naive_utc_datetime(starttime, allow_nat=False)
            t1 = to_naive_utc_datetime(endtime, allow_nat=False)
            if not isinstance(t0, _pd.Timestamp) or not isinstance(t1, _pd.Timestamp):
                raise ValueError("not convertible to Timestamp.")
            if t0 >= t1:
                raise ValueError("starttime is after or equal to endtime.")
        except ValueError as e:
            raise ValueError(f"error parsing starttime and endtime: {e}") from None

        t_idx = _pd.date_range(t0, t1, freq=step)
        _lat = _np.full(len(t_idx), lat)
        _lon = _np.full(len(t_idx), lon)
        _elev = _np.full(len(t_idx), elev)
        if method == "correction":
            tseries = _pd.Series(
                data=self.tidal_correction(
                    site_id=None, lat=_lat, lon=_lon, elev=_elev, date_time=t_idx
                ),
                index=t_idx,
                name=method,
            )
        elif method == "acceleration":
            a, b = self.gravity_accelerations(_lat, _lon, _elev, t_idx)
            tseries = _pd.Series(
                data=a + b,
                index=t_idx,
                name=method,
            )
        else:
            raise ValueError(f"Invalid method: {method}")

        return tseries


@overload
def _decimal_julian_century(dt: DatetimeScalar, **kwargs) -> _np.float64: ...
@overload
def _decimal_julian_century(dt: DatetimeArray, **kwargs) -> NDArray[_np.float64]: ...


def _decimal_julian_century(
    dt: DatetimeScalar | DatetimeArray, **kwargs
) -> NDArray[_np.float64] | _np.float64:
    """
    Convert datetime to Julian Centuries since epoch B1900.0.

    A Julian Century is 100 * 365.25 * 86400 SI seconds. This function
    returns decimal Julian Centuries since 1899-12-31T12:00:00.

    Parameters
    ----------
    dt : str, datetime, array-like
        The date-times to be converted. Can be any format parsable by
        ``pandas.to_datetime`` method. Datetimes without a timezone are
        assumed to be in UTC.

    Returns
    -------
    julian_century : ndarray or scalar
        This is scalar if `dt` is a single value.

    See Also
    --------
    to_naive_utc_datetime : Convert a datetime to a naive UTC datetime.
    pandas.to_datetime : Convert argument to datetime.

    """
    _dt = to_naive_utc_datetime(dt, allow_nat=False, **kwargs)
    if _dt is None or isinstance(_dt, NaTType):
        raise ValueError("dt cannot be NaT or None.")
    if isinstance(_dt, _pd.Timestamp):
        _dt = _pd.DatetimeIndex([_dt])
    elif isinstance(_dt, (_pd.DatetimeIndex, _pd.Series)):
        _dt = _pd.DatetimeIndex(_dt)
    else:
        raise ValueError(
            "dt could not be resolved to a datetime, DatetimeIndex, Series, or array-like of datetimes."
        )

    julian_century_origin = _pd.Timestamp("1899-12-31T12:00:00", tz=None)
    td_seconds = (_dt - julian_century_origin).total_seconds()
    julian_century_seconds: int = 86400 * 36525
    jcentury = _np.atleast_1d(td_seconds / julian_century_seconds)

    return jcentury[0] if jcentury.size == 1 else jcentury


@overload
def _decimal_hour_of_day(date_time: DatetimeScalar) -> _np.float64: ...
@overload
def _decimal_hour_of_day(date_time: DatetimeArray) -> NDArray[_np.float64]: ...


def _decimal_hour_of_day(
    date_time: DatetimeScalar | DatetimeArray,
) -> _np.float64 | NDArray[_np.float64]:
    """Compute time of day in decimal hours.

    Parameters
    ----------
    date_time : str, datetime, array-like
        The date-times to be processed. Can be any format parsable by
        pandas.to_datetime() method.

    Returns
    -------
    hour : ndarray or scalar
        The decimal hour. This is a scalar if `arg` is a single value.

    """
    _dt = to_naive_utc_datetime(date_time, allow_nat=False)
    if isinstance(_dt, NaTType) or _dt is None:
        raise ValueError("date_time cannot be NaT or None.")
    if not isinstance(_dt, (_pd.Timestamp, _pd.Series, _pd.DatetimeIndex)):
        raise ValueError(
            "date_time could not be resolved to a datetime, DatetimeIndex, Series, or array-like of datetimes."
        )

    if isinstance(_dt, _pd.Timestamp):
        _dt = _pd.DatetimeIndex([_dt])
    elif isinstance(_dt, _pd.Series):
        _dt = _pd.DatetimeIndex(_dt)

    _tdelta = ((_dt - _dt.normalize()).total_seconds() / 3600.0).to_numpy(float)

    return _tdelta[0] if _tdelta.size == 1 else _tdelta


class EternaTidalParameters:
    """Class to store tidal parameters for use with ETERNA/pygtide.

    Tidal parameters (``TIDALPARAM`` in ETERNA, ``wave_group`` in pygtide)
    define the ampltude factors and phase leads to be applied to the tidal potentials
    provided by a given tidal catalogue. A single tidal parameter is comprises 4 values

        - ``freq_start``: the start frequency of the tidal constituent in cycles per day (cpd).
        - ``freq_stop``: the stop frequency of the tidal constituent in cpd.
        - ``amplitude_factor``: multiply the amplitude of the tidal constituent.
        - ``phase_lead``: the phase lead in degrees

    An example tidal parameter (from ETERNA documentation) covering M2, the semi-diurnal
    lunar tide constituent is ``[1.914129, 1.950419, 1.18705, 2.0327]``. Multiple tidal
    parameters can be combined to cover the full spectrum of tidal constituents.

    Attributes
    ----------
    data : pd.DataFrame
        A DataFrame containing the tidal parameters under the columns 'freq_start',
        'freq_stop', 'amplitude', and 'phase_lead'. Other columns such as 'wave_group'
        label may be included but are not used.

    Parameters
    ----------
    freq_start : array_like
        The start frequency of the tidal constituent in cycles per day (cpd).
    freq_stop : array_like
        The stop frequency of the tidal constituent in cycles per day (cpd).
    amplitude : array_like
        The amplitude factor to apply to the tidal constituent.
    phase_lead : array_like
        The phase lead in degrees to apply to the tidal constituent.
    wave_group : array_like, optional
        Labels for the tidal constituent, e.g. "M2".
    **kwargs: array_likes, optional
        Other columns to include in the ``data`` DataFrame. Must be array-like of the
        same size as the other parameters.

    Returns
    -------
    EternaTidalParameters
         An instance of the class containing the tidal parameters.

    See Also
    --------
    EternaPredictTidalCorrection : Class to compute tidal corrections using ETERNA/pygtide.
    pygtide : Python package for tidal predictions using ETERNA catalogues.
    """

    def __init__(
        self,
        freq_start: FloatArray,
        freq_stop: FloatArray,
        amplitude: FloatArray,
        phase_lead: FloatArray,
        wave_group: StringArray | None = None,
        **kwargs,
    ) -> None:
        self.data: _pd.DataFrame

        try:
            freq_start = to_1d_ndarray(freq_start).astype(float)
            s = freq_start.size
            freq_stop = to_1d_ndarray(freq_stop, expected_size=s).astype(float)
            amplitude = to_1d_ndarray(amplitude, expected_size=s).astype(float)
            phase_lead = to_1d_ndarray(phase_lead, expected_size=s).astype(float)
            if wave_group is not None:
                wave_group = to_1d_ndarray(wave_group, expected_size=s).astype(str)
        except ValueError as e:
            raise ValueError(f"Error parsing tidal parameters: {e}")

        self.data = _pd.DataFrame.from_dict(
            data={
                "freq_start": freq_start,
                "freq_stop": freq_stop,
                "amplitude": amplitude,
                "phase_lead": phase_lead,
            },
            orient="columns",
        )
        if wave_group is not None:
            self.data["wave_group"] = wave_group
        for k, v in kwargs.items():
            self.data[k] = v

        self.sort_and_validate()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nparam={self.data.shape[0]})"

    def __copy__(self) -> Self:
        return self.__class__.from_dataframe(self.data.copy())

    def copy(self) -> Self:
        return self.__copy__()

    def sort_and_validate(self, gap_threshold: float | None = None) -> None:
        """Sort tidal parameters by ``start_freq`` and validate data.

        This method ensures that the following conditions are met:

        - All data are floats with no NaN.
        - Frequency intervals are positive and increasing
            i.e. ``freq_start`` < ``freq_stop``.
        - Frequency intervals are discrete and do not overlap.
        - Warn if there are gaps between successivefrequency intervals.

        Raises
        ------
        ValueError
            If any of the above validations fail.
        """

        emsg = GSolveDataWarning(prefix=f"{type(self).__name__} error", show=True)
        errors = []
        data_subset = self.data.loc[
            :, ["freq_start", "freq_stop", "amplitude", "phase_lead"]
        ]
        nan_rows = data_subset.index[data_subset.isna().any(axis=1)].to_list()
        if nan_rows:
            emsg(f"NaN values found in rows {nan_rows}")

        bad_freq_rows = self.data.index[
            (self.data["freq_stop"] <= self.data["freq_start"])
        ].to_list()
        if bad_freq_rows:
            emsg(f"'freq_start' >= 'freq_stop' in rows {bad_freq_rows}.")

        self.data = self.data.sort_values(by=["freq_start"], ignore_index=True)

        overlaps = _np.full_like(self.data.index, False, dtype=bool)
        overlaps[:-1] = (
            self.data.iloc[:-1, 1].to_numpy() >= self.data.iloc[1:, 0].to_numpy()
        )
        if overlaps.any():
            overlap_rows = self.data.index[overlaps].to_list()
            emsg(
                "Frequency intervals must be discrete. Overlapping "
                f"frequency found in rows {overlap_rows}."
            )
        if emsg.count > 0:
            raise ValueError(f"Validation failed, {emsg.count} error(s) found.")

        if gap_threshold is not None and self.data.shape[0] > 1:
            gap_threshold = float(gap_threshold)
            gaps = (
                self.data.iloc[:-1, 1] - self.data.iloc[1:, 0] > gap_threshold
            ).to_numpy()
            if gaps.any():
                n_gaps = gaps.sum()
                warnings.warn(
                    f"some frequency intervals separated by greater than {gap_threshold} ",
                    UserWarning,
                )

    @classmethod
    def from_array(cls, arr: ArrayLike) -> Self:
        """
        Create an EternaTidalParameters instance from an array-like object.

        Parameters
        ----------
        arr : array_like
            A 1D array with at least 4 elements or a 2D array with at least 4 columns.
            Cells/columns 0-3 correspond to 'freq_start', 'freq_stop', 'amplitude', and
            'phase_lead' in that order.

        Returns
        -------
        EternaTidalParameters
        """
        if isinstance(arr, _pd.DataFrame):
            return cls.from_dataframe(
                arr,
            )

        arr = _np.atleast_2d(arr).astype(float)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("Input array must be 2D with at least 4 columns.")

        kwargs = {f"col_{i}": arr[:, i] for i in range(4, arr.shape[1])}
        return cls(
            freq_start=arr[:, 0],
            freq_stop=arr[:, 1],
            amplitude=arr[:, 2],
            phase_lead=arr[:, 3],
            **kwargs,
        )

    @classmethod
    def from_dataframe(cls, df: _pd.DataFrame, include_index: bool = False) -> Self:
        """
        Create an EternaTidalParameters object from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing columns 'freq_start', 'freq_stop', 'amplitude', and
            'phase_lead'. Other columns will be included in the object's ``data``
            attribute but are not required.
        include_index : bool, default False
            If True, the index will be treated as a column when parsing the DataFrame.

        Returns
        -------
        EternaTidalParameters
        """
        if include_index:
            df = df.reset_index()

        expected_columns = {"freq_start", "freq_stop", "amplitude", "phase_lead"}
        if not expected_columns.issubset(df.columns):
            missing = list(expected_columns - set(df.columns))
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        return cls(
            freq_start=df["freq_start"],
            freq_stop=df["freq_stop"],
            amplitude=df["amplitude"],
            phase_lead=df["phase_lead"],
            **{k: df[k] for k in df.columns if k not in expected_columns},
        )

    @classmethod
    def from_excel(cls, fname: FilePath, sheet_name: str | int = 0) -> Self:
        """
        Read an EternaTidalParameters object from an Excel worksheet.

        The spreadsheet must contain columns labelled 'freq_start', 'freq_stop',
        'amplitude', and 'phase_lead'. Other columns will be loaded but are not
        used.

        Parameters
        ----------
        fname : FilePath
            The Excel/spreadsheet file to be read. Can be any object type supported by
            or spreadsheet format (e.g. odf) supported by ``pandas.read_excel()``,
        sheet_name : str | int, default is 0
            The name or index of the worksheet to read. Default is 0, the first
            sheet in `fname`.

        Returns
        -------
        EternaTidalParameters

        See Also
        --------
        EternaTidalParameters.from_dataframe : Create an EternaTidalParameters
            object from a DataFrame.
        pandas.read_excel : Read an Excel file into a DataFrame.
        """

        df = _pd.read_excel(fname, sheet_name=sheet_name)
        if isinstance(df, dict):
            raise ValueError(
                "Excel file contains multiple sheets. Please specify sheet_name."
            )

        return cls.from_dataframe(df)

    @classmethod
    def quick_tide_pro_defaults(cls) -> Self:
        """Return an EternaTidalParameters instance with the
        default parameters used by QuickTide Pro.

        Quick tide pro uses the Tamura (1987) tidal potential catalogue.
        Using thee parameters with higer resolution tide catalogues may
        produce unexpected results.

        """
        df = _pd.DataFrame.from_dict(
            data={
                "DC": [0.000000, 0.000001, 1.000000, 0.0000],
                "Long": [0.000002, 0.249951, 1.160000, 0.0000],
                "Q1": [0.721500, 0.906315, 1.154250, 0.0000],
                "O1": [0.921941, 0.974188, 1.154240, 0.0000],
                "P1": [0.989049, 0.998028, 1.149150, 0.0000],
                "K1": [0.999853, 1.216397, 1.134890, 0.0000],
                "N2": [1.719381, 1.906462, 1.161720, 0.0000],
                "M2": [1.923766, 1.976926, 1.161720, 0.0000],
                "S2": [1.991787, 2.002885, 1.161720, 0.0000],
                "K2": [2.003032, 2.182843, 1.161720, 0.0000],
                "M3": [2.753244, 3.081254, 1.07338, 0.0000],
                "other": [3.791964, 3.937897, 1.03900, 0.0000],
            },
            orient="index",
            columns=["freq_start", "freq_stop", "amplitude", "phase_lead"],
        ).rename_axis("wave_group")
        return cls.from_dataframe(df, include_index=True)

    @classmethod
    def default(
        cls,
        freq_start: float = 0.0,
        freq_stop: float = 10.0,
        amplitude_factor: float = 1.16,
        phase_lead: float = 0.0,
    ) -> Self:
        """Return a tidal parameters object with default values used by GSolve."""
        return cls.from_array([freq_start, freq_stop, amplitude_factor, phase_lead])

    def pygtide_wavegroup_arg(self) -> NDArray[_np.float64]:
        """Return a copy of the paramaters as an ndarray (shape= (N, 4))
        for use as the argument to ``pygtide.set_wavegroup()`` method.
        """
        return (
            self.data.loc[:, ["freq_start", "freq_stop", "amplitude", "phase_lead"]]
            .to_numpy()
            .copy()
        )


class EternaPredictTidalCorrection(EarthTideCorrectionProvider):
    """Compute earth tide gravity corrections using PREDICT from ETERNA34.

    The

    Attributes
    ----------
    tidal_params : EternaTidalParameters

    Parameters
    ----------
    tidal_params : array-like | EternaTidalParameters, optional
        The tidal parameters applied to discrete components of the tidal catalogue. This is a 2D array with 4 columns, where each row corresponds to a tidal component and the columns are:
    tidalpoten : int, default 8
        The tide potential catalogue to use. ETERNA/pygtide provides 8 potential
        catalogues of increasing resolution and therefore computational cost.
        The most commonly used cataloges:

        - ``4`` : Tamura (1987), 1200 waves.
        - ``7`` : Hartmann and Wenzel (1995), 12935 waves.
        - ``8`` (Default) : Kudryavtsev (2004), 28806 waves (highest resolution).
        
        See ETERNA/pygtide for the full list of available catalogues.
    tidalcompo : int, default 0
        The tidal component to calculate. The default 0 is earth tide gravity, but other
        components such as displacement, tilt or strain can also be calculated.
        See pygtide or ETERNA documentation for details.
    amtruncate : float, default 1e-10
        Amplitude threshold for components to be included in the tidal calculation.
        Waves with amplitudes below this threshold are excluded. Higher values
        will reduce computation time, but also the accuracy of results.
    poletidecor : float, default 1.16
        Amplitude factor for of pole tide gravity component. Pole tides are caused to
        variations in the Earth's rotation axis (Chandler Wobble) and are not included
        in the standard tidal potential catalogues. Pole tide solutions are dependent
        on observational data provided by IERS, so the user should that they
        periodically run ``pgtide.update()`` to ensure these data are up to date.
    lodtidecor : float, default 1.16
        Amplitude factor for of Length Of Day tide gravity component, due to variations
        in the Earth's rotation rate and are not included in the standard tidal
        potential catalogues.. LOD corrections are depenedent on observational data
        provided by IERS, so the user should that they periodically run
        ``pgtide.update()`` to ensure these data are up to date.

    See Also
    --------
    EternaTidalParameters : Class to store tidal parameters for use with ETERNA/pygtide.
    pygtide : Python package for tidal predictions using ETERNA catalogues.
    """

    def __init__(
        self,
        tidal_params: ArrayLike | EternaTidalParameters | None = None,
        tidalpoten: int = 8,
        tidalcompo: int = 0,
        amtruncate: float = 1e-10,
        poletidecor: float = 1.16,
        lodtidecor: float = 1.16,
        **kwargs,
    ) -> None:
        self._pgt: pygtide.pygtide = pygtide.pygtide(msg=False)
        self._pgt.msg = False
        self._pgt_kwargs: dict[str, float | int] = {
            "tidalpoten": tidalpoten,
            "tidalcompo": tidalcompo,
            "amtruncate": amtruncate,
            "poletidecor": poletidecor,
            "lodtidecor": lodtidecor,
        }
        self.tidal_params: EternaTidalParameters

        if tidal_params is None:
            self.tidal_params = EternaTidalParameters.default()
            self.set_tidal_params(self.tidal_params.pygtide_wavegroup_arg())
        elif isinstance(tidal_params, EternaTidalParameters):
            self.set_tidal_params(tidal_params)
        else:
            self.set_tidal_params(tidal_params)

    def __repr__(self) -> str:
        suffix = ",".join(f"{k}={v}" for k, v in self._pgt_kwargs.items())
        return f"{self.__class__.__name__}({suffix})"

    def set_tidal_params(self, tidal_params: ArrayLike | EternaTidalParameters) -> None:
        """Set the tidal parameters to be applied."""
        if not isinstance(tidal_params, EternaTidalParameters):
            self.tidal_params = EternaTidalParameters.from_array(tidal_params)
        else:
            self.tidal_params = tidal_params.copy()

        self._pgt.set_wavegroup(self.tidal_params.pygtide_wavegroup_arg())

    def identifier(self, **kwargs) -> str:
        if not kwargs:
            return self.__repr__()
        self._pgt_kwargs.update(kwargs)
        suffix = ",".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{self.__repr__()}({suffix})"

    def time_series(
        self,
        lat: float,
        lon: float,
        elev: float,
        starttime: DatetimeScalar,
        duration: TimedeltaScalar,
        sample_interval: int = 60,
        unit: Literal["mgal", "ugal", "nm/s^2"] = "mgal",
        **kwargs,
    ) -> _pd.DataFrame:
        """Compute a time series of tidal corrections at some location.

        A limitation of ETERNA is that tides are always calculated from the start
        of a UTC day.

        Parameters
        ----------
        lat : float
            Latitude of the location in degrees.
        lon : float
            Longitude of the location in degrees.
        elev : float
            Elevation of the location in meters.
        starttime : DatetimeScalar
            Start
        duration : TimedeltaScalar
            Duration of the time series.
        sample_interval : int, optional
            Sampling interval in seconds (default is 60).
        unit : {"mgal", "ugal", "nm/s^2"}, optional
            Unit of the output (default is "mgal").
        **kwargs : dict
            Additional keyword arguments to pass to the underlying pygtide predictor.

        Returns
        -------
        DataFrame
            DataFrame containing the tidal corrections.

        """

        this_run_kwargs = self._pgt_kwargs.copy()
        this_run_kwargs.update(kwargs)

        _starttime = (
            to_naive_utc_datetime(starttime, allow_nat=False)
            .normalize()
            .to_pydatetime()
        )
        if isinstance(duration, (int, float)):
            _duration = int(duration)
        else:
            _duration = int(
                _np.ceil(_pd.to_timedelta(duration).total_seconds() / 3600.0)
            )
            if _duration <= 0:
                raise ValueError(
                    "duration must be a positive timedelta or number of hours."
                )

        sample_interval = int(sample_interval)
        if sample_interval <= 0:
            raise ValueError(
                "sample_interval must be a positive integer number of seconds."
            )

        with warnings.catch_warnings():
            self._pgt.msg = False
            warnings.filterwarnings(
                action="ignore",
                message="Please consider updating the leap second database",
                category=UserWarning,
            )
            self._pgt.predict(
                latitude=lat,
                longitude=lon,
                height=elev,
                startdate=_starttime,
                duration=_duration,
                samprate=sample_interval,
                **this_run_kwargs,
            )

        df = self._pgt.results()
        if not isinstance(df, _pd.DataFrame):
            raise ValueError("No results returned from pygtide prediction.")

        normalised_cols = ["datetime", "signal", "tide", "pole_tide", "lod_tide"]
        df = df.rename(
            columns={a: b for a, b in zip(df.columns, normalised_cols)}
        ).set_index("datetime")
        df = df.set_index(to_naive_utc_datetime(df.index))
        if unit == "nm/s^2":
            return df
        elif unit == "ugal":
            return df * 1e-3
        elif unit == "mgal":
            return df * 1e-4

    def tidal_correction(
        self,
        lat: FloatArray,
        lon: FloatArray,
        elev: FloatArray,
        date_time: DatetimeArray,
        site_id: SiteIDArray | None = None,
        unit: Literal["mgal", "ugal", "nm/s^2"] = "mgal",
        sample_interval: int = 60,
        **kwargs,
    ) -> NDArray[_np.float64]:

        lat = to_1d_ndarray(lat).astype(float)
        lon = to_1d_ndarray(lon, expected_size=lat.size).astype(float)
        elev = to_1d_ndarray(elev, expected_size=lat.size).astype(float)
        date_time = _pd.DatetimeIndex(
            to_naive_utc_datetime(date_time, allow_nat=False)
        ).round(freq="1s")

        # datume for converting date_time to seconds since epoch for interpolation.
        # - set to a day before the minimum ensure that all are captured
        t0 = date_time.floor(freq="s").min() - _pd.Timedelta(days=1)

        # date_time_seconds = (date_time - t0).total_seconds().to_numpy(float)
        date_time_seconds = date_time.astype("int64")

        if site_id is None:
            raise ValueError(
                "site_id is a required parameter for "
                "EternaPredictTidalCorrection tidal_correction method."
            )
        elif isinstance(site_id, str):
            site_id = [site_id] * lat.size
        site_id = to_1d_ndarray(site_id, expected_size=lat.size).astype(str)

        corrs = _np.zeros_like(lat, dtype=float)

        for site in _np.unique(site_id):
            site_mask = site_id == site

            # ensure at least 1 hour of data before and after the range
            # of date_time for this site
            t0 = (date_time[site_mask].min() - _pd.Timedelta(hours=1)).normalize()
            _duration_hrs = (
                int(
                    _np.ceil((date_time[site_mask].max() - t0).total_seconds() / 3600.0)
                )
            ) + 1

            ts = self.time_series(
                lat=lat[site_mask][0],
                lon=lon[site_mask][0],
                elev=elev[site_mask][0],
                starttime=t0,
                duration=_duration_hrs,
                sample_interval=sample_interval,
                unit=unit,
            ).mul(-1.0)

            if not isinstance(ts.index, _pd.DatetimeIndex):
                raise ValueError(
                    "Unexpected time series index type from pygtide results."
                )
            ts = ts.set_index(ts.index.round(freq="1s"))

            if not (corrs[site_mask] == 0.0).all():
                raise ValueError("Unexpected non-zero values in corrs for site mask.")
            corrs[site_mask] = _np.interp(
                x=date_time[site_mask],
                xp=ts.index,
                fp=ts["signal"].to_numpy(),
            )

        return corrs
