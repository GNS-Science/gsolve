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

import datetime

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pytest

from gsolve.tide.earth_tide import (
    LongmanTidalCorrection,
    _decimal_hour_of_day,
    _decimal_julian_century,
    gravimetric_factor,
)


@pytest.fixture
def obs_points() -> pd.DataFrame:
    n = 10
    rng = np.random.default_rng(12345)

    return pd.DataFrame(
        index=pd.date_range(start="2021-07-01", periods=n, freq="1h", name="dt"),
        data={
            "lat": rng.uniform(-90, 90, n),
            "lon": rng.uniform(-180, 180, n),
            "elev": rng.uniform(0, 1000, n),
        },
    )


@pytest.fixture
def expected_corrections() -> np.ndarray:
    return np.array(
        [
            -0.01943523,
            -0.06341408,
            -0.03206663,
            0.06375303,
            0.05473333,
            -0.01921678,
            -0.00626144,
            -0.00982098,
            0.02130002,
            -0.08359675,
        ]
    )


def test_longman_tidal_correction(
    obs_points: pd.DataFrame, expected_corrections: np.ndarray
) -> None:
    expected_amp_factor = 1.2
    corrector = LongmanTidalCorrection()
    assert corrector.amp_factor == expected_amp_factor
    output = corrector.tidal_correction(
        obs_points["lat"], obs_points["lon"], obs_points["elev"], obs_points.index
    )
    nptest.assert_allclose(output, expected_corrections, rtol=1e-6)

    # change amp factor, outputs should be scaled to the new factor
    new_amp_factor = 1.1
    corrector = LongmanTidalCorrection(amp_factor=new_amp_factor)
    assert corrector.amp_factor == new_amp_factor
    output = corrector.tidal_correction(
        obs_points["lat"], obs_points["lon"], obs_points["elev"], obs_points.index
    )
    nptest.assert_allclose(
        output,
        new_amp_factor * expected_corrections / expected_amp_factor,
        rtol=1e-6,
    )


def test_longman_gravity_acceleration_bag_args(
    obs_points: pd.DataFrame, expected_corrections: np.ndarray
) -> None:
    corrector = LongmanTidalCorrection()
    with pytest.raises(ValueError):
        _ = corrector.gravity_accelerations(
            lat=obs_points["lat"].iloc[0],
            lon=obs_points["lon"],
            elev=obs_points["elev"],
            dt=obs_points.index,
        )


def test_longman_tidal_correction_identifier() -> None:
    amp_factor = 1.2
    corrector = LongmanTidalCorrection(amp_factor=amp_factor)
    expected_prefix = "LongmanTidalCorrection"
    assert corrector.identifier() == f"{expected_prefix}(amp_factor={amp_factor})"
    assert (
        corrector.identifier(b=123)
        == f"{expected_prefix}(amp_factor={amp_factor},b=123)"
    )


def test_longman_time_series():
    corrector = LongmanTidalCorrection(amp_factor=1.0)
    args = {
        "starttime": "2020/01/01T00:00",
        "endtime": "2020/01/02T00:00",
        "step": "1s",
        "lat": -45.0,
        "lon": 170.0,
        "elev": 0.0,
    }
    ts1 = corrector.time_series(**args)
    assert ts1.shape[0] == 86400 + 1

    ts2 = corrector.time_series(method="acceleration", **args)
    nptest.assert_array_equal(ts1, ts2)

    with pytest.raises(ValueError):
        ts2 = corrector.time_series(method="bad_method", **args)

    # check bad time arguments
    for bad_arg in [
        pd.NaT,
        "not a date",
        None,
        ["2020/01/01T00:00", "2020/01/01T12:00"],
    ]:
        with pytest.raises(ValueError, match="error parsing starttime and endtime:"):
            _ = corrector.time_series(
                starttime=bad_arg,
                endtime="2020/01/02T00:00",
                step="1s",
                lat=-45.0,
                lon=170.0,
            )

    with pytest.raises(ValueError, match="error parsing starttime and endtime:"):
        _ = corrector.time_series(
            starttime="2020/01/03T00:00",
            endtime="2020/01/02T00:00",
            step="1s",
            lat=-45.0,
            lon=170.0,
        )

    with pytest.raises(ValueError, match="error parsing step:"):
        _ = corrector.time_series(
            starttime="2020/01/01T00:00",
            endtime="2020/01/02T00:00",
            step="not a step",
            lat=-45.0,
            lon=170.0,
        )


def test_longman_repr():
    corrector = LongmanTidalCorrection(amp_factor=1.2)
    expected_repr = "LongmanTidalCorrection(amp_factor=1.2)"
    assert repr(corrector) == expected_repr


class TestLongmanTimeFuncs:
    @pytest.mark.parametrize(
        "dt, expected",
        [
            ("2024-01-01T00:00:00", 0.0),
            ("2024-01-01T12:00:00", 12.0),
            ("2024-01-01T23:59:59", 23.99972222222222),
            (datetime.datetime(2024, 1, 1, 6, 30, 0), 6.5),
            (pd.Timestamp("2024-01-01 18:15:30"), 18.258333333333333),
        ],
    )
    def test_decimal_hour_of_day_scalar(self, dt, expected) -> None:
        result = _decimal_hour_of_day(dt)
        assert abs(result - expected) < 1e-8

    def test_decimal_hour_of_day_array(self):
        dts = [
            "2024-01-01T00:00:00",
            "2024-01-01T06:00:00",
            "2024-01-01T12:30:00",
            "2024-01-01T23:59:59",
        ]
        expected = [0.0, 6.0, 12.5, 23.99972222222222]
        result1 = _decimal_hour_of_day(dts)
        np.testing.assert_allclose(result1, expected, rtol=1e-10)
        result2 = _decimal_hour_of_day(pd.Series(dts))
        np.testing.assert_allclose(result2, expected, rtol=1e-10)

    def test_decimal_hour_of_day_pandas_series(self):
        times = pd.date_range("2024-01-01", periods=3, freq="8h")
        expected = [0.0, 8.0, 16.0]
        result_idx = _decimal_hour_of_day(times)
        result_series = _decimal_hour_of_day(times.to_series())
        np.testing.assert_allclose(result_idx, expected, rtol=1e-10)
        np.testing.assert_allclose(result_series, expected, rtol=1e-10)

    def test_decimal_hour_of_day_with_timezone(self):
        # Should convert to naive UTC
        dt = pd.Timestamp("2024-01-01 03:00:00", tz="Europe/Berlin")
        # 03:00 CET == 02:00 UTC
        expected = 2.0
        result = _decimal_hour_of_day(dt)
        assert abs(result - expected) < 1e-8

    def test_decimal_julian_century(self):
        v = _decimal_julian_century("2020/01/01T00:00:00")
        assert isinstance(v, float)
        v = _decimal_julian_century(["2020/01/01T00:00:00"] * 2)
        assert isinstance(v, np.ndarray)
        dates = pd.date_range("2020/01/01T00:00:00", "2020/01/02T00:00:00", periods=100)
        # test that it works with both DatetimeIndex and Series
        d1 = _decimal_julian_century(dates)
        d2 = _decimal_julian_century(pd.Series(data=dates.to_list()))
        nptest.assert_array_equal(d1, d2)

        # check that bad dates are caught
        # - actually a test of the underlying to_naive_utc_datetime
        for bad_arg in ["not a date", pd.NaT, None, ["2020/01/01T00:00:00", None]]:
            with pytest.raises(expected_exception=ValueError):
                _decimal_julian_century(bad_arg)


def test_gravimetric_factor():
    h2 = 1
    k2 = 2
    assert gravimetric_factor(k2, h2) == 1 + h2 - 1.5 * k2
