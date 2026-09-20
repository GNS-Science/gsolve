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

from __future__ import annotations
from tornado.routing import AnyMatches

import numpy as np
import pandas as pd
import pytest
from scipy.constants import atm

from gsolve.reductions.anomalies import (
    compute_complete_bouguer_anomaly,
    compute_free_air_anomaly,
    compute_simple_bouguer_anomaly,
)


def anomaly_args():
    return {
        "absolute_gravity": np.array([100.0, 200.0]),
        "normal_gravity": np.array([10.0, 20.0]),
        "free_air_correction": np.array([1.0, 2.0]),
        "bouguer_correction": np.array([5.0, 6.0]),
        "terrain_correction": np.array([2.0, 3.0]),
        "atmospheric_correction": np.array([0.5, 0.5]),
        "spherical_bouguer_cap_correction": np.array([0.0, 0.0]),
    }


def test_compute_complete_bouguer_anomaly_basic():
    c = anomaly_args()
    ag = np.array([100.0, 200.0])
    ng = np.array([10.0, 20.0])
    fac = np.array([1.0, 2.0])
    bc = np.array([5.0, 6.0])
    tc = np.array([2.0, 3.0])
    ac = np.array([0.5, 0.5])
    sbc = np.array([0.0, 0.0])
    result = compute_complete_bouguer_anomaly(
        absolute_gravity=ag,
        normal_gravity=ng,
        free_air_correction=fac,
        bouguer_correction=bc,
        terrain_correction=tc,
        atmospheric_correction=ac,
        spherical_bouguer_cap_correction=sbc,
    )
    expected = ag - (ng + fac + ac + bc + sbc - tc)
    np.testing.assert_allclose(result, expected)


def test_compute_complete_bouguer_anomaly_raises_on_nan():
    a = anomaly_args()
    a["absolute_gravity"][1] = np.nan

    with pytest.raises(ValueError):
        compute_complete_bouguer_anomaly(**a)


def test_compute_simple_bouguer_anomaly_basic():
    ag = np.array([100.0, 200.0])
    ng = np.array([10.0, 20.0])
    fac = np.array([1.0, 2.0])
    ac = np.array([0.5, 0.5])
    bc = np.array([5.0, 6.0])
    sbc = np.array([0.0, 0.0])
    result = compute_simple_bouguer_anomaly(
        absolute_gravity=ag,
        normal_gravity=ng,
        free_air_correction=fac,
        atmospheric_correction=ac,
        bouguer_correction=bc,
        spherical_bouguer_cap_correction=sbc,
    )
    expected = compute_complete_bouguer_anomaly(
        absolute_gravity=ag,
        normal_gravity=ng,
        free_air_correction=fac,
        bouguer_correction=bc,
        atmospheric_correction=ac,
        terrain_correction=np.zeros_like(ac),
        spherical_bouguer_cap_correction=sbc,
    )
    np.testing.assert_allclose(result, expected)


def test_compute_simple_bouguer_anomaly_raises_on_nan():
    ag = np.array([100.0, np.nan])
    ng = np.array([10.0, 20.0])
    fac = np.array([1.0, 2.0])
    ac = np.array([0.5, 0.5])
    bc = np.array([5.0, 6.0])
    with pytest.raises(ValueError):
        compute_simple_bouguer_anomaly(
            absolute_gravity=ag,
            normal_gravity=ng,
            free_air_correction=fac,
            atmospheric_correction=ac,
            bouguer_correction=bc,
        )


def test_compute_free_air_anomaly_basic():
    ag = np.array([100.0, 200.0])
    ng = np.array([10.0, 20.0])
    fac = np.array([1.0, 2.0])
    result = compute_free_air_anomaly(ag, ng, fac)
    expected = ag - (ng + fac)
    np.testing.assert_allclose(result, expected)


def test_compute_free_air_anomaly_raises_on_nan():
    ag = np.array([100.0, np.nan])
    ng = np.array([10.0, 20.0])
    fac = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        compute_free_air_anomaly(ag, ng, fac)
