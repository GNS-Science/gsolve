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
"""Tests for gsolve.reductions.anomalies — GravityAnomalies class."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gsolve.reductions.anomalies import GravityAnomalies
from gsolve.reductions.corrections import (
    GravityCorrectionParameters,
    GravityCorrectionProvider,
    GravityCorrections,
)
from gsolve.sites import GravitySites


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SITE_IDS = ["A", "B", "C"]
LATS = [-45.0, -44.0, -43.0]
LONS = [170.0, 171.0, 172.0]
HEIGHTS = [10.0, 50.0, 200.0]
ABS_GRAVITY = [980100.0, 980200.0, 980300.0]


def _make_sites() -> GravitySites:
    return GravitySites(
        site_id=SITE_IDS,
        latitude=LATS,
        longitude=LONS,
        height_ellipsoidal=HEIGHTS,
    )


def _make_abs_gravity_series() -> pd.Series:
    return pd.Series(
        data=ABS_GRAVITY,
        index=pd.Index(SITE_IDS, name="site_id"),
        name="absolute_gravity",
    )


def _make_abs_gravity_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {"absolute_gravity": ABS_GRAVITY},
        index=pd.Index(SITE_IDS, name="site_id"),
    )


def _make_anomalies(corrections_parameters=None) -> GravityAnomalies:
    if corrections_parameters is None:
        corrections_parameters = GravityCorrectionParameters()
    return GravityAnomalies(
        absolute_gravity=_make_abs_gravity_series(),
        sites=_make_sites(),
        corrections_parameters=corrections_parameters,
    )


# ---------------------------------------------------------------------------
# Construction from different absolute_gravity types
# ---------------------------------------------------------------------------


class TestGravityAnomaliesInit:
    def test_from_series(self):
        ga = GravityAnomalies(
            absolute_gravity=_make_abs_gravity_series(),
            sites=_make_sites(),
            corrections_parameters=GravityCorrectionParameters(),
        )
        assert len(ga.data) == 3

    def test_from_dataframe(self):
        ga = GravityAnomalies(
            absolute_gravity=_make_abs_gravity_dataframe(),
            sites=_make_sites(),
            corrections_parameters=GravityCorrectionParameters(),
        )
        assert len(ga.data) == 3

    def test_bad_absolute_gravity_type_raises(self):
        with pytest.raises(TypeError, match="invalid type for arg 'absolute_gravity'"):
            GravityAnomalies(
                absolute_gravity=[1.0, 2.0, 3.0],
                sites=_make_sites(),
                corrections_parameters=GravityCorrectionParameters(),
            )

    def test_sites_as_dataframe(self):
        sites_df = pd.DataFrame(
            {
                "latitude": LATS,
                "longitude": LONS,
                "height_ellipsoidal": HEIGHTS,
            },
            index=pd.Index(SITE_IDS, name="site_id"),
        )
        ga = GravityAnomalies(
            absolute_gravity=_make_abs_gravity_series(),
            sites=sites_df,
            corrections_parameters=GravityCorrectionParameters(),
        )
        assert len(ga.data) == 3

    def test_bad_sites_type_raises(self):
        with pytest.raises(TypeError, match="invalid type for arg 'sites'"):
            GravityAnomalies(
                absolute_gravity=_make_abs_gravity_series(),
                sites="bad_sites",
                corrections_parameters=GravityCorrectionParameters(),
            )

    def test_missing_site_in_sites_raises(self):
        # absolute_gravity has a site not in sites
        ag = pd.Series(
            [980000.0],
            index=pd.Index(["MISSING"], name="site_id"),
            name="absolute_gravity",
        )
        with pytest.raises(ValueError, match="no corresponding site info"):
            GravityAnomalies(
                absolute_gravity=ag,
                sites=_make_sites(),
                corrections_parameters=GravityCorrectionParameters(),
            )


# ---------------------------------------------------------------------------
# Construction from different corrections_parameters types
# ---------------------------------------------------------------------------


class TestCorrectionParameterTypes:
    def test_from_correction_parameters(self):
        ga = _make_anomalies(GravityCorrectionParameters())
        assert isinstance(ga.params, GravityCorrectionParameters)

    def test_from_correction_provider(self):
        provider = GravityCorrectionProvider()
        ga = GravityAnomalies(
            absolute_gravity=_make_abs_gravity_series(),
            sites=_make_sites(),
            corrections_parameters=provider,
        )
        assert isinstance(ga.params, GravityCorrectionParameters)

    def test_from_precomputed_corrections(self):
        sites = _make_sites()
        provider = GravityCorrectionProvider()
        corrs = provider.bouguer_corrections(sites)
        ga = GravityAnomalies(
            absolute_gravity=_make_abs_gravity_series(),
            sites=sites,
            corrections_parameters=corrs,
        )
        assert isinstance(ga.params, GravityCorrectionParameters)
        assert len(ga.data) == 3

    def test_bad_corrections_type_raises(self):
        with pytest.raises(TypeError, match="invalid type for corrections_provider"):
            GravityAnomalies(
                absolute_gravity=_make_abs_gravity_series(),
                sites=_make_sites(),
                corrections_parameters="bad",
            )

    def test_precomputed_corrections_missing_site_raises(self):
        # Precomputed corrections only has 2 of 3 sites
        sites = _make_sites()
        provider = GravityCorrectionProvider()
        corrs = provider.bouguer_corrections(sites)
        # Drop one site from precomputed corrections
        corrs.data = corrs.data.iloc[:2]
        with pytest.raises(
            ValueError, match="do not provide corrections for all sites"
        ):
            GravityAnomalies(
                absolute_gravity=_make_abs_gravity_series(),
                sites=sites,
                corrections_parameters=corrs,
            )


# ---------------------------------------------------------------------------
# Output columns and values
# ---------------------------------------------------------------------------


class TestGravityAnomaliesOutput:
    def test_free_air_anomaly_column_exists(self):
        ga = _make_anomalies()
        assert "free_air_anomaly" in ga.data.columns

    def test_bouguer_anomaly_simple_column_exists(self):
        ga = _make_anomalies()
        assert "bouguer_anomaly_simple" in ga.data.columns

    def test_bouguer_anomaly_complete_is_nan_without_terrain(self):
        ga = _make_anomalies()
        # Without terrain corrections, complete Bouguer should be NaN
        assert ga.data["bouguer_anomaly_complete"].isna().all()

    def test_free_air_anomaly_values(self):
        ga = _make_anomalies()
        # FAA = AG - (NG + FAC);  all values should be finite floats
        assert ga.data["free_air_anomaly"].notna().all()

    def test_bouguer_anomaly_simple_values(self):
        ga = _make_anomalies()
        assert ga.data["bouguer_anomaly_simple"].notna().all()

    def test_absolute_gravity_preserved(self):
        ga = _make_anomalies()
        np.testing.assert_array_almost_equal(
            ga.data["absolute_gravity"].to_numpy(), ABS_GRAVITY
        )

    def test_index_is_site_id(self):
        ga = _make_anomalies()
        assert list(ga.data.index) == SITE_IDS
        assert ga.data.index.name == "site_id"

    def test_with_slab_correction_no_curvature(self):
        params = GravityCorrectionParameters(use_curvature_corrected=False)
        ga = _make_anomalies(params)
        assert "bouguer_slab_correction" in ga.data.columns
        assert ga.data["bouguer_anomaly_simple"].notna().all()

    def test_without_atmospheric_correction(self):
        params = GravityCorrectionParameters(use_atmospheric_correction=False)
        ga = _make_anomalies(params)
        assert (
            "atmospheric_correction" not in ga.data.columns
            or ga.data.get("atmospheric_correction", pd.Series([0.0])).eq(0.0).all()
        )

    def test_params_stored(self):
        params = GravityCorrectionParameters(density_crust=2800.0)
        ga = _make_anomalies(params)
        assert ga.params.density_crust == pytest.approx(2800.0)

    def test_tcorr_params_empty_without_terrain(self):
        ga = _make_anomalies()
        assert ga.tcorr_params == {}

    def test_repr(self):
        ga = _make_anomalies()
        r = repr(ga)
        assert "GravityAnomalies" in r
        assert "n_sites=3" in r
        assert "free_air_anomaly" in r
