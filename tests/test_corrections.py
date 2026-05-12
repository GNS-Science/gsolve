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
"""Tests for gsolve.reductions.corrections."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import boule

from gsolve.reductions.corrections import (
    GravityCorrectionParameters,
    GravityCorrectionProvider,
    GravityCorrections,
    atmospheric_correction,
    bouguer_slab_correction,
    bouguer_slab_curvature_corrected,
    free_air_correction,
    normal_gravity_at_ellipsoid,
    normal_gravity_at_stn_elevation,
    spherical_bouguer_cap_correction,
)
from gsolve.sites import GravitySites


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sites(
    lats=(0.0, -45.0),
    heights=(0.0, 100.0),
) -> GravitySites:
    n = len(lats)
    return GravitySites(
        site_id=[f"S{i}" for i in range(n)],
        latitude=list(lats),
        longitude=[0.0] * n,
        height_ellipsoidal=list(heights),
    )


# ---------------------------------------------------------------------------
# normal_gravity_at_stn_elevation
# ---------------------------------------------------------------------------


class TestNormalGravityAtStnElevation:
    def test_scalar_at_equator_sea_level(self):
        g = normal_gravity_at_stn_elevation(latitude=0.0, height_ellipsoidal=0.0)
        # GRS80 equatorial normal gravity is ~978032.7 mGal
        assert pytest.approx(g, rel=1e-4) == 978032.67715

    def test_si_units(self):
        g_mgal = normal_gravity_at_stn_elevation(0.0, 0.0, si_units=False)
        g_si = normal_gravity_at_stn_elevation(0.0, 0.0, si_units=True)
        assert pytest.approx(g_si, rel=1e-6) == g_mgal * 1e-5

    def test_wgs84_ellipsoid(self):
        g = normal_gravity_at_stn_elevation(0.0, 0.0, ellipsoid="WGS84")
        assert g > 0

    def test_boule_ellipsoid_object(self):
        g = normal_gravity_at_stn_elevation(0.0, 0.0, ellipsoid=boule.GRS80)
        assert g > 0

    def test_negative_height_raises(self):
        with pytest.raises(ValueError, match="heights must be >= 0"):
            normal_gravity_at_stn_elevation(0.0, -1.0)

    def test_bad_ellipsoid_raises(self):
        with pytest.raises(ValueError, match="Unknown ellipsoid"):
            normal_gravity_at_stn_elevation(0.0, 0.0, ellipsoid="GRS67")

    def test_array_input(self):
        lats = np.array([0.0, -45.0, -90.0])
        hts = np.array([0.0, 0.0, 0.0])
        g = normal_gravity_at_stn_elevation(lats, hts)
        assert g.shape == (3,)
        # gravity at pole > gravity at equator
        assert g[2] > g[0]

    def test_gravity_decreases_with_height(self):
        g0 = normal_gravity_at_stn_elevation(0.0, 0.0)
        g100 = normal_gravity_at_stn_elevation(0.0, 100.0)
        assert g100 < g0


# ---------------------------------------------------------------------------
# normal_gravity_at_ellipsoid
# ---------------------------------------------------------------------------


class TestNormalGravityAtEllipsoid:
    def test_grs80_equator(self):
        g = normal_gravity_at_ellipsoid(latitude=0.0, ellipsoid="GRS80")
        assert pytest.approx(g, rel=1e-4) == 978032.67715

    def test_grs80_pole(self):
        g = normal_gravity_at_ellipsoid(latitude=90.0, ellipsoid="GRS80")
        assert pytest.approx(g, rel=1e-4) == 983218.63685

    def test_wgs84_equals_grs80(self):
        # WGS84 is treated as GRS80 in this function
        g80 = normal_gravity_at_ellipsoid(0.0, ellipsoid="GRS80")
        g84 = normal_gravity_at_ellipsoid(0.0, ellipsoid="WGS84")
        assert pytest.approx(g84, rel=1e-6) == g80

    def test_grs67(self):
        g = normal_gravity_at_ellipsoid(0.0, ellipsoid="GRS67")
        assert pytest.approx(g, rel=1e-4) == 978031.84558

    def test_si_units(self):
        g_mgal = normal_gravity_at_ellipsoid(0.0, si_units=False)
        g_si = normal_gravity_at_ellipsoid(0.0, si_units=True)
        assert pytest.approx(g_si, rel=1e-6) == g_mgal * 1e-5

    def test_bad_ellipsoid_raises(self):
        with pytest.raises(ValueError, match="Unknown ellipsoid"):
            normal_gravity_at_ellipsoid(0.0, ellipsoid="SPHERE")

    def test_array_input(self):
        lats = np.array([0.0, 45.0, 90.0])
        g = normal_gravity_at_ellipsoid(lats)
        assert g.shape == (3,)
        assert g[2] > g[0]


# ---------------------------------------------------------------------------
# free_air_correction
# ---------------------------------------------------------------------------


class TestFreeAirCorrection:
    def test_zero_height_returns_zero(self):
        fac = free_air_correction(latitude=0.0, height_ellipsoidal=0.0)
        assert float(fac) == pytest.approx(0.0, abs=1e-10)

    def test_positive_height_gives_negative_fac(self):
        # FAC is negative for positive heights in this sign convention
        fac = free_air_correction(latitude=-45.0, height_ellipsoidal=100.0)
        assert float(fac) < 0

    def test_custom_gradient(self):
        fac_default = free_air_correction(0.0, 100.0)
        fac_custom = free_air_correction(0.0, 100.0, free_air_gradient=0.31)
        # Larger gradient → more negative FAC
        assert float(fac_custom) < float(fac_default)

    def test_array_input(self):
        fac = free_air_correction(
            latitude=np.array([-45.0, 0.0]),
            height_ellipsoidal=np.array([100.0, 200.0]),
        )
        assert fac.shape == (2,)
        assert np.all(fac < 0)


# ---------------------------------------------------------------------------
# atmospheric_correction
# ---------------------------------------------------------------------------


class TestAtmosphericCorrection:
    def test_sea_level_value(self):
        # at h=0: 0.874 * -1 = -0.874 mGal
        ac = atmospheric_correction(0.0)
        assert float(ac) == pytest.approx(-0.874, rel=1e-4)

    def test_increases_with_height(self):
        # atmospheric correction becomes less negative (smaller magnitude) with altitude
        ac0 = atmospheric_correction(0.0)
        ac1000 = atmospheric_correction(1000.0)
        assert float(ac1000) > float(ac0)

    def test_array_input(self):
        ac = atmospheric_correction(np.array([0.0, 1000.0, 5000.0]))
        assert ac.shape == (3,)


# ---------------------------------------------------------------------------
# bouguer_slab_correction
# ---------------------------------------------------------------------------


class TestBouguerSlabCorrection:
    def test_zero_height_returns_zero(self):
        bc = bouguer_slab_correction(0.0)
        assert float(bc) == pytest.approx(0.0, abs=1e-10)

    def test_positive_height_positive_correction(self):
        bc = bouguer_slab_correction(100.0)
        assert float(bc) > 0

    def test_negative_height_ocean_uses_density_contrast(self):
        # ocean case: density contrast (water - crust) → negative correction
        bc = bouguer_slab_correction(-100.0)
        assert float(bc) < 0

    def test_custom_density(self):
        bc_default = bouguer_slab_correction(100.0)
        bc_denser = bouguer_slab_correction(100.0, density_crust=3000.0)
        assert float(bc_denser) > float(bc_default)

    def test_array_input(self):
        bc = bouguer_slab_correction(np.array([0.0, 100.0, -100.0]))
        assert bc.shape == (3,)


# ---------------------------------------------------------------------------
# spherical_bouguer_cap_correction
# ---------------------------------------------------------------------------


class TestSphericalBouguerCapCorrection:
    def test_zero_height_returns_zero(self):
        sbc = spherical_bouguer_cap_correction(0.0)
        assert float(sbc) == pytest.approx(0.0, abs=1e-10)

    def test_positive_at_positive_height(self):
        sbc = spherical_bouguer_cap_correction(100.0)
        assert float(sbc) > 0

    def test_array_input(self):
        sbc = spherical_bouguer_cap_correction(np.array([0.0, 100.0, 1000.0]))
        assert sbc.shape == (3,)


# ---------------------------------------------------------------------------
# bouguer_slab_curvature_corrected
# ---------------------------------------------------------------------------


class TestBouguerSlabCurvatureCorrected:
    def test_string_ellipsoid_grs80(self):
        bc = bouguer_slab_curvature_corrected(
            100.0,
            density_water=1030.0,
            density_crust=2670.0,
            ellipsoid_or_radius="GRS80",
        )
        assert float(bc) > 0

    def test_string_ellipsoid_wgs84(self):
        bc = bouguer_slab_curvature_corrected(
            100.0,
            density_water=1030.0,
            density_crust=2670.0,
            ellipsoid_or_radius="WGS84",
        )
        assert float(bc) > 0

    def test_boule_ellipsoid_object(self):
        bc = bouguer_slab_curvature_corrected(
            100.0,
            density_water=1030.0,
            density_crust=2670.0,
            ellipsoid_or_radius=boule.GRS80,
        )
        assert float(bc) > 0

    def test_float_radius(self):
        bc = bouguer_slab_curvature_corrected(
            100.0,
            density_water=1030.0,
            density_crust=2670.0,
            ellipsoid_or_radius=6371000.0,
        )
        assert float(bc) > 0

    def test_bad_string_ellipsoid_raises(self):
        with pytest.raises(ValueError, match="Unknown ellipsoid"):
            bouguer_slab_curvature_corrected(
                100.0,
                density_water=1030.0,
                density_crust=2670.0,
                ellipsoid_or_radius="BOGUS",
            )

    def test_ocean_negative_height(self):
        bc = bouguer_slab_curvature_corrected(
            -100.0, density_water=1030.0, density_crust=2670.0
        )
        assert float(bc) < 0

    def test_array_input(self):
        bc = bouguer_slab_curvature_corrected(
            np.array([100.0, 200.0]),
            density_water=1030.0,
            density_crust=2670.0,
        )
        assert bc.shape == (2,)


# ---------------------------------------------------------------------------
# GravityCorrectionParameters
# ---------------------------------------------------------------------------


class TestGravityCorrectionParameters:
    def test_defaults(self):
        p = GravityCorrectionParameters()
        assert p.ellipsoid == "GRS80"
        assert p.density_crust == 2670.0
        assert p.density_water == 1030.0
        assert p.use_curvature_corrected is True
        assert p.use_atmospheric_correction is True

    def test_bouguer_correction_type_curvature(self):
        p = GravityCorrectionParameters(use_curvature_corrected=True)
        assert p.bouguer_correction_type() == "bouguer_slab_curvature_corrected"

    def test_bouguer_correction_type_slab(self):
        p = GravityCorrectionParameters(use_curvature_corrected=False)
        assert p.bouguer_correction_type() == "bouguer_slab_correction"

    def test_bouguer_correction_fields_with_atm(self):
        p = GravityCorrectionParameters(
            use_atmospheric_correction=True, use_curvature_corrected=True
        )
        fields = p.bouguer_correction_fields()
        assert "atmospheric_correction" in fields
        assert "bouguer_slab_curvature_corrected" in fields

    def test_bouguer_correction_fields_without_atm(self):
        p = GravityCorrectionParameters(use_atmospheric_correction=False)
        fields = p.bouguer_correction_fields()
        assert "atmospheric_correction" not in fields


# ---------------------------------------------------------------------------
# GravityCorrections
# ---------------------------------------------------------------------------


class TestGravityCorrections:
    def test_init_basic(self):
        gc = GravityCorrections(
            params=None,
            site_id=["A", "B"],
            free_air_correction=[10.0, 20.0],
        )
        assert gc.data.shape[0] == 2
        assert "free_air_correction" in gc.data.columns

    def test_default_params_created_when_none(self):
        gc = GravityCorrections(params=None, site_id=["A"])
        assert isinstance(gc.params, GravityCorrectionParameters)

    def test_bad_correction_key_raises(self):
        with pytest.raises(ValueError, match="Unknown correction type"):
            GravityCorrections(params=None, site_id=["A"], bogus_correction=[1.0])

    def test_empty_site_id_raises(self):
        with pytest.raises(ValueError, match="site_id must not be empty"):
            GravityCorrections(params=None, site_id=[])

    def test_repr(self):
        gc = GravityCorrections(params=None, site_id=["A"], free_air_correction=[1.0])
        assert "GravityCorrections" in repr(gc)


# ---------------------------------------------------------------------------
# GravityCorrectionProvider
# ---------------------------------------------------------------------------


class TestGravityCorrectionProvider:
    def test_init_default(self):
        p = GravityCorrectionProvider()
        assert isinstance(p.params, GravityCorrectionParameters)

    def test_init_with_params(self):
        params = GravityCorrectionParameters(density_crust=2800.0)
        p = GravityCorrectionProvider(params=params)
        assert p.params.density_crust == 2800.0

    def test_init_bad_params_raises(self):
        with pytest.raises(TypeError, match="params must be None"):
            GravityCorrectionProvider(params="bad")

    def test_available_corrections(self):
        corrs = GravityCorrectionProvider.available_corrections()
        assert "free_air_correction" in corrs
        assert "normal_gravity_at_ellipsoid" in corrs
        assert "bouguer_slab_correction" in corrs

    def test_repr(self):
        p = GravityCorrectionProvider()
        assert "GravityCorrectionProvider" in repr(p)

    def test_compute_with_gravity_sites(self):
        sites = _make_sites(lats=(-45.0,), heights=(100.0,))
        provider = GravityCorrectionProvider()
        gc = provider.compute(sites, corrections=["free_air_correction"])
        assert "free_air_correction" in gc.data.columns
        assert gc.data.shape[0] == 1

    def test_compute_with_dataframe(self):
        df = pd.DataFrame(
            {"latitude": [-45.0, 0.0], "height_ellipsoidal": [100.0, 0.0]},
            index=pd.Index(["A", "B"], name="site_id"),
        )
        provider = GravityCorrectionProvider()
        gc = provider.compute(
            df, corrections=["free_air_correction", "atmospheric_correction"]
        )
        assert gc.data.shape[0] == 2
        assert "free_air_correction" in gc.data.columns
        assert "atmospheric_correction" in gc.data.columns

    def test_compute_single_correction_string(self):
        sites = _make_sites()
        provider = GravityCorrectionProvider()
        gc = provider.compute(sites, corrections="free_air_correction")
        assert "free_air_correction" in gc.data.columns

    def test_compute_bad_correction_raises(self):
        sites = _make_sites()
        provider = GravityCorrectionProvider()
        with pytest.raises(ValueError, match="Unrecognised corrections"):
            provider.compute(sites, corrections=["not_a_correction"])

    def test_compute_bad_sites_type_raises(self):
        provider = GravityCorrectionProvider()
        with pytest.raises(TypeError, match="must be a Dataframe or GravitySites"):
            provider.compute("not_a_sites_object")

    def test_compute_include_coords(self):
        sites = _make_sites()
        provider = GravityCorrectionProvider()
        gc = provider.compute(
            sites, corrections=["free_air_correction"], include_coords=True
        )
        assert "latitude" in gc.data.columns
        assert "height_ellipsoidal" in gc.data.columns

    def test_bouguer_corrections(self):
        sites = _make_sites()
        provider = GravityCorrectionProvider()
        gc = provider.bouguer_corrections(sites)
        expected_fields = provider.params.bouguer_correction_fields()
        for f in expected_fields:
            assert f in gc.data.columns

    def test_free_air_corrections(self):
        sites = _make_sites()
        provider = GravityCorrectionProvider()
        gc = provider.free_air_corrections(sites)
        assert "free_air_correction" in gc.data.columns
        assert "normal_gravity_at_ellipsoid" in gc.data.columns

    def test_compute_all_corrections(self):
        sites = _make_sites()
        provider = GravityCorrectionProvider()
        gc = provider.compute(
            sites,
            corrections=list(GravityCorrectionProvider.available_corrections()),
        )
        for c in GravityCorrectionProvider.available_corrections():
            assert c in gc.data.columns

    def test_compute_slab_correction_no_curvature(self):
        params = GravityCorrectionParameters(use_curvature_corrected=False)
        provider = GravityCorrectionProvider(params=params)
        sites = _make_sites()
        gc = provider.bouguer_corrections(sites)
        assert "bouguer_slab_correction" in gc.data.columns
        assert "bouguer_slab_curvature_corrected" not in gc.data.columns
