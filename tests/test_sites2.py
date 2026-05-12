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
"""Tests for gsolve.sites — GravitySites, ReferenceGravity, and combine helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gsolve.sites import (
    GravitySites,
    ReferenceGravity,
    combine_gravity_sites,
    combine_reference_gravity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sites(n: int = 3, with_ref_gravity: bool = False) -> GravitySites:
    site_ids = [f"S{i}" for i in range(n)]
    lats = np.linspace(-45.0, -40.0, n)
    lons = np.linspace(170.0, 175.0, n)
    hts = np.linspace(0.0, 100.0, n)
    ref_grav = np.linspace(980000.0, 980100.0, n) if with_ref_gravity else None
    return GravitySites(
        site_id=site_ids,
        latitude=lats,
        longitude=lons,
        height_ellipsoidal=hts,
        reference_gravity=ref_grav,
    )


def _make_ref_gravity(
    site_ids=("A", "B"), gravities=(980000.0, 980050.0)
) -> ReferenceGravity:
    return ReferenceGravity(site_id=list(site_ids), gravity=list(gravities))


# ---------------------------------------------------------------------------
# GravitySites — construction
# ---------------------------------------------------------------------------


class TestGravitySitesInit:
    def test_basic_construction(self):
        gs = _make_sites(3)
        assert len(gs) == 3
        assert list(gs.data.index) == ["S0", "S1", "S2"]

    def test_repr(self):
        gs = _make_sites(2)
        r = repr(gs)
        assert "GravitySites" in r
        assert "n_sites=2" in r

    def test_duplicate_site_id_raises(self):
        with pytest.raises(ValueError, match="duplicated"):
            GravitySites(
                site_id=["A", "A"],
                latitude=[0.0, 1.0],
                longitude=[0.0, 1.0],
                height_ellipsoidal=[0.0, 0.0],
            )

    def test_reference_gravity_and_tie_defaults(self):
        gs = _make_sites(2)
        assert gs.data["reference_gravity"].isna().all()
        assert gs.data["gsolve_tie"].eq(False).all()

    def test_with_reference_gravity(self):
        gs = _make_sites(2, with_ref_gravity=True)
        assert gs.data["reference_gravity"].notna().all()

    def test_extra_kwargs_stored(self):
        gs = GravitySites(
            site_id=["A"],
            latitude=[0.0],
            longitude=[0.0],
            height_ellipsoidal=[0.0],
            easting=[1000.0],
            northing=[2000.0],
        )
        assert "easting" in gs.data.columns
        assert gs.data.loc["A", "easting"] == 1000.0

    def test_from_dataframe(self):
        df = pd.DataFrame(
            {
                "site_id": ["X", "Y"],
                "latitude": [-45.0, -44.0],
                "longitude": [170.0, 171.0],
                "height_ellipsoidal": [10.0, 20.0],
            }
        )
        gs = GravitySites.from_dataframe(df, use_index=False)
        assert list(gs.data.index) == ["X", "Y"]

    def test_known_fields(self):
        fields = GravitySites.known_fields()
        for f in (
            "latitude",
            "longitude",
            "height_ellipsoidal",
            "reference_gravity",
            "gsolve_tie",
        ):
            assert f in fields


# ---------------------------------------------------------------------------
# GravitySites — get_ties
# ---------------------------------------------------------------------------


class TestGetTies:
    def test_no_ties_returns_empty(self):
        gs = _make_sites(3)
        ties = gs.get_ties(active_only=True)
        assert ties.empty

    def test_active_only_true(self):
        gs = _make_sites(3, with_ref_gravity=True)
        gs.data.loc["S0", "gsolve_tie"] = True
        ties = gs.get_ties(active_only=True)
        assert list(ties.index) == ["S0"]

    def test_active_only_false_returns_all_with_ref_grav(self):
        gs = _make_sites(3, with_ref_gravity=True)
        ties = gs.get_ties(active_only=False)
        assert len(ties) == 3

    def test_gravity_only_false_returns_all_columns(self):
        gs = _make_sites(2, with_ref_gravity=True)
        gs.data.loc["S0", "gsolve_tie"] = True
        ties = gs.get_ties(active_only=True, gravity_only=False)
        assert "latitude" in ties.columns


# ---------------------------------------------------------------------------
# GravitySites — activate_ties / deactivate_ties
# ---------------------------------------------------------------------------


class TestActivateDeactivateTies:
    def test_activate_single_site(self):
        gs = _make_sites(3, with_ref_gravity=True)
        gs.activate_ties("S1")
        assert (
            gs.data.loc["S1", "gsolve_tie"] is True
            or gs.data.loc["S1", "gsolve_tie"] == True
        )

    def test_activate_multiple_sites(self):
        gs = _make_sites(3, with_ref_gravity=True)
        gs.activate_ties(["S0", "S2"])
        assert gs.data.loc["S0", "gsolve_tie"]
        assert gs.data.loc["S2", "gsolve_tie"]

    def test_activate_none_activates_all(self):
        gs = _make_sites(3, with_ref_gravity=True)
        gs.activate_ties(None)
        assert gs.data["gsolve_tie"].all()

    def test_activate_site_without_ref_gravity_raises(self):
        gs = _make_sites(3)  # no ref gravity
        with pytest.raises(ValueError, match="Cannot activate"):
            gs.activate_ties("S0")

    def test_activate_bad_site_id_raises(self):
        gs = _make_sites(3, with_ref_gravity=True)
        with pytest.raises(ValueError, match="not in existing sites"):
            gs.activate_ties("NONEXISTENT")

    def test_deactivate_single_site(self):
        gs = _make_sites(3, with_ref_gravity=True)
        gs.activate_ties(None)
        gs.deactivate_ties("S1")
        assert not gs.data.loc["S1", "gsolve_tie"]

    def test_deactivate_none_deactivates_all(self):
        gs = _make_sites(3, with_ref_gravity=True)
        gs.activate_ties(None)
        gs.deactivate_ties(None)
        assert not gs.data["gsolve_tie"].any()

    def test_deactivate_bad_site_id_raises(self):
        gs = _make_sites(3, with_ref_gravity=True)
        with pytest.raises(ValueError, match="not in existing sites"):
            gs.deactivate_ties("NONEXISTENT")


# ---------------------------------------------------------------------------
# GravitySites — set_reference_gravity
# ---------------------------------------------------------------------------


class TestSetReferenceGravity:
    def test_from_reference_gravity_object(self):
        gs = _make_sites(3)
        ref = _make_ref_gravity(site_ids=("S0", "S1"), gravities=(980000.0, 980050.0))
        gs.set_reference_gravity(ref)
        assert gs.data.loc["S0", "reference_gravity"] == pytest.approx(980000.0)
        assert gs.data.loc["S0", "gsolve_tie"]

    def test_from_dict(self):
        gs = _make_sites(3)
        gs.set_reference_gravity({"S2": 980100.0})
        assert gs.data.loc["S2", "reference_gravity"] == pytest.approx(980100.0)

    def test_from_dataframe(self):
        gs = _make_sites(3)
        df = pd.DataFrame(
            {"gravity": [980000.0]},
            index=pd.Index(["S1"], name="site_id"),
        )
        gs.set_reference_gravity(df)
        assert gs.data.loc["S1", "reference_gravity"] == pytest.approx(980000.0)

    def test_reset_clears_existing(self):
        gs = _make_sites(3, with_ref_gravity=True)
        gs.activate_ties(None)
        gs.set_reference_gravity({"S0": 980000.0}, reset=True)
        # S1 and S2 should be cleared
        assert (
            gs.data.loc["S1", "reference_gravity"]
            != gs.data.loc["S1", "reference_gravity"]
        )  # NaN
        assert gs.data.loc["S0", "reference_gravity"] == pytest.approx(980000.0)

    def test_unmatched_sites_are_ignored(self):
        gs = _make_sites(3)
        gs.set_reference_gravity({"NONEXISTENT": 980000.0})
        assert gs.data["reference_gravity"].isna().all()


# ---------------------------------------------------------------------------
# GravitySites — check_data
# ---------------------------------------------------------------------------


class TestCheckData:
    def test_valid_data_returns_true(self):
        gs = _make_sites(3, with_ref_gravity=True)
        gs.activate_ties("S0")
        assert gs.check_data(warn=False) is True

    def test_no_reference_gravity_returns_false(self):
        gs = _make_sites(3)
        assert gs.check_data(warn=False) is False

    def test_no_ties_returns_false(self):
        gs = _make_sites(3, with_ref_gravity=True)
        # ref gravity set but no tie activated
        assert gs.check_data(warn=False) is False


# ---------------------------------------------------------------------------
# GravitySites — write_to_csv / round-trip
# ---------------------------------------------------------------------------


class TestGravitySitesCsv:
    def test_write_to_csv_and_read_back(self, tmp_path):
        gs = _make_sites(3, with_ref_gravity=True)
        gs.activate_ties("S0")
        fpath = tmp_path / "sites.csv"
        gs.write_to_csv(fpath)
        df = pd.read_csv(fpath, index_col="site_id")
        assert list(df.index) == ["S0", "S1", "S2"]
        assert "latitude" in df.columns


# ---------------------------------------------------------------------------
# ReferenceGravity — construction
# ---------------------------------------------------------------------------


class TestReferenceGravityInit:
    def test_basic_construction(self):
        ref = _make_ref_gravity()
        assert len(ref) == 2
        assert list(ref.data.index) == ["A", "B"]

    def test_active_default_true(self):
        ref = _make_ref_gravity()
        assert ref.data["active"].all()

    def test_active_false(self):
        ref = ReferenceGravity(site_id=["A"], gravity=[980000.0], active=False)
        assert not ref.data.loc["A", "active"]

    def test_duplicate_site_id_raises(self):
        with pytest.raises(ValueError, match="duplicated"):
            ReferenceGravity(site_id=["A", "A"], gravity=[1.0, 2.0])

    def test_empty_site_id_raises(self):
        with pytest.raises(ValueError, match="empty values"):
            ReferenceGravity(site_id=["A", ""], gravity=[1.0, 2.0])

    def test_null_gravity_raises(self):
        with pytest.raises(ValueError, match="null values"):
            ReferenceGravity(site_id=["A"], gravity=[np.nan])

    def test_from_dict_scalar_values(self):
        ref = ReferenceGravity.from_dict({"X": 980000.0, "Y": 980050.0})
        assert list(ref.data.index) == ["X", "Y"]
        assert ref.data.loc["X", "gravity"] == pytest.approx(980000.0)

    def test_from_dict_tuple_values(self):
        ref = ReferenceGravity.from_dict(
            {"X": (980000.0, True), "Y": (980050.0, False)}
        )
        assert ref.data.loc["X", "active"]
        assert not ref.data.loc["Y", "active"]

    def test_from_dataframe(self):
        df = pd.DataFrame(
            {"gravity": [980000.0], "active": [True]},
            index=pd.Index(["A"], name="site_id"),
        )
        ref = ReferenceGravity.from_dataframe(df)
        assert ref.data.loc["A", "gravity"] == pytest.approx(980000.0)

    def test_write_to_csv(self, tmp_path):
        ref = _make_ref_gravity()
        fpath = tmp_path / "ref.csv"
        ref.write_to_csv(fpath)
        df = pd.read_csv(fpath, index_col="site_id")
        assert "gravity" in df.columns


# ---------------------------------------------------------------------------
# combine_gravity_sites
# ---------------------------------------------------------------------------


class TestCombineGravitySites:
    def test_combine_two(self):
        gs1 = GravitySites(
            site_id=["A", "B"],
            latitude=[0.0, 1.0],
            longitude=[0.0, 1.0],
            height_ellipsoidal=[0.0, 0.0],
        )
        gs2 = GravitySites(
            site_id=["C", "D"],
            latitude=[2.0, 3.0],
            longitude=[2.0, 3.0],
            height_ellipsoidal=[0.0, 0.0],
        )
        combined = combine_gravity_sites([gs1, gs2])
        assert len(combined) == 4
        assert set(combined.data.index) == {"A", "B", "C", "D"}

    def test_combine_drops_duplicates_by_default(self):
        gs1 = GravitySites(
            site_id=["A", "B"],
            latitude=[0.0, 1.0],
            longitude=[0.0, 1.0],
            height_ellipsoidal=[0.0, 0.0],
        )
        gs2 = GravitySites(
            site_id=["B", "C"],
            latitude=[9.0, 3.0],
            longitude=[9.0, 3.0],
            height_ellipsoidal=[0.0, 0.0],
        )
        combined = combine_gravity_sites([gs1, gs2])
        assert len(combined) == 3
        # first occurrence of B is kept
        assert combined.data.loc["B", "latitude"] == pytest.approx(1.0)

    def test_combine_error_on_duplicates(self):
        gs1 = GravitySites(
            site_id=["A"], latitude=[0.0], longitude=[0.0], height_ellipsoidal=[0.0]
        )
        gs2 = GravitySites(
            site_id=["A"], latitude=[1.0], longitude=[1.0], height_ellipsoidal=[0.0]
        )
        with pytest.raises(ValueError, match="Duplicate"):
            combine_gravity_sites([gs1, gs2], duplicates="error")

    def test_too_few_sites_raises(self):
        gs = _make_sites(2)
        with pytest.raises(ValueError, match="at least 2"):
            combine_gravity_sites([gs])

    def test_bad_duplicates_arg_raises(self):
        gs1, gs2 = _make_sites(2), _make_sites(2)
        with pytest.raises(ValueError, match="duplicates must be one of"):
            combine_gravity_sites([gs1, gs2], duplicates="invalid")

    def test_non_sites_object_raises(self):
        gs = _make_sites(2)
        with pytest.raises(TypeError):
            combine_gravity_sites([gs, "not_a_sites"])


# ---------------------------------------------------------------------------
# combine_reference_gravity
# ---------------------------------------------------------------------------


class TestCombineReferenceGravity:
    def test_combine_two(self):
        r1 = _make_ref_gravity(("A", "B"), (1.0, 2.0))
        r2 = _make_ref_gravity(("C", "D"), (3.0, 4.0))
        combined = combine_reference_gravity([r1, r2])
        assert len(combined) == 4

    def test_combine_drops_duplicates_by_default(self):
        r1 = _make_ref_gravity(("A",), (1.0,))
        r2 = _make_ref_gravity(("A",), (99.0,))
        combined = combine_reference_gravity([r1, r2])
        assert len(combined) == 1
        assert combined.data.loc["A", "gravity"] == pytest.approx(1.0)

    def test_combine_error_on_duplicates(self):
        r1 = _make_ref_gravity(("A",), (1.0,))
        r2 = _make_ref_gravity(("A",), (2.0,))
        with pytest.raises(ValueError, match="Duplicate"):
            combine_reference_gravity([r1, r2], duplicates="error")

    def test_too_few_raises(self):
        r = _make_ref_gravity()
        with pytest.raises(ValueError, match="at least 2"):
            combine_reference_gravity([r])

    def test_non_ref_gravity_object_raises(self):
        r = _make_ref_gravity()
        with pytest.raises(TypeError):
            combine_reference_gravity([r, "bad"])
