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
"""Unit tests for :mod:`gsolve.reductions.terrain_corrections`.

These tests target the non-numerical "plumbing" of the terrain-correction
machinery: parameter validation and serialization, the ``TerrainCorrector``
zone bookkeeping, and the ``TerrainCorrectionData`` container (correction
accumulation, lookups and CSV round-tripping). The heavy numerical path
(``calculate_terrain_correction`` with real harmonica prisms) is exercised by
``test_terrain_corrections_consistency.py``.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from gsolve.reductions.terrain_corrections import (
    TerrainCorrectionData,
    TerrainCorrectionParameters,
    TerrainCorrector,
    _is_dataarray,
    _to_bool,
)


@pytest.fixture
def flat_dem() -> xr.DataArray:
    """A small flat DEM used as a valid ``dem_source``."""
    x = np.linspace(-100.0, 100.0, 21)
    y = np.linspace(-100.0, 100.0, 21)
    data = np.zeros((21, 21))
    return xr.DataArray(
        data,
        coords={"northing": y, "easting": x},
        dims=("northing", "easting"),
        name="elevation",
    )


def make_params(dem: xr.DataArray, **overrides) -> TerrainCorrectionParameters:
    """Build a valid ``TerrainCorrectionParameters`` with optional overrides."""
    kwargs = {
        "name": "z1",
        "min_dist": 1.0,
        "max_dist": 50.0,
        "dem_source": dem,
        "compute_bathymetry": False,
    }
    kwargs.update(overrides)
    return TerrainCorrectionParameters(**kwargs)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def test_is_dataarray(flat_dem: xr.DataArray) -> None:
    assert _is_dataarray(flat_dem) is True
    assert _is_dataarray(5) is False
    assert _is_dataarray(np.zeros((3, 3))) is False


def test_to_bool() -> None:
    # genuine booleans and numbers behave like builtin bool
    assert _to_bool(True) is True
    assert _to_bool(False) is False
    assert _to_bool(1) is True
    assert _to_bool(0) is False
    assert _to_bool(1.0) is True

    # serialized string forms are interpreted sensibly (the key fix: "False")
    assert _to_bool("True") is True
    assert _to_bool("true") is True
    assert _to_bool("False") is False
    assert _to_bool("false") is False
    assert _to_bool("no") is False
    assert _to_bool("0") is False
    assert _to_bool("") is False
    assert _to_bool("none") is False


# ---------------------------------------------------------------------------
# TerrainCorrectionParameters
# ---------------------------------------------------------------------------


def test_parameters_basic_coercion(flat_dem: xr.DataArray) -> None:
    p = make_params(flat_dem, min_dist=1, max_dist="50")
    # numeric fields are coerced to float
    assert isinstance(p.min_dist, float)
    assert p.max_dist == pytest.approx(50.0)
    # str fields are coerced to str
    assert p.name == "z1"
    assert p.method == "harmonica"
    # bool fields are coerced to bool
    assert p.compute_bathymetry is False
    assert p.compute_topography is True


def test_parameters_setattr_rejects_unknown_field(flat_dem: xr.DataArray) -> None:
    p = make_params(flat_dem)
    # pydantic rejects assignment to an attribute that is not a model field.
    with pytest.raises(ValidationError, match=r"not_a_real_field|no_such_attribute"):
        p.not_a_real_field = 3


def test_parameters_sanity_check_error_on_bad_distances(
    flat_dem: xr.DataArray,
) -> None:
    p = make_params(flat_dem)
    # bypass validation to set an invalid combination, then check explicitly
    p.min_dist = 10.0
    p.max_dist = 5.0
    with pytest.raises(ValueError, match=r"min_dist.*max_dist|max_dist"):
        p._sanity_check(if_errors="error")


def test_parameters_post_init_warns_on_bad_distances(flat_dem: xr.DataArray) -> None:
    # __post_init__ runs _sanity_check(if_errors="warn")
    with pytest.warns(UserWarning, match=r"min_dist"):
        make_params(flat_dem, min_dist=10.0, max_dist=5.0)


def test_parameters_to_dict(flat_dem: xr.DataArray) -> None:
    p = make_params(flat_dem)
    d = p.to_dict()
    assert d["name"] == "z1"
    assert d["min_dist"] == pytest.approx(1.0)
    # with a DataArray source, path2str leaves non-path objects as-is via str()
    d_str = p.to_dict(path2str=True)
    assert isinstance(d_str["dem_source"], str)


def test_parameters_to_series(flat_dem: xr.DataArray) -> None:
    p = make_params(flat_dem)
    s = p.to_series()
    assert isinstance(s, pd.Series)
    assert s["name"] == "z1"

    # index_prefix produces a 2-level MultiIndex
    s2 = p.to_series(index_prefix="zoneA")
    assert s2.index.nlevels == 2
    assert "zoneA" in s2.index.get_level_values(0)

    # a list-like prefix must have length 2
    with pytest.raises(ValueError, match=r"length 2"):
        p.to_series(index_prefix=["a", "b", "c"])


def test_parameters_bool_survives_series_round_trip(flat_dem: xr.DataArray) -> None:
    # regression: compute_bathymetry=False must not become True when the
    # parameters are serialized to strings (as happens via CSV/Excel) and
    # reloaded through from_series.
    p = make_params(flat_dem, name="z1", compute_bathymetry=False)
    s = p.to_series(include_data_array=False).astype(str)
    reloaded = TerrainCorrectionParameters.from_series(s)
    assert reloaded.compute_bathymetry is False
    assert reloaded.compute_topography is True


# ---------------------------------------------------------------------------
# TerrainCorrector
# ---------------------------------------------------------------------------


def test_corrector_init_single_and_sequence(flat_dem: xr.DataArray) -> None:
    p = make_params(flat_dem)
    tc = TerrainCorrector(p)
    assert tc.zones == ["z1"]
    assert set(tc.params) == {"z1"}

    tc2 = TerrainCorrector([make_params(flat_dem, name="a")])
    assert tc2.zones == ["a"]


def test_corrector_init_type_errors(flat_dem: xr.DataArray) -> None:
    with pytest.raises(TypeError, match=r"params arg must be"):
        TerrainCorrector(5)
    with pytest.raises(TypeError, match=r"all items must be"):
        TerrainCorrector([make_params(flat_dem), 5])


def test_corrector_add_zone(flat_dem: xr.DataArray) -> None:
    tc = TerrainCorrector(make_params(flat_dem, name="z1"))
    tc.add_zone(make_params(flat_dem, name="z2", min_dist=5.0, max_dist=30.0))
    assert set(tc.params) == {"z1", "z2"}

    with pytest.raises(TypeError, match=r"must be a TerrainCorrectionParameters"):
        tc.add_zone(5)


def test_corrector_zones_sorted_by_min_dist(flat_dem: xr.DataArray) -> None:
    far = make_params(flat_dem, name="far", min_dist=20.0, max_dist=60.0)
    near = make_params(flat_dem, name="near", min_dist=1.0, max_dist=10.0)
    tc = TerrainCorrector([far, near])
    # zones are returned ordered by increasing min_dist
    assert tc.zones == ["near", "far"]


def test_corrector_constant_coord_source(flat_dem: xr.DataArray) -> None:
    p1 = make_params(flat_dem, name="a")
    p2 = make_params(flat_dem, name="b", min_dist=5.0, max_dist=30.0)
    tc = TerrainCorrector([p1, p2])
    assert tc._has_constant_xy_coord_source() is True
    assert tc._has_constant_z_coord_source() is True

    p3 = make_params(
        flat_dem, name="c", min_dist=5.0, max_dist=30.0, site_easting_field="x2"
    )
    tc2 = TerrainCorrector([p1, p3])
    assert tc2._has_constant_xy_coord_source() is False


# ---------------------------------------------------------------------------
# TerrainCorrectionData
# ---------------------------------------------------------------------------


def test_data_init_empty() -> None:
    tcd = TerrainCorrectionData(site_id=["a", "b", "c"])
    assert tcd.data.index.name == "site_id"
    assert tcd.data.index.tolist() == ["a", "b", "c"]
    assert tcd.data.columns.tolist() == ["tcorr:total"]


def test_data_create_empty_classmethod() -> None:
    tcd = TerrainCorrectionData.create_empty(site_id=["a", "b"])
    assert isinstance(tcd, TerrainCorrectionData)
    assert tcd.data.index.tolist() == ["a", "b"]


def test_data_init_param_corr_mismatch(flat_dem: xr.DataArray) -> None:
    with pytest.raises(ValueError, match=r"must specify both or neither"):
        TerrainCorrectionData(site_id=["a"], params=make_params(flat_dem))
    with pytest.raises(ValueError, match=r"must specify both or neither"):
        TerrainCorrectionData(site_id=["a"], terrain_corrections=[1.0])


def test_data_init_empty_site_id() -> None:
    with pytest.raises(ValueError, match=r"must not be empty"):
        TerrainCorrectionData(site_id=[])


def test_data_set_corrections_accumulates_total(flat_dem: xr.DataArray) -> None:
    tcd = TerrainCorrectionData(site_id=["a", "b"])
    tcd.set_corrections(
        params=make_params(flat_dem, name="z1"),
        topography_corrections=[1.0, 2.0],
    )
    assert "tcorr:z1:topo" in tcd.data.columns
    assert tcd.data["tcorr:total"].tolist() == [1.0, 2.0]

    tcd.set_corrections(
        params=make_params(flat_dem, name="z2", min_dist=5.0, max_dist=30.0),
        topography_corrections=[0.5, 0.5],
    )
    assert "tcorr:z2:topo" in tcd.data.columns
    # total is the row-wise sum of all per-zone corrections
    assert tcd.data["tcorr:total"].tolist() == [1.5, 2.5]


def test_data_set_corrections_errors(flat_dem: xr.DataArray) -> None:
    tcd = TerrainCorrectionData(site_id=["a", "b"])
    with pytest.raises(TypeError, match=r"must be a TerrainCorrectionParameters"):
        tcd.set_corrections(params=5, topography_corrections=[1.0, 2.0])

    with pytest.raises(ValueError, match=r"at least one of"):
        tcd.set_corrections(params=make_params(flat_dem, name="z3"))


def test_data_get_corrections(flat_dem: xr.DataArray) -> None:
    tcd = TerrainCorrectionData(site_id=["a", "b"])
    tcd.set_corrections(
        params=make_params(flat_dem, name="z1"),
        topography_corrections=[1.0, 2.0],
    )

    total = tcd.get_corrections(["a"], total_only=True)
    assert total.columns.tolist() == ["tcorr:total"]
    assert total.loc["a", "tcorr:total"] == pytest.approx(1.0)

    # missing site, raise
    with pytest.raises(ValueError, match=r"no terrain correction data found"):
        tcd.get_corrections(["missing"], if_missing="raise")

    # invalid if_missing
    with pytest.raises(ValueError, match=r"invalid if_missing"):
        tcd.get_corrections(["a"], if_missing="bogus")

    # fill missing
    with pytest.warns(UserWarning):
        filled = tcd.get_corrections(
            ["a", "missing"], if_missing="fill", fill_value=-9.0
        )
    assert set(filled.index) == {"a", "missing"}
    assert filled.loc["missing", "tcorr:total"] == pytest.approx(-9.0)

    # drop missing (default) warns and drops
    with pytest.warns(UserWarning):
        dropped = tcd.get_corrections(["a", "missing"], if_missing="drop")
    assert dropped.index.tolist() == ["a"]


def test_data_repr(flat_dem: xr.DataArray) -> None:
    tcd = TerrainCorrectionData(site_id=["a", "b"])
    tcd.set_corrections(
        params=make_params(flat_dem, name="z1"),
        topography_corrections=[1.0, 2.0],
    )
    r = repr(tcd)
    assert "TerrainCorrectionData" in r
    assert "n_sites=2" in r


def test_data_csv_round_trip(flat_dem: xr.DataArray) -> None:
    tcd = TerrainCorrectionData(site_id=["a", "b"])
    tcd.set_corrections(
        params=make_params(flat_dem, name="z1"),
        topography_corrections=[1.0, 2.0],
    )

    csv = tcd.to_csv()
    assert isinstance(csv, str)
    # parameters are written as commented header lines
    assert any(line.startswith("#") for line in csv.splitlines())


def test_data_csv_round_trip_to_file(tmp_path, flat_dem: xr.DataArray) -> None:
    # regression: round-trips used to fail because (1) a DataArray dem_source
    # was serialized as a multi-line repr that corrupted the parameter header,
    # and (2) compute_bathymetry=False was read back as True.
    tcd = TerrainCorrectionData(site_id=["a", "b"])
    tcd.set_corrections(
        params=make_params(flat_dem, name="z1", compute_bathymetry=False),
        topography_corrections=[1.0, 2.0],
    )
    out = tmp_path / "tcorr.csv"
    ret = tcd.to_csv(out)
    assert ret is None
    assert out.exists()

    text = out.read_text()
    # the DataArray source is stringified, not dumped as a multi-line repr
    assert "xarray.DataArray" in text
    assert "northing  (northing)" not in text

    reloaded = TerrainCorrectionData.from_csv(out)
    assert isinstance(reloaded, TerrainCorrectionData)
    np.testing.assert_allclose(
        reloaded.data["tcorr:z1:topo"].to_numpy(),
        np.array([1.0, 2.0]),
    )
    # the bathymetry flag survived the round-trip
    reloaded_params = next(iter(reloaded.params.values()))
    assert reloaded_params.compute_bathymetry is False
