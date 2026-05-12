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

# test GravitySites class.

from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from gsolve.sites import ReferenceGravity

# @pytest.fixture
# def legacy_gsolve_sites_file(shared_datadir: Path) -> Path:
#     return shared_datadir / "legacy_format.xlsx"


# @pytest.fixture
# def current_gsolve_sites_file(shared_datadir: Path) -> Path:
#     return shared_datadir / "current_format.xlsx"


def test_init_ok() -> None:
    site_ids = [1, 2, 3]
    gravity = [100, 200, 300]
    ReferenceGravity(site_ids, gravity)


def test_init_duplicate_site_id() -> None:
    site_ids = [1, 2, 2]
    gravity = [100, 200, 300]
    with pytest.raises(ValueError, match=r"site_id field contains duplicated values"):
        ReferenceGravity(site_ids, gravity)


def test_init_empty_site_id() -> None:
    site_ids = [1, 2, ""]
    gravity = [100, 200, 300]
    with pytest.raises(
        ValueError, match=r"site_id field contains empty values at rows"
    ):
        ReferenceGravity(site_ids, gravity)


def test_init_empty_gravity_data() -> None:
    site_id = [1, 2, 3]
    gravity = [100, 200, None]
    with pytest.raises(
        ValueError, match=r"gravity field contains null values for sites"
    ):
        ReferenceGravity(site_id=site_id, gravity=gravity)  # ty:ignore[invalid-argument-type]
