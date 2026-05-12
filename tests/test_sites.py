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

from gsolve.sites import GravitySites


@pytest.fixture
def legacy_gsolve_sites_file(shared_datadir: Path) -> Path:
    return shared_datadir / "legacy_format.xlsx"


@pytest.fixture
def current_gsolve_sites_file(shared_datadir: Path) -> Path:
    return shared_datadir / "current_format.xlsx"


def test_read_excel(
    legacy_gsolve_sites_file: Path, current_gsolve_sites_file: Path
) -> None:
    gs_old = GravitySites.from_excel(legacy_gsolve_sites_file)
    assert len(gs_old) == 159

    gs_new = GravitySites.from_excel(current_gsolve_sites_file)
    assert len(gs_new) == 159

    assert_frame_equal(gs_old.data.iloc[:, :3], gs_new.data.iloc[:, :3])
