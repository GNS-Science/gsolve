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

# Description: Test cases for the core.excel_io module.

from pathlib import Path

import pandas.testing
import pytest

from gsolve.core.excel_io import (
    get_excel_worksheets,
    get_true_sheet_name,
    read_excel_worksheet,
)


def test_get_true_sheet_name(shared_datadir: Path) -> None:
    # Test case 1: Get the true sheet name from an excel file using the sheet
    # index
    f1 = shared_datadir / "legacy_format.xlsx"

    assert get_true_sheet_name(f1, 0) == "Survey Data"
    assert get_true_sheet_name(f1, "survey data") == "Survey Data"
    assert get_true_sheet_name(f1, "xxxxx") is None
    assert get_true_sheet_name(f1, 100) is None

    # test bad sheet name name or index is caught
    with pytest.raises(ValueError, match=r"has no sheet at index"):
        get_true_sheet_name(f1, 100, raise_error=True)
    with pytest.raises(ValueError, match=r"has no sheet named"):
        get_true_sheet_name(f1, "xxx", raise_error=True)


def test_read_excel_worksheet(shared_datadir: Path) -> None:
    f1 = shared_datadir / "legacy_format.xlsx"
    df1 = read_excel_worksheet(f1, sheet_name="survey data")
    assert df1.shape == (220, 10)
    df2 = read_excel_worksheet(f1, sheet_name=["xxxx", "survey data"])
    pandas.testing.assert_frame_equal(df1, df2)

    with pytest.raises(ValueError):
        read_excel_worksheet(f1, sheet_name="xxxx")


def test_get_excel_worksheets(shared_datadir: Path) -> None:
    f1 = shared_datadir / "legacy_format.xlsx"
    assert get_excel_worksheets(f1) == ["Survey Data", "Locations", "Tie Data"]
