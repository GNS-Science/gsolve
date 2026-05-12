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

"""Classes and functions to facilitate reading and writing excel files."""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args

import pandas as _pd

from gsolve.core._typing import FilePath, IfSheetExists, IfWorkbookExists
from gsolve.core.utils import is_list_like

__all__ = [
    "get_excel_worksheets",
    "get_true_sheet_name",
    "read_excel_worksheet",
    "write_excel_worksheet",
]


def get_excel_worksheets(excel_file: FilePath, **kwargs) -> list[str | int]:
    """Return list of all worksheets in ``excel_file``

    Parameters
    ----------
    excel_file :
        The excel workbook to read.

    Returns
    -------
    list
        List of worksheet names in the excel file.
    """
    with _pd.ExcelFile(excel_file, **kwargs) as xls:
        return xls.sheet_names


def get_true_sheet_name(
    excel_file: FilePath, sheet_name: str | int, raise_error: bool = False
) -> str | int | None:
    """
    For given `sheet_name` find the corresponding worksheet in `excel_file` ignoring
    capitalisation etc.

    Parameters
    ----------
    excel_file : str or PathLike
        The path to the excel file.
    sheet_name : str or int
        The name or index of the sheet to read.
    raise_error : bool, default is False
        Raise an error if the sheet is not found, otherwise return None.

    Returns
    -------
    str or None
        The true sheet name if found, otherwise None.
    """
    true_sheet_name = None
    sheet_names = get_excel_worksheets(excel_file)
    if isinstance(sheet_name, int):
        try:
            true_sheet_name = sheet_names[sheet_name]
        except IndexError:
            if raise_error:
                raise ValueError(
                    f"excel file {excel_file} has no sheet at index: {sheet_name}"
                )
    else:
        sheet_names_lc = [
            s.lower() for s in get_excel_worksheets(excel_file) if isinstance(s, str)
        ]
        if sheet_name.lower() in sheet_names_lc:
            true_sheet_name = sheet_names[sheet_names_lc.index(sheet_name.lower())]
        elif raise_error:
            raise ValueError(
                f"excel file {excel_file} has no sheet named '{sheet_name}'"
            )

    return true_sheet_name


def parse_sheet_name_arg(
    sheet_name: str | int | Sequence[str | int],
) -> list[str | int]:
    """Parse and validate sheet_name argument."""
    sheet_name_list: list[str | int]
    if is_list_like(sheet_name):
        sheet_name_list = [s for s in sheet_name]  # pyrefly:ignore[not-iterable]
    else:
        sheet_name_list = [sheet_name]  # pyrefly:ignore[bad-assignment]

    if not all(isinstance(s, (str, int)) for s in sheet_name_list):
        raise TypeError(
            "sheet_name args must be either a str (sheet name) or an int (sheet index)"
        )

    for s in sheet_name_list:
        if isinstance(s, str) and not s.strip():
            raise ValueError("sheet_name arg cannot contain empty strings")
        if isinstance(s, int) and s < 0:
            raise ValueError(f"invalid sheet index: {s}")

    return sheet_name_list


def read_excel_worksheet(
    excel_file: FilePath,
    sheet_name: str | int | Sequence[str | int],
    **kwargs,
) -> _pd.DataFrame:
    """
    Read the excel worksheet `sheet_name` from `excel_file` into a DataFrame.

    Parameters
    ----------
    excel_file : FilePath
        The path to the excel file.
    sheet_name : str or int or list-like
        The name (str) or index (int) of the worksheet to read from ``excel_file``.
        If multiple names/indices are provided, then read the first one found in
        ``excel_file``.
    **kwargs
        Additional arguments passed to ``pandas.read_excel`` method.

    Returns
    -------
    DataFrame
        The DataFrame read from the excel sheet.

    See Also
    --------
    pandas.read_excel
        The underlying function used to read the excel file.

    """

    sheet_name_list = parse_sheet_name_arg(sheet_name)

    true_sheet_name = None
    for sname in sheet_name_list:
        true_sheet_name = get_true_sheet_name(excel_file, sname, raise_error=False)
        if true_sheet_name is not None:
            break

    if true_sheet_name is None:
        raise ValueError(f"worksheet(s) {sheet_name} not found in {excel_file}")

    # ignore warnings from openpyxl about data validation
    # they are typically non-fatal and do not affect reading the data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _pd.read_excel(excel_file, sheet_name=true_sheet_name, **kwargs)


def write_excel_worksheet(
    df: _pd.DataFrame,
    excel_file: FilePath,
    sheet_name: str,
    if_workbook_exists: IfWorkbookExists = "error",
    if_sheet_exists: IfSheetExists = "error",
    **kwargs,
) -> None:
    """
    Write a DataFrame to excel workbook `excel_file` in the worksheet `sheet_name`.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to write to the excel file.
    excel_file : str or PathLike
        The path to the excel file.
    sheet_name : str
        The name of the worksheet to write to.
    if_workbook_exists: {"error", "replace", "append"}, default "error"
        Behaviour if the excel file already exists.
        - "error": raise a ValueError.
        - "replace": replace the existing file.
        - "append": append to the existing file.
    if_sheet_exists: {"error", "replace", "new"}, default "error"
        Behaviour if the worksheet already exists. Only applicable when
        `if_workbook_exists='append'`.
        - "error": raise a ValueError.
        - "replace": replace the existing worksheet.
        - "new": create a new worksheet
    **kwargs
        Additional arguments passed to ``pandas.DataFrame.to_excel`` method.

    See Also
    --------
    pandas.DataFrame.to_excel
        The underlying function used to write the DataFrame to the excel file.

    """

    if if_workbook_exists not in get_args(IfWorkbookExists):
        raise ValueError(
            f"invalid value for {if_workbook_exists=}, must be one of "
            f"{get_args(IfWorkbookExists)}"
        )

    if if_sheet_exists not in get_args(IfSheetExists):
        raise ValueError(
            f"invalid value for {if_sheet_exists=}, must be one of "
            f"{get_args(IfSheetExists)}"
        )

    writer_kwargs: dict[str, Any] = {
        "engine": "openpyxl",  # "xlsxwriter", "openpyxl", "xlwt"
        "if_sheet_exists": if_sheet_exists,
        "mode": "w",
    }

    excel_file = Path(excel_file)
    if excel_file.exists():
        if if_workbook_exists == "error":
            raise ValueError(
                f"file {excel_file} already exists, and arg {if_workbook_exists=}"
            )
        elif if_workbook_exists == "append":
            writer_kwargs["mode"] = "a"

    if writer_kwargs["mode"] == "w":
        writer_kwargs["if_sheet_exists"] = None

    try:
        with _pd.ExcelWriter(excel_file, **writer_kwargs) as writer:
            df.to_excel(writer, sheet_name=sheet_name, **kwargs)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to {excel_file}, it is probably open in another application"
        )
