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

"""Funtions used in generating summary statistics for various GSolve classes."""

import pandas as _pd


def stdev_ugal(x: _pd.Series) -> float:
    return round(x.std() * 1000, 2)


def range_ugal(x: _pd.Series) -> float:
    return round((x.max() - x.min()) * 1000, 2)


def n(x: _pd.Series) -> int:
    return x.size


def n_inactive(x: _pd.Series) -> int:
    return x.eq(False).sum()


def n_sites(x: _pd.Series) -> int:
    return x.nunique()


def starttime_utc(x: _pd.Series) -> _pd.Timestamp:
    return x.min()


def endtime_utc(x: _pd.Series) -> _pd.Timestamp:
    return x.max()


def duration_hr(x: _pd.Series) -> float:
    return round((x.max() - x.min()).total_seconds() / 3600, 2)


def in_loops(x: _pd.Series) -> str:
    return ",".join(sorted(x.unique()))
