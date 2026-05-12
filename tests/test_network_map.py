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

import pandas as pd
import pytest

from gsolve import GravityObservations
from gsolve.sites import GravitySites


@pytest.fixture
def sample_data() -> dict:
    return {
        "site_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 1, 1],
        "loop": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C", "A", "A"],
        "meter_reading_mgal": [10, 11, 11, 20, 21, 20, 70, 71, 70, 71, 11, 10],
        "meter_id": [
            "dummy",
            "dummy",
            "dummy",
            "dummy",
            "dummy",
            "dummy",
            "dummy",
            "dummy",
            "dummy",
            "dummy",
            "dummy",
            "dummy",
        ],
        "datetime": [
            "2021-01-01T12:00:00",
            "2021-01-01T12:10:00",
            "2021-01-01T12:20:00",
            "2021-01-01T12:30:00",
            "2021-01-01T12:40:00",
            "2021-01-01T12:50:00",
            "2021-01-01T12:51:00",
            "2021-01-01T12:52:00",
            "2021-01-01T12:53:00",
            "2021-01-01T12:54:00",
            "2021-01-01T12:55:00",
            "2021-01-01T12:56:00",
        ],
    }


# @pytest.fixture
# def sample_sites():
#     return GravitySites.from_dataframe(
#         pd.DataFrame(
#             {
#                 "site_id": [1, 2, 3],
#                 "latitude": [10.1, 20.2, 30.3],
#                 "longitude": [100.1, 200.2, 300.3],
#                 "height_ellipsoidal": [25, 35, 45],
#             }
#         )
#     )


# @pytest.fixture()
# def dummy_observations(sample_data: dict) -> GravityObservations:
#     return GravityObservations(**sample_data)


# def test_make_network(dummy_observations, sample_sites):
#     result = dummy_observations._make_network(sample_sites)

#     expected_output = pd.DataFrame(
#         {
#             "site_id": [1, 2, 3, 1],
#             "loop": ["A", "B", "C", "A"],
#             "latitude": [10.1, 20.2, 30.3, 10.1],
#             "longitude": [100.1, 200.2, 300.3, 100.1],
#         }
#     )
#     result = result.astype(
#         dtype={
#             "site_id": "int64",
#             "loop": "str",
#             "latitude": "float64",
#             "longitude": "float64",
#         }
#     )

#     pd.testing.assert_frame_equal(
#         result,
#         expected_output,
#     )

#     assert sample_sites.data["station_occupations"].to_dict() == {"1": 2, "2": 1, "3": 1}

#     assert "group" not in dummy_observations.data.columns
