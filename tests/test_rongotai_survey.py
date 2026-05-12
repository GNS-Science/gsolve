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

# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:41:37 2024

@author: alisonk

end-to-end test on the Rongotai Isthmus gravity data.


"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from gsolve import (
    GravityObservations,
    GravitySites,
    GravitySurvey,
    LaCosteRombergDialConverter,
    ReferenceGravity,
)


# shared_datadir = pathlib.Path(r'C:\git\gsolve\tests\data')
def test_rongotai_network_adjustment(shared_datadir: pathlib.Path) -> None:
    # set paths etc
    # inputs
    data_path = shared_datadir
    cal_data_path = pathlib.Path.cwd() / "examples" / "correction_tables"
    # cal_data_path = pathlib.Path(r'C:\git\gsolve\examples\correction_tables')
    obs_file = data_path / "RIG_G106_20210702_4_gsolve3v2.xlsx"
    ref_site_file = data_path / "Absolute_rongotai_v2.csv"
    corr_table_file = cal_data_path / "G106.csv"

    for method in [1, 2, 3]:
        for percentile_clipping in [95, 100]:
            # reference site solution file
            suffix = "_ci%1i" % percentile_clipping
            site_solution_file = data_path / (
                "RIG_G106_site_solution_method%1i%s.csv" % (method, suffix)
            )
            # site_solution_file = pathlib.Path(r'C:\tmp\gsolve_files') / 'elev_test' /(
            #     "RIG_G106_site_solution_method%1i%s_new.csv" % (method, suffix)
            # )

            # Read in observations & set reference gravity
            obs = GravityObservations.from_excel(
                obs_file, parse_split_datetime=True, sheet_name="Survey Data"
            )
            sites = GravitySites.from_excel(obs_file, sheet_name="Locations")
            ref_sites = ReferenceGravity.from_csv(ref_site_file)
            _ = sites.set_reference_gravity(ref_sites)

            # Process the observed data
            g106converter = LaCosteRombergDialConverter.from_csv(corr_table_file)
            obs.apply_dial_to_mgal(g106converter)
            obs.set_calibration_factor(1.0019)
            obs.apply_earth_tide_correction(sites)
            obs.calculate_tide_corrected_gravity()

            # create a survey object using observations and site objects
            surv = GravitySurvey(obs, sites)

            # solve
            results = surv.solve_lstsq(
                method=method, use_loops=True, percentile_clipping=percentile_clipping
            )

            # load reference files
            reference_site_solution = pd.read_csv(site_solution_file)

            for param in ["absolute_gravity", "variance", "stdev", "stderr"]:
                assert np.all(
                    np.abs(
                        reference_site_solution[param].to_numpy()
                        - results.site_solution[param].to_numpy()
                    )
                    < 1e-4
                )
