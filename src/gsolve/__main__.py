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
import argparse

from gsolve import GravitySurvey, LaCosteRombergDialConverter, ReferenceGravity
from gsolve.reports import GSolveReport
from gsolve.tide.earth_tide import LongmanTidalCorrection


def parse_args(*args: str) -> argparse.Namespace:
    __app_name__ = "gsolve-cli"
    args_parser = argparse.ArgumentParser(
        prog=__app_name__,
        description=(
            f"{__app_name__} - CLI utility to process relative gravimetric measurements"
        ),
        epilog=None,
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Input parameters
    input = args_parser.add_argument_group("Input")

    input.add_argument(
        "-is",
        "--input-survey",
        action="store",
        required=True,
        metavar="file",
        help=(
            "[Required] Input survey .xlsx-file name with gravimeter readings "
            "and site information."
        ),
    )

    # Optional parameters
    input.add_argument(
        "-ref",
        "--reference",
        action="store",
        metavar="file",
        default=None,
        help="[Optional] Input file with reference gravity. Default: None",
    )
    input.add_argument(
        "-ct",
        "--conversion-table-file",
        action="store",
        metavar="file",
        default=None,
        help="[Optional] Input file with conversion table. Default: None",
    )

    # Output parameter
    output = args_parser.add_argument_group("Output")

    output.add_argument(
        "-or",
        "--output-report",
        action="store",
        required=True,
        metavar="file",
        help=(
            "[Required] Output report .xlsx-file name with adjusted gravity "
            "and observations values."
        ),
    )

    # Processing options
    proc = args_parser.add_argument_group("Processing options")

    proc.add_argument(
        "-m",
        "--method",
        type=int,
        action="store",
        default=2,
        choices=range(1, 4),
        metavar="value",
        help="[Optional] Absolute gravity constrain method: {1, 2, 3}. Default: 2",
    )
    proc.add_argument(
        "-cf",
        "--calibration-factor",
        action="store",
        metavar="value",
        type=float,
        default=1,
        help="[Optional] Define calibration factor. Default: 1.0",
    )
    proc.add_argument(
        "-pc",
        "--percentile-clipping",
        type=float,
        default=100,
        metavar="value",
        help="[Optional] Clip data by percentile value. Default: 100",
    )
    proc.add_argument(
        "--use-loops",
        action="store_true",
        default=False,
        help="[Optional] Use individual loops in the adjustment process. Default: False",
    )
    proc.add_argument(
        "--calculate-calibration-factor",
        action="store_true",
        default=False,
        help="[Optional] Claculate calibration factor. Default: False",
    )

    # Help function
    args_parser.add_argument(
        "-h", "--help", action="help", help="Show help message and exit"
    )

    if args:
        return args_parser.parse_args(args=args)
    return args_parser.parse_args()


def processing(args: argparse.Namespace) -> None:
    # Read survey information
    survey = GravitySurvey.from_excel(fname=args.input_survey)

    # Read in list reference (i.e. absolute) stations
    if args.reference is not None:
        ref_sites = ReferenceGravity.from_csv(csv_file=args.reference)
        # set stations with known reference gravity values
        survey.set_reference_gravity(ref_grav=ref_sites)

    # Convert gravimeter readings
    if args.conversion_table_file is not None:
        g106converter = LaCosteRombergDialConverter.from_csv(
            fname=args.conversion_table_file
        )
        # apply dial conversion to convert values to mGal.
        survey.apply_dial_to_mgal(converter=g106converter)

    # Apply calibration factor
    survey.set_calibration_factor(calibration_factor=args.calibration_factor)

    # Calculate the earth tide correction
    survey.apply_earth_tide_correction(tide_corrector=LongmanTidalCorrection())

    # Calculate corrected gravity
    survey.calculate_tide_corrected_gravity()

    # Perform adjustment
    results = survey.solve_lstsq(
        method=args.method,
        use_loops=args.use_loops,
        calculate_calibration_factor=args.calculate_calibration_factor,
        percentile_clipping=args.percentile_clipping,
    )

    # Save output files
    output_report = args.output_report
    report = GSolveReport(
        observations=survey.observations, sites=survey.sites, results=results
    )
    report.to_excel(filename=output_report)


def main() -> None:
    try:
        args = parse_args()
        processing(args)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
