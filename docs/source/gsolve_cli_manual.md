# GSolve CLI manual

GSolve CLI or gsolve-cli is a comand-line interface to run relative
gravity data adjustment in a terminal.

## Installation

To install the utility you should follow the next steps in the project folder:

1. Create and activate a new virtual environment: `conda create -n gsolve-cli` and `conda activate gsolve-cli`

2. Install pip: `conda install pip`

3. Install build: `pip install --upgrade build`

4. Build the package: `python -m build`

5. Install the package: `pip install .`

6. Run `help` command to test installation: `gsolve-cli --help`

If you see this output the utility has been successfully installed in your environment:

```text
usage: gsolve-cli -is file [-ref file] [-ct file] [-or file] [-m value] [-cf value] [-pc value] [--use-loops] [--calculate-calibration-factor] [-h]

gsolve-cli - CLI utility to process relative gravimetric measurements

options:
  -h, --help            Show help message and exit

Input:
  -is file, --input-survey file
                        [Required] Input survey .xlsx-file name with gravimeter readings and site information.
  -ref file, --reference file
                        [Optional] Input file with reference gravity. Default: None
  -ct file, --conversion-table-file file
                        [Optional] Input file with conversion table. Default: None

Output:
  -or file, --output-report file
                        [Optional] Output report .xlsx-file name with adjusted gravity and observations values. Default: None

Processing options:
  -m value, --method value
                        [Optional] Absolute gravity constrain method: {1, 2, 3}. Default: 2
  -cf value, --calibration-factor value
                        [Optional] Define calibration factor. Default: 1.0
  -pc value, --percentile-clipping value
                        [Optional] Clip data by percentile value. Default: 100
  --use-loops           [Optional] Use individual loops in the adjustment process. Default: False
  --calculate-calibration-factor
                        [Optional] Claculate calibration factor. Default: False
```

## Configuration basics

The `gsolve-cli` utility has three groups of parameters:

1. `Input` - defines filenames of mandatory and optional input files;
2. `Output` - defines filename of an output .xlsx-report;
3. `Processing options` - defines adjustment settings.

### Input

- `--input-survey file` or `-is file` [Mandatory] - defines a name of a `.xlsx`-file with survey information: gravimeter readings and site information.
- `--reference file` or `-ref file` [Optional] - defines a name of a `.csv`-file with reference gravity values. Default: `None`
- `--conversion-table-file` or `-ct file` [Optional] - defines a name of a `.csv`-file with readings conversion values. Default: `None`

### Output

- `--output-report` or `-or` [Mandatory] - defines name of an output `.xlsx`-report with adjusted gravity and observation values. Default: `None`

### Processing options

- `--method` or `-m` [Optional] - defines a reference gravity constraint approach used in the adjustment process. There are three options: `1` - "Unconstrained least squares", `2` - "Partially constrained least squares", `3` - "Constrained least squares". Default: `2`
- `--calibration-factor` or `-cf` [Optional] - defines a calibration factor applied to observations. Default: `1.0`
- `-percentile-clipping` or `-pc` [Optional] - defines percentile values used for data clipping. Default: `100`.
- `--use-loops` [Optional] - defines if individual loops have to be used in the adjustment process. Default: `False`.
- `--calculate-calibration-factor` [Optional] - defines if the calibration factor calculated during the adjustment. Default: `False`.

## Basic usage

1. To run default processing you just need to define names of input and output files:
`gsolve-cli -is examples/surveys/calibration/G106-Data23Nov2018.xlsx  -or report.xlsx`

2. To run processing with particular reference gravity and conversion data you can run:
`gsolve-cli -is examples/surveys/calibration/G106-Data23Nov2018.xlsx -ref examples/absolute_gravity/base_stations_calibration.csv -ct examples/correction_tables/G106.csv -or report.xlsx`

3. To run processing with particular percentile clipping value you can run:
`gsolve-cli -is examples/surveys/calibration/G106-Data23Nov2018.xlsx -ref examples/absolute_gravity/base_stations_calibration.csv -ct examples/correction_tables/G106.csv -or report.xlsx -pc 95`
