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

from typing import TextIO

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pytest

from gsolve.meter_conversion import LaCosteRombergDialConverter


@pytest.fixture
def g936_table() -> list:
    # meter_id, G936
    table = """
    0,000.00,1.01798
    100,101.80,1.01798
    200,203.60,1.01798
    300,305.39,1.01799
    400,407.19,1.01802
    500,508.99,1.01804
    600,610.80,1.01808
    700,712.60,1.01813
    800,814.42,1.01819
    900,916.24,1.01827
    1000,1018.06,1.01834
    """
    return [l.strip().split(",") for l in table.split("\n") if l.strip()]


@pytest.fixture
def g936_df(g936_table: list) -> pd.DataFrame:
    return pd.DataFrame(
        g936_table,
        columns=["counter_reading", "value_mgal", "interval_factor"],
        dtype=float,
    )


def test_lacoste_romberg_dial_converter_init(g936_df: pd.DataFrame) -> None:
    df = g936_df
    kwargs = {
        "counter_reading": df["counter_reading"],
        "value_mgal": df["value_mgal"],
        "interval_factor": df["interval_factor"],
    }

    # Case: all inputs equal length
    kwargs["interval_factor"] = df["interval_factor"].iloc[:-1]

    converter = LaCosteRombergDialConverter("G936", **kwargs)
    assert pd.isna(converter.table.iloc[-1]["interval_factor"])

    # Case: ifactor len is 1 less than counter_reading
    kwargs["interval_factor"] = df["interval_factor"].iloc[:-1]
    converter = LaCosteRombergDialConverter("G936", **kwargs)
    assert pd.isna(converter.table.iloc[-1]["interval_factor"])
    _ = kwargs.pop("interval_factor")

    # Case: no ifactor, so 'value_mgal_from_ifactor' should be all NaN
    converter = LaCosteRombergDialConverter("G936", **kwargs)
    assert converter.table["value_mgal_from_ifactor"].isna().all()


def test_lacoste_romberg_dial_converter_init_bad(g936_df: pd.DataFrame) -> None:
    df = g936_df
    kwargs = {
        "counter_reading": df["counter_reading"],
        "value_mgal": df["value_mgal"],
        "interval_factor": df["interval_factor"],
    }
    # Case: bad dates
    with pytest.raises(ValueError, match=r"invalid time combination"):
        _ = LaCosteRombergDialConverter(
            "G936", **kwargs, starttime="2021-01-01", endtime="2020-01-01"
        )
    with pytest.raises(Exception):
        _ = LaCosteRombergDialConverter("G936", **kwargs, starttime="xxxxx")

    # Case: inconsistent data lengths
    with pytest.raises(ValueError, match="arrays must be the same shape"):
        _ = LaCosteRombergDialConverter(
            "G936",
            counter_reading=df["counter_reading"],
            value_mgal=df["value_mgal"].iloc[-1],
            interval_factor=df["interval_factor"],
        )

    # bad interval factor
    with pytest.raises(ValueError, match="invalid interval_factor: array size"):
        _ = LaCosteRombergDialConverter(
            "G936",
            counter_reading=df["counter_reading"],
            value_mgal=df["value_mgal"],
            interval_factor=df["interval_factor"].iloc[:-2],
        )
    # nan in interval factor
    with pytest.raises(ValueError, match="contains NaN values."):
        ifactor = df["interval_factor"].copy()
        ifactor.iloc[2] = np.nan
        _ = LaCosteRombergDialConverter(
            "G936",
            counter_reading=df["counter_reading"],
            value_mgal=df["value_mgal"],
            interval_factor=ifactor,
        )
    # ifactor empty
    for ifac in [[], np.ones([11, 2])]:
        with pytest.raises(
            ValueError, match="interval_factor must be a non-empty 1-dimensional array."
        ):
            _ = LaCosteRombergDialConverter(
                "G936",
                counter_reading=df["counter_reading"],
                value_mgal=df["value_mgal"],
                interval_factor=ifac,
            )

    # Case: counter_reading not in ascending order
    with pytest.raises(
        ValueError,
        match="counter_reading values must be unique and in ascending order.",
    ):
        _ = LaCosteRombergDialConverter(
            "G936",
            counter_reading=df["counter_reading"].iloc[::-1],
            value_mgal=df["value_mgal"],
        )
        _ = LaCosteRombergDialConverter(
            "G936", counter_reading=[1, 2, 2], value_mgal=[0, 100, 200]
        )

    # Case: nan values in counter readings
    with pytest.raises(ValueError, match="counter_reading contains NaN."):
        _ = LaCosteRombergDialConverter(
            "G936", counter_reading=[0, 100, np.nan], value_mgal=[1, 2, 3]
        )
    # Case: nan values in value_mgal
    with pytest.raises(ValueError, match="value_mgal contains NaN."):
        _ = LaCosteRombergDialConverter(
            "G936", counter_reading=[0, 100, 200], value_mgal=[1, 2, np.nan]
        )


def test_lacoste_romberg_dial_converter_from_csv(tmp_path, g936_table: list) -> None:  # noqa: ANN001
    csv_file = tmp_path / "G936.csv"

    def _write_csv_body(
        fh: TextIO, table: list | None = None, column_labels: bool = True
    ) -> None:
        table = table if table is not None else g936_table
        if column_labels:
            fh.write("counter_reading,value_mgal,interval_factor\n")
        for line in g936_table:
            fh.write(",".join(line) + "\n")

    # Case: std read _csv
    with open(csv_file, "w") as f:
        f.write("# meter_id, G936\n")
        _write_csv_body(f)

    converter = LaCosteRombergDialConverter.from_csv(csv_file)
    assert pd.isna(converter.table.iloc[-1]["interval_factor"])

    # Case: bad header
    with open(csv_file, "w") as f:
        f.write("# bad_header, G936\n")
        _write_csv_body(f)
    with pytest.raises(ValueError, match=r"invalid header key name"):
        _ = LaCosteRombergDialConverter.from_csv(csv_file)

    # Case: extra header values
    with open(csv_file, "w") as f:
        f.write("# meter_id, G936, extra\n")
        _write_csv_body(f)
    with pytest.raises(ValueError, match=r"has multiple corresponding values"):
        _ = LaCosteRombergDialConverter.from_csv(csv_file)


def test_lacoste_romberg_dial_converter_from_dataframe(g936_df: pd.DataFrame) -> None:
    _ = LaCosteRombergDialConverter.from_dataframe(meter_id="G936", table=g936_df)
    with pytest.raises(KeyError, match="counter_reading"):
        _ = LaCosteRombergDialConverter.from_dataframe(
            meter_id="G936",
            table=g936_df.rename(columns={"counter_reading": "counter_readings"}),
        )


def test_lacoste_romberg_dial_converter_correct_readings(g936_df: pd.DataFrame) -> None:
    converter = LaCosteRombergDialConverter.from_dataframe(
        meter_id="G936", table=g936_df
    )
    # Case: single value
    nptest.assert_almost_equal(converter.convert_readings([0.0]), [0], 6)
    nptest.assert_almost_equal(converter.convert_readings([100.0]), [101.798], 6)

    # Case: arrays
    array_vals = [0.0, 100.0]
    array_result = [0, 101.798]
    nptest.assert_array_almost_equal(
        converter.convert_readings(array_vals), array_result, 6
    )
    nptest.assert_array_almost_equal(
        converter.convert_readings(pd.Series(array_vals)), array_result, 6
    )
    nptest.assert_array_almost_equal(
        converter.convert_readings(np.asarray(array_vals)), array_result, 6
    )


def test_lacoste_romberg_dial_converter_correct_readings_exceeds_range(
    g936_df: pd.DataFrame,
) -> None:
    converter = LaCosteRombergDialConverter.from_dataframe(
        meter_id="G936", table=g936_df
    )
    # Case: lower, higher, both
    vals = [converter.table.index.min() - 1, converter.table.index.max() + 1]
    with pytest.raises(ValueError, match="outside range of convertible values"):
        _ = converter.convert_readings(vals[0])
    with pytest.raises(ValueError, match="outside range of convertible values"):
        _ = converter.convert_readings(vals[1])
    with pytest.raises(ValueError, match="outside range of convertible values"):
        _ = converter.convert_readings(vals)


def test_lacoste_romberg_dial_converter_correct_datetimes(
    g936_df: pd.DataFrame,
) -> None:
    converter = LaCosteRombergDialConverter.from_dataframe(
        meter_id="G936", table=g936_df, starttime="2021-01-01", endtime="2022-01-01"
    )
    array_vals = [0.0, 100.0]
    array_results = [0, 101.798]

    # case: normal dates
    dates = ["2021-01-01", "2021-01-02"]
    nptest.assert_array_almost_equal(
        converter.convert_readings(array_vals, date_time=dates), array_results, 6
    )

    # case: date exceeds endtime
    dates = ["2021-01-01", "2024-01-02"]
    v = converter.convert_readings(array_vals, date_time=dates)
    assert pd.isna(v[1])
    nptest.assert_array_almost_equal(v[0], array_results[0], 6)

    # case: date precedes startime
    v = converter.convert_readings([100], date_time="2019-01-01")
    assert pd.isna(converter.convert_readings([100], date_time="2019-01-01"))


def test_lacoste_romberg_dial_converter_correct_meter_id(
    g936_df: pd.DataFrame,
) -> None:
    converter = LaCosteRombergDialConverter.from_dataframe(
        meter_id="G936", table=g936_df, starttime="2021-01-01", endtime="2022-01-01"
    )
    array_vals = [0.0, 100.0]
    array_results = [0, 101.798]
    meter_ids = ["G936", "G936"]

    # case: normal meter_id
    nptest.assert_array_almost_equal(
        converter.convert_readings(array_vals, meter_id=meter_ids), array_results, 6
    )

    # case: meter_id not in table
    meter_ids = ["G936", "G937"]
    v = converter.convert_readings(array_vals, meter_id=meter_ids)
    assert v[0] == array_results[0] and pd.isna(v[1])

    # case: meter_id, single value
    nptest.assert_almost_equal(
        converter.convert_readings([array_vals[1]], meter_id="G936"), array_results[1]
    )
    assert pd.isna(converter.convert_readings([array_vals[1]], meter_id="xxxx"))
    dates = ["2021-01-01", "2024-01-02"]


def test_lacoste_romberg_dial_converter_correct_meter_id_date(
    g936_df: pd.DataFrame,
) -> None:
    array_vals = [0.0, 100.0]
    array_results = [0, 101.798]
    meter_ids = ["G936", "G936"]
    dates = ["2021-01-01", "2021-01-02"]

    converter = LaCosteRombergDialConverter.from_dataframe(
        meter_id="G936", table=g936_df, starttime="2021-01-01", endtime="2022-01-01"
    )
    nptest.assert_array_almost_equal(
        converter.convert_readings(array_vals, meter_id=meter_ids, date_time=dates),
        array_results,
        6,
    )
    dates = ["2021-01-01", "2024-01-02"]
    v = converter.convert_readings(array_vals, meter_id=meter_ids, date_time=dates)
    assert pd.isna(v[1])
    meter_ids = ["bad", "G936"]
    v = converter.convert_readings(array_vals, meter_id=meter_ids, date_time=dates)
    assert pd.isna(v).all()


# def test_dummy_meter_converter_init() -> None:
#     meter_id = "G936"
#     converter = DummyMeterConverter(meter_id)
#     assert converter.meter_id == meter_id
#     assert converter.starttime == pd.Timestamp.min
#     assert converter.endtime == pd.Timestamp.max
#     assert converter.converter_id().startswith(f"Dummy_{meter_id}")


# def test_dummy_meter_converter_correct_readings() -> None:
#     meter_id = "G936"
#     converter = DummyMeterConverter(
#         meter_id, starttime="2021-01-01", endtime="2022-01-01"
#     )
#     assert converter.convert_readings(0.0) == 0.0
#     nptest.assert_array_equal(converter.convert_readings([0.0, 100.0]), [0.0, 100.0])

#     v = converter.convert_readings(readings=[100.0, 200.0], meter_id=[meter_id, "junk"])
#     assert v[0] == 100.0 and pd.isna(v[1])

#     v = converter.convert_readings(
#         readings=[100.0, 200.0], date_time=["2021-01-01", "2024-01-02"]
#     )
#     assert v[0] == 100.0 and pd.isna(v[1])
#     v = converter.convert_readings(
#         readings=[100.0, 200.0], date_time=["2020-01-01", "2024-01-02"]
#     )
#     assert pd.isna(v).all()
