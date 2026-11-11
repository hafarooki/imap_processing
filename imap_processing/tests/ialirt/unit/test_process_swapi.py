import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_swapi import voltages_for_table
from imap_processing.swapi.swapi_utils import read_swapi_lut_table


@pytest.fixture(scope="session")
def esa_unit_conversion_table() -> pd.DataFrame:
    esa_file_path = (
        imap_module_directory
        / "tests/swapi/lut/imap_swapi_esa-unit-conversion_20250626_v001.csv"
    )
    df = read_swapi_lut_table(esa_file_path)
    return df


@pytest.mark.parametrize(
    "table_id, expected_first, expected_last",
    [
        # Sweep 0 has two timestamps (2/11/2025 and 5/19/2025); the latest
        # one (622.52 ... 51.81) should be used, not the older (619 ... 57).
        (0, [622.52, 622.52], [56.5, 51.81]),
        (1, [2994.5, 2994.5], [56.5, 51.81]),
        # This is the one used in flight
        (2, [10240.0, 9389.94292], [56.5, 51.81]),
    ],
)
def test_voltages_for_table(
    esa_unit_conversion_table, table_id, expected_first, expected_last
):
    """Test that voltages_for_table() returns the absolute value of the 62
    coarse ESA step voltages from the most recent LUT entry for the sweep
    table."""
    voltages = voltages_for_table(esa_unit_conversion_table, table_id)

    assert voltages.shape == (62,)
    assert voltages.dtype == np.float64
    assert np.all(voltages > 0)
    np.testing.assert_allclose(voltages[:2], expected_first)
    np.testing.assert_allclose(voltages[-2:], expected_last)


def test_voltages_for_table_unknown_sweep(esa_unit_conversion_table):
    """Test that voltages_for_table() raises for a sweep not in the LUT."""
    with pytest.raises(ValueError, match="No esa unit conversion available"):
        voltages_for_table(esa_unit_conversion_table, 99)
