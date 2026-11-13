import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from imap_processing.swapi.constants import Species


@pytest.fixture
def effective_area(central_effective_area):
    return central_effective_area


def _efficiency_rows(effective_area, ion):
    efficiency = effective_area.relative_efficiency
    return efficiency.efficiency_epoch.values, efficiency.sel(ion=ion).values


def test_central_effective_area(effective_area):
    table = effective_area.lab_area
    assert table.esa_voltage.attrs["units"] == "V"
    # Claude: at the last efficiency row, so only the voltage interpolation
    # matters
    row_epochs, h_efficiency = _efficiency_rows(effective_area, "H")
    epoch = row_epochs[-1]
    esa_voltages = [
        float(table.esa_voltage[0] + table.esa_voltage[1]) / 2,
        # Placeholder: held at the end values outside the table
        float(table.esa_voltage[0]) / 2,
        float(table.esa_voltage[-1]) * 2,
    ]
    expected = (
        np.array([float(table[0] + table[1]) / 2, float(table[0]), float(table[-1])])
        * h_efficiency[-1]
    )

    assert_allclose(
        effective_area.evaluate_adjusted(esa_voltages, epoch, Species.PROTON),
        expected,
    )


def test_central_effective_area_table_is_in_energy(
    effective_area, response_table_paths
):
    table = pd.read_csv(response_table_paths["central_effective_area_path"])

    assert_allclose(
        effective_area.lab_area.esa_voltage.values,
        table["esa_voltage"].to_numpy(),
    )


@pytest.mark.parametrize(
    ("species", "ion"),
    [
        (Species.PROTON, "H"),
        (Species.ALPHA, "He"),
        (Species.HELIUM_PLUS, "He"),
    ],
)
def test_relative_efficiency_is_interpolated_in_time(effective_area, species, ion):
    row_epochs, efficiencies = _efficiency_rows(effective_area, ion)
    epochs = [
        # Placeholder: held at the first row before the table
        row_epochs[0] - 10**18,
        row_epochs[1],
        (row_epochs[1] + row_epochs[2]) // 2,
        row_epochs[2] + (row_epochs[3] - row_epochs[2]) // 4,
        # Placeholder: held at the last row after the table
        row_epochs[-1] + 10**18,
    ]
    expected_efficiencies = [
        efficiencies[0],
        efficiencies[1],
        (efficiencies[1] + efficiencies[2]) / 2,
        efficiencies[2] + (efficiencies[3] - efficiencies[2]) / 4,
        efficiencies[-1],
    ]
    area = float(effective_area.lab_area[0])
    esa_voltage = float(effective_area.lab_area.esa_voltage[0])

    assert_allclose(
        effective_area.evaluate_adjusted(esa_voltage, epochs, species),
        area * np.array(expected_efficiencies),
        rtol=1e-12,
    )
