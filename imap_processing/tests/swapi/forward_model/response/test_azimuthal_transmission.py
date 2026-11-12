"""Placeholder: Ported from L3a's
``tests/swapi/response/test_azimuthal_transmission.py``.

The checks of the plateau invariants are in ``test_response_tables.py``.
"""

import numpy as np
import pandas as pd
import pytest

from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalTransmission,
    interpolate_azimuthal_transmission,
)
from imap_processing.swapi.forward_model.response.response_tables import (
    SwapiResponse,
)

SPACING_DEG = 0.1


@pytest.fixture
def transmission(swapi_response):
    return swapi_response.azimuthal_transmission


def test_table_layout(transmission):
    # Placeholder: the interpolation tests assume 0.1 deg spacing from 0 to 180 deg
    assert transmission.spacing == pytest.approx(SPACING_DEG)
    assert len(transmission.values) == 1801
    # Placeholder: the sunglasses attenuate by ~1/1000 at azimuth 0
    assert transmission.values[0] == pytest.approx(1e-3)


@pytest.mark.parametrize("index", [0, 10, 199, 299, 500, 1000, 1800])
def test_at_grid_point_returns_table_value(transmission, index):
    # Placeholder: indices span the SG plateau, the SG/OA shoulder, the OA plateau, and
    # the zero-padded tail
    assert interpolate_azimuthal_transmission(
        transmission, index * SPACING_DEG
    ) == pytest.approx(transmission.values[index])


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.7, 0.9])
def test_linear_interpolation_between_grid_points(transmission, fraction):
    # Placeholder: on the SG -> OA shoulder (~25 deg), so the endpoints differ
    index = 250
    azimuth = (index + fraction) * SPACING_DEG
    expected = (1 - fraction) * transmission.values[index] + fraction * (
        transmission.values[index + 1]
    )
    assert interpolate_azimuthal_transmission(transmission, azimuth) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("azimuth", [0.37, 12.5, 29.95, 88.123])
def test_symmetric_about_zero(transmission, azimuth):
    assert interpolate_azimuthal_transmission(transmission, azimuth) == pytest.approx(
        interpolate_azimuthal_transmission(transmission, -azimuth)
    )


@pytest.mark.parametrize(
    "azimuth",
    [
        # Placeholder: exactly on the last index
        1800 * SPACING_DEG,
        # Placeholder: just past it
        1800 * SPACING_DEG + 0.07,
        # Placeholder: near the upper edge of the wrap interval
        179.95,
    ],
)
def test_clamps_at_and_past_the_last_index(transmission, azimuth):
    last = transmission.values[-1]
    assert interpolate_azimuthal_transmission(transmission, azimuth) == last


@pytest.mark.parametrize(
    ("azimuth", "equivalent"),
    [
        # Placeholder: (azimuth + 180) % 360 - 180 maps any azimuth into [-180, 180)
        (359.0, 1.0),
        (180.0, 180.0),
        (-180.0, 180.0),
        (-181.0, 179.0),
        (541.0, 179.0),
        (-541.0, 179.0),
    ],
)
def test_wraps_azimuth_into_canonical_interval(transmission, azimuth, equivalent):
    expected = transmission.values[round(equivalent / SPACING_DEG)]
    assert interpolate_azimuthal_transmission(transmission, azimuth) == pytest.approx(
        expected
    )


def test_clamps_lower_index_when_past_table_extent():
    # Placeholder: a table spanning only 3 deg; |azimuth| / spacing = 10 is past index 3
    transmission = AzimuthalTransmission(
        values=np.array([0.1, 0.2, 0.3, 0.4]), spacing=1.0
    )
    assert np.isfinite(interpolate_azimuthal_transmission(transmission, 10.0))


def test_from_files_rejects_a_table_violating_the_plateaus(
    response_table_paths, tmp_path
):
    table = pd.read_csv(response_table_paths["azimuthal_transmission_path"])
    # Placeholder: a single cell inside the OA plateau
    table.loc[500, "transmission"] = 0.5
    bad = tmp_path / "azimuthal-transmission.csv"
    table.to_csv(bad, index=False)

    with pytest.raises(ValueError, match="plateau"):
        SwapiResponse.from_files(
            **{**response_table_paths, "azimuthal_transmission_path": bad}
        )
