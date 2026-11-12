import math

import pytest

from imap_processing.swapi.constants import BOLTZMANN_CONSTANT_J_PER_K, PROTON_MASS_KG
from imap_processing.swapi.forward_model.look_angles import look_angles_to_velocity
from imap_processing.swapi.forward_model.maxwellian.params import (
    make_maxwellian,
)


def test_thermal_speed_matches_analytic_formula():
    """Claude: For a proton at 1e5 K, the thermal speed is sqrt(kT/m) in km/s."""
    expected_km_s = (
        math.sqrt(BOLTZMANN_CONSTANT_J_PER_K * 100_000.0 / PROTON_MASS_KG) / 1000
    )
    params = make_maxwellian(
        5.0, 100_000.0, look_angles_to_velocity(400.0, 2.0, 1.0), PROTON_MASS_KG
    )
    assert params.thermal_speed == pytest.approx(expected_km_s)


def test_bulk_speed_and_look_angles_come_from_the_velocity():
    """Claude: The derived bulk fields invert look_angles_to_velocity."""
    velocity = look_angles_to_velocity(400.0, 25.0, -3.0)
    params = make_maxwellian(5.0, 100_000.0, velocity, PROTON_MASS_KG)
    assert params.density == 5.0
    assert params.bulk_speed == pytest.approx(400.0)
    assert params.bulk_azimuth == pytest.approx(25.0)
    assert params.bulk_elevation == pytest.approx(-3.0)
