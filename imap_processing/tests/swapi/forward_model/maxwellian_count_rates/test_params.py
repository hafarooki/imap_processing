"""Ported from L3a's ``tests/swapi/l3a/science/solar_wind/test_params.py``.

L3a's ``to_vector`` / ``from_vector`` tests have no counterpart: those methods
were dropped.
"""

import math

import numpy as np
import pytest

from imap_processing.swapi.constants import BOLTZMANN_CONSTANT_J_PER_K, PROTON_MASS_KG
from imap_processing.swapi.forward_model.maxwellian_count_rates.params import (
    bulk_speed,
    temperature_to_thermal_speed,
    thermal_speed,
    thermal_speed_to_temperature,
)
from imap_processing.tests.swapi.forward_model.maxwellian_count_rates.conftest import (
    proton_params,
    synthetic_step_response,
)


def test_bulk_speed_is_magnitude_of_velocity():
    """A bulk velocity of (-300, 40, -50) km/s has the Euclidean norm."""
    params = proton_params(velocity_xyz=(-300.0, 40.0, -50.0))
    assert bulk_speed(params) == pytest.approx(math.sqrt(300.0**2 + 40.0**2 + 50.0**2))


def test_bulk_speed_of_zero_velocity_is_zero():
    """A zero bulk velocity gives exactly 0."""
    assert bulk_speed(proton_params(velocity_xyz=(0.0, 0.0, 0.0))) == 0.0


def test_thermal_speed_matches_analytic_formula():
    """For a proton at 1e5 K, the thermal speed is sqrt(kT/m) in km/s."""
    expected_km_s = (
        math.sqrt(BOLTZMANN_CONSTANT_J_PER_K * 100_000.0 / PROTON_MASS_KG) / 1000
    )
    assert temperature_to_thermal_speed(PROTON_MASS_KG, 100_000.0) == pytest.approx(
        expected_km_s
    )


@pytest.mark.parametrize("temperature", [1_000.0, 1e5, 1e7])
def test_temperature_to_thermal_speed_round_trips_via_inverse(temperature):
    """Converting to a thermal speed and back recovers the temperature."""
    speed = temperature_to_thermal_speed(PROTON_MASS_KG, temperature)
    assert thermal_speed_to_temperature(speed, PROTON_MASS_KG) == pytest.approx(
        temperature
    )


def test_thermal_speed_scales_with_sqrt_of_temperature():
    """Quadrupling the temperature doubles the thermal speed."""
    speed_low = temperature_to_thermal_speed(PROTON_MASS_KG, 50_000.0)
    speed_high = temperature_to_thermal_speed(PROTON_MASS_KG, 200_000.0)
    np.testing.assert_allclose(speed_high / speed_low, 2.0)


def test_thermal_speed_uses_params_temperature_and_response_mass():
    """``thermal_speed`` reads T from the moments and m from the response."""
    params = proton_params(temperature=120_000.0)
    response = synthetic_step_response()
    np.testing.assert_allclose(
        thermal_speed(params, response),
        temperature_to_thermal_speed(response.mass_kg, 120_000.0),
    )
