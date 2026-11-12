"""Ported from L3a's ``tests/swapi/l3a/science/solar_wind/test_forward_model.py``.

The L3a test of invariance under a joint rotation of the bulk velocity and the
SWAPI-to-RTN matrix has no counterpart: the model here works in the SWAPI frame.
"""

import numpy as np
import pytest

from imap_processing.swapi.constants import PROTON_MASS_KG, Species
from imap_processing.swapi.forward_model.maxwellian.integration import (
    calculate_integral,
    maxwellian_count_rates,
)
from imap_processing.swapi.forward_model.response.instrument_response import (
    central_speed,
)
from imap_processing.tests.swapi.forward_model.maxwellian.conftest import (
    proton_params,
)

# Reference bulk speed for the slow-wind state shared by every test in this
# module (5 cm^-3, 100 kK protons under the default `proton_params`).
REF_BULK_SPEED_KM_S = 450.0

# Off-axis bulk velocity used for the velocity-component Jacobian tests. Small
# x and z components break the symmetry that makes the vx/vz Jacobian entries
# near zero, so finite-difference noise doesn't dominate.
OFF_AXIS_VELOCITY_XYZ = np.array([30.0, -REF_BULK_SPEED_KM_S, 20.0])

# Jacobian columns
LOG_DENSITY, LOG_TEMPERATURE, VELOCITY_X, VELOCITY_Y, VELOCITY_Z = range(5)

# Small enough that the finite-difference truncation is well below the GL
# quadrature noise floor, large enough that round-off stays well below 1%.
FD_RELATIVE_STEP = 1e-5

# The Jacobian is exact in the continuum but evaluated with the same
# Gauss-Legendre quadrature as the rate; the mismatch comes from the dynamic
# quadrature limits shifting slightly when the state is perturbed. L3a
# measured a worst case of ~2.7% on this fixture; 5% leaves headroom while
# still catching a sign flip or factor of 2.
JACOBIAN_RTOL = 0.05
JACOBIAN_ATOL = 1e-3


def esa_voltage_at_speed(speed_km_s: float) -> float:
    """ESA voltage in V whose central proton speed is ``speed_km_s``."""
    # central speed scales with sqrt(voltage)
    return (speed_km_s / central_speed(1.0, Species.PROTON)) ** 2


# Putting the bulk speed at the passband center maximizes the rate and
# minimizes truncation effects in the quadrature.
ESA_VOLTAGE_AT_REF_SPEED_V = esa_voltage_at_speed(REF_BULK_SPEED_KM_S)


def rates_at_state(state, responses):
    """Rates at ``[ln(density), ln(temperature), vx, vy, vz]``."""
    rates, _ = maxwellian_count_rates(
        np.exp(state[LOG_DENSITY]),
        np.exp(state[LOG_TEMPERATURE]),
        state[VELOCITY_X:],
        responses,
    )
    return rates.copy()


def finite_difference_jacobian_column(state, responses, j):
    """Central difference ``(C(x + eps e_j) - C(x - eps e_j)) / 2 eps``."""
    eps = FD_RELATIVE_STEP * max(abs(state[j]), 1.0)
    forward = state.copy()
    forward[j] += eps
    backward = state.copy()
    backward[j] -= eps
    return (
        rates_at_state(forward, responses) - rates_at_state(backward, responses)
    ) / (2 * eps)


def test_returns_one_rate_and_one_jacobian_row_per_sample(proton_step_responses):
    """Three samples spanning the passband-center voltage give a length-3 rate
    vector and a (3, 5) Jacobian."""
    responses = proton_step_responses(
        ESA_VOLTAGE_AT_REF_SPEED_V * np.array([0.95, 1.0, 1.05])
    )
    params = proton_params()
    rates, jacobian = maxwellian_count_rates(
        params.density, params.temperature, params.velocity_xyz, responses
    )
    assert rates.shape == (3,)
    assert jacobian.shape == (3, 5)


@pytest.fixture
def ref_response(proton_step_responses):
    return proton_step_responses([ESA_VOLTAGE_AT_REF_SPEED_V])[0]


def test_rate_is_positive_when_velocity_aligns_with_boresight(ref_response):
    """With the bulk speed at the passband center and the flow along the
    boresight, the rate is strictly positive."""
    rate, _ = calculate_integral(proton_params(), ref_response)
    assert rate > 0.0


def test_rate_is_linear_in_density(ref_response):
    """Doubling the density doubles the rate exactly."""
    rate_at_5, _ = calculate_integral(proton_params(density=5.0), ref_response)
    rate_at_10, _ = calculate_integral(proton_params(density=10.0), ref_response)
    np.testing.assert_allclose(rate_at_10, 2.0 * rate_at_5, rtol=1e-12)


@pytest.mark.parametrize("bulk_speed", [2000.0, 50.0])
def test_returns_zero_rate_and_jacobian_when_bulk_speed_misses_passband(
    ref_response, bulk_speed
):
    """A bulk speed far above or below the passband short-circuits to rate 0
    and a zero Jacobian row."""
    params = proton_params(velocity_xyz=(0.0, -bulk_speed, 0.0))
    rate, jacobian_row = calculate_integral(params, ref_response)
    assert rate == 0.0
    np.testing.assert_array_equal(jacobian_row, np.zeros(5))


def test_log_density_jacobian_column_equals_the_rate(proton_step_responses):
    """The rate is linear in density, so the ln(density) column equals the rate
    exactly."""
    responses = proton_step_responses(
        ESA_VOLTAGE_AT_REF_SPEED_V * np.array([0.95, 1.0, 1.05])
    )
    params = proton_params(velocity_xyz=OFF_AXIS_VELOCITY_XYZ)
    rates, jacobian = maxwellian_count_rates(
        params.density, params.temperature, params.velocity_xyz, responses
    )
    np.testing.assert_array_equal(jacobian[:, LOG_DENSITY], rates)


@pytest.mark.parametrize(
    "column", [LOG_TEMPERATURE, VELOCITY_X, VELOCITY_Y, VELOCITY_Z]
)
def test_analytic_jacobian_matches_finite_differences(proton_step_responses, column):
    """At an off-axis bulk state, each analytic Jacobian column matches a
    central finite difference of the rate to within GL quadrature noise."""
    # two samples with a small voltage gradient, so each contributes a
    # different, non-degenerate row
    responses = proton_step_responses(
        ESA_VOLTAGE_AT_REF_SPEED_V * np.array([1.0, 1.05])
    )
    params = proton_params(velocity_xyz=OFF_AXIS_VELOCITY_XYZ)
    state = np.concatenate(
        [[np.log(params.density), np.log(params.temperature)], params.velocity_xyz]
    )
    _, analytic_jacobian = maxwellian_count_rates(
        params.density, params.temperature, params.velocity_xyz, responses
    )
    np.testing.assert_allclose(
        analytic_jacobian[:, column],
        finite_difference_jacobian_column(state, responses, column),
        rtol=JACOBIAN_RTOL,
        atol=JACOBIAN_ATOL,
    )


def test_proton_mass_comes_from_the_response(ref_response):
    """The thermal speed uses the response's mass: the same moments with a
    heavier species' response give a different rate."""
    heavier = ref_response._replace(mass_kg=4 * PROTON_MASS_KG)
    rate, _ = calculate_integral(proton_params(), ref_response)
    heavier_rate, _ = calculate_integral(proton_params(), heavier)
    assert rate != heavier_rate
