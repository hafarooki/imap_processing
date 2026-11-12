"""Tests of the Maxwellian count rates and their Jacobian.

Mostly from the ``ialirt-swapi-forward-model`` branch's
``test_maxwellian_count_rates.py``; the output-shape test is from L3a's
``tests/swapi/l3a/science/solar_wind/test_forward_model.py``. L3a's test of
invariance under a joint rotation of the bulk velocity and the SWAPI-to-RTN
matrix has no counterpart: the model here works in the SWAPI frame.
"""

import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose

from imap_processing.swapi.constants import PROTON_MASS_KG, Species
from imap_processing.swapi.forward_model.look_angles import look_angles_to_velocity
from imap_processing.swapi.forward_model.maxwellian.integration import (
    calculate_integral,
    maxwellian_count_rates,
)
from imap_processing.tests.swapi.forward_model.maxwellian.conftest import (
    proton_params,
)

# a 400 km/s proton sits at the center of the passband at ~442 V
ESA_VOLTAGES = np.linspace(420.0, 465.0, 10)
VELOCITY = look_angles_to_velocity(400.0, 2.0, 1.0)
MOMENTS = dict(density=5.0, temperature=100_000.0, velocity_xyz=VELOCITY)


@pytest.fixture
def responses_at(swapi_response, epoch):
    def responses_at(esa_voltages=ESA_VOLTAGES):
        return swapi_response.step_responses(esa_voltages, epoch, Species.PROTON)

    return responses_at


@pytest.fixture
def rates_at(responses_at):
    """Evaluate the rates and Jacobian of a sweep."""

    def rates_at(esa_voltages=ESA_VOLTAGES, **overrides):
        moments = MOMENTS | overrides
        return maxwellian_count_rates(
            moments["density"],
            moments["temperature"],
            np.asarray(moments["velocity_xyz"], dtype=float),
            responses_at(esa_voltages),
        )

    return rates_at


def test_returns_one_rate_and_one_jacobian_row_per_sample(rates_at):
    rates, jacobian = rates_at()
    assert rates.shape == ESA_VOLTAGES.shape
    assert jacobian.shape == (len(ESA_VOLTAGES), 5)


def test_rates_are_positive_in_passband(rates_at):
    rates, _ = rates_at()
    assert np.all(rates > 0)
    # peaked near the 400 km/s central voltage, not at the edges of the sweep
    assert 0 < np.argmax(rates) < len(ESA_VOLTAGES) - 1


def test_rates_match_swapi_tools(rates_at):
    # from swapi_tools.solar_wind.solar_wind_model.ideal_rates_at_look_angles
    # with the same inputs; the relative efficiency is 1 at this epoch
    expected = [
        12790.3372,
        13247.89684,
        13649.45592,
        13978.60541,
        14228.63446,
        14396.8441,
        14481.77537,
        14483.58013,
        14403.82024,
        14246.95474,
    ]
    rates, _ = rates_at()
    assert_allclose(rates, expected, rtol=1e-8)


def test_rates_and_jacobian_are_zero_far_from_bulk_speed(rates_at):
    rates, jacobian = rates_at(np.array([50.0, 5_000.0]))
    assert_allclose(rates, 0.0)
    assert_allclose(jacobian, 0.0)


def test_rates_scale_with_density(rates_at):
    rates, jacobian = rates_at()
    doubled, _ = rates_at(density=2 * MOMENTS["density"])
    assert_allclose(doubled, 2 * rates, rtol=1e-12)
    # so the ln(density) column is the rate
    assert_allclose(jacobian[:, 0], rates)


def test_d_log_temperature_matches_finite_difference(rates_at):
    step = 1e-4
    temperature = MOMENTS["temperature"]
    upper, _ = rates_at(temperature=temperature * np.exp(step))
    lower, _ = rates_at(temperature=temperature * np.exp(-step))
    expected = (upper - lower) / (2 * step)
    _, jacobian = rates_at()
    assert_allclose(
        jacobian[:, 1], expected, rtol=1e-3, atol=1e-3 * np.abs(expected).max()
    )


def test_d_velocity_matches_finite_difference(rates_at):
    step = 1e-2  # km/s
    _, jacobian = rates_at()
    d_velocity = jacobian[:, 2:]
    # the quadrature bounds follow the flow direction, so finite differences
    # carry a small error; scale the tolerance to the whole gradient so the
    # small v_x component (the flow is near the y axis) isn't held to more
    atol = 1e-3 * np.abs(d_velocity).max()
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        upper, _ = rates_at(velocity_xyz=VELOCITY + delta)
        lower, _ = rates_at(velocity_xyz=VELOCITY - delta)
        expected = (upper - lower) / (2 * step)
        assert_allclose(d_velocity[:, axis], expected, rtol=1e-3, atol=atol)


def test_nan_moments_give_nan_rates(rates_at):
    rates, jacobian = rates_at(density=np.nan)
    assert np.all(np.isnan(rates))
    assert np.all(np.isnan(jacobian[:, 2:]))


def test_callable_from_numba(responses_at, rates_at):
    @numba.njit
    def sweep(responses, density, temperature, velocity_xyz):
        rates, _ = maxwellian_count_rates(density, temperature, velocity_xyz, responses)
        return rates

    expected, _ = rates_at()
    assert_allclose(sweep(responses_at(), **MOMENTS), expected)


def test_mass_comes_from_the_response(responses_at):
    """The thermal speed uses the response's mass: the same moments with a
    heavier species' response give a different rate."""
    response = responses_at()[5]
    heavier = response._replace(mass_kg=4 * PROTON_MASS_KG)
    params = proton_params(velocity_xyz=VELOCITY)
    rate, _ = calculate_integral(params, response)
    heavier_rate, _ = calculate_integral(params, heavier)
    assert rate != heavier_rate
