from typing import NamedTuple

import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose

from imap_processing.swapi.constants import Species
from imap_processing.swapi.forward_model.instrument_response import step_response_at
from imap_processing.swapi.forward_model.look_angles import look_angles_to_velocity
from imap_processing.swapi.forward_model.maxwellian_count_rates import (
    JACOBIAN_COLUMNS,
    maxwellian_count_rate,
    maxwellian_count_rates,
    maxwellian_count_rates_at_temperatures,
)

# a 400 km/s proton sits at the center of the passband at ~442 V
ESA_VOLTAGES = np.linspace(420.0, 465.0, 10)
VELOCITY = look_angles_to_velocity(400.0, 2.0, 1.0)
MOMENTS = dict(density=5.0, temperature=100_000.0, velocity_xyz=VELOCITY)


class Sweep(NamedTuple):
    rates: np.ndarray
    d_log_density: np.ndarray
    d_log_temperature: np.ndarray
    d_velocity: np.ndarray


@pytest.fixture
def responses_at(response_tables, epoch):
    def responses_at(esa_voltages=ESA_VOLTAGES):
        return response_tables.step_responses(esa_voltages, epoch, Species.PROTON)

    return responses_at


@pytest.fixture
def rates_at(responses_at):
    """Evaluate the single-sample kernel at each energy of a sweep."""

    def rates_at(esa_voltages=ESA_VOLTAGES, **overrides):
        moments = MOMENTS | overrides
        responses = responses_at(esa_voltages)
        results = [
            maxwellian_count_rate(
                step_response_at(responses, i),
                moments["density"],
                moments["temperature"],
                moments["velocity_xyz"],
            )
            for i in range(len(esa_voltages))
        ]
        return Sweep(
            np.array([r.rate for r in results]),
            np.array([r.jacobian.d_log_density for r in results]),
            np.array([r.jacobian.d_log_temperature for r in results]),
            np.array([r.jacobian.d_velocity for r in results]).reshape(-1, 3),
        )

    return rates_at


def test_rates_are_positive_in_passband(rates_at):
    actual = rates_at()
    assert actual.rates.shape == ESA_VOLTAGES.shape
    assert np.all(actual.rates > 0)
    # peaked near the 400 km/s central voltage, not at the edges of the sweep
    assert 0 < np.argmax(actual.rates) < len(ESA_VOLTAGES) - 1


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
    assert_allclose(rates_at().rates, expected, rtol=1e-8)


def test_rates_are_zero_far_from_bulk_speed(rates_at):
    actual = rates_at(np.array([50.0, 5_000.0]))
    assert_allclose(actual.rates, 0.0)


def test_rates_scale_with_density(rates_at):
    base = rates_at()
    doubled = rates_at(density=2 * MOMENTS["density"])
    assert_allclose(doubled.rates, 2 * base.rates, rtol=1e-12)
    assert_allclose(base.d_log_density, base.rates)


def test_d_log_temperature_matches_finite_difference(rates_at):
    step = 1e-4
    temperature = MOMENTS["temperature"]
    upper = rates_at(temperature=temperature * np.exp(step)).rates
    lower = rates_at(temperature=temperature * np.exp(-step)).rates
    expected = (upper - lower) / (2 * step)
    actual = rates_at().d_log_temperature
    assert_allclose(actual, expected, rtol=1e-3, atol=1e-3 * np.abs(expected).max())


def test_d_velocity_matches_finite_difference(rates_at):
    step = 1e-2  # km/s
    actual = rates_at().d_velocity
    # the quadrature bounds follow the flow direction, so finite differences
    # carry a small error; scale the tolerance to the whole gradient so the
    # small v_x component (the flow is near the y axis) isn't held to more
    atol = 1e-3 * np.abs(actual).max()
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        upper = rates_at(velocity_xyz=VELOCITY + delta).rates
        lower = rates_at(velocity_xyz=VELOCITY - delta).rates
        expected = (upper - lower) / (2 * step)
        assert_allclose(actual[:, axis], expected, rtol=1e-3, atol=atol)


def test_nan_moments_give_nan_rates(rates_at):
    actual = rates_at(density=np.nan)
    assert np.all(np.isnan(actual.rates))
    assert np.all(np.isnan(actual.d_velocity))


def test_batched_rates_match_the_single_sample_kernel(responses_at, rates_at):
    expected = rates_at()
    velocities = np.tile(VELOCITY, (len(ESA_VOLTAGES), 1))

    actual = maxwellian_count_rates(
        responses_at(), MOMENTS["density"], MOMENTS["temperature"], velocities
    )

    assert JACOBIAN_COLUMNS == (
        "log_density",
        "log_temperature",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    )
    assert actual.jacobian.shape == (len(ESA_VOLTAGES), 5)
    assert_allclose(actual.rates, expected.rates, rtol=1e-12)
    assert_allclose(actual.jacobian[:, 0], expected.d_log_density, rtol=1e-12)
    assert_allclose(actual.jacobian[:, 1], expected.d_log_temperature, rtol=1e-12)
    assert_allclose(
        actual.jacobian[:, 2:],
        expected.d_velocity,
        rtol=1e-12,
        atol=1e-12 * np.abs(expected.d_velocity).max(),
    )


def test_batched_rates_take_a_velocity_per_sample(responses_at, rates_at):
    azimuths = np.linspace(-5.0, 5.0, len(ESA_VOLTAGES))
    velocities = np.array(
        [look_angles_to_velocity(400.0, azimuth, 1.0) for azimuth in azimuths]
    )

    actual = maxwellian_count_rates(
        responses_at(), MOMENTS["density"], MOMENTS["temperature"], velocities
    )

    for i, velocity in enumerate(velocities):
        expected = rates_at(velocity_xyz=velocity).rates[i]
        assert actual.rates[i] == pytest.approx(expected, rel=1e-12)


def test_batched_rates_accept_a_broadcast_velocity(responses_at):
    tiled = np.tile(VELOCITY, (len(ESA_VOLTAGES), 1))
    broadcast = np.broadcast_to(VELOCITY, tiled.shape)

    def rates(velocities):
        return maxwellian_count_rates(
            responses_at(), MOMENTS["density"], MOMENTS["temperature"], velocities
        ).rates

    assert_allclose(rates(broadcast), rates(tiled), rtol=1e-12)


def test_batched_rates_of_a_subset_are_the_subset_of_the_rates(responses_at):
    responses = responses_at()
    velocities = np.tile(VELOCITY, (len(ESA_VOLTAGES), 1))
    indices = np.array([7, 2, 3])

    def rates(responses, velocities):
        return maxwellian_count_rates(
            responses, MOMENTS["density"], MOMENTS["temperature"], velocities
        ).rates

    assert_allclose(
        rates(responses.subset(indices), velocities[indices]),
        rates(responses, velocities)[indices],
        rtol=1e-12,
    )


def test_batched_rates_reject_a_velocity_per_sample_mismatch(responses_at):
    with pytest.raises(ValueError, match="velocity"):
        maxwellian_count_rates(
            responses_at(),
            MOMENTS["density"],
            MOMENTS["temperature"],
            np.tile(VELOCITY, (len(ESA_VOLTAGES) - 1, 1)),
        )


def test_callable_from_numba(responses_at, rates_at):
    @numba.njit
    def sweep(responses, density, temperature, velocity_xyz):
        velocities = np.empty((responses.central_speed.shape[0], 3))
        velocities[:] = velocity_xyz
        return maxwellian_count_rates(responses, density, temperature, velocities).rates

    actual = sweep(responses_at(), **MOMENTS)
    assert_allclose(actual, rates_at().rates)


def test_rates_at_temperatures_match_one_call_per_temperature(responses_at):
    responses = responses_at()
    velocities = np.broadcast_to(VELOCITY, (len(ESA_VOLTAGES), 3))
    temperatures = np.array([3e4, 1e5, 1e6])

    actual = maxwellian_count_rates_at_temperatures(
        responses, 5.0, temperatures, velocities
    )

    expected = [
        maxwellian_count_rates(responses, 5.0, temperature, velocities).rates
        for temperature in temperatures
    ]
    np.testing.assert_array_equal(actual, expected)


def test_rates_at_temperatures_reject_a_velocity_per_sample_mismatch(responses_at):
    with pytest.raises(ValueError, match="velocity_xyz"):
        maxwellian_count_rates_at_temperatures(
            responses_at(), 5.0, np.array([1e5]), np.zeros((1, 3))
        )
