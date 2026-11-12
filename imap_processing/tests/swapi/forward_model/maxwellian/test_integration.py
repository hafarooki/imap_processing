"""Placeholder: Tests of the Maxwellian count rates and their Jacobian.

Mostly from the ``ialirt-swapi-forward-model`` branch's
``test_maxwellian_count_rates.py``. L3a's test of
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
    _integrate_region,
    calculate_integral,
    maxwellian_count_rates,
)
from imap_processing.swapi.forward_model.maxwellian.integration_limits import (
    get_angular_quadrature,
)
from imap_processing.swapi.forward_model.response.instrument_response import Region
from imap_processing.tests.swapi.forward_model.maxwellian.conftest import (
    count_rate_factor,
    proton_params,
    synthetic_step_response,
)

# Placeholder: a 400 km/s proton sits at the center of the passband at ~442 V
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
    """Placeholder: Evaluate the rates and Jacobian of a sweep."""

    def rates_at(esa_voltages=ESA_VOLTAGES, **overrides):
        moments = MOMENTS | overrides
        return maxwellian_count_rates(
            moments["density"],
            moments["temperature"],
            np.asarray(moments["velocity_xyz"], dtype=float),
            responses_at(esa_voltages),
        )

    return rates_at


def test_rates_match_swapi_tools(rates_at):
    # Placeholder: from
    # swapi_tools.solar_wind.solar_wind_model.ideal_rates_at_look_angles with the same
    # inputs; the relative efficiency is 1 at this epoch
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
    # Placeholder: so the ln(density) column is the rate
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
    step = 1e-2  # Placeholder: km/s
    _, jacobian = rates_at()
    d_velocity = jacobian[:, 2:]
    # Placeholder: the quadrature bounds follow the flow direction, so finite
    # differences carry a small error; scale the tolerance to the whole gradient so the
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
    """Placeholder: The thermal speed uses the response's mass: the same moments with a
    heavier species' response give a different rate."""
    response = responses_at()[5]
    heavier = response._replace(mass_kg=4 * PROTON_MASS_KG)
    velocity_xyz = np.asarray(VELOCITY, dtype=float)
    (rate,), _ = maxwellian_count_rates(
        5.0, 100_000.0, velocity_xyz, numba.typed.List([response])
    )
    (heavier_rate,), _ = maxwellian_count_rates(
        5.0, 100_000.0, velocity_xyz, numba.typed.List([heavier])
    )
    assert rate != heavier_rate


def _region_rates(maxwellian, response):
    """Claude: Integrate every region with a quadrature, without the OA skip rule."""
    factor = count_rate_factor(maxwellian, response)
    region_rates = {}
    for region in (
        Region.SUNGLASSES,
        Region.OPEN_APERTURE_NEGATIVE,
        Region.OPEN_APERTURE_POSITIVE,
    ):
        quadrature = get_angular_quadrature(maxwellian, response, region)
        if quadrature is None:
            continue
        integral, jac = _integrate_region(maxwellian, response, region, quadrature)
        region_rates[region] = (factor * integral, factor * jac)
    return region_rates


@pytest.mark.parametrize(
    ("velocity_xyz", "temperature", "density", "expected_skipped"),
    [
        # Placeholder: OA+ aligned, SG out of reach: OA+ integrated
        ((-450.0, 0.0, 0.0), 1.0e5, 5.0, []),
        # Placeholder: hot plasma near the SG/OA+ edge: SG and OA+ integrated, the
        # far-side OA- reaches its window but is bounded below the skip threshold
        ((-300.0, -330.0, 0.0), 1.0e6, 5.0, [Region.OPEN_APERTURE_NEGATIVE]),
        # Placeholder: tenuous OA+ plasma: the 0.1 counts/s floor skips OA+
        ((-450.0, 0.0, 0.0), 1.0e5, 1.0e-12, [Region.OPEN_APERTURE_POSITIVE]),
    ],
)
def test_calculate_integral_applies_oa_skip_rule(
    velocity_xyz, temperature, density, expected_skipped
):
    """Claude: ``calculate_integral`` sums exactly the regions the
    ``max(0.1 Hz, 1e-3 C_SG)`` OA skip rule keeps."""
    response = synthetic_step_response()
    maxwellian = proton_params(
        velocity_xyz=velocity_xyz, temperature=temperature, density=density
    )
    region_rates = _region_rates(maxwellian, response)
    # Claude: a skipped region must have something to skip, or the case proves nothing
    for region in expected_skipped:
        assert region_rates[region][0] > 0
    kept = [region_rates[r] for r in region_rates if r not in expected_skipped]
    rate, jacobian = calculate_integral(maxwellian, response)
    assert rate == pytest.approx(sum(r for r, _ in kept), rel=1e-12, abs=0.0)
    assert_allclose(jacobian, sum(j for _, j in kept), rtol=1e-12, atol=0.0)
