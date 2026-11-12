"""Placeholder: Tests of the Maxwellian count rates and their Jacobian.

Mostly from the ``ialirt-swapi-forward-model`` branch's
``test_maxwellian_count_rates.py``, plus L3a's test of invariance under a joint
rotation of the bulk velocity and the RTN-to-SWAPI matrix.
"""

import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose

from imap_processing.swapi.constants import PROTON_MASS_KG, Species
from imap_processing.swapi.forward_model.look_angles import look_angles_to_velocity
from imap_processing.swapi.forward_model.maxwellian.integration import (
    _calculate_integral,
    _integrate_region,
    maxwellian_count_rates,
)
from imap_processing.swapi.forward_model.maxwellian.integration_limits import (
    angular_quadrature_rule,
)
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalRegion,
)
from imap_processing.swapi.forward_model.response.measurement_species_response import (
    measurement_species_responses,
)
from imap_processing.tests.swapi.forward_model.maxwellian.conftest import (
    count_rate_factor,
    proton_params,
    synthetic_measurement_response,
)

# Placeholder: a 400 km/s proton sits at the center of the passband at ~442 V
ESA_VOLTAGES = np.linspace(420.0, 465.0, 10)
VELOCITY = look_angles_to_velocity(400.0, 2.0, 1.0)
MOMENTS = dict(density=5.0, temperature=100_000.0, velocity_rtn_sc=VELOCITY)


def _rotation(axis, angle_deg):
    """Claude: Right-handed rotation matrix about a unit axis."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    angle = np.radians(angle_deg)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


# Claude: I-ALiRT-like attitude, R along SWAPI -y, then tilted off the axes so every
# RTN component reaches every SWAPI component
R_ALONG_MINUS_Y = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
TILTED_RTN_TO_SWAPI = _rotation((1.0, 2.0, 3.0), 25.0) @ R_ALONG_MINUS_Y


@pytest.fixture
def responses_at(swapi_response, epoch):
    def responses_at(esa_voltages=ESA_VOLTAGES, rtn_to_swapi=None):
        # Claude: one epoch or rotation is repeated for every sample
        n_samples = np.size(esa_voltages)
        if rtn_to_swapi is None:
            rtn_to_swapi = np.eye(3)
        return measurement_species_responses(
            swapi_response,
            esa_voltages,
            np.full(n_samples, epoch),
            Species.PROTON,
            np.broadcast_to(rtn_to_swapi, (n_samples, 3, 3)),
        )

    return responses_at


@pytest.fixture
def rates_at(responses_at):
    """Placeholder: Evaluate the rates and Jacobian of a sweep."""

    def rates_at(esa_voltages=ESA_VOLTAGES, rtn_to_swapi=None, **overrides):
        moments = MOMENTS | overrides
        return maxwellian_count_rates(
            moments["density"],
            moments["temperature"],
            np.asarray(moments["velocity_rtn_sc"], dtype=float),
            responses_at(esa_voltages, rtn_to_swapi),
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
    # Placeholder: the integration limits follow the flow direction, so finite
    # differences carry a small error; scale the tolerance to the whole gradient so the
    # small v_x component (the flow is near the y axis) isn't held to more
    atol = 1e-3 * np.abs(d_velocity).max()
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        upper, _ = rates_at(velocity_rtn_sc=VELOCITY + delta)
        lower, _ = rates_at(velocity_rtn_sc=VELOCITY - delta)
        expected = (upper - lower) / (2 * step)
        assert_allclose(d_velocity[:, axis], expected, rtol=1e-3, atol=atol)


def test_rates_depend_only_on_the_swapi_frame_velocity(rates_at):
    """Claude: Rotating the RTN velocity along with the RTN-to-SWAPI matrix leaves
    the SWAPI-frame flow, and so the rates, unchanged."""
    expected, expected_jacobian = rates_at()
    velocity_rtn = TILTED_RTN_TO_SWAPI.T @ VELOCITY
    rates, jacobian = rates_at(
        rtn_to_swapi=TILTED_RTN_TO_SWAPI, velocity_rtn_sc=velocity_rtn
    )
    assert_allclose(rates, expected, rtol=1e-12)
    assert_allclose(jacobian[:, :2], expected_jacobian[:, :2], rtol=1e-12)
    # Claude: the velocity columns are the SWAPI-frame gradient rotated to RTN
    assert_allclose(
        jacobian[:, 2:],
        expected_jacobian[:, 2:] @ TILTED_RTN_TO_SWAPI,
        rtol=1e-12,
        atol=1e-12 * np.abs(expected_jacobian[:, 2:]).max(),
    )


def test_d_rtn_velocity_matches_finite_difference(rates_at):
    """Claude: The velocity columns are derivatives w.r.t. the RTN velocity when the
    rotation mixes every component."""
    step = 1e-2  # Claude: km/s
    velocity_rtn = TILTED_RTN_TO_SWAPI.T @ VELOCITY
    _, jacobian = rates_at(
        rtn_to_swapi=TILTED_RTN_TO_SWAPI, velocity_rtn_sc=velocity_rtn
    )
    d_velocity = jacobian[:, 2:]
    atol = 1e-3 * np.abs(d_velocity).max()
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        upper, _ = rates_at(
            rtn_to_swapi=TILTED_RTN_TO_SWAPI, velocity_rtn_sc=velocity_rtn + delta
        )
        lower, _ = rates_at(
            rtn_to_swapi=TILTED_RTN_TO_SWAPI, velocity_rtn_sc=velocity_rtn - delta
        )
        expected = (upper - lower) / (2 * step)
        assert_allclose(d_velocity[:, axis], expected, rtol=1e-3, atol=atol)


def test_each_sample_uses_its_own_rotation(responses_at, rates_at):
    """Claude: A per-sample rotation applies to its own sample only."""
    expected, _ = rates_at()
    rotations = np.repeat(np.eye(3)[np.newaxis], len(ESA_VOLTAGES), axis=0)
    # Claude: turn the last sample's view of the flow away from the boresight
    rotations[-1] = -np.eye(3)
    rates, _ = rates_at(rtn_to_swapi=rotations)
    assert_allclose(rates[:-1], expected[:-1], rtol=1e-12)
    assert rates[-1] == 0.0


def test_callable_from_numba(responses_at, rates_at):
    @numba.njit
    def sweep(responses, density, temperature, velocity_rtn_sc):
        rates, _ = maxwellian_count_rates(
            density, temperature, velocity_rtn_sc, responses
        )
        return rates

    expected, _ = rates_at()
    assert_allclose(sweep(responses_at(), **MOMENTS), expected)


def test_plain_list_is_rejected(responses_at):
    with pytest.raises(TypeError, match=r"numba\.typed\.List"):
        maxwellian_count_rates(
            MOMENTS["density"],
            MOMENTS["temperature"],
            np.asarray(VELOCITY, dtype=float),
            list(responses_at()),
        )


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


def _region_rates(params, response):
    """Claude: Integrate every region with a quadrature rule, without the OA skip."""
    factor = count_rate_factor(params, response)
    region_rates = {}
    for region in (
        AzimuthalRegion.SUNGLASSES,
        AzimuthalRegion.OPEN_APERTURE_NEGATIVE,
        AzimuthalRegion.OPEN_APERTURE_POSITIVE,
    ):
        rule = angular_quadrature_rule(
            params, response, region, 0.0, oa_skip_floor_counts_per_s=0.0
        )
        if rule is None:
            continue
        integral, jac = _integrate_region(params, response, region, rule)
        region_rates[region] = (factor * integral, factor * jac)
    return region_rates


@pytest.mark.parametrize(
    ("velocity_xyz", "temperature", "density", "expected_skipped"),
    [
        # Placeholder: OA+ aligned, SG out of reach: OA+ integrated
        ((-450.0, 0.0, 0.0), 1.0e5, 5.0, []),
        # Placeholder: hot plasma near the SG/OA+ edge: SG and OA+ integrated, the
        # far-side OA- reaches its window but is bounded below the skip threshold
        ((-300.0, -330.0, 0.0), 1.0e6, 5.0, [AzimuthalRegion.OPEN_APERTURE_NEGATIVE]),
        # Placeholder: tenuous OA+ plasma: the 0.1 counts/s floor skips OA+
        ((-450.0, 0.0, 0.0), 1.0e5, 1.0e-12, [AzimuthalRegion.OPEN_APERTURE_POSITIVE]),
    ],
)
def test_calculate_integral_applies_oa_skip_rule(
    velocity_xyz, temperature, density, expected_skipped
):
    """Claude: ``calculate_integral`` sums exactly the regions the
    ``max(0.1 Hz, 1e-3 C_SG)`` OA skip rule keeps."""
    response = synthetic_measurement_response()
    params = proton_params(
        velocity_xyz=velocity_xyz, temperature=temperature, density=density
    )
    region_rates = _region_rates(params, response)
    # Claude: a skipped region must have something to skip, or the case proves nothing
    for region in expected_skipped:
        assert region_rates[region][0] > 0
    kept = [region_rates[r] for r in region_rates if r not in expected_skipped]
    rate, jacobian = _calculate_integral(params, response)
    assert rate == pytest.approx(sum(r for r, _ in kept), rel=1e-12, abs=0.0)
    assert_allclose(jacobian, sum(j for _, j in kept), rtol=1e-12, atol=0.0)
