"""Ported from L3a's
``tests/swapi/l3a/science/solar_wind/test_integration_limits.py``."""

import numpy as np
import pytest

from imap_processing.swapi.constants import PROTON_MASS_KG
from imap_processing.swapi.forward_model.maxwellian_count_rates.integration_limits import (  # noqa: E501
    SPEED_HALF_WIDTH_VTH,
    get_angular_quadrature,
    get_speed_quadrature,
    speed_window_misses_passband,
)
from imap_processing.swapi.forward_model.maxwellian_count_rates.params import (
    bulk_speed,
    temperature_to_thermal_speed,
    thermal_speed_to_temperature,
)
from imap_processing.swapi.forward_model.response.instrument_response import Region
from imap_processing.swapi.forward_model.response.passband import (
    interpolate_passband,
    speed_ratio_range_at_elevation,
)
from imap_processing.tests.swapi.forward_model.maxwellian_count_rates.conftest import (
    ESA_VOLTAGE_1KEV,
    proton_params,
)

# Region azimuth bands
SG_AZIMUTH_LO_DEG = -20.0
SG_AZIMUTH_HI_DEG = +20.0
OA_AZIMUTH_INNER_DEG = 20.0
OA_AZIMUTH_OUTER_DEG = 150.0

BULK_SPEED_KM_S = 450.0
# flows whose look direction is at azimuth +90 deg and -90 deg, elevation 0
VELOCITY_AT_PLUS_90_XYZ = (-BULK_SPEED_KM_S, 0.0, 0.0)
VELOCITY_AT_MINUS_90_XYZ = (+BULK_SPEED_KM_S, 0.0, 0.0)


@pytest.fixture
def response(proton_step_responses):
    return proton_step_responses([ESA_VOLTAGE_1KEV])[0]


# --- speed_window_misses_passband ---------------------------------------------
# The cheap pre-check that returns True only when `bulk_speed +/- k vth` is
# fully disjoint from the widest passband speed range, central_speed [0.9, 1.1].


def test_window_does_not_miss_when_bulk_speed_sits_in_passband(response):
    """A 450 km/s bulk at the 1 keV step sits well inside the passband."""
    assert not speed_window_misses_passband(proton_params(), response)


@pytest.mark.parametrize("bulk_speed_km_s", [1500.0, 100.0])
def test_window_misses_when_bulk_speed_outside_passband(response, bulk_speed_km_s):
    """A 1500 or 100 km/s bulk puts the whole window above or below the
    passband."""
    params = proton_params(velocity_xyz=(0.0, -bulk_speed_km_s, 0.0))
    assert speed_window_misses_passband(params, response)


@pytest.mark.parametrize(
    ("k_sigma_km_s", "misses"),
    [
        # k sigma = 50 km/s reaches back into the passband
        (50.0, False),
        # k sigma = 5 km/s stays clear of it
        (5.0, True),
    ],
)
def test_window_overlap_scales_with_speed_half_width(response, k_sigma_km_s, misses):
    """With the bulk 30 km/s above the passband's upper edge, the window
    overlaps the passband only when ``SPEED_HALF_WIDTH_VTH`` thermal speeds
    exceed 30 km/s."""
    bulk = response.central_speed * 1.1 + 30.0
    sigma = k_sigma_km_s / SPEED_HALF_WIDTH_VTH
    params = proton_params(
        velocity_xyz=(0.0, -bulk, 0.0),
        temperature=thermal_speed_to_temperature(sigma, PROTON_MASS_KG),
    )
    assert speed_window_misses_passband(params, response) == misses


# --- get_angular_quadrature: sunglasses ---------------------------------------
# The 21 x 21 GL rectangle clamped to the SG elevation range and the
# [-20 deg, +20 deg] azimuth band.


def test_sg_region_is_active_for_boresight_bulk(response):
    """A boresight bulk with a narrow Maxwellian falls inside the SG band."""
    skip, quadrature = get_angular_quadrature(
        proton_params(), response, Region.SUNGLASSES, 0.0
    )
    assert not skip
    assert quadrature is not None


def test_sg_azimuth_window_is_inside_plus_minus_20_degrees(response):
    """For a boresight bulk, every azimuth node lies inside the SG band."""
    _, quadrature = get_angular_quadrature(
        proton_params(), response, Region.SUNGLASSES, 0.0
    )
    assert quadrature.azimuth_points.min() >= SG_AZIMUTH_LO_DEG
    assert quadrature.azimuth_points.max() <= SG_AZIMUTH_HI_DEG


def test_sg_azimuth_window_clamps_to_plus_minus_20_for_broad_distribution(response):
    """A very hot plasma's 180 deg angular extent clamps the azimuth window
    to the SG band, with the outermost nodes hugging both edges."""
    _, quadrature = get_angular_quadrature(
        proton_params(temperature=1e10), response, Region.SUNGLASSES, 0.0
    )
    # GL nodes lie strictly inside the band
    np.testing.assert_array_less(quadrature.azimuth_points, SG_AZIMUTH_HI_DEG + 1e-9)
    np.testing.assert_array_less(SG_AZIMUTH_LO_DEG - 1e-9, quadrature.azimuth_points)
    # The outermost of 21 GL nodes sits ~0.125 deg inside the edge of a 40 deg
    # window; 1.5 deg only catches a window that collapses or shifts away.
    assert quadrature.azimuth_points.max() > SG_AZIMUTH_HI_DEG - 1.5
    assert quadrature.azimuth_points.min() < SG_AZIMUTH_LO_DEG + 1.5


def test_sg_elevation_window_clamps_into_passband_elevation_range(response):
    """A hot plasma's saturated angular extent is clipped to the SG passband's
    elevation range."""
    _, quadrature = get_angular_quadrature(
        proton_params(temperature=1e10), response, Region.SUNGLASSES, 0.0
    )
    sg_el_lo, sg_el_hi = response.sg_passband.elevation_range
    assert quadrature.elevation_points.min() >= sg_el_lo
    assert quadrature.elevation_points.max() <= sg_el_hi


def test_sg_quadrature_has_21_nodes_in_each_angular_dimension(response):
    """The angular quadrature has (21, 21) Gauss-Legendre nodes."""
    _, quadrature = get_angular_quadrature(
        proton_params(), response, Region.SUNGLASSES, 0.0
    )
    assert quadrature.elevation_points.shape == (21,)
    assert quadrature.azimuth_points.shape == (21,)


def test_sin_cos_caches_match_their_angle_arrays(response):
    """The precomputed sin/cos arrays match the node angles."""
    _, quadrature = get_angular_quadrature(
        proton_params(), response, Region.SUNGLASSES, 0.0
    )
    elevation = np.radians(quadrature.elevation_points)
    azimuth = np.radians(quadrature.azimuth_points)
    np.testing.assert_allclose(quadrature.sin_elevation, np.sin(elevation))
    np.testing.assert_allclose(quadrature.cos_elevation, np.cos(elevation))
    np.testing.assert_allclose(quadrature.sin_azimuth, np.sin(azimuth))
    np.testing.assert_allclose(quadrature.cos_azimuth, np.cos(azimuth))


def test_sg_region_skipped_when_bulk_far_from_sg_band(response):
    """A bulk at azimuth 90 deg is 70 deg from the SG band, which a narrow
    Maxwellian can't bridge."""
    skip, quadrature = get_angular_quadrature(
        proton_params(velocity_xyz=VELOCITY_AT_PLUS_90_XYZ),
        response,
        Region.SUNGLASSES,
        0.0,
    )
    assert skip
    assert quadrature is None


# --- get_angular_quadrature: open aperture ------------------------------------


@pytest.mark.parametrize(
    ("velocity_xyz", "region", "azimuth_lo", "azimuth_hi"),
    [
        (
            VELOCITY_AT_PLUS_90_XYZ,
            Region.OPEN_APERTURE_POSITIVE,
            OA_AZIMUTH_INNER_DEG,
            OA_AZIMUTH_OUTER_DEG,
        ),
        (
            VELOCITY_AT_MINUS_90_XYZ,
            Region.OPEN_APERTURE_NEGATIVE,
            -OA_AZIMUTH_OUTER_DEG,
            -OA_AZIMUTH_INNER_DEG,
        ),
    ],
)
def test_oa_region_is_active_inside_its_band(
    response, velocity_xyz, region, azimuth_lo, azimuth_hi
):
    """A bulk at azimuth +/-90 deg activates the matching OA half, whose
    azimuth nodes all lie inside its band."""
    skip, quadrature = get_angular_quadrature(
        proton_params(velocity_xyz=velocity_xyz), response, region, 0.0
    )
    assert not skip
    assert quadrature.azimuth_points.min() >= azimuth_lo
    assert quadrature.azimuth_points.max() <= azimuth_hi


@pytest.mark.parametrize(
    ("velocity_xyz", "region"),
    [
        (VELOCITY_AT_PLUS_90_XYZ, Region.OPEN_APERTURE_NEGATIVE),
        (VELOCITY_AT_MINUS_90_XYZ, Region.OPEN_APERTURE_POSITIVE),
    ],
)
def test_opposite_oa_region_is_skipped(response, velocity_xyz, region):
    """A bulk at azimuth +/-90 deg is too far from the opposite OA half for a
    narrow Maxwellian to reach."""
    skip, quadrature = get_angular_quadrature(
        proton_params(velocity_xyz=velocity_xyz), response, region, 0.0
    )
    assert skip
    assert quadrature is None


# --- get_speed_quadrature -----------------------------------------------------
# The 15-node GL quadrature over the intersection of `bulk_speed +/- k vth` and
# the passband's speed range at the elevation.


def cold_params_at_center(response):
    """Cold plasma (480 K) at the passband center: `bulk +/- k sigma` binds."""
    return proton_params(
        velocity_xyz=(0.0, -response.central_speed, 0.0), temperature=480.0
    )


def test_speed_window_lies_inside_passband_at_elevation(response):
    """For warm plasma the passband binds, so every node lies inside
    ``[r_min, r_max] * v0`` at the elevation."""
    _, quadrature = get_speed_quadrature(
        proton_params(), response, Region.SUNGLASSES, 0.0
    )
    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(response.sg_passband, 0.0)
    assert quadrature.points.min() >= response.central_speed * ratio_lo
    assert quadrature.points.max() <= response.central_speed * ratio_hi


def test_cold_plasma_window_is_bulk_plus_minus_k_vth(response):
    """For cold plasma, every node lies inside `bulk +/- k sigma`, and the
    outermost nodes sit within ~2 km/s of its edges."""
    params = cold_params_at_center(response)
    sigma = temperature_to_thermal_speed(PROTON_MASS_KG, params.temperature)
    speed = bulk_speed(params)
    _, quadrature = get_speed_quadrature(params, response, Region.SUNGLASSES, 0.0)
    window_lo = speed - SPEED_HALF_WIDTH_VTH * sigma
    window_hi = speed + SPEED_HALF_WIDTH_VTH * sigma
    assert quadrature.points.min() >= window_lo
    assert quadrature.points.max() <= window_hi
    assert window_hi - quadrature.points.max() < 2.0
    assert quadrature.points.min() - window_lo < 2.0


def test_speed_quadrature_arrays_are_all_15_long(response):
    """Points, weights and ``speed_cubed_times_passband`` have 15 nodes."""
    _, quadrature = get_speed_quadrature(
        proton_params(), response, Region.SUNGLASSES, 0.0
    )
    assert quadrature.points.shape == (15,)
    assert quadrature.weights.shape == (15,)
    assert quadrature.speed_cubed_times_passband.shape == (15,)


def test_speed_quadrature_skips_when_window_misses_passband(response):
    """A bulk so far above the passband that `bulk - k sigma` exceeds its
    upper edge gives skip=True and no quadrature."""
    params = proton_params(velocity_xyz=(0.0, -1500.0, 0.0), temperature=10_000.0)
    skip, quadrature = get_speed_quadrature(params, response, Region.SUNGLASSES, 0.0)
    assert skip
    assert quadrature is None


def test_speed_quadrature_uses_region_specific_passband(response):
    """With a hot plasma so the passband binds in both regions, the SG and OA
    windows start at their own passband's lower edge, and differ."""
    params = proton_params(temperature=1_000_000.0)
    _, sg_quad = get_speed_quadrature(params, response, Region.SUNGLASSES, 0.0)
    _, oa_quad = get_speed_quadrature(
        params, response, Region.OPEN_APERTURE_POSITIVE, 0.0
    )
    sg_ratio_lo, _ = speed_ratio_range_at_elevation(response.sg_passband, 0.0)
    oa_ratio_lo, _ = speed_ratio_range_at_elevation(response.oa_passband, 0.0)
    assert sg_quad.points.min() >= response.central_speed * sg_ratio_lo
    assert oa_quad.points.min() >= response.central_speed * oa_ratio_lo
    # a region mix-up would make these equal
    assert sg_quad.points.min() != oa_quad.points.min()


def test_speed_cubed_times_passband_at_each_node(response):
    """``speed_cubed_times_passband`` equals ``v^3 P(theta, v / v0)`` at every
    node."""
    _, quadrature = get_speed_quadrature(
        proton_params(), response, Region.SUNGLASSES, 0.0
    )
    expected = [
        v**3
        * interpolate_passband(response.sg_passband, 0.0, v / response.central_speed)
        for v in quadrature.points
    ]
    np.testing.assert_allclose(quadrature.speed_cubed_times_passband, expected)
