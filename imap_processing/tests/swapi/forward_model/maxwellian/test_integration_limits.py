"""Placeholder: Ported from L3a's
``tests/swapi/l3a/science/solar_wind/test_integration_limits.py``."""

import numpy as np
import pytest

from imap_processing.swapi.constants import PROTON_MASS_KG
from imap_processing.swapi.forward_model.maxwellian.integration_limits import (
    angular_quadrature_rule,
    speed_quadrature_rule,
    speed_window_misses_passband,
)
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalRegion,
)
from imap_processing.swapi.forward_model.response.passband import (
    speed_ratio_range_at_elevation,
)
from imap_processing.tests.swapi.forward_model.maxwellian.conftest import (
    ESA_VOLTAGE_1KEV,
    VELOCITY_AT_MINUS_90_XYZ,
    VELOCITY_AT_PLUS_90_XYZ,
    proton_params,
    thermal_speed_to_temperature,
)

# Placeholder: default ``n_sigma`` of ``speed_window_misses_passband`` and
# ``speed_quadrature_rule``
N_SIGMA = 6.0

# Placeholder: AzimuthalRegion azimuth bands
SG_AZIMUTH_LO_DEG = -20.0
SG_AZIMUTH_HI_DEG = +20.0
OA_AZIMUTH_INNER_DEG = 20.0
OA_AZIMUTH_OUTER_DEG = 150.0


@pytest.fixture
def response(proton_measurement_responses):
    return proton_measurement_responses([ESA_VOLTAGE_1KEV])[0]


# Placeholder: --- speed_window_misses_passband ----------------------------------------


def test_window_misses_when_bulk_speed_below_passband(response):
    """Claude: A 100 km/s bulk puts the whole window below the passband; the overlap
    test below covers the window above it."""
    params = proton_params(velocity_xyz=(0.0, -100.0, 0.0))
    assert speed_window_misses_passband(params, response)


@pytest.mark.parametrize(
    ("k_sigma_km_s", "misses"),
    [
        # Placeholder: k sigma = 50 km/s reaches back into the passband
        (50.0, False),
        # Placeholder: k sigma = 5 km/s stays clear of it
        (5.0, True),
    ],
)
def test_window_overlap_scales_with_speed_half_width(response, k_sigma_km_s, misses):
    """Placeholder: With the bulk 30 km/s above the passband's upper edge, the window
    overlaps the passband only when ``N_SIGMA`` thermal speeds
    exceed 30 km/s."""
    bulk = response.central_speed * 1.1 + 30.0
    sigma = k_sigma_km_s / N_SIGMA
    params = proton_params(
        velocity_xyz=(0.0, -bulk, 0.0),
        temperature=thermal_speed_to_temperature(sigma, PROTON_MASS_KG),
    )
    assert speed_window_misses_passband(params, response) == misses


# Placeholder: --- get_angular_quadrature_rule: sunglasses -----------------------------


def test_sg_azimuth_window_clamps_to_plus_minus_20_for_broad_distribution(response):
    """Placeholder: A very hot plasma's 180 deg angular extent clamps the azimuth window
    to the SG band, with the outermost nodes hugging both edges."""
    rule = angular_quadrature_rule(
        proton_params(temperature=1e10), response, AzimuthalRegion.SUNGLASSES, 0.0
    )
    # Placeholder: GL nodes lie strictly inside the band
    np.testing.assert_array_less(rule.azimuth_points, SG_AZIMUTH_HI_DEG + 1e-9)
    np.testing.assert_array_less(SG_AZIMUTH_LO_DEG - 1e-9, rule.azimuth_points)
    # Placeholder: The outermost of 21 GL nodes sits ~0.125 deg inside the edge of a 40
    # deg window; 1.5 deg only catches a window that collapses or shifts away.
    assert rule.azimuth_points.max() > SG_AZIMUTH_HI_DEG - 1.5
    assert rule.azimuth_points.min() < SG_AZIMUTH_LO_DEG + 1.5


def test_sg_elevation_window_clamps_into_passband_elevation_range(response):
    """Placeholder: A hot plasma's saturated angular extent is clipped to the SG
    passband's elevation range."""
    rule = angular_quadrature_rule(
        proton_params(temperature=1e10), response, AzimuthalRegion.SUNGLASSES, 0.0
    )
    sg_el_lo, sg_el_hi = response.sg_passband.elevation_range
    assert rule.elevation_points.min() >= sg_el_lo
    assert rule.elevation_points.max() <= sg_el_hi


def test_sg_region_skipped_when_bulk_far_from_sg_band(response):
    """Placeholder: A bulk at azimuth 90 deg is 70 deg from the SG band, which a narrow
    Maxwellian can't bridge."""
    rule = angular_quadrature_rule(
        proton_params(velocity_xyz=VELOCITY_AT_PLUS_90_XYZ),
        response,
        AzimuthalRegion.SUNGLASSES,
        0.0,
    )
    assert rule is None


# Placeholder: --- get_angular_quadrature_rule: open aperture --------------------------


@pytest.mark.parametrize(
    ("velocity_xyz", "region", "azimuth_lo", "azimuth_hi"),
    [
        (
            VELOCITY_AT_PLUS_90_XYZ,
            AzimuthalRegion.OPEN_APERTURE_POSITIVE,
            OA_AZIMUTH_INNER_DEG,
            OA_AZIMUTH_OUTER_DEG,
        ),
        (
            VELOCITY_AT_MINUS_90_XYZ,
            AzimuthalRegion.OPEN_APERTURE_NEGATIVE,
            -OA_AZIMUTH_OUTER_DEG,
            -OA_AZIMUTH_INNER_DEG,
        ),
    ],
)
def test_oa_region_is_active_inside_its_band(
    response, velocity_xyz, region, azimuth_lo, azimuth_hi
):
    """Placeholder: A bulk at azimuth +/-90 deg activates the matching OA half, whose
    azimuth nodes all lie inside its band."""
    rule = angular_quadrature_rule(
        proton_params(velocity_xyz=velocity_xyz), response, region, 0.0
    )
    assert rule.azimuth_points.min() >= azimuth_lo
    assert rule.azimuth_points.max() <= azimuth_hi


@pytest.mark.parametrize(
    ("velocity_xyz", "region"),
    [
        (VELOCITY_AT_PLUS_90_XYZ, AzimuthalRegion.OPEN_APERTURE_NEGATIVE),
        (VELOCITY_AT_MINUS_90_XYZ, AzimuthalRegion.OPEN_APERTURE_POSITIVE),
    ],
)
def test_opposite_oa_region_is_skipped(response, velocity_xyz, region):
    """Placeholder: A bulk at azimuth +/-90 deg is too far from the opposite OA half for
    a narrow Maxwellian to reach."""
    rule = angular_quadrature_rule(
        proton_params(velocity_xyz=velocity_xyz), response, region, 0.0
    )
    assert rule is None


# Placeholder: --- get_speed_quadrature_rule -------------------------------------------


def cold_params_at_center(response):
    """
    Placeholder: Cold plasma (480 K) at the passband center: `bulk +/- k sigma` binds.
    """
    return proton_params(
        velocity_xyz=(0.0, -response.central_speed, 0.0), temperature=480.0
    )


def test_speed_window_lies_inside_passband_at_elevation(response):
    """Placeholder: For warm plasma the passband binds, so every node lies inside
    ``[r_min, r_max] * v0`` at the elevation."""
    rule = speed_quadrature_rule(
        proton_params(), response, AzimuthalRegion.SUNGLASSES, 0.0
    )
    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(response.sg_passband, 0.0)
    assert rule.points.min() >= response.central_speed * ratio_lo
    assert rule.points.max() <= response.central_speed * ratio_hi


def test_cold_plasma_window_is_bulk_plus_minus_k_vth(response):
    """Placeholder: For cold plasma, every node lies inside `bulk +/- k sigma`, and the
    outermost nodes sit within ~2 km/s of its edges."""
    params = cold_params_at_center(response)
    sigma = params.thermal_speed_sigma
    speed = params.bulk_speed
    rule = speed_quadrature_rule(params, response, AzimuthalRegion.SUNGLASSES, 0.0)
    window_lo = speed - N_SIGMA * sigma
    window_hi = speed + N_SIGMA * sigma
    assert rule.points.min() >= window_lo
    assert rule.points.max() <= window_hi
    assert window_hi - rule.points.max() < 2.0
    assert rule.points.min() - window_lo < 2.0


def test_speed_nodes_skips_when_window_misses_passband(response):
    """Placeholder: A bulk so far above the passband that `bulk - k sigma` exceeds its
    upper edge gives no quadrature rule."""
    params = proton_params(velocity_xyz=(0.0, -1500.0, 0.0), temperature=10_000.0)
    rule = speed_quadrature_rule(params, response, AzimuthalRegion.SUNGLASSES, 0.0)
    assert rule is None


def test_speed_nodes_uses_region_specific_passband(response):
    """Placeholder: With a hot plasma so the passband binds in both regions, the OA
    window starts at its own passband's lower edge, not the SG one."""
    params = proton_params(temperature=1_000_000.0)
    sg_rule = speed_quadrature_rule(params, response, AzimuthalRegion.SUNGLASSES, 0.0)
    oa_rule = speed_quadrature_rule(
        params, response, AzimuthalRegion.OPEN_APERTURE_POSITIVE, 0.0
    )
    oa_ratio_lo, _ = speed_ratio_range_at_elevation(response.oa_passband, 0.0)
    assert oa_rule.points.min() >= response.central_speed * oa_ratio_lo
    # Placeholder: a region mix-up would make these equal
    assert sg_rule.points.min() != oa_rule.points.min()
