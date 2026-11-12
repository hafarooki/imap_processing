"""Placeholder: Ported from L3a's
``tests/swapi/l3a/science/solar_wind/test_trim_open_aperture.py``."""

import math

import numpy as np
import pytest

from imap_processing.swapi.forward_model.maxwellian.integration import (
    OA_SKIP_FLOOR_COUNTS_PER_S,
)
from imap_processing.swapi.forward_model.maxwellian.integration_limits import (
    OA_SCAN_RESOLUTION,
    OA_SCAN_THRESHOLD,
    AngularWindow,
    trim_open_aperture,
)
from imap_processing.tests.swapi.forward_model.maxwellian.conftest import (
    VELOCITY_AT_MINUS_90_XYZ,
    VELOCITY_AT_PLUS_90_XYZ,
    count_rate_factor,
    proton_params,
    synthetic_step_response,
)


def params_at_elevation(speed, elevation_deg, temperature=1.0e5):
    """
    Placeholder: Protons flowing in at look azimuth +90 deg and the given elevation.
    """
    elevation = math.radians(elevation_deg)
    return proton_params(
        velocity_xyz=(
            -speed * math.cos(elevation),
            0.0,
            -speed * math.sin(elevation),
        ),
        temperature=temperature,
    )


def trim(
    response,
    params,
    *,
    azimuth_lo,
    azimuth_hi,
    min_elevation=-10.0,
    max_elevation=10.0,
):
    """Placeholder: Return the trimmed (lo, hi) and the rate bound in counts/s."""
    window = AngularWindow(min_elevation, max_elevation, azimuth_lo, azimuth_hi)
    lo, hi, integral_bound = trim_open_aperture(params, response, window)
    rate_bound = integral_bound * count_rate_factor(params, response)
    return (lo, hi), rate_bound


@pytest.fixture
def response():
    return synthetic_step_response()


def test_cold_boresight_bulk_bound_is_below_skip_floor(response):
    """Placeholder: A cold boresight bulk far from the OA bounds the OA rate below the
    0.1 counts/s skip floor."""
    _, rate_bound = trim(
        response,
        proton_params(temperature=1.0e4),
        azimuth_lo=20.0,
        azimuth_hi=150.0,
    )
    assert 0.0 <= rate_bound < OA_SKIP_FLOOR_COUNTS_PER_S


def test_aligned_bulk_bound_is_above_skip_floor(response):
    """Placeholder: A 5 cm^-3, 100 kK plasma well aligned with OA+ easily clears the
    0.1 counts/s skip floor."""
    _, rate_bound = trim(
        response,
        proton_params(velocity_xyz=VELOCITY_AT_PLUS_90_XYZ),
        azimuth_lo=20.0,
        azimuth_hi=150.0,
    )
    assert rate_bound > OA_SKIP_FLOOR_COUNTS_PER_S


@pytest.fixture
def window_at_plus_90(response):
    window, _ = trim(
        response,
        proton_params(velocity_xyz=VELOCITY_AT_PLUS_90_XYZ),
        azimuth_lo=20.0,
        azimuth_hi=150.0,
    )
    return window


def test_endpoints_expand_one_node_past_threshold_crossings(
    response, window_at_plus_90
):
    """Placeholder: The window extends one scan node past the first and last nodes where
    the integrand exceeds ``OA_SCAN_THRESHOLD`` times its max."""
    # Placeholder: The transmission is 1 on the OA plateau [31 deg, 115 deg] and in the
    # synthetic table, so the integrand is the Maxwellian centered at the bulk azimuth,
    # recomputed here.
    params = proton_params(velocity_xyz=VELOCITY_AT_PLUS_90_XYZ)
    scan = np.linspace(20.0, 150.0, OA_SCAN_RESOLUTION)
    sigma = params.thermal_speed
    speed = params.bulk_speed
    central = response.central_speed
    delta_v_sq = (
        central**2 + speed**2 - 2.0 * central * speed * np.cos(np.radians(scan - 90.0))
    )
    integrand = np.exp(-delta_v_sq / (2.0 * sigma**2))
    above = np.flatnonzero(integrand > OA_SCAN_THRESHOLD * integrand.max())
    assert above.size > 0
    lo, hi = window_at_plus_90
    assert lo == pytest.approx(scan[max(above[0] - 1, 0)])
    assert hi == pytest.approx(scan[min(above[-1] + 1, OA_SCAN_RESOLUTION - 1)])


def test_mirror_symmetric_bulk_yields_mirror_symmetric_windows(
    response, window_at_plus_90
):
    """
    Placeholder: Bulks at +/-90 deg give OA+ and OA- windows that mirror each other.
    """
    lo_pos, hi_pos = window_at_plus_90
    (lo_neg, hi_neg), _ = trim(
        response,
        proton_params(velocity_xyz=VELOCITY_AT_MINUS_90_XYZ),
        azimuth_lo=-150.0,
        azimuth_hi=-20.0,
    )
    # Placeholder: linspace(20, 150, N) mirrors linspace(-150, -20, N) up to one ULP
    np.testing.assert_allclose(lo_neg, -hi_pos, atol=1e-12, rtol=0)
    np.testing.assert_allclose(hi_neg, -lo_pos, atol=1e-12, rtol=0)


def test_bulk_elevation_above_range_yields_non_empty_window(response):
    """Placeholder: A bulk at +20 deg elevation, above the +10 deg cap, still gives a
    window once the scan elevation is clamped to +10 deg."""
    (lo, hi), _ = trim(
        response,
        params_at_elevation(speed=450.0, elevation_deg=20.0),
        azimuth_lo=20.0,
        azimuth_hi=150.0,
    )
    assert hi > lo
