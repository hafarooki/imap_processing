"""Placeholder: Ported from L3a's
``tests/swapi/l3a/science/solar_wind/test_trim_open_aperture.py``."""

import math

import numpy as np
import pytest

from imap_processing.swapi.forward_model.maxwellian.trim_open_aperture import (
    OA_SCAN_RESOLUTION,
    OA_SCAN_THRESHOLD,
    trim_open_aperture,
)
from imap_processing.tests.swapi.forward_model.maxwellian.conftest import (
    proton_params,
    synthetic_step_response,
)

# Placeholder: flows whose look direction is at azimuth +90 deg and -90 deg, elevation 0
VELOCITY_AT_PLUS_90_XYZ = (-450.0, 0.0, 0.0)
VELOCITY_AT_MINUS_90_XYZ = (450.0, 0.0, 0.0)


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
    sg_rate,
    min_elevation=-10.0,
    max_elevation=10.0,
):
    return trim_open_aperture(
        response,
        params.density,
        params.velocity_xyz,
        params.bulk_speed,
        params.thermal_speed,
        min_elevation,
        max_elevation,
        azimuth_lo,
        azimuth_hi,
        sg_rate,
    )


@pytest.fixture
def response():
    return synthetic_step_response()


@pytest.mark.parametrize(("azimuth_lo", "azimuth_hi"), [(30.0, 30.0), (50.0, 20.0)])
def test_skips_when_azimuth_window_is_empty(response, azimuth_lo, azimuth_hi):
    """Placeholder: A zero-width or inverted azimuth window short-circuits to (0, 0)."""
    window = trim(
        response,
        proton_params(),
        azimuth_lo=azimuth_lo,
        azimuth_hi=azimuth_hi,
        sg_rate=10.0,
    )
    assert window == (0.0, 0.0)


@pytest.mark.parametrize(
    "sg_rate",
    [
        # Placeholder: the OA upper bound falls below 1e-3 sg_rate
        1.0e6,
        # Placeholder: the 0.1 counts/s absolute floor alone forces the skip
        0.0,
        # Placeholder: 1e-3 sg_rate = 0.05 < 0.1, so the absolute floor still applies
        50.0,
    ],
)
def test_skips_cold_boresight_bulk(response, sg_rate):
    """Placeholder: A cold boresight bulk far from the OA falls below the skip floor."""
    window = trim(
        response,
        proton_params(temperature=1.0e4),
        azimuth_lo=20.0,
        azimuth_hi=150.0,
        sg_rate=sg_rate,
    )
    assert window == (0.0, 0.0)


@pytest.fixture
def window_at_plus_90(response):
    # Placeholder: zero sg_rate keeps only the 0.1 counts/s floor, which a 5 cm^-3, 100
    # kK plasma well aligned with OA+ easily exceeds
    return trim(
        response,
        proton_params(velocity_xyz=VELOCITY_AT_PLUS_90_XYZ),
        azimuth_lo=20.0,
        azimuth_hi=150.0,
        sg_rate=0.0,
    )


def test_trimmed_window_is_inside_input_and_brackets_bulk(window_at_plus_90):
    """Placeholder: A bulk deep inside OA+ gives a non-empty window inside the input
    clamp that strictly contains the bulk azimuth."""
    lo, hi = window_at_plus_90
    assert hi > lo
    assert lo >= 20.0
    assert hi <= 150.0
    assert lo < 90.0 < hi


def test_trimmed_endpoints_lie_on_the_scan_grid(window_at_plus_90):
    """
    Placeholder: Both endpoints are nodes of ``linspace(lo, hi, OA_SCAN_RESOLUTION)``.
    """
    lo, hi = window_at_plus_90
    scan = np.linspace(20.0, 150.0, OA_SCAN_RESOLUTION)
    assert np.any(np.isclose(scan, lo))
    assert np.any(np.isclose(scan, hi))


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
    lo_neg, hi_neg = trim(
        response,
        proton_params(velocity_xyz=VELOCITY_AT_MINUS_90_XYZ),
        azimuth_lo=-150.0,
        azimuth_hi=-20.0,
        sg_rate=0.0,
    )
    # Placeholder: linspace(20, 150, N) mirrors linspace(-150, -20, N) up to one ULP
    np.testing.assert_allclose(lo_neg, -hi_pos, atol=1e-12, rtol=0)
    np.testing.assert_allclose(hi_neg, -lo_pos, atol=1e-12, rtol=0)


def test_high_sg_rate_forces_skip_for_otherwise_aligned_bulk(
    response, window_at_plus_90
):
    """Placeholder: A bulk that gives a window at sg_rate=0 is skipped once sg_rate is
    far above the OA upper bound."""
    lo_open, hi_open = window_at_plus_90
    assert hi_open > lo_open
    window = trim(
        response,
        proton_params(velocity_xyz=VELOCITY_AT_PLUS_90_XYZ),
        azimuth_lo=20.0,
        azimuth_hi=150.0,
        sg_rate=1.0e15,
    )
    assert window == (0.0, 0.0)


def test_bulk_elevation_above_range_yields_non_empty_window(response):
    """Placeholder: A bulk at +20 deg elevation, above the +10 deg cap, still gives a
    window once the scan elevation is clamped to +10 deg."""
    lo, hi = trim(
        response,
        params_at_elevation(speed=450.0, elevation_deg=20.0),
        azimuth_lo=20.0,
        azimuth_hi=150.0,
        sg_rate=0.0,
    )
    assert hi > lo
