"""Limits for integrating the Maxwellian VDF with the instrument response."""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.forward_model.maxwellian.params import MaxwellianParams
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalRegion,
    interpolate_azimuthal_transmission,
)
from imap_processing.swapi.forward_model.response.measurement_species_response import (
    MeasurementSpeciesResponse,
)
from imap_processing.swapi.forward_model.response.passband import (
    Passband,
    interpolate_passband,
    speed_ratio_range_at_elevation,
)


class AngularQuadratureRule(NamedTuple):
    """Elevation and azimuth [degrees] integration points and weights."""

    elevation_points: npt.NDArray[np.float64]
    elevation_weights: npt.NDArray[np.float64]
    azimuth_points: npt.NDArray[np.float64]
    azimuth_weights: npt.NDArray[np.float64]


class SpeedQuadratureRule(NamedTuple):
    """Speed [km/s] integration points and weights."""

    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    speed_cubed_times_passband: npt.NDArray[np.float64]


@numba.njit
def speed_window_misses_passband(
    params: MaxwellianParams,
    response: MeasurementSpeciesResponse,
    n_sigma: float = 6.0,
) -> bool:
    """
    Check whether the distribution is out of range entirely for a particular ESA step.

    Based on the bulk speed and n_sigma*sigma thermal width, determines if
    there is overlap with the ESA step's conservative passband range (0.9, 1.1)*v_0.
    This is separate from the per-elevation speed range of the passband, which is
    narrower.
    """
    half_width = n_sigma * params.thermal_speed_sigma
    sw_window_lo = params.bulk_speed - half_width
    sw_window_hi = params.bulk_speed + half_width

    central_speed = response.central_speed
    passband_lo = central_speed * 0.9
    passband_hi = central_speed * 1.1
    return bool(sw_window_hi < passband_lo or sw_window_lo > passband_hi)


@numba.njit
def angular_quadrature_rule(
    params: MaxwellianParams,
    response: MeasurementSpeciesResponse,
    region: AzimuthalRegion,
    sg_rate: float,
    oa_skip_floor_counts_per_s: float = 0.1,
    oa_skip_fraction: float = 1e-3,
) -> AngularQuadratureRule | None:
    """
    Compute the angular quadrature rule for a particular region.

    Returns None if integrating this region should be skipped because the estimated
    contribution to the integral is too low.
    """
    window = _angular_limits(params, response, region)
    min_el, max_el, min_az, max_az = window
    if max_el <= min_el or max_az <= min_az:
        return None

    if region != AzimuthalRegion.SUNGLASSES:
        oa_azimuths, oa_integrand = _trim_open_aperture(params, response, window)
        min_az, max_az = oa_azimuths[0], oa_azimuths[-1]
        if max_az <= min_az:
            return None
        rate_upper_bound = _oa_rate_upper_bound(
            params, response, window, oa_azimuths, oa_integrand
        )
        if rate_upper_bound < max(
            oa_skip_floor_counts_per_s, oa_skip_fraction * sg_rate
        ):
            return None

    elevation_points, elevation_weights = _rescale(
        _GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION, min_el, max_el
    )
    azimuth_points, azimuth_weights = _rescale(
        _GL_NODES_AZIMUTH, _GL_WEIGHTS_AZIMUTH, min_az, max_az
    )
    return AngularQuadratureRule(
        elevation_points=elevation_points,
        elevation_weights=elevation_weights,
        azimuth_points=azimuth_points,
        azimuth_weights=azimuth_weights,
    )


@numba.njit
def speed_quadrature_rule(
    params: MaxwellianParams,
    response: MeasurementSpeciesResponse,
    region: AzimuthalRegion,
    elevation: float,
    n_sigma: float = 6.0,
) -> SpeedQuadratureRule | None:
    """
    Compute the speed quadrature rule for a particular region and elevation.

    Returns None if the elevation should be skipped for this region.
    """
    passband = _passband_for(response, region)
    central_speed = response.central_speed

    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(passband, elevation)
    passband_lo = central_speed * ratio_lo
    passband_hi = central_speed * ratio_hi
    half_width = n_sigma * params.thermal_speed_sigma
    min_speed = max(params.bulk_speed - half_width, passband_lo)
    max_speed = min(params.bulk_speed + half_width, passband_hi)
    if max_speed <= min_speed:
        return None

    points, weights = _rescale(_GL_NODES_SPEED, _GL_WEIGHTS_SPEED, min_speed, max_speed)

    speed_cubed_times_passband = np.empty_like(points)
    for i, v in enumerate(points):
        speed_cubed_times_passband[i] = v**3 * interpolate_passband(
            passband, elevation, v / central_speed
        )

    return SpeedQuadratureRule(points, weights, speed_cubed_times_passband)


# Gauss-Legendre quadrature with fixed numbers of integration points,
# variable integration limits.
_GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION = np.polynomial.legendre.leggauss(21)
_GL_NODES_AZIMUTH, _GL_WEIGHTS_AZIMUTH = np.polynomial.legendre.leggauss(21)
_GL_NODES_SPEED, _GL_WEIGHTS_SPEED = np.polynomial.legendre.leggauss(15)


@numba.njit
def _rescale(
    nodes: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    lower: float,
    upper: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Scale the Gauss-Legendre quadrature rule for the integration limits."""
    half = 0.5 * (upper - lower)
    mid = 0.5 * (upper + lower)
    return mid + half * nodes, half * weights


class _AngularWindow(NamedTuple):
    """Elevation and azimuth [degrees] integration limits."""

    min_elevation: float
    max_elevation: float
    min_azimuth: float
    max_azimuth: float


@numba.njit
def _angular_limits(
    params: MaxwellianParams,
    response: MeasurementSpeciesResponse,
    region: AzimuthalRegion,
    epsilon: float = 1e-6,
) -> _AngularWindow:
    """
    Compute the angular integration limits.

    Accounts for the thermal width of the distribution and the instrument response for
    the given region. The window is the bulk direction +/- the half-width of the cone
    [degrees] beyond which the VDF falls below ``epsilon`` of its peak.
    """
    cos_theta = (
        params.thermal_speed_sigma**2
        * np.log(epsilon)
        / (response.central_speed * params.bulk_speed)
        + 1
    )
    half_width = float(np.degrees(np.arccos(_clamp(cos_theta, -1, +1))))

    el_lo, el_hi = _passband_for(response, region).elevation_range
    az_lo, az_hi = region.value

    return _AngularWindow(
        min_elevation=_clamp(params.bulk_elevation - half_width, el_lo, el_hi),
        max_elevation=_clamp(params.bulk_elevation + half_width, el_lo, el_hi),
        min_azimuth=_clamp(params.bulk_azimuth - half_width, az_lo, az_hi),
        max_azimuth=_clamp(params.bulk_azimuth + half_width, az_lo, az_hi),
    )


@numba.njit
def _clamp(x: float, lower: float, upper: float) -> float:
    """Clamp a value to [lower, upper]."""
    return min(max(x, lower), upper)


@numba.njit
def _trim_open_aperture(
    params: MaxwellianParams,
    response: MeasurementSpeciesResponse,
    window: _AngularWindow,
    oa_scan_resolution: int = 64,
    oa_scan_threshold: float = 1e-6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the reduced open aperture azimuthal integration range.

    Returns the scan azimuths [degrees] spanning the trimmed range and the approximate
    integrand at those azimuths.
    """
    scan_elevation = min(
        max(params.bulk_elevation, window.min_elevation), window.max_elevation
    )
    scan_azimuths = np.linspace(
        window.min_azimuth, window.max_azimuth, oa_scan_resolution
    )
    approximate_integrand = _evaluate_oa_integrand_along_azimuth(
        params, response, scan_azimuths, scan_elevation
    )

    threshold = oa_scan_threshold * np.max(approximate_integrand)
    n = approximate_integrand.shape[0]
    lower_index = 0
    for i in range(n):
        if approximate_integrand[i] > threshold:
            lower_index = max(i - 1, 0)
            break
    upper_index = n - 1
    for i in range(n - 1, -1, -1):
        if approximate_integrand[i] > threshold:
            upper_index = min(i + 1, n - 1)
            break

    return (
        scan_azimuths[lower_index : upper_index + 1],
        approximate_integrand[lower_index : upper_index + 1],
    )


@numba.njit
def _evaluate_oa_integrand_along_azimuth(
    params: MaxwellianParams,
    response: MeasurementSpeciesResponse,
    scan_azimuths: npt.NDArray[np.float64],
    scan_elevation: float,
) -> npt.NDArray[np.float64]:
    """Compute the approximate integrand as a function of azimuth."""
    central_speed = response.central_speed
    bulk_speed = params.bulk_speed
    thermal_speed_sigma = params.thermal_speed_sigma
    bulk_az = params.bulk_azimuth
    bulk_el = params.bulk_elevation

    sin_bulk_el = math.sin(np.radians(bulk_el))
    cos_bulk_el = math.cos(np.radians(bulk_el))
    sin_scan_el = math.sin(np.radians(scan_elevation))
    cos_scan_el = math.cos(np.radians(scan_elevation))
    cos_view_to_bulk = sin_bulk_el * sin_scan_el + cos_bulk_el * cos_scan_el * np.cos(
        np.radians(scan_azimuths - bulk_az)
    )
    delta_v_sq = (
        central_speed**2
        + bulk_speed**2
        - 2.0 * central_speed * bulk_speed * cos_view_to_bulk
    )
    integrand = np.exp(-delta_v_sq / (2.0 * thermal_speed_sigma**2))

    for i in range(scan_azimuths.shape[0]):
        integrand[i] *= interpolate_azimuthal_transmission(
            response.azimuthal_transmission,
            scan_azimuths[i],
        )

    return integrand


@numba.njit
def _oa_rate_upper_bound(
    params: MaxwellianParams,
    response: MeasurementSpeciesResponse,
    window: _AngularWindow,
    azimuths: npt.NDArray[np.float64],
    approximate_integrand: npt.NDArray[np.float64],
) -> float:
    """Heuristic estimate of the open aperture contribution [counts/s]."""
    transmission_az_integral = np.trapezoid(approximate_integrand, x=azimuths)
    central_speed = response.central_speed
    delta_theta_deg = window.max_elevation - window.min_elevation
    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(response.oa_passband, 0.0)
    delta_v = central_speed * (ratio_hi - ratio_lo)
    integral_upper_bound = (
        central_speed**3 * delta_theta_deg * delta_v * transmission_az_integral
    )
    count_rate_factor = (
        response.scaled_central_effective_area
        * params.density
        * (math.sqrt(2 * math.pi) * params.thermal_speed_sigma) ** -3
        * (math.pi / 180.0) ** 2
        * 1e5  # km -> cm
    )
    return float(count_rate_factor * integral_upper_bound)


@numba.njit
def _passband_for(
    response: MeasurementSpeciesResponse, region: AzimuthalRegion
) -> Passband:
    """Select passband by azimuthal region."""
    if region == AzimuthalRegion.SUNGLASSES:
        return response.sg_passband
    return response.oa_passband
