"""Quadrature limits of the response integral (L3a ``integration_limits.py``)."""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.forward_model.look_angles import velocity_to_look_angles
from imap_processing.swapi.forward_model.maxwellian_count_rates.params import (
    _MaxwellianParams,
    bulk_speed,
    thermal_speed,
)
from imap_processing.swapi.forward_model.maxwellian_count_rates.trim_open_aperture import (  # noqa: E501
    trim_open_aperture,
)
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    interpolate_azimuthal_transmission,
)
from imap_processing.swapi.forward_model.response.instrument_response import (
    Region,
    StepResponse,
    azimuth_range,
    passband_for,
)
from imap_processing.swapi.forward_model.response.passband import (
    interpolate_passband,
    speed_ratio_range_at_elevation,
)


class AngularQuadrature(NamedTuple):
    """Elevation and azimuth quadrature over one aperture region."""

    elevation_points: npt.NDArray[np.float64]
    elevation_weights: npt.NDArray[np.float64]
    azimuth_points: npt.NDArray[np.float64]
    azimuth_weights: npt.NDArray[np.float64]
    sin_elevation: npt.NDArray[np.float64]
    cos_elevation: npt.NDArray[np.float64]
    sin_azimuth: npt.NDArray[np.float64]
    cos_azimuth: npt.NDArray[np.float64]
    transmission_azimuth: npt.NDArray[np.float64]


class SpeedQuadrature(NamedTuple):
    """Speed quadrature of one region at one elevation."""

    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    speed_cubed_times_passband: npt.NDArray[np.float64]


_GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION = np.polynomial.legendre.leggauss(21)
_GL_NODES_AZIMUTH, _GL_WEIGHTS_AZIMUTH = np.polynomial.legendre.leggauss(21)
_GL_NODES_SPEED, _GL_WEIGHTS_SPEED = np.polynomial.legendre.leggauss(15)


EPSILON = 1e-6
SPEED_HALF_WIDTH_VTH = 6.0


@numba.njit
def speed_window_misses_passband(
    sw_params: _MaxwellianParams, response: StepResponse
) -> bool:  # numpydoc ignore=PR01,RT01
    """Check whether bulk speed +/- 6 thermal speeds misses the passband."""
    sigma = thermal_speed(sw_params, response)
    speed = bulk_speed(sw_params)

    sw_window_lo = speed - SPEED_HALF_WIDTH_VTH * sigma
    sw_window_hi = speed + SPEED_HALF_WIDTH_VTH * sigma

    # The passband grid is zero outside speed_ratio ∈ [0.9, 1.1] at every elevation
    central_speed = response.central_speed
    passband_lo = central_speed * 0.9
    passband_hi = central_speed * 1.1
    return bool(sw_window_hi < passband_lo or sw_window_lo > passband_hi)


@numba.njit
def _rescale(
    nodes: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    lower: float,
    upper: float,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64]
]:  # numpydoc ignore=PR01,RT01
    """Map a Gauss-Legendre rule from [-1, 1] onto [lower, upper]."""
    half = 0.5 * (upper - lower)
    mid = 0.5 * (upper + lower)
    return mid + half * nodes, half * weights


@numba.njit
def get_angular_quadrature(
    sw_params: _MaxwellianParams,
    response: StepResponse,
    region: Region,
    sg_rate: float,
) -> tuple[bool, AngularQuadrature | None]:  # numpydoc ignore=PR01,RT01
    """Build a region's angular quadrature, or flag the region as skipped."""
    min_el, max_el, min_az, max_az = _angular_limits(sw_params, region, response)
    if max_el <= min_el or max_az <= min_az:
        return True, None

    if region != Region.SUNGLASSES:
        min_az, max_az = trim_open_aperture(
            response,
            sw_params,
            min_el,
            max_el,
            min_az,
            max_az,
            sg_rate,
        )
        if max_az <= min_az:
            return True, None

    elevation_points, elevation_weights = _rescale(
        _GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION, min_el, max_el
    )
    azimuth_points, azimuth_weights = _rescale(
        _GL_NODES_AZIMUTH, _GL_WEIGHTS_AZIMUTH, min_az, max_az
    )
    transmission_azimuth = np.empty_like(azimuth_points)
    for i in range(azimuth_points.shape[0]):
        transmission_azimuth[i] = interpolate_azimuthal_transmission(
            response.azimuthal_transmission, azimuth_points[i]
        )
    return False, AngularQuadrature(
        elevation_points=elevation_points,
        elevation_weights=elevation_weights,
        azimuth_points=azimuth_points,
        azimuth_weights=azimuth_weights,
        sin_elevation=np.sin(np.radians(elevation_points)),
        cos_elevation=np.cos(np.radians(elevation_points)),
        sin_azimuth=np.sin(np.radians(azimuth_points)),
        cos_azimuth=np.cos(np.radians(azimuth_points)),
        transmission_azimuth=transmission_azimuth,
    )


@numba.njit
def get_speed_quadrature(
    sw_params: _MaxwellianParams,
    response: StepResponse,
    region: Region,
    elevation: float,
) -> tuple[bool, SpeedQuadrature | None]:  # numpydoc ignore=PR01,RT01
    """Build a region's speed quadrature at one elevation, or flag it skipped."""
    passband = passband_for(response, region)
    central_speed = response.central_speed
    sigma = thermal_speed(sw_params, response)
    bulk_v = bulk_speed(sw_params)

    # `bulk_v +/- k sigma` Maxwellian window intersected with the region's passband
    # at this elevation
    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(passband, elevation)
    passband_lo = central_speed * ratio_lo
    passband_hi = central_speed * ratio_hi
    min_speed = max(bulk_v - SPEED_HALF_WIDTH_VTH * sigma, passband_lo)
    max_speed = min(bulk_v + SPEED_HALF_WIDTH_VTH * sigma, passband_hi)
    if max_speed <= min_speed:
        return True, None

    points, weights = _rescale(_GL_NODES_SPEED, _GL_WEIGHTS_SPEED, min_speed, max_speed)

    speed_cubed_times_passband = np.empty_like(points)
    for i, v in enumerate(points):
        speed_cubed_times_passband[i] = v**3 * interpolate_passband(
            passband, elevation, v / central_speed
        )

    return False, SpeedQuadrature(points, weights, speed_cubed_times_passband)


@numba.njit
def _angular_limits(
    sw_params: _MaxwellianParams,
    region: Region,
    response: StepResponse,
) -> tuple[float, float, float, float]:  # numpydoc ignore=PR01,RT01
    """Find a region's (min_el, max_el, min_az, max_az) seeing the Maxwellian."""
    half_width = _maxwellian_angular_extent(sw_params, response, EPSILON)
    v = sw_params.velocity_xyz
    bulk_az, bulk_el = velocity_to_look_angles(v[0], v[1], v[2])

    el_lo, el_hi = passband_for(response, region).elevation_range
    az_lo, az_hi = azimuth_range(region)

    min_el, max_el = _clamp_window(bulk_el, half_width, el_lo, el_hi)
    min_az, max_az = _clamp_window(bulk_az, half_width, az_lo, az_hi)
    return min_el, max_el, min_az, max_az


@numba.njit
def _maxwellian_angular_extent(
    sw_params: _MaxwellianParams, response: StepResponse, epsilon: float
) -> float:  # numpydoc ignore=PR01,RT01
    """Find the cone half-width (deg) beyond which the Maxwellian is negligible."""
    sigma = thermal_speed(sw_params, response)
    speed = bulk_speed(sw_params)
    cos_theta = sigma**2 * np.log(epsilon) / (response.central_speed * speed) + 1
    return float(np.degrees(np.arccos(_clamp(cos_theta, -1, +1))))


@numba.njit
def _clamp(x: float, lower: float, upper: float) -> float:  # numpydoc ignore=PR01,RT01
    """Clamp a value to [lower, upper]."""
    return min(max(x, lower), upper)


@numba.njit
def _clamp_window(
    center: float, half_width: float, lower_bound: float, upper_bound: float
) -> tuple[float, float]:  # numpydoc ignore=PR01,RT01
    """Clamp the window center +/- half_width to [lower_bound, upper_bound]."""
    return (
        _clamp(center - half_width, lower_bound, upper_bound),
        _clamp(center + half_width, lower_bound, upper_bound),
    )
