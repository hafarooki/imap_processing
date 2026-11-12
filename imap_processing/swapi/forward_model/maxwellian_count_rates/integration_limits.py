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
) -> bool:
    """
    Check whether the Maxwellian's speed window misses the passband entirely.

    Parameters
    ----------
    sw_params : _MaxwellianParams
        Maxwellian moments.
    response : StepResponse
        Instrument response at the ESA step.

    Returns
    -------
    bool
        True if ``bulk speed +/- 6 thermal speeds`` doesn't overlap the
        passband's speed range.
    """
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
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Map a Gauss-Legendre rule from [-1, 1] onto [lower, upper].

    Parameters
    ----------
    nodes : numpy.ndarray
        Nodes on [-1, 1].
    weights : numpy.ndarray
        Weights on [-1, 1].
    lower : float
        Lower bound of the interval.
    upper : float
        Upper bound of the interval.

    Returns
    -------
    nodes : numpy.ndarray
        Nodes on [lower, upper].
    weights : numpy.ndarray
        Weights on [lower, upper].
    """
    half = 0.5 * (upper - lower)
    mid = 0.5 * (upper + lower)
    return mid + half * nodes, half * weights


@numba.njit
def get_angular_quadrature(
    sw_params: _MaxwellianParams,
    response: StepResponse,
    region: Region,
    sg_rate: float,
) -> tuple[bool, AngularQuadrature | None]:
    """
    Build the elevation and azimuth quadrature over one aperture region.

    Parameters
    ----------
    sw_params : _MaxwellianParams
        Maxwellian moments.
    response : StepResponse
        Instrument response at the ESA step.
    region : Region
        Azimuthal region.
    sg_rate : float
        Rate in counts/s already found in the sunglasses, used to skip
        negligible open-aperture halves.

    Returns
    -------
    skip_region : bool
        True if the region sees none of the Maxwellian.
    angular_quadrature : AngularQuadrature or None
        The quadrature, or None if ``skip_region``.
    """
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
) -> tuple[bool, SpeedQuadrature | None]:
    """
    Build the speed quadrature of one region at one elevation.

    The quadrature spans the Maxwellian window around the bulk speed
    intersected with the region's passband at the elevation.

    Parameters
    ----------
    sw_params : _MaxwellianParams
        Maxwellian moments.
    response : StepResponse
        Instrument response at the ESA step.
    region : Region
        Azimuthal region.
    elevation : float
        Look elevation in deg.

    Returns
    -------
    skip_elevation : bool
        True if the window misses the passband at this elevation.
    speed_quadrature : SpeedQuadrature or None
        The quadrature, or None if ``skip_elevation``.
    """
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
) -> tuple[float, float, float, float]:
    """
    Find the angular window of one region that can see the Maxwellian.

    Parameters
    ----------
    sw_params : _MaxwellianParams
        Maxwellian moments.
    region : Region
        Azimuthal region.
    response : StepResponse
        Instrument response at the ESA step.

    Returns
    -------
    min_el : float
        Lower elevation bound in deg.
    max_el : float
        Upper elevation bound in deg.
    min_az : float
        Lower azimuth bound in deg.
    max_az : float
        Upper azimuth bound in deg.
    """
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
) -> float:
    """
    Find the angle from the bulk flow beyond which the Maxwellian is negligible.

    Parameters
    ----------
    sw_params : _MaxwellianParams
        Maxwellian moments.
    response : StepResponse
        Instrument response at the ESA step.
    epsilon : float
        Fraction of the Maxwellian's peak, at the central speed, below which
        look directions are dropped.

    Returns
    -------
    float
        Half-width of the cone in deg.
    """
    sigma = thermal_speed(sw_params, response)
    speed = bulk_speed(sw_params)
    cos_theta = sigma**2 * np.log(epsilon) / (response.central_speed * speed) + 1
    return float(np.degrees(np.arccos(_clamp(cos_theta, -1, +1))))


@numba.njit
def _clamp(x: float, lower: float, upper: float) -> float:
    """
    Clamp a value to an interval.

    Parameters
    ----------
    x : float
        The value.
    lower : float
        Lower bound.
    upper : float
        Upper bound.

    Returns
    -------
    float
        ``x`` clamped to [lower, upper].
    """
    return min(max(x, lower), upper)


@numba.njit
def _clamp_window(
    center: float, half_width: float, lower_bound: float, upper_bound: float
) -> tuple[float, float]:
    """
    Clamp the window ``center +/- half_width`` to an interval.

    Parameters
    ----------
    center : float
        Center of the window.
    half_width : float
        Half-width of the window.
    lower_bound : float
        Lower bound of the interval.
    upper_bound : float
        Upper bound of the interval.

    Returns
    -------
    lower : float
        Lower edge of the clamped window.
    upper : float
        Upper edge of the clamped window.
    """
    return (
        _clamp(center - half_width, lower_bound, upper_bound),
        _clamp(center + half_width, lower_bound, upper_bound),
    )
