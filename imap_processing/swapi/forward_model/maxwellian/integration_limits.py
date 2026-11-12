"""
Placeholder: Quadrature limits of the response integral.

Ported from L3a ``integration_limits.py``, with L3a ``trim_open_aperture.py`` (the
open-aperture azimuth trim) folded in.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.forward_model.maxwellian.params import MaxwellianParams
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


class AngularWindow(NamedTuple):
    """Placeholder: Elevation and azimuth limits (deg) of a region's integral."""

    min_elevation: float
    max_elevation: float
    min_azimuth: float
    max_azimuth: float


class AngularQuadrature(NamedTuple):
    """Placeholder: Elevation and azimuth quadrature over one aperture region."""

    elevation_points: npt.NDArray[np.float64]
    elevation_weights: npt.NDArray[np.float64]
    azimuth_points: npt.NDArray[np.float64]
    azimuth_weights: npt.NDArray[np.float64]
    sin_elevation: npt.NDArray[np.float64]
    cos_elevation: npt.NDArray[np.float64]
    sin_azimuth: npt.NDArray[np.float64]
    cos_azimuth: npt.NDArray[np.float64]
    transmission_azimuth: npt.NDArray[np.float64]
    # Placeholder: upper bound on the region's integral before its conversion to
    # counts/s; inf for the sunglasses, which are never skipped
    integral_upper_bound: float


class SpeedQuadrature(NamedTuple):
    """Placeholder: Speed quadrature of one region at one elevation."""

    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    speed_cubed_times_passband: npt.NDArray[np.float64]


# Placeholder: From the L3a docs (``docs/swapi/solar-wind-forward-model.md``,
# "Integral"): "Each region is integrated using nested Gauss-Legendre quadrature with
# fixed numbers of integration points: (N_theta, N_phi, N_v) = (21, 21, 15)."
_GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION = np.polynomial.legendre.leggauss(21)
_GL_NODES_AZIMUTH, _GL_WEIGHTS_AZIMUTH = np.polynomial.legendre.leggauss(21)
_GL_NODES_SPEED, _GL_WEIGHTS_SPEED = np.polynomial.legendre.leggauss(15)


EPSILON = 1e-6
SPEED_HALF_WIDTH_VTH = 6.0
OA_SCAN_THRESHOLD = 1e-6
OA_SCAN_RESOLUTION = 64


@numba.njit
def speed_window_misses_passband(
    params: MaxwellianParams, response: StepResponse
) -> bool:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Check whether bulk speed +/- 6 thermal speeds misses the passband.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Speed Limits"): "The
    integral is skipped early if there is no overlap with the conservative range
    [0.9v_0, 1.1v_0]."
    """
    half_width = SPEED_HALF_WIDTH_VTH * params.thermal_speed
    sw_window_lo = params.bulk_speed - half_width
    sw_window_hi = params.bulk_speed + half_width

    # Placeholder: The passband grid is zero outside speed_ratio ∈ [0.9, 1.1] at every
    # elevation
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
    """Placeholder: Map a Gauss-Legendre rule from [-1, 1] onto [lower, upper]."""
    half = 0.5 * (upper - lower)
    mid = 0.5 * (upper + lower)
    return mid + half * nodes, half * weights


@numba.njit
def get_angular_quadrature(
    params: MaxwellianParams,
    response: StepResponse,
    region: Region,
) -> AngularQuadrature | None:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Build a region's angular quadrature, or None if the region is skipped.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Angular Limits"):
    "The azimuthal region is skipped if the window is clamped to zero. For OA+/-, the
    azimuth window is further reduced with a 64-point scan of the azimuth-only
    integrand"
    """
    window = _angular_limits(params, response, region)
    min_el, max_el, min_az, max_az = window
    if max_el <= min_el or max_az <= min_az:
        return None

    integral_upper_bound = np.inf
    if region != Region.SUNGLASSES:
        min_az, max_az, integral_upper_bound = trim_open_aperture(
            params, response, window
        )
        if max_az <= min_az:
            return None

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
    return AngularQuadrature(
        elevation_points=elevation_points,
        elevation_weights=elevation_weights,
        azimuth_points=azimuth_points,
        azimuth_weights=azimuth_weights,
        sin_elevation=np.sin(np.radians(elevation_points)),
        cos_elevation=np.cos(np.radians(elevation_points)),
        sin_azimuth=np.sin(np.radians(azimuth_points)),
        cos_azimuth=np.cos(np.radians(azimuth_points)),
        transmission_azimuth=transmission_azimuth,
        integral_upper_bound=integral_upper_bound,
    )


@numba.njit
def get_speed_quadrature(
    params: MaxwellianParams,
    response: StepResponse,
    region: Region,
    elevation: float,
) -> SpeedQuadrature | None:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Build a region's speed quadrature at one elevation, or None if skipped.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Speed Limits"): "The
    speed integral is limited to the overlap between the passband's speed width and the
    VDF-driven interval [v_b - Delta v_VDF, v_b + Delta v_VDF], Delta v_VDF = 6v_th,
    [...] Any theta where v_hi(theta) <= v_lo(theta) is skipped."
    """
    passband = passband_for(response, region)
    central_speed = response.central_speed

    # Placeholder: `bulk_speed +/- k thermal_speed` Maxwellian window intersected with
    # the region's passband at this elevation
    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(passband, elevation)
    passband_lo = central_speed * ratio_lo
    passband_hi = central_speed * ratio_hi
    half_width = SPEED_HALF_WIDTH_VTH * params.thermal_speed
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

    return SpeedQuadrature(points, weights, speed_cubed_times_passband)


@numba.njit
def _angular_limits(
    params: MaxwellianParams,
    response: StepResponse,
    region: Region,
) -> AngularWindow:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Find a region's (min_el, max_el, min_az, max_az) seeing the Maxwellian.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Angular Limits"):
    "The window is centered on the bulk-flow direction in the instrument frame. [...]
    [It] is then clamped to the passband elevation range and azimuth range for the
    region. The azimuthal region is skipped if the window is clamped to zero."
    """
    half_width = _maxwellian_angular_extent(params, response, EPSILON)

    el_lo, el_hi = passband_for(response, region).elevation_range
    az_lo, az_hi = azimuth_range(region)

    min_el, max_el = _clamp_window(params.bulk_elevation, half_width, el_lo, el_hi)
    min_az, max_az = _clamp_window(params.bulk_azimuth, half_width, az_lo, az_hi)
    return AngularWindow(min_el, max_el, min_az, max_az)


@numba.njit
def _maxwellian_angular_extent(
    params: MaxwellianParams, response: StepResponse, epsilon: float
) -> float:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Find the cone half-width (deg) past which the Maxwellian is negligible.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Angular Limits"):
    "The angular extent of the VDF's extent within a factor of epsilon = 10^-6 is given
    by Delta alpha = arccos(clamp(v_th^2 ln(epsilon) / (v_0 v_b) + 1; -1, 1))"
    """
    cos_theta = (
        params.thermal_speed**2
        * np.log(epsilon)
        / (response.central_speed * params.bulk_speed)
        + 1
    )
    return float(np.degrees(np.arccos(_clamp(cos_theta, -1, +1))))


@numba.njit
def _clamp(x: float, lower: float, upper: float) -> float:  # numpydoc ignore=PR01,RT01
    """Placeholder: Clamp a value to [lower, upper]."""
    return min(max(x, lower), upper)


@numba.njit
def _clamp_window(
    center: float, half_width: float, lower_bound: float, upper_bound: float
) -> tuple[float, float]:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Clamp the window center +/- half_width to [lower_bound, upper_bound].

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Angular Limits"):
    "[The rectangular window] is then clamped to the passband elevation range and
    azimuth range for the region."
    """
    return (
        _clamp(center - half_width, lower_bound, upper_bound),
        _clamp(center + half_width, lower_bound, upper_bound),
    )


# Placeholder: --- open-aperture azimuth trim (L3a ``trim_open_aperture.py``) ---


@numba.njit
def trim_open_aperture(
    params: MaxwellianParams,
    response: StepResponse,
    window: AngularWindow,
) -> tuple[float, float, float]:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Trim an open-aperture half's azimuths (deg) to where it sees counts.

    Also returns an upper bound on the half's response integral, in the units of the
    integral before its conversion to counts/s; the caller decides whether to
    skip the half from it. The caller must pass a non-empty azimuth window.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Angular Limits"):
    "For OA+/-, the azimuth window is further reduced with a 64-point scan of the
    azimuth-only integrand [...] The reduced window includes the range where h(phi) >
    10^-6 max h padded by one sample on each side."
    """
    scan_elevation = min(
        max(params.bulk_elevation, window.min_elevation), window.max_elevation
    )
    scan_azimuths = np.linspace(
        window.min_azimuth, window.max_azimuth, OA_SCAN_RESOLUTION
    )
    transmission_x_maxwellian = _evaluate_oa_integrand_along_azimuth(
        params, response, scan_azimuths, scan_elevation
    )

    threshold = OA_SCAN_THRESHOLD * np.max(transmission_x_maxwellian)
    n = transmission_x_maxwellian.shape[0]
    lower_index = 0
    for i in range(n):
        if transmission_x_maxwellian[i] > threshold:
            lower_index = max(i - 1, 0)
            break
    upper_index = n - 1
    for i in range(n - 1, -1, -1):
        if transmission_x_maxwellian[i] > threshold:
            upper_index = min(i + 1, n - 1)
            break

    dphi_deg = scan_azimuths[1] - scan_azimuths[0]
    transmission_az_integral = float(
        np.trapezoid(
            transmission_x_maxwellian[lower_index : upper_index + 1], dx=dphi_deg
        )
    )
    integral_bound = _oa_integral_upper_bound(
        response, window, transmission_az_integral
    )
    return scan_azimuths[lower_index], scan_azimuths[upper_index], integral_bound


@numba.njit
def _evaluate_oa_integrand_along_azimuth(
    params: MaxwellianParams,
    response: StepResponse,
    scan_azimuths: npt.NDArray[np.float64],
    scan_elevation: float,
) -> npt.NDArray[np.float64]:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Evaluate the transmission-weighted Maxwellian at the central speed.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Angular Limits"):
    "where v = v_0, theta_b' is the bulk elevation clamped into the OA passband's
    elevation range"
    """
    central_speed = response.central_speed
    bulk_speed = params.bulk_speed
    thermal_speed = params.thermal_speed
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
    integrand = np.exp(-delta_v_sq / (2.0 * thermal_speed**2))

    for i in range(scan_azimuths.shape[0]):
        integrand[i] *= interpolate_azimuthal_transmission(
            response.azimuthal_transmission,
            scan_azimuths[i],
        )

    return integrand


@numba.njit
def _oa_integral_upper_bound(
    response: StepResponse,
    window: AngularWindow,
    transmission_az_integral: float,
) -> float:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Bound from above an open-aperture half's integral, before conversion.

    ``calculate_integral`` multiplies it by the counts/s conversion factor.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Angular Limits"):
    "The OA integral is skipped if the heuristic estimate of its contribution to the
    total integral [...] is below max(0.1 Hz, 10^-3 C_SG). Here Delta theta is the
    clamped OA elevation width (in units of radians) and Delta v = (r_max(0) -
    r_min(0))v_0 is the passband width at theta = 0."
    """
    central_speed = response.central_speed
    delta_theta_deg = window.max_elevation - window.min_elevation
    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(response.oa_passband, 0.0)
    delta_v = central_speed * (ratio_hi - ratio_lo)
    return float(
        central_speed**3 * delta_theta_deg * delta_v * transmission_az_integral
    )
