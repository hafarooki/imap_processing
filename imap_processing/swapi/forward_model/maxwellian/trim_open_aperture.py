"""
Placeholder: Trimming of the open-aperture azimuths.

Ported from L3a ``trim_open_aperture.py``.
"""

from __future__ import annotations

import math

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.forward_model.look_angles import velocity_to_look_angles
from imap_processing.swapi.forward_model.maxwellian.utils import (
    count_rate_conversion_factor,
)
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    interpolate_azimuthal_transmission,
)
from imap_processing.swapi.forward_model.response.instrument_response import (
    StepResponse,
)
from imap_processing.swapi.forward_model.response.passband import (
    speed_ratio_range_at_elevation,
)

OA_SCAN_THRESHOLD = 1e-6
OA_SCAN_RESOLUTION = 64
OA_SKIP_FRACTION = 1e-3


@numba.njit
def trim_open_aperture(
    response: StepResponse,
    density: float,
    velocity_xyz: npt.NDArray[np.float64],
    bulk_speed: float,
    thermal_speed: float,
    min_elevation: float,
    max_elevation: float,
    azimuth_lo: float,
    azimuth_hi: float,
    sg_rate: float,
) -> tuple[float, float]:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Trim an open-aperture half's azimuths (deg) to where it sees counts.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Angular Limits"):
    "For OA+/-, the azimuth window is further reduced with a 64-point scan of the
    azimuth-only integrand [...] The reduced window includes the range where h(phi) >
    10^-6 max h padded by one sample on each side."
    """
    if azimuth_hi <= azimuth_lo:
        return 0.0, 0.0

    v = velocity_xyz
    _, bulk_el = velocity_to_look_angles(v[0], v[1], v[2])
    scan_elevation = min(max(bulk_el, min_elevation), max_elevation)
    scan_azimuths = np.linspace(azimuth_lo, azimuth_hi, OA_SCAN_RESOLUTION)
    transmission_x_maxwellian = _evaluate_oa_integrand_along_azimuth(
        response,
        velocity_xyz,
        bulk_speed,
        thermal_speed,
        scan_azimuths,
        scan_elevation,
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
    upper_bound = _oa_rate_upper_bound(
        response,
        density,
        thermal_speed,
        transmission_az_integral,
        min_elevation,
        max_elevation,
    )
    if upper_bound < max(0.1, OA_SKIP_FRACTION * sg_rate):
        return 0.0, 0.0
    return scan_azimuths[lower_index], scan_azimuths[upper_index]


@numba.njit
def _evaluate_oa_integrand_along_azimuth(
    response: StepResponse,
    velocity_xyz: npt.NDArray[np.float64],
    bulk_speed: float,
    thermal_speed: float,
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
    v = velocity_xyz
    bulk_az, bulk_el = velocity_to_look_angles(v[0], v[1], v[2])

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
    maxwellian = np.exp(-delta_v_sq / (2.0 * thermal_speed**2))

    for i in range(scan_azimuths.shape[0]):
        maxwellian[i] *= interpolate_azimuthal_transmission(
            response.azimuthal_transmission,
            scan_azimuths[i],
        )

    return maxwellian


@numba.njit
def _oa_rate_upper_bound(
    response: StepResponse,
    density: float,
    thermal_speed: float,
    transmission_az_integral: float,
    min_elevation: float,
    max_elevation: float,
) -> float:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Bound from above the rate (counts/s) of an open-aperture half.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Angular Limits"):
    "The OA integral is skipped if the heuristic estimate of its contribution to the
    total integral [...] is below max(0.1 Hz, 10^-3 C_SG). Here Delta theta is the
    clamped OA elevation width (in units of radians) and Delta v = (r_max(0) -
    r_min(0))v_0 is the passband width at theta = 0."
    """
    central_speed = response.central_speed
    delta_theta_deg = max_elevation - min_elevation
    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(response.oa_passband, 0.0)
    delta_v = central_speed * (ratio_hi - ratio_lo)
    return float(
        count_rate_conversion_factor(density, thermal_speed, response)
        * central_speed**3
        * delta_theta_deg
        * delta_v
        * transmission_az_integral
    )
