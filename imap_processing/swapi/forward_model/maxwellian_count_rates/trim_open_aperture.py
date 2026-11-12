"""Trimming of the open-aperture azimuths (L3a ``trim_open_aperture.py``)."""

from __future__ import annotations

import math

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.forward_model.look_angles import velocity_to_look_angles
from imap_processing.swapi.forward_model.maxwellian_count_rates.params import (
    _MaxwellianParams,
    bulk_speed,
    thermal_speed,
)
from imap_processing.swapi.forward_model.maxwellian_count_rates.utils import (
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
    sw_params: _MaxwellianParams,
    min_elevation: float,
    max_elevation: float,
    azimuth_lo: float,
    azimuth_hi: float,
    sg_rate: float,
) -> tuple[float, float]:
    """
    Trim the azimuth range of an open-aperture half to where it sees counts.

    Parameters
    ----------
    response : StepResponse
        Instrument response at the ESA step.
    sw_params : _MaxwellianParams
        Maxwellian moments.
    min_elevation : float
        Lower elevation bound of the quadrature in deg.
    max_elevation : float
        Upper elevation bound of the quadrature in deg.
    azimuth_lo : float
        Lower azimuth bound of the half in deg.
    azimuth_hi : float
        Upper azimuth bound of the half in deg.
    sg_rate : float
        Rate in counts/s found in the sunglasses.

    Returns
    -------
    azimuth_lo : float
        Lower bound of the trimmed range in deg.
    azimuth_hi : float
        Upper bound of the trimmed range in deg; equal to ``azimuth_lo`` if
        the half is negligible.
    """
    if azimuth_hi <= azimuth_lo:
        return 0.0, 0.0

    v = sw_params.velocity_xyz
    _, bulk_el = velocity_to_look_angles(v[0], v[1], v[2])
    scan_elevation = min(max(bulk_el, min_elevation), max_elevation)
    scan_azimuths = np.linspace(azimuth_lo, azimuth_hi, OA_SCAN_RESOLUTION)
    transmission_x_maxwellian = _evaluate_oa_integrand_along_azimuth(
        response, sw_params, scan_azimuths, scan_elevation
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
        sw_params,
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
    sw_params: _MaxwellianParams,
    scan_azimuths: npt.NDArray[np.float64],
    scan_elevation: float,
) -> npt.NDArray[np.float64]:
    """
    Evaluate the transmission-weighted Maxwellian at the central speed.

    Parameters
    ----------
    response : StepResponse
        Instrument response at the ESA step.
    sw_params : _MaxwellianParams
        Maxwellian moments.
    scan_azimuths : numpy.ndarray
        Azimuths to evaluate at in deg.
    scan_elevation : float
        Elevation to evaluate at in deg.

    Returns
    -------
    numpy.ndarray
        Azimuthal transmission times the (unnormalized) Maxwellian at each
        azimuth.
    """
    sigma = thermal_speed(sw_params, response)
    speed = bulk_speed(sw_params)
    central_speed = response.central_speed
    v = sw_params.velocity_xyz
    bulk_az, bulk_el = velocity_to_look_angles(v[0], v[1], v[2])

    sin_bulk_el = math.sin(np.radians(bulk_el))
    cos_bulk_el = math.cos(np.radians(bulk_el))
    sin_scan_el = math.sin(np.radians(scan_elevation))
    cos_scan_el = math.cos(np.radians(scan_elevation))
    cos_view_to_bulk = sin_bulk_el * sin_scan_el + cos_bulk_el * cos_scan_el * np.cos(
        np.radians(scan_azimuths - bulk_az)
    )
    delta_v_sq = (
        central_speed**2 + speed**2 - 2.0 * central_speed * speed * cos_view_to_bulk
    )
    maxwellian = np.exp(-delta_v_sq / (2.0 * sigma**2))

    for i in range(scan_azimuths.shape[0]):
        maxwellian[i] *= interpolate_azimuthal_transmission(
            response.azimuthal_transmission,
            scan_azimuths[i],
        )

    return maxwellian


@numba.njit
def _oa_rate_upper_bound(
    response: StepResponse,
    sw_params: _MaxwellianParams,
    transmission_az_integral: float,
    min_elevation: float,
    max_elevation: float,
) -> float:
    """
    Bound from above the rate of an open-aperture half.

    Parameters
    ----------
    response : StepResponse
        Instrument response at the ESA step.
    sw_params : _MaxwellianParams
        Maxwellian moments.
    transmission_az_integral : float
        Azimuthal integral of the transmission-weighted Maxwellian in deg.
    min_elevation : float
        Lower elevation bound in deg.
    max_elevation : float
        Upper elevation bound in deg.

    Returns
    -------
    float
        Upper bound on the half's rate in counts/s.
    """
    central_speed = response.central_speed
    delta_theta_deg = max_elevation - min_elevation
    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(response.oa_passband, 0.0)
    delta_v = central_speed * (ratio_hi - ratio_lo)
    return float(
        count_rate_conversion_factor(sw_params, response)
        * central_speed**3
        * delta_theta_deg
        * delta_v
        * transmission_az_integral
    )
