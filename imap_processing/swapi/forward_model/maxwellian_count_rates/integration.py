"""Ideal coincidence rates and their Jacobian (L3a ``forward_model.py``)."""

from __future__ import annotations

import math

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.forward_model.maxwellian_count_rates.integration_limits import (  # noqa: E501
    AngularQuadrature,
    get_angular_quadrature,
    get_speed_quadrature,
    speed_window_misses_passband,
)
from imap_processing.swapi.forward_model.maxwellian_count_rates.params import (
    _MaxwellianParams,
    bulk_speed,
    thermal_speed,
)
from imap_processing.swapi.forward_model.maxwellian_count_rates.utils import (
    count_rate_conversion_factor,
)
from imap_processing.swapi.forward_model.response.instrument_response import (
    Region,
    StepResponse,
)


@numba.njit
def maxwellian_count_rates(
    density: float,
    temperature: float,
    velocity_xyz: npt.NDArray[np.float64],
    responses: list[StepResponse],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the ideal coincidence rates of a chunk's samples and their Jacobian.

    Parameters
    ----------
    density : float
        Number density in cm^-3.
    temperature : float
        Temperature in K.
    velocity_xyz : numpy.ndarray
        Bulk flow velocity in the SWAPI frame in km/s (3,).
    responses : numba.typed.List[StepResponse]
        Instrument response at each sample's ESA step, e.g. from
        ``SwapiResponse.step_responses``.

    Returns
    -------
    rates : numpy.ndarray
        Ideal (pre-deadtime) coincidence rates in counts/s (n,).
    jacobian : numpy.ndarray
        Derivatives of ``rates`` (n, 5) w.r.t. ln(density), ln(temperature)
        and the SWAPI-frame bulk velocity (x, y, z).
    """
    sw_params = _MaxwellianParams(density, velocity_xyz, temperature)
    n = len(responses)
    rates = np.empty(n)
    jacobian = np.empty((n, 5))
    for i in range(n):
        rates[i], jacobian[i] = calculate_integral(sw_params, responses[i])
    return rates, jacobian


@numba.njit(fastmath=True)
def calculate_integral(
    sw_params: _MaxwellianParams, response: StepResponse
) -> tuple[float, npt.NDArray[np.float64]]:
    """
    Compute the ideal coincidence rate at one ESA step and its Jacobian.

    Parameters
    ----------
    sw_params : _MaxwellianParams
        Maxwellian moments.
    response : StepResponse
        Instrument response at the ESA step.

    Returns
    -------
    rate : float
        Ideal (pre-deadtime) coincidence rate in counts/s.
    jacobian_row : numpy.ndarray
        Derivatives of ``rate`` (5,) w.r.t. ln(density), ln(temperature) and
        the SWAPI-frame bulk velocity (x, y, z).
    """
    jacobian_row = np.zeros(5)
    if speed_window_misses_passband(sw_params, response):
        return 0.0, jacobian_row

    rate = 0.0
    sg_rate = 0.0
    for region in (
        Region.SUNGLASSES,
        Region.OPEN_APERTURE_NEGATIVE,
        Region.OPEN_APERTURE_POSITIVE,
    ):
        skip_region, angular_quadrature = get_angular_quadrature(
            sw_params, response, region, sg_rate
        )
        if not skip_region:
            region_rate, region_jac = _integrate_region(
                sw_params, response, region, angular_quadrature
            )
            rate += region_rate
            jacobian_row += region_jac
            if region == Region.SUNGLASSES:
                sg_rate = region_rate
    return rate, jacobian_row


@numba.njit(fastmath=True)
def _integrate_region(
    sw_params: _MaxwellianParams,
    response: StepResponse,
    region: Region,
    angular_quadrature: AngularQuadrature,
) -> tuple[float, npt.NDArray[np.float64]]:
    """
    Integrate one region's rate and Jacobian over elevation, azimuth and speed.

    Parameters
    ----------
    sw_params : _MaxwellianParams
        Maxwellian moments.
    response : StepResponse
        Instrument response at the ESA step.
    region : Region
        Azimuthal region.
    angular_quadrature : AngularQuadrature
        The region's elevation and azimuth quadrature.

    Returns
    -------
    rate : float
        The region's rate in counts/s.
    jacobian : numpy.ndarray
        Derivatives of ``rate`` (5,) w.r.t. ln(density), ln(temperature) and
        the SWAPI-frame bulk velocity (x, y, z).
    """
    velocity = sw_params.velocity_xyz
    bulk_x = velocity[0]
    bulk_y = velocity[1]
    bulk_z = velocity[2]

    elevation_rate = 0.0
    elevation_log_temperature_jacobian = 0.0
    elevation_jacobian_x = 0.0
    elevation_jacobian_y = 0.0
    elevation_jacobian_z = 0.0
    for i_elevation in range(angular_quadrature.elevation_points.shape[0]):
        skip_elevation, speed_quadrature = get_speed_quadrature(
            sw_params,
            response,
            region,
            angular_quadrature.elevation_points[i_elevation],
        )
        if skip_elevation:
            continue

        azimuth_rate = 0.0
        azimuth_log_temperature_jacobian = 0.0
        azimuth_jacobian_x = 0.0
        azimuth_jacobian_y = 0.0
        azimuth_jacobian_z = 0.0
        for i_azimuth in range(angular_quadrature.azimuth_points.shape[0]):
            # unit flow direction of particles seen at this look direction
            cos_elevation = angular_quadrature.cos_elevation[i_elevation]
            direction_x = -cos_elevation * angular_quadrature.sin_azimuth[i_azimuth]
            direction_y = -cos_elevation * angular_quadrature.cos_azimuth[i_azimuth]
            direction_z = -angular_quadrature.sin_elevation[i_elevation]

            velocity_along_direction = (
                direction_x * bulk_x + direction_y * bulk_y + direction_z * bulk_z
            )

            rate_integral = 0.0
            log_temperature_jacobian_integral = 0.0
            velocity_along_direction_integral = 0.0
            for i_speed in range(speed_quadrature.points.shape[0]):
                speed = speed_quadrature.points[i_speed]
                exponent = (
                    (
                        speed**2
                        + bulk_speed(sw_params) ** 2
                        - 2 * speed * velocity_along_direction
                    )
                    * 1.0
                    / (2 * thermal_speed(sw_params, response) ** 2)
                )
                weighted_integrand = (
                    speed_quadrature.weights[i_speed]
                    * speed_quadrature.speed_cubed_times_passband[i_speed]
                    * math.exp(-exponent)
                )
                rate_integral += weighted_integrand

                log_temperature_jacobian_integral += weighted_integrand * exponent
                velocity_along_direction_integral += weighted_integrand * speed

            azimuth_weight = (
                angular_quadrature.azimuth_weights[i_azimuth]
                * angular_quadrature.transmission_azimuth[i_azimuth]
            )

            azimuth_rate += azimuth_weight * rate_integral

            azimuth_log_temperature_jacobian += (
                azimuth_weight * log_temperature_jacobian_integral
            )
            azimuth_jacobian_x += azimuth_weight * (
                direction_x * velocity_along_direction_integral - bulk_x * rate_integral
            )
            azimuth_jacobian_y += azimuth_weight * (
                direction_y * velocity_along_direction_integral - bulk_y * rate_integral
            )
            azimuth_jacobian_z += azimuth_weight * (
                direction_z * velocity_along_direction_integral - bulk_z * rate_integral
            )

        elevation_weight = (
            angular_quadrature.elevation_weights[i_elevation]
            * angular_quadrature.cos_elevation[i_elevation]
        )
        elevation_rate += azimuth_rate * elevation_weight
        elevation_log_temperature_jacobian += (
            azimuth_log_temperature_jacobian * elevation_weight
        )
        elevation_jacobian_x += azimuth_jacobian_x * elevation_weight
        elevation_jacobian_y += azimuth_jacobian_y * elevation_weight
        elevation_jacobian_z += azimuth_jacobian_z * elevation_weight

    count_rate_factor = count_rate_conversion_factor(sw_params, response)
    rate = elevation_rate * count_rate_factor

    jacobian = np.empty(5)
    jacobian[0] = rate
    jacobian[1] = elevation_log_temperature_jacobian * count_rate_factor - 1.5 * rate
    velocity_factor = count_rate_factor / thermal_speed(sw_params, response) ** 2
    jacobian[2] = elevation_jacobian_x * velocity_factor
    jacobian[3] = elevation_jacobian_y * velocity_factor
    jacobian[4] = elevation_jacobian_z * velocity_factor

    return rate, jacobian
