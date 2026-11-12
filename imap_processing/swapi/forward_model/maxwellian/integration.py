"""Optimized code to integrate the Maxwellian VDF with the instrument response."""

from __future__ import annotations

import math

import numba
import numpy as np
import numpy.typing as npt
from numba.typed.typedlist import List as TypedList

from imap_processing.swapi.forward_model.maxwellian.integration_limits import (
    AngularQuadratureRule,
    angular_quadrature_rule,
    speed_quadrature_rule,
    speed_window_misses_passband,
)
from imap_processing.swapi.forward_model.maxwellian.params import (
    MaxwellianParams,
    make_maxwellian,
)
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalRegion,
    interpolate_azimuthal_transmission,
)
from imap_processing.swapi.forward_model.response.measurement_species_response import (
    MeasurementSpeciesResponse,
)


@numba.njit
def maxwellian_count_rates(
    density: float,
    temperature: float,
    velocity_rtn_sc: npt.NDArray[np.float64],
    responses: TypedList[MeasurementSpeciesResponse],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the SWAPI model count rates from a Maxwellian distribution.

    This function computes the ideal (non-deadtime-corrected) coincidence rate
    contribution from a single Maxwellian distribution. It also computes the analytic
    Jacobian. It accounts for the 3D instrument response.

    Parameters
    ----------
    density : float
        Number density in cm^-3.
    temperature : float
        Temperature in K.
    velocity_rtn_sc : numpy.ndarray
        Bulk flow velocity relative to the spacecraft in RTN in km/s (3,).
    responses : numba.typed.List of MeasurementSpeciesResponse
        Instrument response at each sample's ESA step.

    Returns
    -------
    rates : numpy.ndarray
        Ideal (pre-deadtime) coincidence rates in counts/s (n,).
    jacobian : numpy.ndarray
        Derivatives of ``rates`` (n, 5) w.r.t. ln(density), ln(temperature)
        and the spacecraft-frame RTN bulk velocity (R, T, N).
    """
    if not isinstance(responses, TypedList):
        raise TypeError("responses must be a numba.typed.List")
    n = len(responses)
    rates = np.empty(n)
    jacobian = np.empty((n, 5))
    for i in range(n):
        response = responses[i]
        velocity_xyz = response.rtn_to_swapi @ velocity_rtn_sc
        params = make_maxwellian(density, temperature, velocity_xyz, response.mass_kg)
        rates[i], jacobian_row = _calculate_integral(params, response)
        jacobian[i, :2] = jacobian_row[:2]
        # chain rule through velocity_xyz = rtn_to_swapi @ velocity_rtn_sc
        # TODO double check this
        jacobian[i, 2:] = jacobian_row[2:] @ response.rtn_to_swapi
    return rates, jacobian


@numba.njit(fastmath=True)
def _calculate_integral(
    params: MaxwellianParams,
    response: MeasurementSpeciesResponse,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Compute the count rate [count/s] and Jacobian for one ESA step."""
    jacobian_row = np.zeros(5)
    if speed_window_misses_passband(params, response):
        return 0.0, jacobian_row

    # converts the response integral to counts/s
    count_rate_factor = (
        response.central_effective_area
        * params.density
        * (math.sqrt(2 * math.pi) * params.thermal_speed_sigma) ** -3
        * (math.pi / 180.0) ** 2
        * 1e5  # km -> cm
    )
    rate = 0.0
    sg_rate = 0.0
    for region in (
        AzimuthalRegion.SUNGLASSES,
        AzimuthalRegion.OPEN_APERTURE_NEGATIVE,
        AzimuthalRegion.OPEN_APERTURE_POSITIVE,
    ):
        angular_rule = angular_quadrature_rule(params, response, region, sg_rate)
        if angular_rule is None:
            continue
        region_integral, region_jac = _integrate_region(
            params, response, region, angular_rule
        )
        region_rate = region_integral * count_rate_factor
        rate += region_rate
        jacobian_row += region_jac * count_rate_factor
        if region == AzimuthalRegion.SUNGLASSES:
            sg_rate = region_rate
    return rate, jacobian_row


@numba.njit(fastmath=True)
def _integrate_region(
    params: MaxwellianParams,
    response: MeasurementSpeciesResponse,
    region: AzimuthalRegion,
    angular_rule: AngularQuadratureRule,
) -> tuple[float, npt.NDArray[np.float64]]:  # numpydoc ignore=PR01,RT01
    """Compute response integral and Jacobian for one azimuthal region."""
    bulk_x = params.velocity_xyz[0]
    bulk_y = params.velocity_xyz[1]
    bulk_z = params.velocity_xyz[2]
    bulk_speed = params.bulk_speed
    thermal_speed_sigma = params.thermal_speed_sigma
    bulk_speed_squared = bulk_speed**2
    inverse_two_thermal_speed_squared = 1.0 / (2 * thermal_speed_sigma**2)
    n_azimuth = angular_rule.azimuth_points.shape[0]

    azimuth_transmission = np.empty(n_azimuth)
    for i_azimuth in range(n_azimuth):
        azimuth_transmission[i_azimuth] = interpolate_azimuthal_transmission(
            response.azimuthal_transmission, angular_rule.azimuth_points[i_azimuth]
        )
    azimuth_radians = np.radians(angular_rule.azimuth_points)
    sin_azimuth = np.sin(azimuth_radians)
    cos_azimuth = np.cos(azimuth_radians)
    elevation_radians = np.radians(angular_rule.elevation_points)
    sin_elevation = np.sin(elevation_radians)
    cos_elevation_points = np.cos(elevation_radians)

    rate_sum = 0.0
    log_temperature_sum = 0.0
    direction_speed_sum_x = 0.0
    direction_speed_sum_y = 0.0
    direction_speed_sum_z = 0.0
    for i_elevation in range(angular_rule.elevation_points.shape[0]):
        speed_rule = speed_quadrature_rule(
            params,
            response,
            region,
            angular_rule.elevation_points[i_elevation],
        )
        if speed_rule is None:
            continue

        cos_elevation = cos_elevation_points[i_elevation]
        elevation_weight = angular_rule.elevation_weights[i_elevation] * cos_elevation
        direction_z = -sin_elevation[i_elevation]
        for i_azimuth in range(n_azimuth):
            direction_x = -cos_elevation * sin_azimuth[i_azimuth]
            direction_y = -cos_elevation * cos_azimuth[i_azimuth]

            velocity_along_direction = (
                direction_x * bulk_x + direction_y * bulk_y + direction_z * bulk_z
            )

            rate_integral = 0.0
            log_temperature_jacobian_integral = 0.0
            velocity_along_direction_integral = 0.0
            for i_speed in range(speed_rule.points.shape[0]):
                speed = speed_rule.points[i_speed]
                exponent = (
                    speed**2 + bulk_speed_squared - 2 * speed * velocity_along_direction
                ) * inverse_two_thermal_speed_squared
                weighted_integrand = (
                    speed_rule.weights[i_speed]
                    * speed_rule.speed_cubed_times_passband[i_speed]
                    * math.exp(-exponent)
                )
                rate_integral += weighted_integrand

                log_temperature_jacobian_integral += weighted_integrand * exponent
                velocity_along_direction_integral += weighted_integrand * speed

            angular_weight = (
                elevation_weight
                * angular_rule.azimuth_weights[i_azimuth]
                * azimuth_transmission[i_azimuth]
            )
            rate_sum += angular_weight * rate_integral
            log_temperature_sum += angular_weight * log_temperature_jacobian_integral
            speed_weight = angular_weight * velocity_along_direction_integral
            direction_speed_sum_x += speed_weight * direction_x
            direction_speed_sum_y += speed_weight * direction_y
            direction_speed_sum_z += speed_weight * direction_z

    jacobian = np.empty(5)
    jacobian[0] = rate_sum
    jacobian[1] = log_temperature_sum - 1.5 * rate_sum
    jacobian[2] = (direction_speed_sum_x - bulk_x * rate_sum) / thermal_speed_sigma**2
    jacobian[3] = (direction_speed_sum_y - bulk_y * rate_sum) / thermal_speed_sigma**2
    jacobian[4] = (direction_speed_sum_z - bulk_z * rate_sum) / thermal_speed_sigma**2

    return rate_sum, jacobian
