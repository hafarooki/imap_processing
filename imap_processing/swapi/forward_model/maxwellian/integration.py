"""
Placeholder: Ideal coincidence rates and their Jacobian.

Ported from L3a ``forward_model.py``.
"""

from __future__ import annotations

import math

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import BOLTZMANN_CONSTANT_J_PER_K
from imap_processing.swapi.forward_model.maxwellian.integration_limits import (
    AngularQuadrature,
    get_angular_quadrature,
    get_speed_quadrature,
    speed_window_misses_passband,
)
from imap_processing.swapi.forward_model.maxwellian.utils import (
    count_rate_conversion_factor,
)
from imap_processing.swapi.forward_model.response.instrument_response import (
    METERS_PER_KILOMETER,
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
    Placeholder: Compute ideal coincidence rates of chunk samples and their Jacobian.

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
    bulk_speed = math.sqrt(
        velocity_xyz[0] ** 2 + velocity_xyz[1] ** 2 + velocity_xyz[2] ** 2
    )
    n = len(responses)
    rates = np.empty(n)
    jacobian = np.empty((n, 5))
    for i in range(n):
        response = responses[i]
        # Placeholder: the thermal speed depends on the response's species mass
        thermal_speed = temperature_to_thermal_speed(response.mass_kg, temperature)
        rates[i], jacobian[i] = calculate_integral(
            density, velocity_xyz, bulk_speed, thermal_speed, response
        )
    return rates, jacobian


@numba.njit
def temperature_to_thermal_speed(
    mass: float, temperature: float
) -> float:  # numpydoc ignore=PR01,RT01
    """Placeholder: Convert a temperature (K) to a thermal speed sqrt(kT/m) (km/s)."""
    return (
        math.sqrt(BOLTZMANN_CONSTANT_J_PER_K * temperature / mass)
        / METERS_PER_KILOMETER
    )


@numba.njit(fastmath=True)
def calculate_integral(
    density: float,
    velocity_xyz: npt.NDArray[np.float64],
    bulk_speed: float,
    thermal_speed: float,
    response: StepResponse,
) -> tuple[float, npt.NDArray[np.float64]]:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Compute the ideal rate (counts/s) at one ESA step and its Jacobian row.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Integral"): "The
    region sum includes the sunglasses (SG, |phi| <= 20 deg) and the two sides of the
    open aperture (OA+/-, 20 deg <= |phi| <= 150 deg)."
    """
    jacobian_row = np.zeros(5)
    if speed_window_misses_passband(bulk_speed, thermal_speed, response):
        return 0.0, jacobian_row

    rate = 0.0
    sg_rate = 0.0
    for region in (
        Region.SUNGLASSES,
        Region.OPEN_APERTURE_NEGATIVE,
        Region.OPEN_APERTURE_POSITIVE,
    ):
        angular_quadrature = get_angular_quadrature(
            density,
            velocity_xyz,
            bulk_speed,
            thermal_speed,
            response,
            region,
            sg_rate,
        )
        if angular_quadrature is not None:
            region_rate, region_jac = _integrate_region(
                density,
                velocity_xyz,
                bulk_speed,
                thermal_speed,
                response,
                region,
                angular_quadrature,
            )
            rate += region_rate
            jacobian_row += region_jac
            if region == Region.SUNGLASSES:
                sg_rate = region_rate
    return rate, jacobian_row


@numba.njit(fastmath=True)
def _integrate_region(
    density: float,
    velocity_xyz: npt.NDArray[np.float64],
    bulk_speed: float,
    thermal_speed: float,
    response: StepResponse,
    region: Region,
    angular_quadrature: AngularQuadrature,
) -> tuple[float, npt.NDArray[np.float64]]:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Integrate a region's rate (counts/s) and Jacobian over angle and speed.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Analytic Jacobian"):
    "Thus, the same integral used for C can evaluate all five columns of the Jacobian in
    one pass."
    """
    bulk_x = velocity_xyz[0]
    bulk_y = velocity_xyz[1]
    bulk_z = velocity_xyz[2]
    # Placeholder: hoisted out of the speed loop; numba doesn't do it on its own
    bulk_speed_squared = bulk_speed**2
    inverse_two_thermal_speed_squared = 1.0 / (2 * thermal_speed**2)

    rate_sum = 0.0
    log_temperature_sum = 0.0
    direction_speed_sum_x = 0.0
    direction_speed_sum_y = 0.0
    direction_speed_sum_z = 0.0
    for i_elevation in range(angular_quadrature.elevation_points.shape[0]):
        speed_quadrature = get_speed_quadrature(
            bulk_speed,
            thermal_speed,
            response,
            region,
            angular_quadrature.elevation_points[i_elevation],
        )
        if speed_quadrature is None:
            continue

        cos_elevation = angular_quadrature.cos_elevation[i_elevation]
        elevation_weight = (
            angular_quadrature.elevation_weights[i_elevation] * cos_elevation
        )
        # Placeholder: unit flow direction of particles seen at this look direction
        direction_z = -angular_quadrature.sin_elevation[i_elevation]
        for i_azimuth in range(angular_quadrature.azimuth_points.shape[0]):
            direction_x = -cos_elevation * angular_quadrature.sin_azimuth[i_azimuth]
            direction_y = -cos_elevation * angular_quadrature.cos_azimuth[i_azimuth]

            velocity_along_direction = (
                direction_x * bulk_x + direction_y * bulk_y + direction_z * bulk_z
            )

            rate_integral = 0.0
            log_temperature_jacobian_integral = 0.0
            velocity_along_direction_integral = 0.0
            for i_speed in range(speed_quadrature.points.shape[0]):
                speed = speed_quadrature.points[i_speed]
                exponent = (
                    speed**2 + bulk_speed_squared - 2 * speed * velocity_along_direction
                ) * inverse_two_thermal_speed_squared
                weighted_integrand = (
                    speed_quadrature.weights[i_speed]
                    * speed_quadrature.speed_cubed_times_passband[i_speed]
                    * math.exp(-exponent)
                )
                rate_integral += weighted_integrand

                log_temperature_jacobian_integral += weighted_integrand * exponent
                velocity_along_direction_integral += weighted_integrand * speed

            angular_weight = (
                elevation_weight
                * angular_quadrature.azimuth_weights[i_azimuth]
                * angular_quadrature.transmission_azimuth[i_azimuth]
            )
            rate_sum += angular_weight * rate_integral
            log_temperature_sum += angular_weight * log_temperature_jacobian_integral
            speed_weight = angular_weight * velocity_along_direction_integral
            direction_speed_sum_x += speed_weight * direction_x
            direction_speed_sum_y += speed_weight * direction_y
            direction_speed_sum_z += speed_weight * direction_z

    count_rate_factor = count_rate_conversion_factor(density, thermal_speed, response)
    rate = rate_sum * count_rate_factor

    jacobian = np.empty(5)
    jacobian[0] = rate
    jacobian[1] = log_temperature_sum * count_rate_factor - 1.5 * rate
    # Placeholder: the bulk term of d/dv sums to -bulk * rate_sum over all angles
    velocity_factor = count_rate_factor / thermal_speed**2
    jacobian[2] = (direction_speed_sum_x - bulk_x * rate_sum) * velocity_factor
    jacobian[3] = (direction_speed_sum_y - bulk_y * rate_sum) * velocity_factor
    jacobian[4] = (direction_speed_sum_z - bulk_z * rate_sum) * velocity_factor

    return rate, jacobian
