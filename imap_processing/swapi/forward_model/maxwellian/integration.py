"""
Placeholder: Ideal coincidence rates and their Jacobian.

Ported from L3a ``forward_model.py``.
"""

from __future__ import annotations

import math

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.forward_model.maxwellian.distribution import (
    Maxwellian,
    make_maxwellian,
)
from imap_processing.swapi.forward_model.maxwellian.integration_limits import (
    AngularQuadrature,
    get_angular_quadrature,
    get_speed_quadrature,
    speed_window_misses_passband,
)
from imap_processing.swapi.forward_model.response.instrument_response import (
    Region,
    StepResponse,
)

# Placeholder: From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Angular
# Limits"): "The OA integral is skipped if the heuristic estimate of its contribution
# to the total integral [...] is below max(0.1 Hz, 10^-3 C_SG)."
OA_SKIP_FLOOR_COUNTS_PER_S = 0.1
OA_SKIP_FRACTION = 1e-3


@numba.njit
def maxwellian_count_rates(
    density: float,
    temperature: float,
    velocity_rtn_sc: npt.NDArray[np.float64],
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
    velocity_rtn_sc : numpy.ndarray
        Bulk flow velocity relative to the spacecraft in RTN in km/s (3,).
    responses : numba.typed.List[StepResponse]
        Instrument response at each sample's ESA step, e.g. from
        ``SwapiResponse.step_responses``.

    Returns
    -------
    rates : numpy.ndarray
        Ideal (pre-deadtime) coincidence rates in counts/s (n,).
    jacobian : numpy.ndarray
        Derivatives of ``rates`` (n, 5) w.r.t. ln(density), ln(temperature)
        and the spacecraft-frame RTN bulk velocity (R, T, N).
    """
    n = len(responses)
    rates = np.empty(n)
    jacobian = np.empty((n, 5))
    for i in range(n):
        response = responses[i]
        rotation = response.rtn_to_swapi
        velocity_xyz = _rotate(rotation, velocity_rtn_sc)
        # Placeholder: the thermal speed depends on the response's species mass
        maxwellian = make_maxwellian(
            density, temperature, velocity_xyz, response.mass_kg
        )
        rates[i], jacobian_row = calculate_integral(maxwellian, response)
        jacobian[i, :2] = jacobian_row[:2]
        # Claude: chain rule, d/dv_rtn = d/dv_xyz @ rtn_to_swapi
        for j in range(3):
            jacobian[i, 2 + j] = (
                jacobian_row[2] * rotation[0, j]
                + jacobian_row[3] * rotation[1, j]
                + jacobian_row[4] * rotation[2, j]
            )
    return rates, jacobian


@numba.njit
def _rotate(
    rotation: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:  # numpydoc ignore=PR01,RT01
    """
    Claude: Multiply a (3, 3) matrix by a (3,) vector.

    Written out so numba doesn't need BLAS for a 3 x 3 product.
    """
    rotated = np.empty(3)
    for i in range(3):
        rotated[i] = (
            rotation[i, 0] * vector[0]
            + rotation[i, 1] * vector[1]
            + rotation[i, 2] * vector[2]
        )
    return rotated


@numba.njit(fastmath=True)
def calculate_integral(
    maxwellian: Maxwellian,
    response: StepResponse,
) -> tuple[float, npt.NDArray[np.float64]]:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Compute the ideal rate (counts/s) at one ESA step and its Jacobian row.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Integral"): "The
    region sum includes the sunglasses (SG, |phi| <= 20 deg) and the two sides of the
    open aperture (OA+/-, 20 deg <= |phi| <= 150 deg)."
    """
    jacobian_row = np.zeros(5)
    if speed_window_misses_passband(maxwellian, response):
        return 0.0, jacobian_row

    # Placeholder: converts the response integral to counts/s
    count_rate_factor = (
        response.central_effective_area
        * maxwellian.density
        * (math.sqrt(2 * math.pi) * maxwellian.thermal_speed) ** -3
        * (math.pi / 180.0) ** 2
        * 1e5  # Placeholder: km -> cm
    )
    rate = 0.0
    sg_rate = 0.0
    for region in (
        Region.SUNGLASSES,
        Region.OPEN_APERTURE_NEGATIVE,
        Region.OPEN_APERTURE_POSITIVE,
    ):
        angular_quadrature = get_angular_quadrature(maxwellian, response, region)
        if angular_quadrature is None:
            continue
        if region != Region.SUNGLASSES and _skip_open_aperture(
            count_rate_factor * angular_quadrature.integral_upper_bound, sg_rate
        ):
            continue
        region_integral, region_jac = _integrate_region(
            maxwellian, response, region, angular_quadrature
        )
        region_rate = region_integral * count_rate_factor
        rate += region_rate
        jacobian_row += region_jac * count_rate_factor
        if region == Region.SUNGLASSES:
            sg_rate = region_rate
    return rate, jacobian_row


@numba.njit
def _skip_open_aperture(
    rate_upper_bound: float, sg_rate: float
) -> bool:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Whether an OA half's rate bound (counts/s) is too small to integrate.

    Kept out of the ``fastmath`` caller, which may assume away the NaNs that NaN
    moments produce here.
    """
    return rate_upper_bound < max(
        OA_SKIP_FLOOR_COUNTS_PER_S, OA_SKIP_FRACTION * sg_rate
    )


@numba.njit(fastmath=True)
def _integrate_region(
    maxwellian: Maxwellian,
    response: StepResponse,
    region: Region,
    angular_quadrature: AngularQuadrature,
) -> tuple[float, npt.NDArray[np.float64]]:  # numpydoc ignore=PR01,RT01
    """
    Placeholder: Integrate a region's response integral and Jacobian.

    Both are before the conversion to counts/s, which the caller applies.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Analytic Jacobian"):
    "Thus, the same integral used for C can evaluate all five columns of the Jacobian in
    one pass."
    """
    # Claude: unpacked into locals so the loops below read plain floats
    bulk_x = maxwellian.velocity_xyz[0]
    bulk_y = maxwellian.velocity_xyz[1]
    bulk_z = maxwellian.velocity_xyz[2]
    bulk_speed = maxwellian.bulk_speed
    thermal_speed = maxwellian.thermal_speed
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
            maxwellian,
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

    jacobian = np.empty(5)
    jacobian[0] = rate_sum
    jacobian[1] = log_temperature_sum - 1.5 * rate_sum
    # Placeholder: the bulk term of d/dv sums to -bulk * rate_sum over all angles
    inverse_thermal_var = 1.0 / thermal_speed**2
    jacobian[2] = (direction_speed_sum_x - bulk_x * rate_sum) * inverse_thermal_var
    jacobian[3] = (direction_speed_sum_y - bulk_y * rate_sum) * inverse_thermal_var
    jacobian[4] = (direction_speed_sum_z - bulk_z * rate_sum) * inverse_thermal_var

    return rate_sum, jacobian
