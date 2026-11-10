"""
Forward model of SWAPI count rates for a Maxwellian solar wind.

The kernel was ported from ``swapi_tools.solar_wind.solar_wind_model``, which in
turn was vendored from imap_l3_processing.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import (
    BOLTZMANN_CONSTANT_J_PER_K,
    PASSBAND_SPEED_RATIOS,
)
from imap_processing.swapi.forward_model.azimuthal_transmission import (
    interpolate_azimuthal_transmission,
)
from imap_processing.swapi.forward_model.instrument_response import (
    METERS_PER_KILOMETER,
    Region,
    StepResponse,
    StepResponses,
    azimuth_range,
    flatten_step_responses,
    passband_for,
    step_response_at,
    unflatten_step_responses,
)
from imap_processing.swapi.forward_model.look_angles import (
    velocity_to_look_angles,
)
from imap_processing.swapi.forward_model.passband import (
    interpolate_passband,
    speed_ratio_range_at_elevation,
)

CENTIMETERS_PER_KILOMETER = 1e5

# columns of the Jacobian returned by maxwellian_count_rates
JACOBIAN_COLUMNS = (
    "log_density",
    "log_temperature",
    "velocity_x",
    "velocity_y",
    "velocity_z",
)


class Jacobian(NamedTuple):
    """
    Derivatives of a count rate w.r.t. the Maxwellian moments.

    Attributes
    ----------
    d_log_density : float
        Derivative w.r.t. ln(density) in counts/s.
    d_log_temperature : float
        Derivative w.r.t. ln(temperature) in counts/s.
    d_velocity : numpy.ndarray
        Derivative w.r.t. the bulk velocity (x, y, z) in the SWAPI frame, in
        counts/s per km/s (3,).
    """

    d_log_density: float
    d_log_temperature: float
    d_velocity: npt.NDArray[np.float64]


class RateAndJacobian(NamedTuple):
    """
    A count rate and its derivatives w.r.t. the Maxwellian moments.

    Attributes
    ----------
    rate : float
        Ideal (pre-deadtime) coincidence rate in counts/s.
    jacobian : Jacobian
        Derivatives of ``rate`` w.r.t. the moments.
    """

    rate: float
    jacobian: Jacobian


class RatesAndJacobian(NamedTuple):
    """
    Count rates of a chunk's samples and their derivatives w.r.t. the moments.

    Attributes
    ----------
    rates : numpy.ndarray
        Ideal (pre-deadtime) coincidence rates in counts/s (n,).
    jacobian : numpy.ndarray
        Derivatives of ``rates`` (n, 5), with columns ``JACOBIAN_COLUMNS``:
        w.r.t. ln(density) and ln(temperature) in counts/s, then w.r.t. the
        SWAPI-frame bulk velocity (x, y, z) in counts/s per km/s.
    """

    rates: npt.NDArray[np.float64]
    jacobian: npt.NDArray[np.float64]


@numba.njit(parallel=True)  # type: ignore[untyped-decorator]
def maxwellian_count_rates(
    responses: StepResponses,
    density: float,
    temperature: float,
    velocity_xyz: npt.NDArray[np.float64],
) -> RatesAndJacobian:
    """
    Compute SWAPI's ideal coincidence rate at each sample of a chunk.

    Compiled with numba; one call evaluates every sample, in parallel across
    threads, so a fitter pays the Python-to-numba overhead once per evaluation.

    Parameters
    ----------
    responses : StepResponses
        Instrument response of each sample, from
        :meth:`~imap_processing.swapi.forward_model.response_tables.SwapiResponse.step_responses`.
    density : float
        Number density in cm^-3.
    temperature : float
        Temperature in K.
    velocity_xyz : numpy.ndarray
        Bulk flow velocity in the SWAPI frame of each sample in km/s (n, 3).
        A velocity shared by every sample can be passed as
        ``numpy.broadcast_to(velocity, (n, 3))``.

    Returns
    -------
    RatesAndJacobian
        Ideal (pre-deadtime) coincidence rate of each sample and its
        derivatives w.r.t. the moments, with the velocity derivatives in the
        SWAPI frame.

    Raises
    ------
    ValueError
        If ``velocity_xyz`` doesn't have one velocity per sample.
    """
    n_samples = responses.central_speed.shape[0]
    if velocity_xyz.shape[0] != n_samples or velocity_xyz.shape[1] != 3:
        raise ValueError("velocity_xyz must have shape (n_samples, 3).")

    # numba can't pass the nested tuples into the parallel loop as they are
    flat_responses = flatten_step_responses(responses)
    rates = np.empty(n_samples)
    jacobian = np.empty((n_samples, 5))
    for i in numba.prange(n_samples):
        rate, sample_jacobian = maxwellian_count_rate(
            step_response_at(unflatten_step_responses(flat_responses), i),
            density,
            temperature,
            velocity_xyz[i],
        )
        rates[i] = rate
        jacobian[i, 0] = sample_jacobian.d_log_density
        jacobian[i, 1] = sample_jacobian.d_log_temperature
        jacobian[i, 2:] = sample_jacobian.d_velocity
    return RatesAndJacobian(rates, jacobian)


@numba.njit(parallel=True)  # type: ignore[untyped-decorator]
def maxwellian_count_rates_at_temperatures(
    responses: StepResponses,
    density: float,
    temperatures: npt.NDArray[np.float64],
    velocity_xyz: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute SWAPI's ideal coincidence rate at each sample for many temperatures.

    Equivalent to :func:`maxwellian_count_rates` at each temperature, without
    the jacobian, in one parallel loop, so a grid of temperatures pays numba's
    parallel-launch overhead once rather than once per temperature.

    Parameters
    ----------
    responses : StepResponses
        Instrument response of each sample, from
        :meth:`~imap_processing.swapi.forward_model.response_tables.SwapiResponse.step_responses`.
    density : float
        Number density in cm^-3.
    temperatures : numpy.ndarray
        Temperatures in K (m,).
    velocity_xyz : numpy.ndarray
        Bulk flow velocity in the SWAPI frame of each sample in km/s (n, 3).

    Returns
    -------
    numpy.ndarray
        Ideal (pre-deadtime) coincidence rates in counts/s (m, n).

    Raises
    ------
    ValueError
        If ``velocity_xyz`` doesn't have one velocity per sample.
    """
    n_samples = responses.central_speed.shape[0]
    if velocity_xyz.shape[0] != n_samples or velocity_xyz.shape[1] != 3:
        raise ValueError("velocity_xyz must have shape (n_samples, 3).")

    flat_responses = flatten_step_responses(responses)
    rates = np.empty((temperatures.shape[0], n_samples))
    for j in numba.prange(rates.size):
        # prange's index is unsigned, and unsigned // signed is a float in numba
        i_temperature, i = divmod(np.int64(j), n_samples)
        rates[i_temperature, i] = maxwellian_count_rate(
            step_response_at(unflatten_step_responses(flat_responses), i),
            density,
            temperatures[i_temperature],
            velocity_xyz[i],
        )[0]
    return rates


@numba.njit(fastmath=True)  # type: ignore[untyped-decorator]
def maxwellian_count_rate(
    response: StepResponse,
    density: float,
    temperature: float,
    velocity_xyz: npt.NDArray[np.float64],
) -> RateAndJacobian:
    """
    Compute SWAPI's ideal coincidence rate for a Maxwellian solar wind.

    Compiled with numba, so a numba-compiled fitter can call it directly. The
    rate and its derivatives are summed over the aperture regions.

    Parameters
    ----------
    response : StepResponse
        Instrument response at the sample's ESA voltage, e.g. from
        :func:`~imap_processing.swapi.forward_model.instrument_response.step_response_at`.
        It carries the species.
    density : float
        Number density in cm^-3.
    temperature : float
        Temperature in K.
    velocity_xyz : numpy.ndarray
        Bulk flow velocity in the SWAPI frame in km/s, a float array (3,); see
        :func:`~imap_processing.swapi.forward_model.look_angles.look_angles_to_velocity`.

    Returns
    -------
    RateAndJacobian
        Ideal (pre-deadtime) coincidence rate and its derivatives w.r.t. the
        moments, with ``d_velocity`` in the SWAPI frame.
    """
    # Maxwellian standard deviation sqrt(kT/m) in km/s
    thermal_speed = (
        math.sqrt(BOLTZMANN_CONSTANT_J_PER_K * temperature / response.mass_kg)
        / METERS_PER_KILOMETER
    )
    # counts/s per unit of the (deg, deg, km/s) integral over the response
    count_rate_factor = (
        response.central_effective_area
        * density
        * (math.sqrt(2 * math.pi) * thermal_speed) ** -3
        * (math.pi / 180.0) ** 2
        * CENTIMETERS_PER_KILOMETER
    )
    bulk_speed = math.sqrt(
        velocity_xyz[0] ** 2 + velocity_xyz[1] ** 2 + velocity_xyz[2] ** 2
    )

    d_log_temperature = 0.0
    d_velocity = np.zeros(3)
    # the passband grid is zero outside its speed-ratio span at every
    # elevation, so a Maxwellian window that misses it gives no counts
    window = _SPEED_HALF_WIDTH_VTH * thermal_speed
    central_speed = response.central_speed
    if (
        bulk_speed + window < central_speed * _MIN_SPEED_RATIO
        or bulk_speed - window > central_speed * _MAX_SPEED_RATIO
    ):
        return RateAndJacobian(0.0, Jacobian(0.0, d_log_temperature, d_velocity))

    bulk_az, bulk_el = velocity_to_look_angles(
        velocity_xyz[0], velocity_xyz[1], velocity_xyz[2]
    )

    rate = 0.0
    sg_rate = 0.0
    for region in _REGIONS:
        angular_quadrature = _angular_quadrature(
            response,
            bulk_speed=bulk_speed,
            thermal_speed=thermal_speed,
            count_rate_factor=count_rate_factor,
            region=region,
            bulk_az=bulk_az,
            bulk_el=bulk_el,
            sg_rate=sg_rate,
            epsilon=_ANGULAR_EPSILON,
        )
        region_rate, region_d_log_temperature, region_d_velocity = _integrate_region(
            response,
            velocity_xyz=velocity_xyz,
            bulk_speed=bulk_speed,
            thermal_speed=thermal_speed,
            count_rate_factor=count_rate_factor,
            region=region,
            angular_quadrature=angular_quadrature,
            speed_half_width_vth=_SPEED_HALF_WIDTH_VTH,
        )
        rate += region_rate
        d_log_temperature += region_d_log_temperature
        d_velocity += region_d_velocity
        if region == Region.SUNGLASSES:
            sg_rate = region_rate
    # the rate is proportional to the density
    return RateAndJacobian(rate, Jacobian(rate, d_log_temperature, d_velocity))


_GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION = np.polynomial.legendre.leggauss(21)
_GL_NODES_AZIMUTH, _GL_WEIGHTS_AZIMUTH = np.polynomial.legendre.leggauss(21)
_GL_NODES_SPEED, _GL_WEIGHTS_SPEED = np.polynomial.legendre.leggauss(15)

# sunglasses first: its rate decides whether the open-aperture halves matter
_REGIONS = (
    Region.SUNGLASSES,
    Region.OPEN_APERTURE_NEGATIVE,
    Region.OPEN_APERTURE_POSITIVE,
)
_MIN_SPEED_RATIO = float(PASSBAND_SPEED_RATIOS[0])
_MAX_SPEED_RATIO = float(PASSBAND_SPEED_RATIOS[-1])
# fraction of the Maxwellian's peak below which look directions are dropped
_ANGULAR_EPSILON = 1e-6
# half-width of the speed window around the bulk speed, in thermal speeds
_SPEED_HALF_WIDTH_VTH = 6.0


class _AngularQuadrature(NamedTuple):
    """
    Elevation and azimuth quadrature over one aperture region.

    Attributes
    ----------
    elevation_points : numpy.ndarray
        Elevation nodes in deg.
    elevation_weights : numpy.ndarray
        Gauss-Legendre weights times cos(elevation), the solid-angle element.
    azimuth_points : numpy.ndarray
        Azimuth nodes in deg.
    azimuth_weights : numpy.ndarray
        Gauss-Legendre weights times the azimuthal transmission.
    """

    elevation_points: npt.NDArray[np.float64]
    elevation_weights: npt.NDArray[np.float64]
    azimuth_points: npt.NDArray[np.float64]
    azimuth_weights: npt.NDArray[np.float64]


@numba.njit  # type: ignore[untyped-decorator]
def _angular_quadrature(
    response: StepResponse,
    bulk_speed: float,
    thermal_speed: float,
    count_rate_factor: float,
    region: Region,
    bulk_az: float,
    bulk_el: float,
    sg_rate: float,
    epsilon: float,
) -> _AngularQuadrature:
    """
    Build the elevation and azimuth quadrature over one aperture region.

    The quadrature is limited to the cone around the bulk flow outside which
    the Maxwellian at the central speed is below ``epsilon`` of its peak.

    Parameters
    ----------
    response : StepResponse
        The sample's instrument response.
    bulk_speed : float
        Norm of ``velocity_xyz`` in km/s.
    thermal_speed : float
        Maxwellian standard deviation sqrt(kT/m) in km/s.
    count_rate_factor : float
        Counts/s per unit of the (deg, deg, km/s) integral over the response.
    region : Region
        Azimuthal region.
    bulk_az : float
        Look azimuth of the bulk flow in deg.
    bulk_el : float
        Look elevation of the bulk flow in deg.
    sg_rate : float
        Rate through the sunglasses in counts/s, for skipping negligible
        open-aperture halves.
    epsilon : float
        Fraction of the Maxwellian's peak below which look directions are
        dropped.

    Returns
    -------
    _AngularQuadrature
        The quadrature; empty if the region sees none of the Maxwellian.
    """
    cos_half_width = (
        thermal_speed**2 * np.log(epsilon) / (response.central_speed * bulk_speed) + 1
    )
    half_width = float(np.degrees(np.arccos(min(max(cos_half_width, -1), +1))))

    el_lo, el_hi = passband_for(response, region).elevation_range
    az_lo, az_hi = azimuth_range(region)

    min_el = min(max(bulk_el - half_width, el_lo), el_hi)
    max_el = min(max(bulk_el + half_width, el_lo), el_hi)
    min_az = min(max(bulk_az - half_width, az_lo), az_hi)
    max_az = min(max(bulk_az + half_width, az_lo), az_hi)
    if max_el > min_el and max_az > min_az and region != Region.SUNGLASSES:
        min_az, max_az = _trim_open_aperture(
            response,
            bulk_speed=bulk_speed,
            thermal_speed=thermal_speed,
            count_rate_factor=count_rate_factor,
            bulk_az=bulk_az,
            bulk_el=bulk_el,
            min_elevation=min_el,
            max_elevation=max_el,
            azimuth_lo=min_az,
            azimuth_hi=max_az,
            sg_rate=sg_rate,
        )
    if max_el <= min_el or max_az <= min_az:
        empty = np.empty(0)
        return _AngularQuadrature(empty, empty, empty, empty)

    elevation_points, elevation_weights = _rescale(
        _GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION, min_el, max_el
    )
    azimuth_points, azimuth_weights = _rescale(
        _GL_NODES_AZIMUTH, _GL_WEIGHTS_AZIMUTH, min_az, max_az
    )
    for i in range(azimuth_points.shape[0]):
        azimuth_weights[i] *= interpolate_azimuthal_transmission(
            response.azimuthal_transmission, azimuth_points[i]
        )
    return _AngularQuadrature(
        elevation_points,
        elevation_weights * np.cos(np.radians(elevation_points)),
        azimuth_points,
        azimuth_weights,
    )


@numba.njit  # type: ignore[untyped-decorator]
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


@numba.njit  # type: ignore[untyped-decorator]
def _trim_open_aperture(
    response: StepResponse,
    bulk_speed: float,
    thermal_speed: float,
    count_rate_factor: float,
    bulk_az: float,
    bulk_el: float,
    min_elevation: float,
    max_elevation: float,
    azimuth_lo: float,
    azimuth_hi: float,
    sg_rate: float,
    scan_threshold: float = 1e-6,
    scan_resolution: int = 64,
    skip_fraction: float = 1e-3,
) -> tuple[float, float]:
    """
    Trim the azimuth range of an open-aperture half to where it sees counts.

    Parameters
    ----------
    response : StepResponse
        The sample's instrument response.
    bulk_speed : float
        Norm of ``velocity_xyz`` in km/s.
    thermal_speed : float
        Maxwellian standard deviation sqrt(kT/m) in km/s.
    count_rate_factor : float
        Counts/s per unit of the (deg, deg, km/s) integral over the response.
    bulk_az : float
        Look azimuth of the bulk flow in deg.
    bulk_el : float
        Look elevation of the bulk flow in deg.
    min_elevation : float
        Lower elevation bound of the quadrature in deg.
    max_elevation : float
        Upper elevation bound of the quadrature in deg.
    azimuth_lo : float
        Lower azimuth bound of the half in deg.
    azimuth_hi : float
        Upper azimuth bound of the half in deg.
    sg_rate : float
        Rate through the sunglasses in counts/s.
    scan_threshold : float
        Fraction of the peak of the transmission-weighted Maxwellian, scanned
        at the bulk elevation, below which azimuths are dropped.
    scan_resolution : int
        Number of azimuths in the scan.
    skip_fraction : float
        Fraction of ``sg_rate`` below which an upper bound on the half's rate
        is negligible.

    Returns
    -------
    azimuth_lo : float
        Lower bound of the trimmed range in deg.
    azimuth_hi : float
        Upper bound of the trimmed range in deg; equal to ``azimuth_lo`` if
        the half is negligible.
    """
    sigma = thermal_speed
    speed = bulk_speed
    central_speed = response.central_speed

    scan_elevation = min(max(bulk_el, min_elevation), max_elevation)
    scan_azimuths = np.linspace(azimuth_lo, azimuth_hi, scan_resolution)
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
    transmission_x_maxwellian = np.exp(-delta_v_sq / (2.0 * sigma**2))
    for i in range(scan_azimuths.shape[0]):
        transmission_x_maxwellian[i] *= interpolate_azimuthal_transmission(
            response.azimuthal_transmission, scan_azimuths[i]
        )

    threshold = scan_threshold * np.max(transmission_x_maxwellian)
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
    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(response.oa_passband, 0.0)
    upper_bound = (
        count_rate_factor
        * central_speed**3
        * (max_elevation - min_elevation)
        * (central_speed * (ratio_hi - ratio_lo))
        * transmission_az_integral
    )
    if upper_bound < max(0.1, skip_fraction * sg_rate):
        return 0.0, 0.0
    return scan_azimuths[lower_index], scan_azimuths[upper_index]


@numba.njit(fastmath=True)  # type: ignore[untyped-decorator]
def _integrate_region(
    response: StepResponse,
    velocity_xyz: npt.NDArray[np.float64],
    bulk_speed: float,
    thermal_speed: float,
    count_rate_factor: float,
    region: Region,
    angular_quadrature: _AngularQuadrature,
    speed_half_width_vth: float,
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """
    Integrate one region's rate and derivatives over elevation, azimuth, speed.

    Parameters
    ----------
    response : StepResponse
        The sample's instrument response.
    velocity_xyz : numpy.ndarray
        Bulk flow in the SWAPI frame in km/s (3,).
    bulk_speed : float
        Norm of ``velocity_xyz`` in km/s.
    thermal_speed : float
        Maxwellian standard deviation sqrt(kT/m) in km/s.
    count_rate_factor : float
        Counts/s per unit of the (deg, deg, km/s) integral over the response.
    region : Region
        Azimuthal region.
    angular_quadrature : _AngularQuadrature
        Elevation and azimuth quadrature over the region.
    speed_half_width_vth : float
        Half-width of the speed window around the bulk speed, in thermal
        speeds.

    Returns
    -------
    rate : float
        The region's rate in counts/s.
    d_log_temperature : float
        Derivative of the rate w.r.t. ln(temperature) in counts/s.
    d_velocity : numpy.ndarray
        Derivative of the rate w.r.t. the SWAPI-frame flow in counts/s per
        km/s (3,).
    """
    bulk_x = velocity_xyz[0]
    bulk_y = velocity_xyz[1]
    bulk_z = velocity_xyz[2]
    sin_elevation = np.sin(np.radians(angular_quadrature.elevation_points))
    cos_elevations = np.cos(np.radians(angular_quadrature.elevation_points))
    sin_azimuth = np.sin(np.radians(angular_quadrature.azimuth_points))
    cos_azimuth = np.cos(np.radians(angular_quadrature.azimuth_points))

    elevation_rate = 0.0
    elevation_log_temperature_jacobian = 0.0
    elevation_jacobian_x = 0.0
    elevation_jacobian_y = 0.0
    elevation_jacobian_z = 0.0
    for i_elevation in range(angular_quadrature.elevation_points.shape[0]):
        speed_points, speed_weights = _speed_quadrature(
            response,
            bulk_speed=bulk_speed,
            thermal_speed=thermal_speed,
            region=region,
            elevation=angular_quadrature.elevation_points[i_elevation],
            speed_half_width_vth=speed_half_width_vth,
        )
        if speed_points.shape[0] == 0:
            continue

        cos_elevation = cos_elevations[i_elevation]
        azimuth_rate = 0.0
        azimuth_log_temperature_jacobian = 0.0
        azimuth_jacobian_x = 0.0
        azimuth_jacobian_y = 0.0
        azimuth_jacobian_z = 0.0
        for i_azimuth in range(angular_quadrature.azimuth_weights.shape[0]):
            # unit flow direction of particles seen at this look direction
            direction_x = -cos_elevation * sin_azimuth[i_azimuth]
            direction_y = -cos_elevation * cos_azimuth[i_azimuth]
            direction_z = -sin_elevation[i_elevation]

            velocity_along_direction = (
                direction_x * bulk_x + direction_y * bulk_y + direction_z * bulk_z
            )

            rate_integral = 0.0
            log_temperature_jacobian_integral = 0.0
            velocity_along_direction_integral = 0.0
            for i_speed in range(speed_points.shape[0]):
                speed = speed_points[i_speed]
                exponent = (
                    speed**2 + bulk_speed**2 - 2 * speed * velocity_along_direction
                ) / (2 * thermal_speed**2)
                weighted_integrand = speed_weights[i_speed] * math.exp(-exponent)
                rate_integral += weighted_integrand
                log_temperature_jacobian_integral += weighted_integrand * exponent
                velocity_along_direction_integral += weighted_integrand * speed

            azimuth_weight = angular_quadrature.azimuth_weights[i_azimuth]
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

        elevation_weight = angular_quadrature.elevation_weights[i_elevation]
        elevation_rate += azimuth_rate * elevation_weight
        elevation_log_temperature_jacobian += (
            azimuth_log_temperature_jacobian * elevation_weight
        )
        elevation_jacobian_x += azimuth_jacobian_x * elevation_weight
        elevation_jacobian_y += azimuth_jacobian_y * elevation_weight
        elevation_jacobian_z += azimuth_jacobian_z * elevation_weight

    rate = elevation_rate * count_rate_factor

    d_log_temperature = (
        elevation_log_temperature_jacobian * count_rate_factor - 1.5 * rate
    )
    velocity_factor = count_rate_factor / thermal_speed**2
    d_velocity = np.empty(3)
    d_velocity[0] = elevation_jacobian_x * velocity_factor
    d_velocity[1] = elevation_jacobian_y * velocity_factor
    d_velocity[2] = elevation_jacobian_z * velocity_factor
    return rate, d_log_temperature, d_velocity


@numba.njit  # type: ignore[untyped-decorator]
def _speed_quadrature(
    response: StepResponse,
    bulk_speed: float,
    thermal_speed: float,
    region: Region,
    elevation: float,
    speed_half_width_vth: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Build the speed quadrature of one region at one elevation.

    The quadrature spans the Maxwellian window around the bulk speed
    intersected with the region's passband at the elevation.

    Parameters
    ----------
    response : StepResponse
        The sample's instrument response.
    bulk_speed : float
        Norm of ``velocity_xyz`` in km/s.
    thermal_speed : float
        Maxwellian standard deviation sqrt(kT/m) in km/s.
    region : Region
        Azimuthal region.
    elevation : float
        Look elevation in deg.
    speed_half_width_vth : float
        Half-width of the speed window around the bulk speed, in thermal
        speeds.

    Returns
    -------
    points : numpy.ndarray
        Speed nodes in km/s; empty if the window misses the passband.
    weights : numpy.ndarray
        Weights, each carrying ``v^3 * passband(v / central_speed)``.
    """
    passband = passband_for(response, region)
    central_speed = response.central_speed
    sigma = thermal_speed
    bulk_v = bulk_speed

    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(passband, elevation)
    min_speed = max(bulk_v - speed_half_width_vth * sigma, central_speed * ratio_lo)
    max_speed = min(bulk_v + speed_half_width_vth * sigma, central_speed * ratio_hi)
    if max_speed <= min_speed:
        return np.empty(0), np.empty(0)

    points, weights = _rescale(_GL_NODES_SPEED, _GL_WEIGHTS_SPEED, min_speed, max_speed)
    for i, v in enumerate(points):
        weights[i] *= v**3 * interpolate_passband(
            passband, elevation, v / central_speed
        )
    return points, weights
