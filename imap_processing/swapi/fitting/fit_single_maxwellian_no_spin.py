"""Fit a single proton Maxwellian flowing along the spin axis to SWAPI rates."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import (
    BOLTZMANN_CONSTANT_J_PER_K,
    PROTON_CHARGE_C,
)
from imap_processing.swapi.fitting.least_squares import (
    least_squares_with_jacobian,
    mean_residuals_and_jacobian,
)
from imap_processing.swapi.forward_model.deadtime import deadtime_factor
from imap_processing.swapi.forward_model.instrument_response import (
    InstrumentResponses,
)
from imap_processing.swapi.forward_model.maxwellian_count_rates import (
    maxwellian_count_rates,
)

# floor of the initial temperature guess, 1 eV in K
MIN_INITIAL_TEMPERATURE_K = PROTON_CHARGE_C / BOLTZMANN_CONSTANT_J_PER_K
# floor of the initial density guess in cm^-3
MIN_INITIAL_DENSITY = 0.01


def fit_single_maxwellian_no_spin(
    responses: InstrumentResponses,
    count_rates: npt.NDArray[np.floating],
    *,
    max_evaluations: int = 100,
) -> dict[str, float | npt.NDArray[np.float64]]:
    """
    Fit a chunk of sweeps with one proton Maxwellian aligned with the spin axis.

    The flow is along the boresight (-y in the SWAPI frame), so it is the same
    at every spin angle and every sweep sees the same rates. The model's
    post-deadtime rates are fit to the observed rates averaged over sweeps,
    with state (ln density, ln temperature, ln speed), so the optimizer's
    tolerances are relative for every moment.

    Parameters
    ----------
    responses : InstrumentResponses
        Proton instrument response of each sample of the chunk, flattened in C
        order from (sweep, step).
    count_rates : numpy.ndarray
        Observed coincidence rates in counts/s (sweep, step), NaN where invalid.
    max_evaluations : int
        Maximum number of model evaluations.

    Returns
    -------
    dict[str, float | numpy.ndarray]
        ``speed`` [km/s], ``density`` [cm^-3], ``temperature`` [K], and
        ``model_count_rates``, the post-deadtime model rates at the best fit
        [counts/s] shaped like ``count_rates``.

    Raises
    ------
    ValueError
        If there isn't one response per count rate.
    """
    rates = np.asarray(count_rates, dtype=float)
    if responses.central_speed.shape[0] != rates.size:
        raise ValueError(
            f"got {responses.central_speed.shape[0]} responses for "
            f"{rates.size} count rates"
        )

    result = least_squares_with_jacobian(
        mean_residuals_and_jacobian,
        _initial_state(responses, rates),
        args=(boresight_rates_and_jacobian, responses, rates),
        method="lm",
        xtol=1e-4,
        x_scale=1.0,
        max_nfev=max_evaluations,
    )

    log_density, log_temperature, log_speed = result.x
    ideal = boresight_rates_and_jacobian(responses, result.x)[0]
    return {
        "speed": float(np.exp(log_speed)),
        "density": float(np.exp(log_density)),
        "temperature": float(np.exp(log_temperature)),
        "model_count_rates": (ideal * deadtime_factor(ideal)).reshape(rates.shape),
    }


def boresight_rates_and_jacobian(
    responses: InstrumentResponses, state: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the ideal rates of a flow along the boresight and their jacobian.

    Parameters
    ----------
    responses : InstrumentResponses
        Instrument response of each sample (n,).
    state : numpy.ndarray
        (ln density, ln temperature, ln speed) with density in cm^-3,
        temperature in K and speed in km/s (3,).

    Returns
    -------
    rates : numpy.ndarray
        Ideal (pre-deadtime) rates in counts/s (n,).
    jacobian : numpy.ndarray
        Derivatives of ``rates`` w.r.t. ``state`` (n, 3).
    """
    log_density, log_temperature, log_speed = state
    speed = np.exp(log_speed)
    n_samples = responses.central_speed.shape[0]
    # the flow is (0, -speed, 0) in the SWAPI frame
    velocity = np.broadcast_to(np.array([0.0, -speed, 0.0]), (n_samples, 3))
    rates, jacobian = maxwellian_count_rates(
        responses, np.exp(log_density), np.exp(log_temperature), velocity
    )
    return rates, np.stack(
        # d/d(ln speed) = speed d/d(speed) = -speed d/d(v_y)
        [jacobian[:, 0], jacobian[:, 1], -speed * jacobian[:, 3]],
        axis=-1,
    )


def _initial_state(
    responses: InstrumentResponses, rates: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Guess the state from the peak of the observed rates.

    The speed is the central speed of the peak step and the temperature scales
    with the speed squared (60,000 K at 400 km/s, at least 1 eV). The density
    is the least-squares scale of the rates at unit density.

    Parameters
    ----------
    responses : InstrumentResponses
        Instrument response of each sample, flattened in C order from
        (sweep, step).
    rates : numpy.ndarray
        Observed coincidence rates in counts/s (sweep, step), NaN where invalid.

    Returns
    -------
    numpy.ndarray
        Initial (ln density, ln temperature, ln speed).
    """
    central_speeds = responses.central_speed.reshape(rates.shape).mean(axis=0)
    speed = central_speeds[np.nanargmax(np.nanmean(rates, axis=0))]
    temperature = max(60_000.0 * (speed / 400.0) ** 2, MIN_INITIAL_TEMPERATURE_K)

    unit_rates = boresight_rates_and_jacobian(
        responses, np.array([0.0, np.log(temperature), np.log(speed)])
    )[0]
    valid = np.isfinite(rates.ravel())
    observed, unit_rates = rates.ravel()[valid], unit_rates[valid]
    density = max(
        (observed @ unit_rates) / (unit_rates @ unit_rates), MIN_INITIAL_DENSITY
    )
    return np.array([np.log(density), np.log(temperature), np.log(speed)])
