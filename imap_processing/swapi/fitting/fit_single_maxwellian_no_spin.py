"""Fit a single proton Maxwellian flowing along the spin axis to SWAPI rates."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import (
    BOLTZMANN_CONSTANT_J_PER_K,
    PROTON_CHARGE_C,
    Species,
)
from imap_processing.swapi.fitting.least_squares import (
    least_squares_with_jacobian,
    mean_residuals_and_jacobian,
)
from imap_processing.swapi.forward_model.deadtime import deadtime_factor
from imap_processing.swapi.forward_model.instrument_response import (
    StepResponses,
)
from imap_processing.swapi.forward_model.maxwellian_count_rates import (
    maxwellian_count_rates,
)
from imap_processing.swapi.forward_model.response_tables import SwapiResponse

# floor of the initial temperature guess, 1 eV in K
MIN_INITIAL_TEMPERATURE_K = PROTON_CHARGE_C / BOLTZMANN_CONSTANT_J_PER_K
# floor of the initial density guess in cm^-3
MIN_INITIAL_DENSITY = 0.01


def fit_single_maxwellian_no_spin(
    esa_voltages: npt.NDArray[np.floating],
    count_rates: npt.NDArray[np.floating],
    *,
    swapi_response: SwapiResponse,
    epoch: int,
    max_evaluations: int = 100,
) -> dict[str, float | npt.NDArray[np.float64]]:
    """
    Fit a chunk of sweeps with one proton Maxwellian aligned with the spin axis.

    The flow is along the boresight (-y in the SWAPI frame), so it is the same
    at every spin angle and every sweep sees the same rates. The chunk is
    collapsed to one sweep by averaging the valid rates of each step over
    sweeps (see :func:`mean_sweep`), and the model's post-deadtime rates are fit
    to that mean sweep with state (ln density, ln temperature, ln speed), so
    the optimizer's tolerances are relative for every moment.

    Parameters
    ----------
    esa_voltages : numpy.ndarray
        ESA voltage magnitude of each sample in V (sweep, step); every
        sweep must have the same voltages.
    count_rates : numpy.ndarray
        Observed coincidence rates in counts/s (sweep, step), NaN where invalid.
    swapi_response : SwapiResponse
        SWAPI instrument response tables.
    epoch : int
        Time of the chunk in TT J2000 ns, for the relative efficiency lookup.
    max_evaluations : int
        Maximum number of model evaluations.

    Returns
    -------
    dict[str, float | numpy.ndarray]
        ``speed`` [km/s], ``density`` [cm^-3], ``temperature`` [K], and
        ``model_count_rates``, the post-deadtime model rates at the best fit
        [counts/s] shaped like ``count_rates`` (the same for every sweep).
    """
    sweep_voltages, rates = mean_sweep(esa_voltages, count_rates)
    responses = chunk_responses(sweep_voltages, rates, swapi_response, epoch)
    # the model is only evaluated at steps with a valid rate while fitting
    fit_steps = np.isfinite(rates[0])
    fit_responses = responses.subset(fit_steps)
    fit_rates = rates[:, fit_steps]

    result = least_squares_with_jacobian(
        mean_residuals_and_jacobian,
        _initial_state(fit_responses, fit_rates),
        args=(boresight_rates_and_jacobian, fit_responses, fit_rates),
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
        "model_count_rates": np.broadcast_to(
            ideal * deadtime_factor(ideal), np.shape(count_rates)
        ).copy(),
    }


def mean_sweep(
    esa_voltages: npt.ArrayLike, count_rates: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Collapse a chunk of sweeps with the same ESA voltages to their mean sweep.

    Parameters
    ----------
    esa_voltages : numpy.ndarray
        ESA voltage magnitude of each sample in V (sweep, step).
    count_rates : numpy.ndarray
        Observed coincidence rates in counts/s (sweep, step), NaN where invalid.

    Returns
    -------
    esa_voltages : numpy.ndarray
        ESA voltage magnitude of each step in V (1, step).
    count_rates : numpy.ndarray
        Mean of each step's valid rates in counts/s, NaN where a step has no
        valid rate (1, step).

    Raises
    ------
    ValueError
        If the shapes differ, aren't (sweep, step), or the sweeps have
        different ESA voltages.
    """
    voltages = np.asarray(esa_voltages, dtype=float)
    rates = np.asarray(count_rates, dtype=float)
    if voltages.shape != rates.shape or rates.ndim != 2:
        raise ValueError(
            f"expected ESA voltages and count rates of the same (sweep, step) "
            f"shape, got {voltages.shape} and {rates.shape}"
        )
    if not np.all(voltages == voltages[0]):
        raise ValueError("every sweep of the chunk must have the same ESA voltages")
    valid = np.isfinite(rates)
    n_valid = valid.sum(axis=0)
    total = np.where(valid, rates, 0.0).sum(axis=0)
    # steps with no valid rate stay NaN, without nanmean's empty-slice warning
    mean = np.divide(
        total, n_valid, out=np.full(total.shape, np.nan), where=n_valid > 0
    )
    return voltages[:1], mean[np.newaxis]


def chunk_responses(
    esa_voltages: npt.ArrayLike,
    count_rates: npt.NDArray[np.float64],
    swapi_response: SwapiResponse,
    epoch: int,
) -> StepResponses:
    """
    Get the proton response of each sample of a chunk.

    Parameters
    ----------
    esa_voltages : numpy.ndarray
        ESA voltage magnitude of each sample in V (sweep, step).
    count_rates : numpy.ndarray
        Observed coincidence rates in counts/s (sweep, step).
    swapi_response : SwapiResponse
        SWAPI instrument response tables.
    epoch : int
        Time of the chunk in TT J2000 ns, for the relative efficiency lookup.

    Returns
    -------
    StepResponses
        Proton response of each sample, flattened in C order from
        (sweep, step).

    Raises
    ------
    ValueError
        If ``esa_voltages`` and ``count_rates`` have different shapes.
    """
    if np.shape(esa_voltages) != np.shape(count_rates):
        raise ValueError(
            f"got ESA voltages of shape {np.shape(esa_voltages)} for count "
            f"rates of shape {np.shape(count_rates)}"
        )
    return swapi_response.step_responses(esa_voltages, epoch, Species.PROTON)


def boresight_rates_and_jacobian(
    responses: StepResponses, state: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the ideal rates of a flow along the boresight and their jacobian.

    Parameters
    ----------
    responses : StepResponses
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
    responses: StepResponses, rates: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Guess the state from the peak of the observed rates.

    The speed is the central speed of the peak step and the temperature scales
    with the speed squared (60,000 K at 400 km/s, at least 1 eV). The density
    is the least-squares scale of the rates at unit density.

    Parameters
    ----------
    responses : StepResponses
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
    # mean of each step's valid rates, NaN (without nanmean's empty-slice
    # warning) for steps with none
    valid = np.isfinite(rates)
    n_valid = valid.sum(axis=0)
    mean_rates = np.divide(
        np.where(valid, rates, 0.0).sum(axis=0),
        n_valid,
        out=np.full(n_valid.shape, np.nan),
        where=n_valid > 0,
    )
    speed = central_speeds[np.nanargmax(mean_rates)]
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
