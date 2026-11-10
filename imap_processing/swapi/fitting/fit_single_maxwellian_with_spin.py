"""Fit a single proton Maxwellian flowing off the spin axis to SWAPI rates."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from imap_processing.swapi.constants import NUM_ENERGY_STEPS
from imap_processing.swapi.fitting.fit_single_maxwellian_no_spin import (
    MIN_INITIAL_DENSITY,
    fit_single_maxwellian_no_spin,
)
from imap_processing.swapi.fitting.least_squares import least_squares_with_jacobian
from imap_processing.swapi.forward_model.deadtime import deadtime_factor
from imap_processing.swapi.forward_model.instrument_response import (
    InstrumentResponses,
)
from imap_processing.swapi.forward_model.maxwellian_count_rates import (
    maxwellian_count_rates,
)

# nominal spin period of IMAP and timing of the SWAPI sweep
SPIN_PERIOD_S = 15.0
SWEEP_PERIOD_S = 12.0
BINS_PER_SWEEP = NUM_ENERGY_STEPS


def fit_single_maxwellian_with_spin(
    responses: InstrumentResponses,
    count_rates: npt.NDArray[np.floating],
    *,
    max_evaluations: int = 100,
    max_restarts: int = 6,
) -> dict[str, float | npt.NDArray[np.float64]]:
    """
    Fit a chunk of consecutive sweeps with one proton Maxwellian off the spin axis.

    The fit starts from :func:`fit_single_maxwellian_no_spin` and fits the
    post-deadtime rates of every sample, with state (ln density,
    ln temperature, v_x, v_y, v_z) and the velocity in the SWAPI frame at spin
    angle 0 (see :func:`spin_rotations`).

    A deflected flow and its mirror image about the spin axis fit similarly, so
    the fit is restarted from the mirrored flow while that improves it.

    Parameters
    ----------
    responses : InstrumentResponses
        Proton instrument response of each sample of the chunk, flattened in C
        order from (sweep, step).
    count_rates : numpy.ndarray
        Observed coincidence rates in counts/s (sweep, step), NaN where invalid.
        Each step is one bin after the last and each sweep starts one sweep
        period after the last.
    max_evaluations : int
        Maximum number of model evaluations per fit.
    max_restarts : int
        Maximum number of refits from the mirrored flow.

    Returns
    -------
    dict[str, float | numpy.ndarray]
        ``speed`` [km/s], ``density`` [cm^-3], ``temperature`` [K], and
        ``model_count_rates``, the post-deadtime model rates at the best fit
        [counts/s] shaped like ``count_rates``.
    """
    rates = np.asarray(count_rates, dtype=float).ravel()
    rotations = spin_rotations(*np.shape(count_rates)).reshape(-1, 3, 3)
    fit_args = (responses, rotations, rates)
    fit_kwargs = dict(method="lm", xtol=1e-4, x_scale=1.0, max_nfev=max_evaluations)

    # the no-spin fit is a flow along the boresight, -y, which is the spin axis,
    # so it is the same at every spin angle
    no_spin = fit_single_maxwellian_no_spin(
        responses, count_rates, max_evaluations=max_evaluations
    )
    initial = np.array(
        [
            np.log(no_spin["density"]),
            np.log(no_spin["temperature"]),
            0.0,
            -no_spin["speed"],
            0.0,
        ]
    )
    current = least_squares_with_jacobian(
        _residuals_and_jacobian, initial, args=fit_args, **fit_kwargs
    ).x

    for _ in range(max_restarts):
        current_error = _mean_squared_error(current, *fit_args)

        # the spin axis is SWAPI +y, so mirroring about it flips x and z
        mirrored = current.copy()
        mirrored[2:] *= [-1.0, 1.0, -1.0]
        mirrored[0] = np.log(_optimal_density(mirrored, *fit_args))
        # skip if the mirrored flow fits much worse
        if _mean_squared_error(mirrored, *fit_args) >= 100 * current_error:
            break

        candidate = least_squares_with_jacobian(
            _residuals_and_jacobian, mirrored, args=fit_args, **fit_kwargs
        ).x
        # don't replace if it's not an improvement
        if _mean_squared_error(candidate, *fit_args) > current_error:
            break
        current = candidate

    return {
        "speed": float(np.linalg.norm(current[2:])),
        "density": float(np.exp(current[0])),
        "temperature": float(np.exp(current[1])),
        "model_count_rates": _post_deadtime_rates(
            current, responses, rotations
        ).reshape(np.shape(count_rates)),
    }


def spin_rates_and_jacobian(
    responses: InstrumentResponses,
    rotations: npt.NDArray[np.float64],
    state: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the ideal rates of a flow seen from the spinning spacecraft.

    Parameters
    ----------
    responses : InstrumentResponses
        Instrument response of each sample (n,).
    rotations : numpy.ndarray
        Rotation of each sample from :func:`spin_rotations` (n, 3, 3).
    state : numpy.ndarray
        (ln density, ln temperature, v_x, v_y, v_z) with density in cm^-3,
        temperature in K and the velocity in the SWAPI frame at spin angle 0
        in km/s (5,).

    Returns
    -------
    rates : numpy.ndarray
        Ideal (pre-deadtime) rates in counts/s (n,).
    jacobian : numpy.ndarray
        Derivatives of ``rates`` w.r.t. ``state`` (n, 5).
    """
    log_density, log_temperature = state[:2]
    # flow seen by SWAPI at each sample: R^T v
    velocity = np.ascontiguousarray(np.einsum("nji,j->ni", rotations, state[2:]))
    rates, jacobian = maxwellian_count_rates(
        responses, np.exp(log_density), np.exp(log_temperature), velocity
    )
    # d/dv = R d/dv_swapi
    d_velocity = np.einsum("nij,nj->ni", rotations, jacobian[:, 2:])
    return rates, np.concatenate([jacobian[:, :2], d_velocity], axis=-1)


def spin_rotations(n_sweeps: int, n_steps: int) -> npt.NDArray[np.float64]:
    """
    Compute the rotation of each step from the SWAPI frame to spin angle 0.

    The spacecraft spins right-handed about its sunward +Z, which is SWAPI +Y
    (imap_001.tf), so R is the right-handed rotation by the spin angle about +Y
    and a flow v (in the SWAPI frame at spin angle 0) is seen by SWAPI as
    R^T v.

    Each step of a sweep is one bin (sweep period / bins per sweep) after the
    last, and consecutive sweeps start one sweep period apart. The angle is 0
    at the first step of the first sweep, which is arbitrary anyway.

    Parameters
    ----------
    n_sweeps : int
        Number of sweeps.
    n_steps : int
        Number of consecutive steps per sweep.

    Returns
    -------
    numpy.ndarray
        Rotation matrices (sweep, step, 3, 3).
    """
    bins = BINS_PER_SWEEP * np.arange(n_sweeps)[:, None] + np.arange(n_steps)
    seconds = bins * SWEEP_PERIOD_S / BINS_PER_SWEEP
    angle = 2 * np.pi * seconds / SPIN_PERIOD_S
    cos, sin = np.cos(angle), np.sin(angle)
    zero, one = np.zeros_like(angle), np.ones_like(angle)
    return np.stack(
        [
            np.stack([cos, zero, sin], axis=-1),
            np.stack([zero, one, zero], axis=-1),
            np.stack([-sin, zero, cos], axis=-1),
        ],
        axis=-2,
    )


def _post_deadtime_rates(
    state: npt.NDArray[np.float64],
    responses: InstrumentResponses,
    rotations: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute the post-deadtime model rates of every sample.

    Parameters
    ----------
    state : numpy.ndarray
        State as in :func:`spin_rates_and_jacobian` (5,).
    responses : InstrumentResponses
        Instrument response of each sample (n,).
    rotations : numpy.ndarray
        Rotation of each sample from :func:`spin_rotations` (n, 3, 3).

    Returns
    -------
    numpy.ndarray
        Model rates in counts/s (n,).
    """
    ideal = spin_rates_and_jacobian(responses, rotations, state)[0]
    return ideal * deadtime_factor(ideal)


def _mean_squared_error(
    state: npt.NDArray[np.float64],
    responses: InstrumentResponses,
    rotations: npt.NDArray[np.float64],
    rates: npt.NDArray[np.float64],
) -> float:
    """
    Compute the mean squared error of the model rates over the valid samples.

    Parameters
    ----------
    state : numpy.ndarray
        State as in :func:`spin_rates_and_jacobian` (5,).
    responses : InstrumentResponses
        Instrument response of each sample (n,).
    rotations : numpy.ndarray
        Rotation of each sample from :func:`spin_rotations` (n, 3, 3).
    rates : numpy.ndarray
        Observed coincidence rates in counts/s (n,), NaN where invalid.

    Returns
    -------
    float
        Mean squared residual in (counts/s)^2.
    """
    modeled = _post_deadtime_rates(state, responses, rotations)
    return float(np.nanmean((modeled - rates) ** 2))


def _residuals_and_jacobian(
    state: npt.NDArray[np.float64],
    responses: InstrumentResponses,
    rotations: npt.NDArray[np.float64],
    rates: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute residuals of the post-deadtime model rates of every valid sample.

    Parameters
    ----------
    state : numpy.ndarray
        State as in :func:`spin_rates_and_jacobian` (5,).
    responses : InstrumentResponses
        Instrument response of each sample (n,).
    rotations : numpy.ndarray
        Rotation of each sample from :func:`spin_rotations` (n, 3, 3).
    rates : numpy.ndarray
        Observed coincidence rates in counts/s (n,), NaN where invalid.

    Returns
    -------
    residuals : numpy.ndarray
        Model minus observed rate of each valid sample.
    jacobian : numpy.ndarray
        Derivatives of ``residuals`` w.r.t. ``state``.
    """
    valid = np.isfinite(rates)
    ideal, jacobian = spin_rates_and_jacobian(responses, rotations, state)
    # measured = r / (1 + tau r), so d(measured)/dr = deadtime_factor**2
    dead = deadtime_factor(ideal)
    return (
        (ideal * dead - rates)[valid],
        (jacobian * (dead**2)[:, np.newaxis])[valid],
    )


def _optimal_density(
    state: npt.NDArray[np.float64],
    responses: InstrumentResponses,
    rotations: npt.NDArray[np.float64],
    rates: npt.NDArray[np.float64],
) -> float:
    """
    Find the density whose post-deadtime rates best fit the observed rates.

    The temperature and velocity are taken from ``state``.

    Parameters
    ----------
    state : numpy.ndarray
        State as in :func:`spin_rates_and_jacobian`; its density is ignored.
    responses : InstrumentResponses
        Instrument response of each sample (n,).
    rotations : numpy.ndarray
        Rotation of each sample from :func:`spin_rotations` (n, 3, 3).
    rates : numpy.ndarray
        Observed coincidence rates in counts/s (n,), NaN where invalid.

    Returns
    -------
    float
        Best-fit density in cm^-3, at least ``MIN_INITIAL_DENSITY``.
    """
    valid = np.isfinite(rates)
    # the ideal rates scale with density, so one evaluation at unit density
    # suffices
    unit_state = state.copy()
    unit_state[0] = 0.0
    unit_rates = spin_rates_and_jacobian(responses, rotations, unit_state)[0][valid]
    (density,) = curve_fit(
        lambda unit_rate, density: (
            density * unit_rate * deadtime_factor(density * unit_rate)
        ),
        unit_rates,
        rates[valid],
        p0=[1.0],
    )[0]
    return max(float(density), MIN_INITIAL_DENSITY)
