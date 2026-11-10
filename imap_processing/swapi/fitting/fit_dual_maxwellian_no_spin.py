"""Fit a core plus suprathermal proton Maxwellian along the spin axis."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import nnls

from imap_processing.swapi.fitting.fit_single_maxwellian_no_spin import (
    boresight_rates_and_jacobian,
)
from imap_processing.swapi.fitting.least_squares import (
    least_squares_with_jacobian,
    mean_residuals_and_jacobian,
)
from imap_processing.swapi.forward_model.deadtime import (
    SWAPI_DEADTIME_S,
    deadtime_factor,
)
from imap_processing.swapi.forward_model.instrument_response import (
    InstrumentResponses,
)

# floor on the grid search's starting densities, relative to their total,
# since NNLS can return zero and the fit's state is log density
MIN_INITIAL_DENSITY_RATIO = 1e-3

# log-spaced trial temperatures in K for the core+suprathermal grid search; the
# suprathermal/core ratios searched are those the fit's bounds allow
GRID_TEMPERATURES_K = np.logspace(4, 8, 101)

# largest suprathermal/core temperature ratio the fit (and grid search) allows
MAX_TEMPERATURE_RATIO = 20.0

# core temperatures in K the fit may reach, keeping trf's steps away from
# temperatures where the model's jacobian is NaN
MIN_FIT_TEMPERATURE_K = 1e3
MAX_FIT_TEMPERATURE_K = 1e8

# bounds on the state (ln n_core, ln T_core, ln speed, ln n_supra, ln T_supra/T_core):
# the suprathermal temperature is at least the core's and at most
# MAX_TEMPERATURE_RATIO times it
LOWER_BOUNDS = np.array([-np.inf, np.log(MIN_FIT_TEMPERATURE_K), -np.inf, -np.inf, 0.0])
UPPER_BOUNDS = np.array(
    [
        np.inf,
        np.log(MAX_FIT_TEMPERATURE_K),
        np.inf,
        np.inf,
        np.log(MAX_TEMPERATURE_RATIO),
    ]
)


class GridSearch(NamedTuple):
    """
    The core+suprathermal grid search that starts the dual fit.

    Cell (i, k) has core temperature ``GRID_TEMPERATURES_K[i]`` and
    suprathermal temperature ``GRID_TEMPERATURES_K[i + k]``, i.e. ratio
    ``ratios[k]``; cells past the end of the grid are NaN.

    Attributes
    ----------
    speed : float
        Speed of the peak mean rate in km/s, shared by every cell.
    ratios : numpy.ndarray
        Suprathermal/core temperature ratio of each ratio index (ratio,).
    rmse : numpy.ndarray
        RMS of post-deadtime model minus observed mean rates in counts/s, the
        residual the dual fit minimizes (temperature, ratio).
    densities : numpy.ndarray
        Each cell's (core, suprathermal) densities in cm^-3
        (temperature, ratio, 2).
    is_local_min : numpy.ndarray
        Whether each cell is a local minimum of ``rmse`` in ratio
        (temperature, ratio).
    k_first : numpy.ndarray
        Each core temperature's first local minimum in ratio, -1 if none
        (temperature,).
    i_best : int
        Core temperature index of the chosen cell: the lowest RMSE among the
        first local minima.
    k_best : int
        Ratio index of the chosen cell.
    """

    speed: float
    ratios: npt.NDArray[np.float64]
    rmse: npt.NDArray[np.float64]
    densities: npt.NDArray[np.float64]
    is_local_min: npt.NDArray[np.bool_]
    k_first: npt.NDArray[np.int64]
    i_best: int
    k_best: int


def fit_dual_maxwellian_no_spin(
    responses: InstrumentResponses,
    count_rates: npt.NDArray[np.floating],
    *,
    max_evaluations: int = 200,
) -> dict[str, float | npt.NDArray[np.float64]]:
    """
    Fit a chunk of sweeps with a core plus suprathermal proton Maxwellian.

    Both populations flow along the spin axis and share one speed. The fit
    starts from :func:`grid_search` and fits the model's post-deadtime rates to
    the observed rates averaged over sweeps, with state (ln n_core, ln T_core,
    ln speed, ln n_supra, ln T_supra/T_core), the suprathermal temperature
    constrained to 1-``MAX_TEMPERATURE_RATIO`` times the core's.

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
        ``speed`` [km/s]; ``core_density``, ``suprathermal_density`` [cm^-3]
        and ``core_temperature``, ``suprathermal_temperature`` [K]; the
        ``density`` [cm^-3] and ``temperature`` [K] of both populations
        measured together, ``n_core + n_supra`` and
        ``(n_core T_core + n_supra T_supra) / n`` since they share one speed;
        and ``model_count_rates``, the post-deadtime model rates at the best
        fit [counts/s] shaped like ``count_rates``.

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
        _initial_state(grid_search(responses, rates)),
        args=(dual_rates_and_jacobian, responses, rates),
        method="trf",
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        xtol=1e-4,
        x_scale=1.0,
        max_nfev=max_evaluations,
    )

    log_n_core, log_t_core, log_speed, log_n_supra, log_t_ratio = result.x
    ideal = dual_rates_and_jacobian(responses, result.x)[0]
    core_density, core_temperature = np.exp(log_n_core), np.exp(log_t_core)
    supra_density = np.exp(log_n_supra)
    supra_temperature = np.exp(log_t_core + log_t_ratio)
    density = core_density + supra_density
    return {
        "speed": float(np.exp(log_speed)),
        "core_density": float(core_density),
        "core_temperature": float(core_temperature),
        "suprathermal_density": float(supra_density),
        "suprathermal_temperature": float(supra_temperature),
        "density": float(density),
        "temperature": float(
            (core_density * core_temperature + supra_density * supra_temperature)
            / density
        ),
        "model_count_rates": (ideal * deadtime_factor(ideal)).reshape(rates.shape),
    }


def dual_rates_and_jacobian(
    responses: InstrumentResponses, state: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the ideal rates of a core plus suprathermal along the boresight.

    Parameters
    ----------
    responses : InstrumentResponses
        Instrument response of each sample (n,).
    state : numpy.ndarray
        (ln n_core, ln T_core, ln speed, ln n_supra, ln T_supra/T_core) with
        densities in cm^-3, temperatures in K and speed in km/s (5,).

    Returns
    -------
    rates : numpy.ndarray
        Ideal (pre-deadtime) rates in counts/s (n,).
    jacobian : numpy.ndarray
        Derivatives of ``rates`` w.r.t. ``state`` (n, 5).
    """
    log_n_core, log_t_core, log_speed, log_n_supra, log_t_ratio = state
    core_rates, core_jacobian = boresight_rates_and_jacobian(
        responses, np.array([log_n_core, log_t_core, log_speed])
    )
    supra_rates, supra_jacobian = boresight_rates_and_jacobian(
        responses, np.array([log_n_supra, log_t_core + log_t_ratio, log_speed])
    )
    # columns of each population's jacobian are (ln n, ln T, ln speed); chain
    # rule to the state, where ln T_supra = ln T_core + ln ratio and the speed
    # is shared
    return core_rates + supra_rates, np.stack(
        [
            core_jacobian[:, 0],
            core_jacobian[:, 1] + supra_jacobian[:, 1],
            core_jacobian[:, 2] + supra_jacobian[:, 2],
            supra_jacobian[:, 0],
            supra_jacobian[:, 1],
        ],
        axis=-1,
    )


def grid_search(
    responses: InstrumentResponses, count_rates: npt.NDArray[np.floating]
) -> GridSearch:
    """
    Search core temperature and suprathermal/core temperature ratio.

    The speed is the central speed of the peak mean rate. The ratios searched
    are those the dual fit's bounds allow, within the grid, and cells are
    scored by the dual fit's residual: post-deadtime model minus observed rate,
    averaged over sweeps, unweighted, over every step with a finite mean rate.

    The residual isn't linear in the densities, so each cell's densities are
    solved by NNLS on ideal rates, with the observed rates un-deadtimed and each
    step weighted by d(observed)/d(ideal) = deadtime_factor**2, which matches
    the residual to first order; then the cell is scored by the exact residual.

    Parameters
    ----------
    responses : InstrumentResponses
        Proton instrument response of each sample of the chunk, flattened in C
        order from (sweep, step).
    count_rates : numpy.ndarray
        Observed coincidence rates in counts/s (sweep, step), NaN where invalid.

    Returns
    -------
    GridSearch
        Every cell's score and densities, and the chosen cell.
    """
    rates = np.asarray(count_rates, dtype=float)
    valid = np.isfinite(rates)
    mean_rates = np.nanmean(rates, axis=0)
    central_speeds = responses.central_speed.reshape(rates.shape).mean(axis=0)
    speed = float(central_speeds[np.nanargmax(mean_rates)])

    # rates at unit density averaged over the valid samples of each step, like
    # the fit's model (temperature, step)
    unit_rates = np.stack(
        [
            np.nanmean(
                np.where(
                    valid,
                    boresight_rates_and_jacobian(
                        responses, np.array([0.0, np.log(temperature), np.log(speed)])
                    )[0].reshape(rates.shape),
                    np.nan,
                ),
                axis=0,
            )
            for temperature in GRID_TEMPERATURES_K
        ]
    )

    # the grid is log-spaced, so ratio index k maps core index i to
    # suprathermal i + k; ratios start at 1, the fit's lower bound on ln ratio,
    # and stop at its upper bound
    n = len(GRID_TEMPERATURES_K)
    ratios = GRID_TEMPERATURES_K / GRID_TEMPERATURES_K[0]
    ratios = ratios[ratios <= MAX_TEMPERATURE_RATIO * (1 + 1e-9)]
    n_ratios = len(ratios)

    is_used = np.isfinite(mean_rates)
    observed_ideal = mean_rates / (1 - SWAPI_DEADTIME_S * mean_rates)
    weights = deadtime_factor(observed_ideal) ** 2
    rmse = np.full((n, n_ratios), np.nan)
    densities = np.full((n, n_ratios, 2), np.nan)
    for i in range(n):
        for k in range(min(n_ratios, n - i)):
            unit = np.stack([unit_rates[i], unit_rates[i + k]], axis=-1)
            is_good = np.isfinite(unit).all(axis=1) & is_used
            w = weights[is_good]
            densities[i, k] = nnls(
                unit[is_good] * w[:, None], observed_ideal[is_good] * w
            )[0]
            modeled = unit[is_good] @ densities[i, k]
            residual = modeled * deadtime_factor(modeled) - mean_rates[is_good]
            rmse[i, k] = np.sqrt(np.mean(residual**2))

    # cells off the grid count as infinite RMSE
    padded = np.pad(
        np.nan_to_num(rmse, nan=np.inf), ((0, 0), (1, 1)), constant_values=np.inf
    )
    is_local_min = (
        np.isfinite(rmse)
        & (padded[:, 1:-1] <= padded[:, :-2])
        & (padded[:, 1:-1] <= padded[:, 2:])
    )
    has_min = is_local_min.any(axis=1)
    k_first = is_local_min.argmax(axis=1)
    first_min_rmse = np.where(has_min, rmse[np.arange(n), k_first], np.nan)
    i_best = int(np.nanargmin(first_min_rmse))
    return GridSearch(
        speed=speed,
        ratios=ratios,
        rmse=rmse,
        densities=densities,
        is_local_min=is_local_min,
        k_first=np.where(has_min, k_first, -1),
        i_best=i_best,
        k_best=int(k_first[i_best]),
    )


def _initial_state(grid: GridSearch) -> npt.NDArray[np.float64]:
    """
    Take the dual fit's starting state from the grid search's chosen cell.

    Parameters
    ----------
    grid : GridSearch
        Result of :func:`grid_search`.

    Returns
    -------
    numpy.ndarray
        Initial (ln n_core, ln T_core, ln speed, ln n_supra, ln T_supra/T_core).
    """
    i_best, k_best = grid.i_best, grid.k_best
    # e.g. a spectrum with no suprathermal tail gives zero suprathermal density
    core_density, supra_density = np.maximum(
        grid.densities[i_best, k_best],
        MIN_INITIAL_DENSITY_RATIO * grid.densities[i_best, k_best].sum(),
    )
    return np.array(
        [
            np.log(core_density),
            np.log(GRID_TEMPERATURES_K[i_best]),
            np.log(grid.speed),
            np.log(supra_density),
            np.log(grid.ratios[k_best]),
        ]
    )
