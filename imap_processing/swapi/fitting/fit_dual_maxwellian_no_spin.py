"""Fit a core plus suprathermal proton Maxwellian along the spin axis."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from imap_processing.swapi.fitting.fit_single_maxwellian_no_spin import (
    boresight_rates_and_jacobian,
    chunk_responses,
    mean_sweep,
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
    StepResponses,
)
from imap_processing.swapi.forward_model.maxwellian_count_rates import (
    maxwellian_count_rates_at_temperatures,
)
from imap_processing.swapi.forward_model.response_tables import SwapiResponse

# floor on the grid search's starting densities, relative to their total,
# since NNLS can return zero and the fit's state is log density
MIN_INITIAL_DENSITY_RATIO = 1e-3

# largest suprathermal/core density ratio of a grid search cell that can start
# the fit; the suprathermal is typically a few percent of the core, and cells
# with comparable densities are spurious minima, e.g. the populations swapped
MAX_GRID_DENSITY_RATIO = 0.5

# two-column NNLS problems whose Gram determinant is below this fraction of
# the product of its diagonal are treated as having parallel columns
NNLS_SINGULAR_TOLERANCE = 1e-10
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
    esa_voltages: npt.NDArray[np.floating],
    count_rates: npt.NDArray[np.floating],
    *,
    swapi_response: SwapiResponse,
    epoch: int,
    max_evaluations: int = 200,
) -> dict[str, float | npt.NDArray[np.float64]]:
    """
    Fit a chunk of sweeps with a core plus suprathermal proton Maxwellian.

    Both populations flow along the spin axis and share one speed, so every
    sweep sees the same rates and the chunk is collapsed to one sweep by
    averaging the valid rates of each step over sweeps (see
    :func:`~imap_processing.swapi.fitting.fit_single_maxwellian_no_spin.mean_sweep`).
    The fit starts from :func:`grid_search` and fits the model's post-deadtime
    rates to that mean sweep, with state (ln n_core, ln T_core,
    ln speed, ln n_supra, ln T_supra/T_core), the suprathermal temperature
    constrained to 1-``MAX_TEMPERATURE_RATIO`` times the core's.

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
        ``speed`` [km/s]; ``core_density``, ``suprathermal_density`` [cm^-3]
        and ``core_temperature``, ``suprathermal_temperature`` [K]; the
        ``density`` [cm^-3] and ``temperature`` [K] of both populations
        measured together, ``n_core + n_supra`` and
        ``(n_core T_core + n_supra T_supra) / n`` since they share one speed;
        and ``model_count_rates``, the post-deadtime model rates at the best
        fit [counts/s] shaped like ``count_rates`` (the same for every sweep).
    """
    sweep_voltages, rates = mean_sweep(esa_voltages, count_rates)
    responses = chunk_responses(sweep_voltages, rates, swapi_response, epoch)

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
        "model_count_rates": np.broadcast_to(
            ideal * deadtime_factor(ideal), np.shape(count_rates)
        ).copy(),
    }


def dual_rates_and_jacobian(
    responses: StepResponses, state: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the ideal rates of a core plus suprathermal along the boresight.

    Parameters
    ----------
    responses : StepResponses
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
    responses: StepResponses, count_rates: npt.NDArray[np.floating]
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
    responses : StepResponses
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
    velocity = np.broadcast_to(
        np.array([0.0, -speed, 0.0]), (responses.central_speed.shape[0], 3)
    )
    all_unit_rates = maxwellian_count_rates_at_temperatures(
        responses, 1.0, GRID_TEMPERATURES_K, velocity
    ).reshape((len(GRID_TEMPERATURES_K), *rates.shape))
    unit_rates = np.nanmean(np.where(valid, all_unit_rates, np.nan), axis=1)

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
    # cell (i, k)'s core and suprathermal unit rates (temperature, ratio, step)
    supra_index = np.arange(n)[:, None] + np.arange(n_ratios)
    is_on_grid = supra_index < n
    core = np.broadcast_to(unit_rates[:, None], (n, n_ratios, len(mean_rates)))
    supra = unit_rates[np.minimum(supra_index, n - 1)]
    is_good = np.isfinite(core) & np.isfinite(supra) & is_used
    core = np.where(is_good, core, 0.0)
    supra = np.where(is_good, supra, 0.0)
    densities = two_column_nnls(
        np.stack([core, supra], axis=-1),
        np.where(is_used, observed_ideal, 0.0),
        np.where(is_good, weights, 0.0),
    )
    modeled = core * densities[..., :1] + supra * densities[..., 1:]
    residual = np.where(
        is_good, modeled * deadtime_factor(modeled) - np.nan_to_num(mean_rates), 0.0
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt((residual**2).sum(axis=-1) / is_good.sum(axis=-1))
    rmse[~is_on_grid] = np.nan
    densities[~is_on_grid] = np.nan

    # cells off the grid count as infinite RMSE
    padded = np.pad(
        np.nan_to_num(rmse, nan=np.inf), ((0, 0), (1, 1)), constant_values=np.inf
    )
    is_local_min = (
        np.isfinite(rmse)
        & (padded[:, 1:-1] <= padded[:, :-2])
        & (padded[:, 1:-1] <= padded[:, 2:])
        & (densities[..., 1] <= MAX_GRID_DENSITY_RATIO * densities[..., 0])
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


def two_column_nnls(
    columns: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Solve many weighted two-column non-negative least-squares problems at once.

    Each problem minimizes ``||weights * (columns @ x - targets)||`` over
    ``x >= 0``, like ``scipy.optimize.nnls`` on the weighted columns and
    targets. With two unknowns the solution is either the unconstrained one,
    if it's non-negative, or the better of the two single-column solutions.
    Parallel columns fit equally well alone, and as with ``nnls`` the one more
    aligned with the targets is used, the first on a tie.

    Parameters
    ----------
    columns : numpy.ndarray
        The two columns of each problem (..., row, 2).
    targets : numpy.ndarray
        Targets, broadcast against each problem's rows (..., row).
    weights : numpy.ndarray
        Weight of each row; 0 leaves it out (..., row).

    Returns
    -------
    numpy.ndarray
        Each problem's solution (..., 2).
    """
    weighted = columns * weights[..., None]
    weighted_targets = targets * weights
    gram_11 = (weighted[..., 0] ** 2).sum(axis=-1)
    gram_22 = (weighted[..., 1] ** 2).sum(axis=-1)
    gram_12 = (weighted[..., 0] * weighted[..., 1]).sum(axis=-1)
    projection_1 = (weighted[..., 0] * weighted_targets).sum(axis=-1)
    projection_2 = (weighted[..., 1] * weighted_targets).sum(axis=-1)

    determinant = gram_11 * gram_22 - gram_12**2
    is_singular = determinant <= NNLS_SINGULAR_TOLERANCE * gram_11 * gram_22
    with np.errstate(invalid="ignore", divide="ignore"):
        unconstrained_1 = (
            gram_22 * projection_1 - gram_12 * projection_2
        ) / determinant
        unconstrained_2 = (
            gram_11 * projection_2 - gram_12 * projection_1
        ) / determinant
        single_1 = np.where(gram_11 > 0, np.maximum(projection_1, 0) / gram_11, 0.0)
        single_2 = np.where(gram_22 > 0, np.maximum(projection_2, 0) / gram_22, 0.0)
    is_interior = ~is_singular & (unconstrained_1 >= 0) & (unconstrained_2 >= 0)
    # a single-column solution x reduces the squared residual by x * projection;
    # parallel columns reduce it equally, and nnls takes the larger projection
    use_first = np.where(
        is_singular,
        projection_1 >= projection_2,
        single_1 * projection_1 >= single_2 * projection_2,
    )
    return np.stack(
        [
            np.where(is_interior, unconstrained_1, np.where(use_first, single_1, 0.0)),
            np.where(is_interior, unconstrained_2, np.where(use_first, 0.0, single_2)),
        ],
        axis=-1,
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
