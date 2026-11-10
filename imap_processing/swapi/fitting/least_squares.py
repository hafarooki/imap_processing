"""Least-squares helpers shared by the SWAPI Maxwellian fits."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, least_squares

from imap_processing.swapi.forward_model.deadtime import deadtime_factor
from imap_processing.swapi.forward_model.instrument_response import (
    StepResponses,
)


def least_squares_with_jacobian(
    residuals_and_jacobian: Callable[..., tuple[np.ndarray, np.ndarray]],
    x0: npt.NDArray[np.float64],
    *,
    args: tuple = (),
    **kwargs: Any,
) -> OptimizeResult:
    """
    Run ``scipy.optimize.least_squares`` on a function giving both outputs.

    ``least_squares`` asks for the residuals and the jacobian in separate
    calls, but the forward model computes both at once, so the last evaluation
    is reused when both are asked for at the same state.

    Parameters
    ----------
    residuals_and_jacobian : callable
        ``residuals_and_jacobian(x, *args)`` returning the residuals (m,) and
        their jacobian w.r.t. ``x`` (m, n).
    x0 : numpy.ndarray
        Initial state (n,).
    args : tuple
        Extra positional arguments to ``residuals_and_jacobian``.
    **kwargs : dict
        Passed on to ``scipy.optimize.least_squares``.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The result of ``least_squares``.
    """
    last_evaluation: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def evaluate(x: npt.NDArray[np.float64]) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the residuals and jacobian, reusing the last evaluation.

        Parameters
        ----------
        x : numpy.ndarray
            State.

        Returns
        -------
        residuals : numpy.ndarray
            Residuals at ``x``.
        jacobian : numpy.ndarray
            Derivatives of the residuals w.r.t. ``x``.
        """
        key = x.tobytes()
        if key not in last_evaluation:
            last_evaluation.clear()
            last_evaluation[key] = residuals_and_jacobian(x, *args)
        return last_evaluation[key]

    return least_squares(
        lambda x: evaluate(x)[0], x0, jac=lambda x: evaluate(x)[1], **kwargs
    )


def mean_residuals_and_jacobian(
    state: npt.NDArray[np.float64],
    rates_and_jacobian: Callable[
        [StepResponses, npt.NDArray[np.float64]],
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ],
    responses: StepResponses,
    count_rates: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute residuals of the post-deadtime model rates averaged over sweeps.

    The model and observed rates are each averaged over the valid samples of
    each step, and steps with no valid sample are dropped.

    Parameters
    ----------
    state : numpy.ndarray
        Fit state (n,).
    rates_and_jacobian : callable
        ``rates_and_jacobian(responses, state)`` returning the ideal rates of
        every sample in counts/s and their jacobian w.r.t. ``state``.
    responses : StepResponses
        Instrument response of each sample, flattened in C order from
        (sweep, step).
    count_rates : numpy.ndarray
        Observed coincidence rates in counts/s (sweep, step), NaN where invalid.

    Returns
    -------
    residuals : numpy.ndarray
        Model minus observed mean rate of each step with a valid sample.
    jacobian : numpy.ndarray
        Derivatives of ``residuals`` w.r.t. ``state``.
    """
    valid = np.isfinite(count_rates)
    # steps with no valid sample are dropped before averaging, which avoids
    # nanmean's empty-slice warning
    fit_mask = valid.any(axis=0)
    valid = valid[:, fit_mask]
    observed = np.nanmean(count_rates[:, fit_mask], axis=0)

    ideal, jacobian = rates_and_jacobian(responses, state)
    # measured = r / (1 + tau r), so d(measured)/dr = deadtime_factor**2
    dead = deadtime_factor(ideal)
    modeled = (ideal * dead).reshape(count_rates.shape)[:, fit_mask]
    modeled_jacobian = (jacobian * (dead**2)[:, np.newaxis]).reshape(
        *count_rates.shape, -1
    )[:, fit_mask]
    modeled = np.nanmean(np.where(valid, modeled, np.nan), axis=0)
    modeled_jacobian = np.nanmean(
        np.where(valid[..., np.newaxis], modeled_jacobian, np.nan), axis=0
    )
    return modeled - observed, modeled_jacobian
