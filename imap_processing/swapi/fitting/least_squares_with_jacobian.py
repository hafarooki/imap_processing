"""Convenience least_squares wrapper for a function that provides a Jacobian."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, least_squares


def least_squares_with_jacobian(
    residuals_and_jacobian: Callable[..., tuple[np.ndarray, np.ndarray]],
    x0: npt.NDArray[np.float64],
    **kwargs: Any,
) -> OptimizeResult:
    """
    Run ``scipy.optimize.least_squares`` for a function that provides Jacobians.

    The problem is that ``scipy.optimize.least_squares`` takes a separate function for
    the residuals and the Jacobian, but our forward model calculates them both in one
    go. The workaround is to cache the most recent evaluation and reuse it if the
    parameters remain the same.

    Parameters
    ----------
    residuals_and_jacobian : x -> (residuals, jacobians)
        The function that takes the model parameters ``x`` (n,) and provides the
        residuals (m,) and their Jacobians (m, n) w.r.t. ``x``.
    x0 : numpy.ndarray
        Initial state (n,).
    **kwargs : dict
        Extra arguments for ``scipy.optimize.least_squares``.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The result of ``scipy.optimize.least_squares``.s
    """
    last_evaluation: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def evaluate(x: npt.NDArray[np.float64]) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the residuals and jacobian, reusing the last evaluation.

        Parameters
        ----------
        x : numpy.ndarray
            The model parameters.

        Returns
        -------
        residuals : numpy.ndarray
            Residuals for model parameters ``x``.
        jacobian : numpy.ndarray
            Derivatives of the residuals w.r.t. ``x``.
        """
        key = x.tobytes()
        if key not in last_evaluation:
            last_evaluation.clear()
            last_evaluation[key] = residuals_and_jacobian(x)
        return last_evaluation[key]

    return least_squares(
        lambda x: evaluate(x)[0], x0, jac=lambda x: evaluate(x)[1], **kwargs
    )
