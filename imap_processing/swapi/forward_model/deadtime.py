"""SWAPI deadtime correction."""

import numpy as np
import numpy.typing as npt

SWAPI_DEADTIME_S = 183.7e-9


def deadtime_factor(true_rate: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Compute the ratio of "measured" to "true" coincidence rate.

    Parameters
    ----------
    true_rate : float or numpy.ndarray
        "True" coincidence rate in counts/s.

    Returns
    -------
    numpy.ndarray
        "Measured"/"true" ratio, in range (0, 1].
    """
    return 1.0 / (1.0 + SWAPI_DEADTIME_S * np.asarray(true_rate, dtype=float))
