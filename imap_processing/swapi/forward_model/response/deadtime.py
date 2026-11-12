"""
SWAPI deadtime correction.

The detector deadtime is ``tau = 183.7 ns``. Following Tsoulfanidis (1995),
p. 74, the true count rate ``n`` and measured rate ``g`` are related by
``n = g / (1 - g tau)``. Rearranged for the forward model, the true model rate
``C^model`` is mapped to the predicted observed rate before computing residuals.
This deadtime correction is often non-negligible. It reaches 5% at
``C ~ 2.7 x 10^5 Hz``, a routine value for high-flux solar wind.
"""

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
