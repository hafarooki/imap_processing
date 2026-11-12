"""Conversion of the response integral to counts/s (L3a ``utils.py``)."""

from __future__ import annotations

import math

import numba
import numpy as np

from imap_processing.swapi.forward_model.maxwellian.params import (
    _MaxwellianParams,
    thermal_speed,
)
from imap_processing.swapi.forward_model.response.instrument_response import (
    StepResponse,
)


@numba.njit
def count_rate_conversion_factor(
    sw_params: _MaxwellianParams, response: StepResponse
) -> float:  # numpydoc ignore=PR01,RT01
    """Compute the factor converting the response integral to counts/s."""
    sigma = thermal_speed(sw_params, response)
    return (
        response.central_effective_area
        * sw_params.density
        * (np.sqrt(2 * np.pi) * sigma) ** -3
        * (math.pi / 180.0) ** 2
        * 1e5  # km -> cm
    )
