"""Ported from L3a's ``tests/swapi/l3a/science/solar_wind/test_utils.py``.

L3a's ``average_spin_axis_rtn`` tests have no counterpart: that function wasn't
ported.
"""

import math

import numpy as np

from imap_processing.swapi.constants import PROTON_MASS_KG
from imap_processing.swapi.forward_model.maxwellian.params import (
    temperature_to_thermal_speed,
)
from imap_processing.swapi.forward_model.maxwellian.utils import (
    count_rate_conversion_factor,
)
from imap_processing.tests.swapi.forward_model.maxwellian.conftest import (
    proton_params,
    synthetic_step_response,
)


def test_count_rate_conversion_factor_matches_closed_form_formula():
    """A representative proton case reproduces
    ``A_eff n (sqrt(2 pi) sigma)^-3 (pi / 180)^2 1e5``."""
    params = proton_params(density=5.0, temperature=100_000.0)
    response = synthetic_step_response(central_effective_area=0.5)
    sigma = temperature_to_thermal_speed(PROTON_MASS_KG, 100_000.0)
    expected = (
        response.central_effective_area
        * params.density
        * (math.sqrt(2 * math.pi) * sigma) ** -3
        * (math.pi / 180.0) ** 2
        * 1e5
    )
    np.testing.assert_allclose(
        count_rate_conversion_factor(params, response), expected, rtol=1e-12
    )
