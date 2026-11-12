"""
Placeholder: Forward model of SWAPI count rates for a Maxwellian solar wind.

Ported from imap_l3_processing's L3a solar-wind forward model
(``swapi/l3a/science/solar_wind/``), keeping its module layout except that
``params.py``'s speed conversions moved into ``distribution.py``,
``trim_open_aperture.py`` was folded into ``integration_limits.py``, and
``utils.py``'s count-rate factor was inlined into ``integration.py``. The algorithm
is unchanged; the only differences are that the bulk flow is given in the SWAPI
frame (no RTN rotation) and that the instrument response is a
:class:`~imap_processing.swapi.forward_model.response.instrument_response.StepResponse`
rather than an L3 ``ResponseGrid``.

From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Integral"): "The solar
wind velocity distribution function (VDF) is modeled as a drifting Maxwellian"
"""

from imap_processing.swapi.forward_model.maxwellian.integration import (
    maxwellian_count_rates,
)

__all__ = ["maxwellian_count_rates"]
