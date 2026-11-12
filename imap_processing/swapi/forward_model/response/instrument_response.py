"""
Placeholder: SWAPI's response at one ESA voltage, as the forward model kernel sees it.

From the L3a docs (``docs/swapi/response.md``, "Effective Area Function"): "In
instrument coordinates (speed v, elevation angle theta, azimuth angle phi; Rankin et al.
2025), A^s is decomposed as A^s(v, theta, phi, V) = A_0^s(V) * P_region(phi)(v/v_0^s,
theta, V) * T(phi)"
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import numba
import numpy as np

from imap_processing.swapi.constants import (
    OA_AZIMUTH_MAX_DEG,
    PROTON_CHARGE_C,
    PROTON_MASS_KG,
    SG_AZIMUTH_MAX_DEG,
    SIMION_K_FACTOR,
    Species,
)
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.response.passband import Passband

METERS_PER_KILOMETER = 1_000


class Region(IntEnum):
    """
    Placeholder: Azimuthal region of the aperture, named by the sign of its azimuths.

    The sunglasses are centered on the boresight, flanked by the two halves of
    the open aperture.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Integral"): "The
    region sum includes the sunglasses (SG, |phi| <= 20 deg) and the two sides of the
    open aperture (OA+/-, 20 deg <= |phi| <= 150 deg)."
    """

    OPEN_APERTURE_NEGATIVE = -1
    SUNGLASSES = 0
    OPEN_APERTURE_POSITIVE = 1


class StepResponse(NamedTuple):
    """Placeholder: SWAPI's response at one ESA step for one species."""

    mass_kg: float
    central_speed: float
    central_effective_area: float
    sg_passband: Passband
    oa_passband: Passband
    azimuthal_transmission: AzimuthalTransmission


def central_speed(esa_voltage: float, species: Species) -> float:
    """
    Placeholder: Compute the speed at the center of the passband at an ESA voltage.

    The central energy per charge is ``SIMION_K_FACTOR`` times the ESA voltage.

    From the L3a docs (``docs/swapi/response.md``, "Effective Area Function"): "v_0^s =
    sqrt(2 k* q^s |V| / m^s) is the central speed [...] k* = 1.89 eV/V/e, the peak
    (E/q)/|V| at theta = 0 deg based on high-resolution SIMION simulations."

    Parameters
    ----------
    esa_voltage : float
        ESA voltage magnitude in V.
    species : Species
        Ion species.

    Returns
    -------
    float
        ``sqrt(2 (E/q) (e / m_p) / (m/q))`` in km/s, with
        ``E/q = SIMION_K_FACTOR * esa_voltage``.
    """
    return float(
        np.sqrt(
            2.0
            * SIMION_K_FACTOR
            * esa_voltage
            * (PROTON_CHARGE_C / PROTON_MASS_KG)
            / species.mass_per_charge
        )
        / METERS_PER_KILOMETER
    )


@numba.njit
def passband_for(response: StepResponse, region: Region) -> Passband:
    """
    Placeholder: Select the passband of an azimuthal region.

    Parameters
    ----------
    response : StepResponse
        The instrument response.
    region : Region
        Azimuthal region.

    Returns
    -------
    Passband
        The sunglasses' passband for the sunglasses, else the open aperture's.
    """
    if region == Region.SUNGLASSES:
        return response.sg_passband
    return response.oa_passband


@numba.njit
def azimuth_range(region: Region) -> tuple[float, float]:
    """
    Placeholder: Find the azimuths spanned by an azimuthal region.

    Parameters
    ----------
    region : Region
        Azimuthal region.

    Returns
    -------
    lower : float
        Lowest azimuth in deg.
    upper : float
        Highest azimuth in deg.
    """
    if region == Region.SUNGLASSES:
        return -SG_AZIMUTH_MAX_DEG, SG_AZIMUTH_MAX_DEG
    if region == Region.OPEN_APERTURE_POSITIVE:
        return SG_AZIMUTH_MAX_DEG, OA_AZIMUTH_MAX_DEG
    return -OA_AZIMUTH_MAX_DEG, -SG_AZIMUTH_MAX_DEG
