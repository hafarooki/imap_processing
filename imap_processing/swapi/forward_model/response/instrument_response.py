"""SWAPI's response at one ESA voltage, as the forward model kernel sees it."""

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
    Azimuthal region of the aperture, named by the sign of its azimuths.

    The sunglasses are centered on the boresight, flanked by the two halves of
    the open aperture.
    """

    OPEN_APERTURE_NEGATIVE = -1
    SUNGLASSES = 0
    OPEN_APERTURE_POSITIVE = 1


class StepResponse(NamedTuple):
    """SWAPI's response at one ESA step for one species."""

    mass_kg: float
    central_speed: float
    central_effective_area: float
    sg_passband: Passband
    oa_passband: Passband
    azimuthal_transmission: AzimuthalTransmission


def central_speed(esa_voltage: float, species: Species) -> float:
    """
    Compute the speed at the center of the passband at an ESA voltage.

    The central energy per charge is ``SIMION_K_FACTOR`` times the ESA voltage.

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
    Select the passband of an azimuthal region.

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
    Find the azimuths spanned by an azimuthal region.

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
