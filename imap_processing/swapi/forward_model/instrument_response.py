"""SWAPI's response at one ESA voltage, as the forward model kernel sees it."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import (
    OA_AZIMUTH_MAX_DEG,
    PROTON_CHARGE_C,
    PROTON_MASS_KG,
    SG_AZIMUTH_MAX_DEG,
    Species,
)
from imap_processing.swapi.forward_model.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.passband import (
    Passband,
    Passbands,
    passband_at,
)

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


class InstrumentResponse(NamedTuple):
    """
    SWAPI's response at one ESA voltage for one species.

    Attributes
    ----------
    mass_kg : float
        Ion mass in kg.
    central_speed : float
        Speed in km/s at the center of the passband.
    central_effective_area : float
        Effective area at elevation 0, scaled for the species.
    sg_passband : Passband
        Passband of the sunglasses.
    oa_passband : Passband
        Passband of the open aperture.
    azimuthal_transmission : AzimuthalTransmission
        Azimuthal transmission.
    """

    mass_kg: float
    central_speed: float
    central_effective_area: float
    sg_passband: Passband
    oa_passband: Passband
    azimuthal_transmission: AzimuthalTransmission


class InstrumentResponses(NamedTuple):
    """
    SWAPI's response at each sample of a chunk, for one species.

    Built by :math:`SwapiCalibration.responses`. The passbands are shared
    between samples; each sample indexes its own.

    Attributes
    ----------
    mass_kg : float
        Ion mass in kg.
    central_speed : numpy.ndarray
        Speed in km/s at the center of each sample's passband (n,).
    central_effective_area : numpy.ndarray
        Effective area at elevation 0 of each sample, scaled for the species
        (n,).
    passband_index : numpy.ndarray
        Index of each sample's passband in ``sg_passbands`` and
        ``oa_passbands`` (n,).
    sg_passbands : Passbands
        Passbands of the sunglasses.
    oa_passbands : Passbands
        Passbands of the open aperture.
    azimuthal_transmission : AzimuthalTransmission
        Azimuthal transmission.
    """

    mass_kg: float
    central_speed: npt.NDArray[np.float64]
    central_effective_area: npt.NDArray[np.float64]
    passband_index: npt.NDArray[np.int64]
    sg_passbands: Passbands
    oa_passbands: Passbands
    azimuthal_transmission: AzimuthalTransmission

    def subset(self, indices: npt.ArrayLike) -> InstrumentResponses:
        """
        Select samples, sharing the passbands.

        Parameters
        ----------
        indices : numpy.ndarray
            Indices (or a boolean mask) of the samples to keep.

        Returns
        -------
        InstrumentResponses
            The responses of the selected samples, in the order given.
        """
        indices = np.asarray(indices)
        return self._replace(
            central_speed=self.central_speed[indices],
            central_effective_area=self.central_effective_area[indices],
            passband_index=self.passband_index[indices],
        )


@numba.njit  # type: ignore[untyped-decorator]
def response_at(responses: InstrumentResponses, index: int) -> InstrumentResponse:
    """
    Select one sample's response, without copying the passbands.

    Parameters
    ----------
    responses : InstrumentResponses
        The responses of a chunk.
    index : int
        Index of the sample.

    Returns
    -------
    InstrumentResponse
        The sample's response.
    """
    i_passband = responses.passband_index[index]
    return InstrumentResponse(
        responses.mass_kg,
        responses.central_speed[index],
        responses.central_effective_area[index],
        passband_at(responses.sg_passbands, i_passband),
        passband_at(responses.oa_passbands, i_passband),
        responses.azimuthal_transmission,
    )


def central_speed(esa_energy: float, species: Species) -> float:
    """
    Compute the speed at the center of the passband at an ESA energy.

    Parameters
    ----------
    esa_energy : float
        ESA energy per charge in eV/q.
    species : Species
        Ion species.

    Returns
    -------
    float
        ``sqrt(2 (E/q) (e / m_p) / (m/q))`` in km/s.
    """
    return float(
        np.sqrt(
            2.0
            * esa_energy
            * (PROTON_CHARGE_C / PROTON_MASS_KG)
            / species.mass_per_charge
        )
        / METERS_PER_KILOMETER
    )


@numba.njit  # type: ignore[untyped-decorator]
def passband_for(response: InstrumentResponse, region: Region) -> Passband:
    """
    Select the passband of an azimuthal region.

    Parameters
    ----------
    response : InstrumentResponse
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


@numba.njit  # type: ignore[untyped-decorator]
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
