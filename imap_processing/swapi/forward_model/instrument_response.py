"""SWAPI's response at one ESA voltage, as the forward model kernel sees it."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import (
    OA_AZIMUTH_MAX_DEG,
    PROTON_CHARGE_C,
    PROTON_MASS_KG,
    SG_AZIMUTH_MAX_DEG,
    SIMION_K_FACTOR,
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


class StepResponse(NamedTuple):
    """
    SWAPI's response at one ESA step for one species.

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


class StepResponses(NamedTuple):
    """
    SWAPI's response at each ESA step of a chunk, for one species.

    Built by :math:`SwapiResponse.step_responses`. The passbands are shared
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

    def subset(self, indices: npt.ArrayLike) -> StepResponses:
        """
        Select samples, sharing the passbands.

        Parameters
        ----------
        indices : numpy.ndarray
            Indices (or a boolean mask) of the samples to keep.

        Returns
        -------
        StepResponses
            The responses of the selected samples, in the order given.
        """
        indices = np.asarray(indices)
        return self._replace(
            central_speed=self.central_speed[indices],
            central_effective_area=self.central_effective_area[indices],
            passband_index=self.passband_index[indices],
        )


# boundaries of the nested tuples in flatten_step_responses' flat tuple
_SG_START = 4
_OA_START = _SG_START + len(Passbands._fields)
_TRANSMISSION_START = _OA_START + len(Passbands._fields)


@numba.njit  # type: ignore[untyped-decorator]
def flatten_step_responses(responses: StepResponses) -> tuple[Any, ...]:
    """
    Flatten the responses' nested tuples into one flat tuple.

    numba (0.68) can't pass a tuple of tuples holding arrays into a parallel
    (``prange``) loop, but can pass a flat one, which
    :func:`unflatten_step_responses` rebuilds inside the loop without copying.

    Parameters
    ----------
    responses : StepResponses
        The responses of a chunk.

    Returns
    -------
    tuple
        The fields of ``responses`` then of each of its nested tuples, in order.
    """
    return (
        responses[:_SG_START]
        + responses.sg_passbands[:]
        + responses.oa_passbands[:]
        + responses.azimuthal_transmission[:]
    )


@numba.njit  # type: ignore[untyped-decorator]
def unflatten_step_responses(flat: tuple[Any, ...]) -> StepResponses:
    """
    Rebuild the responses flattened by :func:`flatten_step_responses`.

    Parameters
    ----------
    flat : tuple
        The flat tuple.

    Returns
    -------
    StepResponses
        The responses, sharing the flat tuple's arrays.
    """
    return StepResponses(
        flat[0],
        flat[1],
        flat[2],
        flat[3],
        Passbands(*flat[_SG_START:_OA_START]),
        Passbands(*flat[_OA_START:_TRANSMISSION_START]),
        AzimuthalTransmission(*flat[_TRANSMISSION_START:]),
    )


@numba.njit  # type: ignore[untyped-decorator]
def step_response_at(responses: StepResponses, index: int) -> StepResponse:
    """
    Select one sample's response, without copying the passbands.

    Parameters
    ----------
    responses : StepResponses
        The responses of a chunk.
    index : int
        Index of the sample.

    Returns
    -------
    StepResponse
        The sample's response.
    """
    i_passband = responses.passband_index[index]
    return StepResponse(
        responses.mass_kg,
        responses.central_speed[index],
        responses.central_effective_area[index],
        passband_at(responses.sg_passbands, i_passband),
        passband_at(responses.oa_passbands, i_passband),
        responses.azimuthal_transmission,
    )


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


@numba.njit  # type: ignore[untyped-decorator]
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
