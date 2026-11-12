"""Placeholder: SWAPI's response at one ESA step for one species."""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_C,
    PROTON_MASS_KG,
    SIMION_K_FACTOR,
    Species,
)
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.response.passband import Passband
from imap_processing.swapi.forward_model.response.response_tables import SwapiResponse


class StepResponse(NamedTuple):
    """
    Placeholder: SWAPI's response at one ESA step for one species.

    Attributes
    ----------
    mass_kg : float
        Species mass in kg.
    central_speed : float
        Speed at the center of the passband in km/s.
    central_effective_area : float
        Effective area at elevation 0, scaled by the relative efficiency.
    sg_passband : Passband
        Sunglasses passband.
    oa_passband : Passband
        Open aperture passband.
    azimuthal_transmission : AzimuthalTransmission
        Azimuthal transmission.
    rtn_to_swapi : numpy.ndarray
        Rotation matrix (3, 3) taking RTN vectors to the SWAPI frame at the
        time of the sample, ``v_swapi = rtn_to_swapi @ v_rtn``.
    """

    mass_kg: float
    central_speed: float
    central_effective_area: float
    sg_passband: Passband
    oa_passband: Passband
    azimuthal_transmission: AzimuthalTransmission
    rtn_to_swapi: npt.NDArray[np.float64]


def step_responses(
    swapi_response: SwapiResponse,
    esa_voltages: npt.ArrayLike,
    epoch: int,
    species: Species,
    rtn_to_swapi: npt.ArrayLike,
) -> numba.typed.List:
    """
    Placeholder: Compute SWAPI's response at each sample of a chunk for one species.

    The passbands are cached by ESA voltages in ``swapi_response``, so chunks
    with the same sweep don't re-evaluate them; they are shared between calls
    and must not be modified. The responses themselves are rebuilt on each call,
    which is cheap next to evaluating the passbands, so build them once
    per chunk rather than once per model evaluation.

    Parameters
    ----------
    swapi_response : SwapiResponse
        The instrument response tables.
    esa_voltages : numpy.ndarray
        ESA voltage magnitude of each sample in V; flattened in C order.
    epoch : int
        Time in TT J2000 ns, for the relative efficiency lookup.
    species : Species
        Ion species.
    rtn_to_swapi : numpy.ndarray
        Rotation matrix from RTN to the SWAPI frame, either one (3, 3)
        shared by every sample or one per sample (n, 3, 3), in the same
        order as the flattened ``esa_voltages``.

    Returns
    -------
    numba.typed.List[StepResponse]
        The response of each sample, sharing the passbands of samples at
        the same voltage. The central effective area is
        interpolated linearly in voltage and held at the table's end values
        outside it, then scaled by the species' relative efficiency in
        effect at ``epoch``.

    Raises
    ------
    ValueError
        If there are no voltages, a voltage is not positive and finite, or
        the rotations' shape matches neither (3, 3) nor (n, 3, 3).
    """
    esa_voltages = np.ascontiguousarray(esa_voltages, dtype=float).ravel()
    if esa_voltages.size == 0:
        raise ValueError("No ESA voltages given.")
    if not np.all(np.isfinite(esa_voltages) & (esa_voltages > 0)):
        raise ValueError("ESA voltages must be positive and finite.")
    rotations = np.asarray(rtn_to_swapi, dtype=float)
    if rotations.shape not in ((3, 3), (esa_voltages.size, 3, 3)):
        raise ValueError(
            f"RTN-to-SWAPI rotations must have shape (3, 3) or "
            f"({esa_voltages.size}, 3, 3), not {rotations.shape}."
        )
    # Claude: a C-contiguous copy, so every response's matrix has the same
    # numba type and later changes to the caller's array can't reach it
    rotations = np.array(
        np.broadcast_to(rotations, (esa_voltages.size, 3, 3)), order="C"
    )
    efficiency = swapi_response.relative_efficiency_at(epoch, species)

    sg_passbands, oa_passbands = swapi_response.sweep_passbands(esa_voltages)
    table = swapi_response.central_effective_area
    effective_areas = (
        np.interp(esa_voltages, table.esa_voltage.values, table.values) * efficiency
    )
    # Placeholder: from the L3a docs ("Effective Area Function"), the central
    # speed is v_0^s = sqrt(2 k* q^s |V| / m^s), where k* = SIMION_K_FACTOR is
    # the peak (E/q)/|V| at theta = 0 deg from high-resolution SIMION
    # simulations
    central_speeds = (
        np.sqrt(
            2.0
            * SIMION_K_FACTOR
            * esa_voltages
            * (PROTON_CHARGE_C / PROTON_MASS_KG)
            / species.mass_per_charge
        )
        / METERS_PER_KILOMETER
    )
    return numba.typed.List(
        [
            StepResponse(
                mass_kg=float(species.mass_kg),
                central_speed=float(speed),
                central_effective_area=float(area),
                sg_passband=sg_passband,
                oa_passband=oa_passband,
                azimuthal_transmission=swapi_response.azimuthal_transmission,
                rtn_to_swapi=rotation,
            )
            for speed, area, sg_passband, oa_passband, rotation in zip(
                central_speeds,
                effective_areas,
                sg_passbands,
                oa_passbands,
                rotations,
                strict=True,
            )
        ]
    )
