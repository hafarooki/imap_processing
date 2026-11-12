"""Response function information per measurement and species."""

from __future__ import annotations

from typing import NamedTuple, cast

import numpy as np
import numpy.typing as npt
from numba.typed.typedlist import List as TypedList

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


class MeasurementResponse(NamedTuple):
    """
    Response function information for a particular measurement and species.

    A measurement refers to a particular coincidence rate for a specific sweep and ESA
    step, associated with a particular measurement time.

    Attributes
    ----------
    esa_voltage : float
        ESA voltage magnitude of the measurement in V. Identifies the
        measurement; not used by the integration.
    epoch : int
        Measurement time in TT J2000 ns. Identifies the measurement; not used
        by the integration.
    mass_kg : float
        Species mass in kg.
    central_speed : float
        Speed at the center of the passband in km/s.
    central_effective_area : float
        Effective area at 0 elevation for the measurement's ESA voltage,
        accounting for time-variable relative efficiency.
    sg_passband : Passband
        Sunglasses passband for the measurement's ESA voltage.
    oa_passband : Passband
        Open aperture passband for the measurement's ESA voltage.
    azimuthal_transmission : AzimuthalTransmission
        Azimuthal transmission (included for convenience; independent of ESA
        voltage).
    rtn_to_swapi : numpy.ndarray
        Rotation matrix (3, 3) from RTN to SWAPI coordinates, accounting for the
        measurement time.
    """

    esa_voltage: float
    epoch: int
    mass_kg: float
    central_speed: float
    central_effective_area: float
    sg_passband: Passband
    oa_passband: Passband
    azimuthal_transmission: AzimuthalTransmission
    rtn_to_swapi: npt.NDArray[np.float64]


def measurement_responses(
    swapi_response: SwapiResponse,
    esa_voltages: npt.ArrayLike,
    epoch: int | npt.ArrayLike,
    species: Species,
    rtn_to_swapi: npt.ArrayLike,
) -> TypedList:
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
    epoch : int or numpy.ndarray
        Measurement time in TT J2000 ns, either one shared by every sample or
        one per sample (n,), in the same order as the flattened
        ``esa_voltages``. Sets each sample's relative efficiency.
    species : Species
        Ion species.
    rtn_to_swapi : numpy.ndarray
        Rotation matrix from RTN to the SWAPI frame, either one (3, 3)
        shared by every sample or one per sample (n, 3, 3), in the same
        order as the flattened ``esa_voltages``.

    Returns
    -------
    numba.typed.List[MeasurementResponse]
        The response of each sample, sharing the passbands of samples at
        the same voltage. The central effective area is
        interpolated linearly in voltage and held at the table's end values
        outside it, then scaled by the species' relative efficiency in
        effect at the sample's ``epoch``.

    Raises
    ------
    ValueError
        If there are no voltages, a voltage is not positive and finite, the
        epochs' shape matches neither () nor (n,), or the rotations' shape
        matches neither (3, 3) nor (n, 3, 3).
    """
    esa_voltages = np.ascontiguousarray(esa_voltages, dtype=float).ravel()
    if esa_voltages.size == 0:
        raise ValueError("No ESA voltages given.")
    if not np.all(np.isfinite(esa_voltages) & (esa_voltages > 0)):
        raise ValueError("ESA voltages must be positive and finite.")
    epochs = np.asarray(epoch, dtype=np.int64)
    if epochs.shape not in ((), (esa_voltages.size,)):
        raise ValueError(
            f"Epochs must have shape () or ({esa_voltages.size},), not {epochs.shape}."
        )
    epochs = np.broadcast_to(epochs, esa_voltages.shape)
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
    # Claude: a sweep's samples mostly share one efficiency row, so look each
    # distinct epoch up once
    unique_epochs, i_unique = np.unique(epochs, return_inverse=True)
    efficiencies = np.array(
        [swapi_response.relative_efficiency_at(int(e), species) for e in unique_epochs]
    )[i_unique]

    sg_passbands, oa_passbands = swapi_response.sweep_passbands(esa_voltages)
    table = swapi_response.central_effective_area
    effective_areas = (
        np.interp(esa_voltages, table.esa_voltage.values, table.values) * efficiencies
    )
    # The central speed is v_0^s = sqrt(2 k* q^s |V| / m^s), where
    # k* = SIMION_K_FACTOR(1.88) is the peak (E/q)/|V| at 0 elevation from
    # high-resolution SIMION simulations.
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
    # Claude: numba's List.__new__ returns a plain list when NUMBA_DISABLE_JIT
    # is set, so its inferred type is ``list | List``; the cast narrows it
    return cast(
        TypedList,
        TypedList(
            [
                MeasurementResponse(
                    esa_voltage=float(esa_voltage),
                    epoch=int(sample_epoch),
                    mass_kg=float(species.mass_kg),
                    central_speed=float(speed),
                    central_effective_area=float(area),
                    sg_passband=sg_passband,
                    oa_passband=oa_passband,
                    azimuthal_transmission=swapi_response.azimuthal_transmission,
                    rtn_to_swapi=rotation,
                )
                for (
                    esa_voltage,
                    sample_epoch,
                    speed,
                    area,
                    sg_passband,
                    oa_passband,
                    rotation,
                ) in zip(
                    esa_voltages,
                    epochs,
                    central_speeds,
                    effective_areas,
                    sg_passbands,
                    oa_passbands,
                    rotations,
                    strict=True,
                )
            ]
        ),
    )
