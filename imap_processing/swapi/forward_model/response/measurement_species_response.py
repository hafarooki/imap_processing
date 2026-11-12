"""Response function information per measurement and species."""

from __future__ import annotations

from typing import NamedTuple

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
from imap_processing.swapi.forward_model.response.response_tables import (
    SwapiResponseTables,
)


class MeasurementSpeciesResponse(NamedTuple):
    """
    Response function information for a particular measurement and species.

    A measurement refers to a particular coincidence rate for a specific sweep and ESA
    step, associated with a particular measurement time.

    Attributes
    ----------
    esa_voltage : float
        ESA voltage magnitude of the measurement in V.
    epoch : int
        Measurement time in TT J2000 ns.
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


def measurement_species_responses(
    swapi_response_tables: SwapiResponseTables,
    esa_voltages: npt.ArrayLike,
    epochs: npt.ArrayLike,
    species: Species,
    rtn_to_swapi: npt.ArrayLike,
) -> TypedList[MeasurementSpeciesResponse]:
    """
    Compute response to a given species for a particular set of measurements.

    Parameters
    ----------
    swapi_response_tables : SwapiResponseTables
        Instrument response tables.
    esa_voltages : numpy.ndarray
        ESA voltage magnitude (in volts) for each sample.
    epochs : numpy.ndarray
        Measurement time in TT J2000 ns of each sample (n,).
    species : Species
        Ion species.
    rtn_to_swapi : numpy.ndarray
        Rotation matrix from RTN to the SWAPI frame of each sample (n, 3, 3).

    Returns
    -------
    numba.typed.List[MeasurementResponse]
        The response of each sample, sharing the passbands of samples at
        the same voltage. The central effective area is from
        :meth:`SwapiResponseTables.central_effective_area_at`.

    Raises
    ------
    ValueError
        If there are no voltages, a voltage is not positive and finite, the
        epochs' shape is not (n,), or the rotations' shape is not (n, 3, 3).
    RuntimeError
        If numba's JIT is disabled, so no typed list can be built.
    """
    esa_voltages = np.ascontiguousarray(esa_voltages, dtype=float).ravel()
    if esa_voltages.size == 0:
        raise ValueError("No ESA voltages given.")
    if not np.all(np.isfinite(esa_voltages) & (esa_voltages > 0)):
        raise ValueError("ESA voltages must be positive and finite.")

    epochs = np.asarray(epochs, dtype=np.int64)
    if epochs.shape != esa_voltages.shape:
        raise ValueError(
            f"Epochs must have shape {esa_voltages.shape}, not {epochs.shape}."
        )

    # C-ordered float64 for performance and stability reasons
    rotations = np.array(rtn_to_swapi, dtype=float, order="C")
    if rotations.shape != (esa_voltages.size, 3, 3):
        raise ValueError(
            f"RTN-to-SWAPI rotations must have shape ({esa_voltages.size}, 3, 3), "
            f"not {rotations.shape}."
        )

    sg_passbands, oa_passbands = swapi_response_tables.sweep_passbands(esa_voltages)
    effective_areas = swapi_response_tables.central_effective_area_at(
        esa_voltages, epochs, species
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

    responses = TypedList(
        [
            MeasurementSpeciesResponse(
                esa_voltage=float(esa_voltages[i]),
                epoch=int(epochs[i]),
                mass_kg=float(species.mass_kg),
                central_speed=float(central_speeds[i]),
                central_effective_area=float(effective_areas[i]),
                sg_passband=sg_passbands[i],
                oa_passband=oa_passbands[i],
                azimuthal_transmission=swapi_response_tables.azimuthal_transmission,
                rtn_to_swapi=rotations[i],
            )
            for i in range(esa_voltages.size)
        ]
    )

    if not isinstance(responses, TypedList):
        raise RuntimeError("Response list must be a numba typed List."
                           "Please ensure NUMBA_DISABLE_JIT is disabled.")

    return responses
