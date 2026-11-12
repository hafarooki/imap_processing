"""Placeholder: SWAPI's instrument response ancillary tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.response.effective_area import (
    CentralEffectiveArea,
)
from imap_processing.swapi.forward_model.response.passband import PassbandFits


@dataclass(frozen=True, eq=False)
class SwapiResponseTables:
    """
    Placeholder: SWAPI's instrument response model, read from the ancillary tables.

    ESA steps are identified by their ESA voltage magnitude in V, as in the
    ancillary tables. ``SIMION_K_FACTOR`` converts a voltage to the central
    energy per charge of its passband only where an energy or speed is needed
    (:func:`~.measurement_species_response.measurement_species_responses` and the
    passband fits).

    Typical use: read the tables once, then build the responses of each chunk
    from them with
    :func:`~.measurement_species_response.measurement_species_responses`::

        tables = SwapiResponseTables.from_files(...)
        responses = measurement_species_responses(
            tables, chunk_esa_voltages, chunk_epochs, species, chunk_rtn_to_swapi
        )

    :meth:`.PassbandFits.sweep_passbands` caches the passbands of each sweep, so
    chunks that share ESA voltages reuse them.

    From the L3a docs (``docs/swapi/response.md``, "Effective Area Function"): "The
    production code loads them from ancillary files. They are processed and cached in
    the ``SwapiResponse`` class."

    Attributes
    ----------
    azimuthal_transmission : AzimuthalTransmission
        Azimuthal transmission.
    central_effective_area : CentralEffectiveArea
        Central effective area and its time-variable relative efficiency.
    passband_fits : PassbandFits
        Passband fit of each aperture.
    """

    azimuthal_transmission: AzimuthalTransmission
    central_effective_area: CentralEffectiveArea
    passband_fits: PassbandFits

    @classmethod
    def from_files(
        cls,
        azimuthal_transmission_path: Path,
        central_effective_area_path: Path,
        passband_fit_coefficients_path: Path,
        efficiency_path: Path,
    ) -> SwapiResponseTables:
        """
        Placeholder: Load the ancillary response tables.

        Parameters
        ----------
        azimuthal_transmission_path : pathlib.Path
            The ``azimuthal-transmission`` ancillary CSV.
        central_effective_area_path : pathlib.Path
            The ``central-effective-area`` ancillary CSV.
        passband_fit_coefficients_path : pathlib.Path
            The ``passband-fit-coefficients`` ancillary CSV.
        efficiency_path : pathlib.Path
            The ``efficiency-lut`` ancillary file.

        Returns
        -------
        SwapiResponseTables
            The tables.
        """
        return cls(
            azimuthal_transmission=AzimuthalTransmission.from_csv(
                azimuthal_transmission_path
            ),
            central_effective_area=CentralEffectiveArea.from_files(
                central_effective_area_path, efficiency_path
            ),
            passband_fits=PassbandFits.from_csv(passband_fit_coefficients_path),
        )
