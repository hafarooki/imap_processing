"""Placeholder: SWAPI's central effective area, scaled by relative efficiency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from imap_processing.spice.time import et_to_ttj2000ns, str_to_et
from imap_processing.swapi.constants import Species


@dataclass(frozen=True, eq=False)
class CentralEffectiveArea:
    """
    Placeholder: Central effective area, scaled by the time-variable efficiency.

    Attributes
    ----------
    unscaled_area : xarray.DataArray
        Effective area at elevation 0 along an ``esa_voltage`` coordinate in
        V, as tabulated, before scaling by the relative efficiency. Use
        ``evaluate_scaled`` for the scaled area.
    relative_efficiency : xarray.DataArray
        Relative efficiency (efficiency_epoch, ion) with ``efficiency_epoch``
        in TT J2000 ns, sorted.
    """

    unscaled_area: xr.DataArray
    relative_efficiency: xr.DataArray

    @classmethod
    def from_files(
        cls, central_effective_area_path: Path, efficiency_path: Path
    ) -> CentralEffectiveArea:
        """
        Placeholder: Load the central effective area and efficiency tables.

        Parameters
        ----------
        central_effective_area_path : pathlib.Path
            The ``central-effective-area`` ancillary CSV.
        efficiency_path : pathlib.Path
            The ``efficiency-lut`` ancillary file.

        Returns
        -------
        CentralEffectiveArea
            The central effective area.
        """
        return cls(
            unscaled_area=_read_central_effective_area(central_effective_area_path),
            relative_efficiency=_read_relative_efficiency(efficiency_path),
        )

    def evaluate_scaled(
        self,
        esa_voltages: npt.ArrayLike,
        epochs: npt.ArrayLike,
        species: Species,
    ) -> npt.NDArray[np.float64]:
        """
        Placeholder: Evaluate the scaled central effective area of each sample.

        The tabulated area is interpolated linearly in voltage and held at the
        table's end values outside it, as in the L3a docs (``docs/swapi/response.md``,
        "Central Effective Area and Azimuthal Transmission"): "``SwapiResponse``
        evaluates this curve by linear interpolation in |V| and uses the nearest
        tabulated value outside the CSV range."

        It is then scaled by the species' relative efficiency, interpolated
        linearly in time between the efficiency table's rows and held at the
        first or last row's value outside them. This deviates from the L3a docs,
        which take "the most recent entry whose timestamp precedes t". He+ and He++
        share the helium column.

        Parameters
        ----------
        esa_voltages : numpy.ndarray
            ESA voltage magnitude of each sample in V.
        epochs : numpy.ndarray
            Time of each sample in TT J2000 ns, broadcastable against
            ``esa_voltages``.
        species : Species
            Ion species.

        Returns
        -------
        numpy.ndarray
            The scaled effective area of each sample, with the broadcast shape of
            ``esa_voltages`` and ``epochs``.
        """
        efficiency = self.relative_efficiency
        row_epochs = efficiency.efficiency_epoch.values
        epochs = np.asarray(epochs, dtype=np.int64)
        ion = "H" if species is Species.PROTON else "He"
        # Placeholder: interpolate relative to the first row so the int64 ns
        # epochs keep their precision as float64
        relative_efficiencies = np.interp(
            (epochs - row_epochs[0]).astype(float),
            (row_epochs - row_epochs[0]).astype(float),
            efficiency.sel(ion=ion).values,
        )

        unscaled_areas = np.interp(
            esa_voltages,
            self.unscaled_area.esa_voltage.values,
            self.unscaled_area.values,
        )
        return np.asarray(unscaled_areas * relative_efficiencies, dtype=float)


def _read_central_effective_area(central_effective_area_path: Path) -> xr.DataArray:
    """
    Placeholder: Read the central effective area table.

    Parameters
    ----------
    central_effective_area_path : pathlib.Path
        The ``central-effective-area`` ancillary CSV.

    Returns
    -------
    xarray.DataArray
        Effective area at elevation 0 along an ``esa_voltage`` coordinate in
        V.
    """
    table = pd.read_csv(central_effective_area_path)
    return xr.DataArray(
        table["effective_area"].to_numpy(dtype=float),
        dims=["esa_voltage"],
        coords={
            "esa_voltage": (
                "esa_voltage",
                table["esa_voltage"].to_numpy(dtype=float),
                {"units": "V"},
            )
        },
    )


def _read_relative_efficiency(efficiency_path: Path) -> xr.DataArray:
    """
    Placeholder: Read the relative efficiency table.

    From the L3a docs (``docs/swapi/response.md``, "Central Effective Area and Azimuthal
    Transmission"): "The efficiency calibration table has two columns: epsilon_H for
    hydrogen and epsilon_He for helium."

    Parameters
    ----------
    efficiency_path : pathlib.Path
        The ``efficiency-lut`` ancillary file: whitespace-separated columns of
        UTC time, MET, H efficiency and He efficiency, with ``#`` comments.

    Returns
    -------
    xarray.DataArray
        Relative efficiency (efficiency_epoch, ion) with ``efficiency_epoch``
        in TT J2000 ns.
    """
    table = pd.read_csv(
        efficiency_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["utc", "met", "H", "He"],
    ).sort_values("utc")
    epochs = et_to_ttj2000ns(str_to_et(table["utc"].to_numpy(dtype=str)))
    return xr.DataArray(
        table[["H", "He"]].to_numpy(dtype=float),
        dims=["efficiency_epoch", "ion"],
        coords={
            "efficiency_epoch": (
                "efficiency_epoch",
                np.asarray(epochs, dtype=np.int64),
                {"units": "ns"},
            ),
            "ion": ["H", "He"],
        },
    )
