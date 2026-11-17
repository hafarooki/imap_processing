"""
Handling for the central effective area component of SWAPI's instrument response.

Central effective area ``Aeff(V)`` is defined as the ratio of count rate (C) to particle
flux ``nv`` for a uniform beam of energy ``k * |V|`` incident on SWAPI through the open
aperture at elevation angle ``theta = 0``, where ``k = 1.88`` is the SIMION-calibrated
k-factor, provided that SWAPI's ESA voltage setting is ``V``.

This module handles the ESA voltage-dependent central effective area from before launch,
as well as the time variability as determined through changes in the efficiency table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from imap_processing.spice.time import et_to_ttj2000ns, str_to_et
from imap_processing.swapi.constants import Species


class CentralEffectiveAreaTable:
    """
    Central effective area tables and interpolation.

    Handles storing and interpolating the central effective area-related tables: lab
    central effective area (as a function of ESA voltage) and relative efficiency (as a
    function of time).

    Parameters
    ----------
    central_effective_area_path : pathlib.Path
        The ``central-effective-area`` ancillary file path.
    efficiency_path : pathlib.Path
        The ``efficiency-lut`` ancillary file path.

    Attributes
    ----------
    lab_area : xarray.DataArray
        Lab-calibrated central effective area.
    relative_efficiency : xarray.DataArray
        Relative efficiency for hydrogen and helium.
    """

    def __init__(
        self, central_effective_area_path: Path, efficiency_path: Path
    ) -> None:
        self.lab_area = _read_central_effective_area(central_effective_area_path)
        self.relative_efficiency = _read_relative_efficiency(efficiency_path)

    def evaluate(
        self,
        esa_voltage: npt.ArrayLike,
        epoch: npt.ArrayLike,
        species: Species,
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate the adjusted central effective area of each sample.

        The tabulated area is interpolated linearly in voltage and held at the
        table's end values outside it. It is then multiplied by the species'
        relative efficiency, taken from the most recent row of the efficiency table
        whose timestamp is at or before the sample time.

        ``esa_voltage`` and ``epoch`` may have any shape but their shapes must match.

        Parameters
        ----------
        esa_voltage : numpy.ndarray
            ESA voltage magnitude of each sample in V.
        epoch : numpy.ndarray
            Time of each sample in TT J2000 ns.
        species : Species
            Species being modeled.

        Returns
        -------
        numpy.ndarray
            The adjusted effective area, matching the input shape.

        Raises
        ------
        ValueError
            If ``esa_voltage`` and ``epoch`` have different shapes.
        KeyError
            If any epoch precedes the first row of the efficiency table.
        """
        esa_voltage = np.asarray(esa_voltage, dtype=float)
        epoch = np.asarray(epoch, dtype=np.int64)
        if esa_voltage.shape != epoch.shape:
            raise ValueError(
                f"esa_voltage.shape {esa_voltage.shape} does not match "
                f"epoch.shape {epoch.shape}."
            )

        element = "H" if species is Species.PROTON else "He"
        relative_efficiency = self.relative_efficiency.sel(
            element=element,
            # Wrapped in a DataArray in case of multidimensional inputs
            efficiency_epoch=xr.DataArray(epoch),
            method="ffill",
        ).values

        lab_area = self.lab_area.interp(
            # Wrapped in a DataArray in case of multidimensional inputs
            esa_voltage=xr.DataArray(esa_voltage),
            kwargs={
                "fill_value": (
                    self.lab_area.values[0],
                    self.lab_area.values[-1],
                )
            },
        ).values
        return np.asarray(lab_area * relative_efficiency, dtype=float)


def _read_central_effective_area(central_effective_area_path: Path) -> xr.DataArray:
    """Read the central effective area table."""
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
    """Read the relative efficiency table."""
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
        dims=["efficiency_epoch", "element"],
        coords={
            "efficiency_epoch": (
                "efficiency_epoch",
                np.asarray(epochs, dtype=np.int64),
                {"units": "ns"},
            ),
            "element": ["H", "He"],
        },
    )
