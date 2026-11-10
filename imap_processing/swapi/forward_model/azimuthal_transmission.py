"""SWAPI's azimuthal transmission."""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import (
    OA_PLATEAU_AZIMUTH_MAX_DEG,
    OA_PLATEAU_AZIMUTH_MIN_DEG,
    OA_PLATEAU_TRANSMISSION,
    SG_PLATEAU_AZIMUTH_MAX_DEG,
    SG_PLATEAU_TRANSMISSION,
)


class AzimuthalTransmission(NamedTuple):
    """
    Azimuthal transmission tabulated on a uniform grid starting at 0 deg.

    Attributes
    ----------
    values : numpy.ndarray
        Transmission at azimuths ``0, spacing, 2 * spacing, ...``.
    spacing : float
        Azimuth spacing of ``values`` in deg.
    """

    values: npt.NDArray[np.float64]
    spacing: float

    @classmethod
    def from_table(
        cls, azimuths: npt.ArrayLike, transmission: npt.ArrayLike
    ) -> AzimuthalTransmission:
        """
        Build from a table, checking its grid and plateaus.

        The table must be on a uniform grid from 0 deg, and its values inside
        the sunglasses and open-aperture plateaus must equal the plateau
        transmissions that ``interpolate_azimuthal_transmission`` returns there.

        Parameters
        ----------
        azimuths : numpy.ndarray
            Absolute azimuths in deg.
        transmission : numpy.ndarray
            Transmission at ``azimuths``.

        Returns
        -------
        AzimuthalTransmission
            The transmission.

        Raises
        ------
        ValueError
            If ``azimuths`` is not ``0, spacing, 2 * spacing, ...``, if
            ``transmission`` does not match ``azimuths`` in size, or if the
            table disagrees with a plateau transmission.
        """
        azimuths = np.asarray(azimuths, dtype=float)
        if azimuths.size < 2:
            raise ValueError("Azimuthal transmission needs at least two azimuths.")
        spacing = (azimuths[-1] - azimuths[0]) / (azimuths.size - 1)
        if not np.allclose(
            azimuths, spacing * np.arange(azimuths.size), rtol=0, atol=1e-9
        ):
            raise ValueError(
                "Azimuthal transmission must be on a uniform grid starting at 0 deg."
            )
        values = np.ascontiguousarray(transmission, dtype=float)
        if values.shape != azimuths.shape:
            raise ValueError(
                "Azimuthal transmission must have one value per azimuth, got "
                f"{values.size} values for {azimuths.size} azimuths."
            )
        plateaus = (
            ("sunglasses", 0.0, SG_PLATEAU_AZIMUTH_MAX_DEG, SG_PLATEAU_TRANSMISSION),
            (
                "open-aperture",
                OA_PLATEAU_AZIMUTH_MIN_DEG,
                OA_PLATEAU_AZIMUTH_MAX_DEG,
                OA_PLATEAU_TRANSMISSION,
            ),
        )
        for name, lo, hi, plateau in plateaus:
            in_plateau = (azimuths >= lo - 1e-9) & (azimuths <= hi + 1e-9)
            if not np.allclose(values[in_plateau], plateau):
                raise ValueError(
                    f"Azimuthal transmission must be {plateau} on the {name} "
                    f"plateau ({lo} to {hi} deg)."
                )
        return cls(values=values, spacing=float(spacing))


@numba.njit  # type: ignore[untyped-decorator]
def interpolate_azimuthal_transmission(
    transmission: AzimuthalTransmission, azimuth: float
) -> float:
    """
    Interpolate the azimuthal transmission at an azimuth.

    Parameters
    ----------
    transmission : AzimuthalTransmission
        The azimuthal transmission table.
    azimuth : float
        Look azimuth in deg.

    Returns
    -------
    float
        Transmission, the plateau values inside the plateaus and linear
        interpolation in ``abs(azimuth)`` elsewhere.
    """
    azimuth = (azimuth + 180) % 360 - 180
    abs_azimuth = abs(azimuth)

    if OA_PLATEAU_AZIMUTH_MIN_DEG <= abs_azimuth <= OA_PLATEAU_AZIMUTH_MAX_DEG:
        return OA_PLATEAU_TRANSMISSION
    if abs_azimuth <= SG_PLATEAU_AZIMUTH_MAX_DEG:
        return SG_PLATEAU_TRANSMISSION

    i_float = abs_azimuth / transmission.spacing
    i_lower = math.floor(i_float)
    i_upper = i_lower + 1

    n = len(transmission.values)
    i_lower = min(i_lower, n - 1)
    i_upper = min(i_upper, n - 1)

    weight_lower = float(i_upper) - i_float
    weight_upper = i_float - float(i_lower)
    return (
        transmission.values[i_lower] * weight_lower
        + transmission.values[i_upper] * weight_upper
    )
