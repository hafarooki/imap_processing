"""Functions to support I-ALiRT SWAPI processing."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from scipy.optimize import least_squares

from imap_processing.ialirt.constants import IalirtSwapiConstants as Consts
from imap_processing.ialirt.utils.grouping import (
    _populate_instrument_header_items,
    find_groups,
)
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.swapi.l1.swapi_l1 import process_sweep_data
from imap_processing.swapi.l2.swapi_l2 import SWAPI_LIVETIME

logger = logging.getLogger(__name__)


def process_swapi_ialirt(
    unpacked_data: xr.Dataset, calibration_lut_table: pd.DataFrame
) -> list[dict]:
    """
    Extract I-ALiRT variables and calculate coincidence count rate.

    Parameters
    ----------
    unpacked_data : xr.Dataset
        SWAPI I-ALiRT data that has been parsed from the spacecraft packet.
    calibration_lut_table : pd.DataFrame
        DataFrame containing the contents of the SWAPI esa-unit-conversion lookup table.

    Returns
    -------
    swapi_data : dict
        Dictionary containing all data variables for SWAPI I-ALiRT product.
    """
    logger.info("Processing SWAPI.")

    sci_dataset = unpacked_data.sortby("epoch", ascending=True)

    met = calculate_time(
        sci_dataset["sc_sclk_sec"], sci_dataset["sc_sclk_sub_sec"], 256
    )

    # Add required parameters.
    sci_dataset["met"] = met

    # list for accumulating incomplete groups that need to be logged for
    # diagnostic purposes
    incomplete_groups = []

    # list for accumulating sweeps to be fit once a full chunk of 5 sweeps
    # is accumulated
    sweeps = []

    # list of fit outputs
    swapi_data = []

    # Extract energy values from the calibration lookup table file
    calibration_lut_table["timestamp"] = pd.to_datetime(
        calibration_lut_table["timestamp"]
    )

    grouped_dataset = find_groups(sci_dataset, (0, 11), "swapi_seq_number", "met")

    if grouped_dataset.group.size == 0:
        logger.warning(
            "There was an issue with the SWAPI grouping process, returning empty data."
        )
        return []

    for group in np.unique(grouped_dataset["group"]):
        # Sequence values for the group should be 0-11 with no duplicates.
        seq_values = grouped_dataset["swapi_seq_number"][
            (grouped_dataset["group"] == group)
        ]
        met = grouped_dataset["met"][(grouped_dataset["group"] == group).values]

        swapi_met = grouped_dataset["swapi_acq"][
            (grouped_dataset["group"] == group).values
        ]
        # TODO confirm that this is the center MET
        mid_measurement = int((swapi_met[0] + swapi_met[-1]) // 2)

        # Ensure no duplicates and all values from 0 to 11 are present
        if not np.array_equal(seq_values.values.astype(int), np.arange(12)):
            incomplete_groups.append(group)
            continue

        grouped_subset = grouped_dataset.sel(epoch=grouped_dataset.group == group)

        compressed_counts = process_sweep_data(grouped_subset, "swapi_coin_cnt")
        # I-ALiRT packets have counts compressed by a factor of 16.
        # Add 8 to avoid having counts truncated to 0 and to avoid
        # counts being systematically too low
        # Subset to only the relevant I-ALiRT energy steps
        count_rates = (
            compressed_counts * Consts.compression_factor + Consts.compression_offset
        ) / SWAPI_LIVETIME

        sweep_id = int(grouped_subset["swapi_version"].values[0])
        subset_sweep = calibration_lut_table[
            calibration_lut_table["Sweep #"] == sweep_id
        ]

        # Find the sweep's energy data for the latest time
        subset = subset_sweep[
            (subset_sweep["timestamp"] == subset_sweep["timestamp"].max())
        ]
        if subset.empty:
            raise ValueError(
                f"No esa unit conversion available for sweep {sweep_id}. "
                f"Check lookup table?"
            )

        subset = subset.sort_values(["timestamp", "ESA Step #"])
        esa_energies = subset["Energy"].to_numpy().astype(float)

        sweeps.append(
            xr.Dataset(
                data_vars={
                    "count_rate": (["sweep", "esa_step"], [count_rates]),
                    "esa_energy": (["sweep", "esa_step"], [esa_energies]),
                    "sweep_center_met": (["sweep"], [mid_measurement]),
                },
                coords={
                    "esa_step": np.arange(72),
                },
            )
        )

        # can't fit if 5 sweeps haven't yet accumulated
        if len(sweeps) < 5:
            continue

        # skip sets of 5 sweeps that aren't consecutive
        chunk = xr.concat(sweeps[-5:], dim="sweep")
        is_consecutive = np.isclose(np.diff(chunk.sweep_center_met), 12.0, atol=0.05)
        if not np.all(is_consecutive):
            continue

        fit_result = fit_chunk(
            esa_energies=chunk.esa_energy.values,
            count_rates=chunk.count_rate.values,
        )

        if not fit_result["success"]:
            # TODO confirm this behavior
            fit_result = {"speed": np.nan, "density": np.nan, "temperature": np.nan}

        # TODO fit and convert to strings using Decimal (or f string?) as was
        # done before
        swapi_data.append(
            _populate_instrument_header_items(met)
            | {
                "instrument": "swapi",
                "swapi_epoch": int(met_to_ttj2000ns(chunk.sweep_center_met.mean().item())),
                "swapi_pseudo_proton_speed": fit_result["speed"],
                "swapi_pseudo_proton_density": fit_result["density"],
                "swapi_pseudo_proton_temperature": fit_result["temperature"],
            }
        )
    if incomplete_groups:
        logger.info(
            f"The following swapi groups were skipped due to "
            f"missing or duplicate pkt_counter values: "
            f"{incomplete_groups}"
        )

    return swapi_data


def fit_chunk(
    *,
    esa_energies: NDArray[np.floating],
    count_rates: NDArray[np.floating],
):
    """
    Fit Maxwellian to the observed chunks using the SWAPI forward model.

    Parameters
    ----------
    esa_energies : numpy.ndarray
        ESA energy of each step [eV], with shape (sweep, step) = (5, 72).
    count_rates : numpy.ndarray
        Decompressed observed coincidence rates [counts/s], with shape
        (sweep, step) = (5, 72).

    Returns
    -------
    A dict containing speed, density, and temperature.
    """
    # first, fit analytic gaussian to 5-sweep mean
    initial_guess = gaussian_initial_guess(
        esa_energies=esa_energies, count_rates=count_rates
    )

    if not initial_guess["success"]:
        return {"success": False}

    # then, fit forward model with boresight-aligned assumption to 5-sweep mean
    model = maxwellian_model(esa_energies)

    # curve fit with deadtime correction applied
    initial_guess["density"] = ...

    # TODO finish implementing this fit
    # TODO use analytic jacobian
    least_squares(
        lambda x: model(
            speed=np.exp(x[0]),
            density=np.exp(x[1]),
            temperature=np.exp(x[2]),
            azimuth=0,
            elevation=0,
        ),
        x0=[
            np.log(initial_guess["speed"]),
            np.log(initial_guess["density"]),
            np.log(initial_guess["temperature"]),
        ],
        xtol=1e-4,
    )

    # TODO use forward model to get final fit with bulk deflection accounted for

    # TODO basin hopping if necessary to avoid local minima... grid search
    # might be better though

    return initial_guess


def gaussian_initial_guess(
    *,
    esa_energies: NDArray[np.floating],
    count_rates: NDArray[np.floating],
) -> dict[str, float | bool]:
    """
    Estimate speed and temperature with a simple Gaussian fit.

    The count rate is modeled as:

        C(E) = A * E^2 * exp(-(v(E) - u)^2 / T),   v(E) = sqrt(2 e E / m_p)

    where ``A`` is an arbitrary coefficient.

    Parameters
    ----------
    esa_energies : numpy.ndarray
        ESA energy per charge [eV/e], with shape (sweep, step).
    count_rates : numpy.ndarray
        Observed coincidence rates [counts/s], with shape (sweep, step).

    Returns
    -------
    dict[str, float | bool]
        On success, ``{"success": True, "speed": [km/s], "temperature": [K]}``.
        On failure, ``{"success": False}``.
    """
    energy = np.asarray(esa_energies, dtype=float).mean(axis=0)
    rate = np.asarray(count_rates, dtype=float).mean(axis=0)

    valid = np.isfinite(energy) & np.isfinite(rate) & (energy > 0) & (rate > 0)
    if not np.any(valid):
        logger.warning("SWAPI I-ALiRT initial guess received no valid inputs.")
        return {"success": False}

    energy, rate = energy[valid], rate[valid]
    speed = np.sqrt(2 * energy * Consts.e_charge / Consts.prot_mass)  # m/s

    # log(R / E^2) = log(A) - beta * (v - u)^2 = a2 v^2 + a1 v + a0.
    y = np.log(rate / energy**2)

    # linear and quadratic term kept; constant coefficient is dropped
    try:
        a2, a1, _ = np.polyfit(speed, y, deg=2)
    except np.linalg.LinAlgError:
        logger.warning("SWAPI I-ALiRT initial guess fit did not converge.")
        return {"success": False}

    beta = -a2  # 1 / w^2 [s^2/m^2]
    if not beta > 0:
        logger.warning("SWAPI I-ALiRT initial guess gave invalid temperature.")
        return {"success": False}
    bulk_speed = a1 / (2 * beta)
    if not bulk_speed > 0:
        logger.warning("SWAPI I-ALiRT initial guess gave invalid bulk speed.")
        return {"success": False}
    temperature = Consts.prot_mass / (2 * Consts.boltz * beta)

    return {
        "success": True,
        "speed": float(bulk_speed / 1e3),
        "temperature": float(temperature),
    }
