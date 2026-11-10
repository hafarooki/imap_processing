"""Functions to support I-ALiRT SWAPI processing."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.ialirt.constants import IalirtSwapiConstants as Consts
from imap_processing.ialirt.utils.grouping import (
    _populate_instrument_header_items,
    find_groups,
)
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.swapi.constants import Species
from imap_processing.swapi.fitting.fit_dual_maxwellian_no_spin import (
    fit_dual_maxwellian_no_spin,
)
from imap_processing.swapi.fitting.fit_single_maxwellian_with_spin import (
    fit_single_maxwellian_with_spin,
)
from imap_processing.swapi.forward_model.response_tables import SwapiResponse
from imap_processing.swapi.l1.swapi_l1 import process_sweep_data
from imap_processing.swapi.l2.swapi_l2 import SWAPI_LIVETIME

logger = logging.getLogger(__name__)

# single-Maxwellian temperature above which the chunk is refit with a core plus
# suprathermal Maxwellian
DUAL_FIT_TEMPERATURE_THRESHOLD_K = 3e5


def process_swapi_ialirt(
    unpacked_data: xr.Dataset,
    calibration_lut_table: pd.DataFrame,
    response: SwapiResponse,
) -> list[dict]:
    """
    Extract I-ALiRT variables and calculate coincidence count rate.

    Parameters
    ----------
    unpacked_data : xr.Dataset
        SWAPI I-ALiRT data that has been parsed from the spacecraft packet.
    calibration_lut_table : pd.DataFrame
        DataFrame containing the contents of the SWAPI esa-unit-conversion lookup table.
    response : SwapiResponse
        SWAPI instrument response tables used by the forward-model fits.

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

        chunk_epoch = int(met_to_ttj2000ns(chunk.sweep_center_met.mean().item()))
        fit_result = fit_chunk(
            response=response,
            esa_energies=chunk.esa_energy.values,
            count_rates=chunk.count_rate.values,
            epoch=chunk_epoch,
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
                "swapi_epoch": chunk_epoch,
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
    response: SwapiResponse,
    esa_energies: NDArray[np.floating],
    count_rates: NDArray[np.floating],
    epoch: int,
) -> dict[str, float | bool]:
    """
    Fit proton Maxwellian(s) to a chunk of sweeps using the SWAPI forward model.

    A single Maxwellian flowing off the spin axis is fit first. If its
    temperature exceeds ``DUAL_FIT_TEMPERATURE_THRESHOLD_K``, the chunk is refit
    with a core plus suprathermal Maxwellian along the spin axis, and the
    combined density and temperature of that fit are reported instead.

    Parameters
    ----------
    response : SwapiResponse
        SWAPI instrument response tables.
    esa_energies : numpy.ndarray
        ESA energy of each step [eV], with shape (sweep, step) = (5, 72).
    count_rates : numpy.ndarray
        Decompressed observed coincidence rates [counts/s], with shape
        (sweep, step) = (5, 72).
    epoch : int
        Center time of the chunk in TT J2000 ns, for the efficiency lookup.

    Returns
    -------
    dict[str, float | bool]
        ``success``, and ``speed`` [km/s], ``density`` [cm^-3] and
        ``temperature`` [K] when successful.
    """
    try:
        passbands = response.passbands(esa_energies)
    except ValueError:
        logger.warning("SWAPI I-ALiRT chunk has invalid ESA energies.")
        return {"success": False}
    responses = response.step_responses(passbands, esa_energies, epoch, Species.PROTON)

    fit = fit_single_maxwellian_with_spin(responses, count_rates)
    if fit["temperature"] > DUAL_FIT_TEMPERATURE_THRESHOLD_K:
        logger.debug(
            f"SWAPI I-ALiRT single fit temperature {fit['temperature']:.3g} K "
            f"exceeds {DUAL_FIT_TEMPERATURE_THRESHOLD_K:.3g} K; using dual fit."
        )
        fit = fit_dual_maxwellian_no_spin(responses, count_rates)

    result = {key: float(fit[key]) for key in ("speed", "density", "temperature")}
    if not all(np.isfinite(value) for value in result.values()):
        logger.warning("SWAPI I-ALiRT fit gave non-finite moments.")
        return {"success": False}
    return {"success": True, **result}
