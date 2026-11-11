"""Functions to support I-ALiRT SWAPI processing."""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.swapi.l1.swapi_l1 import process_sweep_data
from imap_processing.swapi.l2.swapi_l2 import SWAPI_LIVETIME

logger = logging.getLogger(__name__)

# TODO figure out where this belongs
SWEEP_CADENCE_SECONDS = 12.0


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

    sweeps: xr.Dataset | None = complete_sweeps(unpacked_data, calibration_lut_table)

    # return empty if fewer than 5 sweeps
    if sweeps is None or sweeps.sizes["sweep"] < 5:
        return []

    # All windows of 5 consecutive sweeps
    windows = (
        sweeps.reset_coords().rename(sweep="chunk").rolling(chunk=5).construct("sweep")
    )
    # A window is usable if the sweeps are consecutive
    gaps = windows["sweep_start_time"].diff("sweep")
    usable = (abs(gaps - SWEEP_CADENCE_SECONDS) < 0.05).all("sweep")
    chunks = (
        windows.isel(chunk=usable)
        .set_coords("sweep_start_time")
        .transpose("chunk", "sweep", ...)
    )


def complete_sweeps(
    unpacked_data: xr.Dataset, calibration_lut_table: pd.DataFrame
) -> xr.Dataset | None:
    """
    Extract per-sweep data from raw I-ALiRT data (coarse steps only).

    Parameters
    ----------
    unpacked_data : xr.Dataset
        SWAPI I-ALiRT data that has been parsed from the spacecraft packet.
    calibration_lut_table : pd.DataFrame
        DataFrame containing the contents of the SWAPI esa-unit-conversion lookup table.

    Returns
    -------
    sweeps : xr.Dataset or None
        All complete sweeps, sorted by time, with 62 coarse ESA steps.
        None if there are no complete sweeps.
        - Data vars
            * count_rate (sweep, esa_step=62)
            * esa_energy (sweep, esa_step=62)
        - Coords
            * sweep_start_time (sweep)
    """
    sci_dataset = unpacked_data.sortby("epoch", ascending=True)

    # add met to sci_dataset since it's needed by find_groups
    sci_dataset["met"] = calculate_time(
        sci_dataset["sc_sclk_sec"], sci_dataset["sc_sclk_sub_sec"], 256
    )

    grouped_dataset = find_groups(sci_dataset, (0, 11), "swapi_seq_number", "met")

    if grouped_dataset.group.size == 0:
        logger.warning(
            "There was an issue with the SWAPI grouping process, returning None."
        )
        return None

    complete_groups = []
    incomplete_groups = []
    for group, sweep in grouped_dataset.groupby("group"):
        # Sequence values for the group. Should be 0-11 with no duplicates.
        seq_values = sweep["swapi_seq_number"].values.astype(int)

        # Complete if all sequence numbers 0-11 are present, in order, with no
        # duplicates. Otherwise, the sweep is incomplete.
        if np.array_equal(seq_values, np.arange(12)):
            complete_groups.append(int(group))
        else:
            incomplete_groups.append(int(group))

    if incomplete_groups:
        logger.info(
            f"The following swapi groups were skipped due to "
            f"missing or duplicate pkt_counter values: "
            f"{incomplete_groups}"
        )

    if not complete_groups:
        return None

    is_complete_packet = grouped_dataset["group"].isin(complete_groups)
    complete_sweep_packets = grouped_dataset.isel(epoch=is_complete_packet)

    # process_sweep_data reorders all complete sweeps at once -> (n_sweeps, 72).
    compressed_counts = np.asarray(
        process_sweep_data(complete_sweep_packets, "swapi_coin_cnt")
    )
    decompressed_counts = compressed_counts * 16
    count_rates = decompressed_counts[:, 1:63] / SWAPI_LIVETIME

    # Take one packet per sweep (the first one, seq no. 0);
    # assume that the metadata is the same for all of them
    first_packets = complete_sweep_packets.isel(
        epoch=complete_sweep_packets["swapi_seq_number"] == 0
    )
    table_ids = first_packets["swapi_version"].values.astype(int)
    esa_energies = np.stack(
        [voltages_for_table(calibration_lut_table, table_id) for table_id in table_ids]
    )

    start_times = first_packets["swapi_acq"].values.astype(float)

    return xr.Dataset(
        data_vars={
            "count_rate": (["sweep", "esa_step"], count_rates),
            "esa_energy": (["sweep", "esa_step"], esa_energies),
        },
        coords={"sweep_start_time": ("sweep", start_times)},
    )


def voltages_for_table(
    calibration_lut_table: pd.DataFrame, table_id: int
) -> np.ndarray:
    """
    Look up the coarse ESA energies for a sweep table.

    Parameters
    ----------
    calibration_lut_table : pd.DataFrame
        SWAPI esa-unit-conversion lookup table.
    table_id : int
        ID of the ESA voltage sweep table. The packet refers to this as
        ``swapi_version``. The table refers to it as "Sweep #".

    Returns
    -------
    esa_voltages : np.ndarray
        Absolute value ESA voltage [V] for the 62 coarse steps.
    """
    subset = calibration_lut_table[calibration_lut_table["Sweep #"] == table_id]
    # Use the most recent entry for this sweep table.
    timestamps = pd.to_datetime(subset["timestamp"])
    subset = subset[timestamps == timestamps.max()]
    if subset.empty:
        raise ValueError(
            f"No esa unit conversion available for sweep table {table_id}. "
            f"Check lookup table?"
        )
    subset = subset.sort_values("ESA Step #")
    esa_voltages: np.ndarray = np.abs(subset["Voltage"].to_numpy()[1:63].astype(float))
    return esa_voltages
