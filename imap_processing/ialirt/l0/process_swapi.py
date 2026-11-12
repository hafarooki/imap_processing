"""Functions to support I-ALiRT SWAPI processing."""

import logging
from decimal import Decimal

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.ialirt.utils.grouping import (
    _populate_instrument_header_items,
    find_groups,
)
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns
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
    swapi_data : list[dict]
        A list of dictionaries, each dictionary containing all of the data variables for
        the SWAPI I-ALiRT product.
    """
    logger.info("Processing SWAPI.")

    sweeps: xr.Dataset | None = complete_sweeps(unpacked_data, calibration_lut_table)

    if sweeps is None:
        logger.info("no complete sweeps could be parsed, returning empty list")
        return []

    swapi_data: list[dict] = []

    for i in range(sweeps.sizes["sweep"] - 4):
        chunk = sweeps.isel(sweep=slice(i, i + 5))

        # skip if the 5 sweeps are not consecutive
        if not np.allclose(
            np.diff(chunk["sweep_start_met"].values),
            SWEEP_CADENCE_SECONDS,
            atol=0.05,
        ):
            continue

        speed, density, temperature = fit_chunk(chunk)

        # TODO this matches the behavior of the previous code, and the other
        #   instruments, not sure if intended
        sc_met = np.array(
            [
                chunk["sweep_start_sc_met"].values[0],
                chunk["sweep_end_sc_met"].values[-1],
            ]
        )

        # center of chunk time
        swapi_met = np.mean(chunk["sweep_start_met"].values) + SWEEP_CADENCE_SECONDS / 2

        swapi_data.append(
            _populate_instrument_header_items(sc_met)
            | {
                "instrument": "swapi",
                "swapi_epoch": int(met_to_ttj2000ns(swapi_met)),
                "swapi_pseudo_proton_speed": Decimal(f"{speed:.3f}"),
                "swapi_pseudo_proton_density": Decimal(f"{density:.3f}"),
                "swapi_pseudo_proton_temperature": Decimal(f"{temperature:.3f}"),
            }
        )

    return swapi_data


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
            * sweep_start_met (sweep): SWAPI acquisition time of the first
              packet of the sweep, MET [s]
            * sweep_start_sc_met (sweep): spacecraft time of the first packet
              of the sweep, MET [s]
            * sweep_end_sc_met (sweep): spacecraft time of the last packet
              of the sweep, MET [s]
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
    last_packets = complete_sweep_packets.isel(
        epoch=complete_sweep_packets["swapi_seq_number"] == 11
    )
    table_ids = first_packets["swapi_version"].values.astype(int)
    esa_energies = np.stack(
        [voltages_for_table(calibration_lut_table, table_id) for table_id in table_ids]
    )

    sweep_start_met = first_packets["swapi_acq"].values.astype(float)

    return xr.Dataset(
        data_vars={
            "count_rate": (["sweep", "esa_step"], count_rates),
            "esa_energy": (["sweep", "esa_step"], esa_energies),
        },
        coords={
            "sweep_start_met": ("sweep", sweep_start_met),
            "sweep_start_sc_met": ("sweep", first_packets["met"].values),
            "sweep_end_sc_met": ("sweep", last_packets["met"].values),
        },
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


def fit_chunk(chunk: xr.Dataset) -> tuple[float, float, float]:
    """
    Fit pseudo moments to a chunk of 5 consecutive sweeps (placeholder).

    Placeholder that inlines the per-sweep fit and geometric mean from
    ``process_swapi_old``: each sweep is fit separately, then the fits are
    combined with a geometric mean over the sweeps whose fit succeeded.

    Parameters
    ----------
    chunk : xr.Dataset
        Sweeps from ``complete_sweeps()``, with ``count_rate`` and
        ``esa_energy`` (ESA voltage) of shape (sweep, esa_step=62).

    Returns
    -------
    pseudo_moments : tuple[float, float, float]
        Pseudo proton speed [km/s], density [cm^-3] and temperature [K],
        each the geometric mean over the sweeps whose fit succeeded. All NaN
        if no sweep's fit succeeded.
    """
    # Claude: imports are local so that only this function changes; move them
    # Claude: to the top of the module when this stops being a placeholder.
    from scipy.optimize import curve_fit  # noqa: PLC0415
    from scipy.special import erf  # noqa: PLC0415

    from imap_processing.ialirt.constants import (  # noqa: PLC0415
        IalirtSwapiConstants as Consts,
    )

    # Claude: placeholder K factor to turn ESA voltage into energy per charge
    # Claude: [eV/q]. It is 1.93 for every coarse step of the flight sweep
    # Claude: tables, but should come from the LUT.
    k_factor = 1.93

    def model_count_rate(
        energy_pass: np.ndarray, speed: float, density: float, temp: float
    ) -> np.ndarray:
        # Claude: count_rate() from process_swapi_old, unchanged.
        thermal_velocity = np.sqrt(2 * Consts.boltz * temp / Consts.prot_mass)
        beta = 1 / (thermal_velocity**2)
        center_speed = np.sqrt(2 * energy_pass * 1.60218e-19 / Consts.prot_mass)
        speed = speed * 1000
        density = density * 1e6 * Consts.temporary_density_factor
        rate: np.ndarray = (
            (density * Consts.eff_area * (beta / np.pi) ** (3 / 2))
            * np.exp(-beta * (center_speed**2 + speed**2 - 2 * center_speed * speed))
            * np.sqrt(np.pi / (beta * speed * center_speed))
            * erf(np.sqrt(beta * speed * center_speed) * (Consts.az_fov / 2))
            * (
                center_speed**4
                * Consts.speed_ew
                * np.arcsin(thermal_velocity / center_speed)
            )
        )
        return rate

    speeds, densities, temperatures = [], [], []
    for i_sweep in range(chunk.sizes["sweep"]):
        sweep = chunk.isel(sweep=i_sweep)
        energy_passbands = sweep["esa_energy"].values * k_factor
        # Claude: process_swapi_old added 8 to the decompressed counts (half the
        # Claude: compression step of 16) so they are not biased low or zero.
        counts = sweep["count_rate"].values * SWAPI_LIVETIME + 8
        count_rates = counts / SWAPI_LIVETIME
        count_rate_error = np.sqrt(counts) / SWAPI_LIVETIME

        # Claude: optimize_pseudo_parameters() from process_swapi_old: fit the
        # Claude: 5 points around the peak; report speed only if the fit fails.
        max_index = int(np.argmax(count_rates))
        initial_speed_guess = np.sqrt(energy_passbands[max_index]) * Consts.speed_coeff
        initial_param_guess = np.array(
            [
                initial_speed_guess,
                5 * (400 / initial_speed_guess) ** 2,
                60000 * (initial_speed_guess / 400) ** 2,
            ]
        )
        sol = None
        try:
            five_point_range = range(max_index - 2, max_index + 2 + 1)
            xdata = energy_passbands.take(five_point_range, mode="clip")
            ydata = count_rates.take(five_point_range, mode="clip")
            sigma = count_rate_error.take(five_point_range, mode="clip")
            popt, pcov = curve_fit(
                f=model_count_rate,
                xdata=xdata,
                ydata=ydata,
                sigma=sigma,
                p0=initial_param_guess,
            )
            yfit = model_count_rate(xdata, *popt)
            r2 = 1 - np.sum((ydata - yfit) ** 2) / np.sum((ydata - ydata.mean()) ** 2)
            if np.all(np.isfinite(pcov)) and r2 >= 0.7:
                sol = popt
        except RuntimeError:
            logger.error("curve_fit failed")
        if sol is None:
            sol = initial_param_guess.copy()
            sol[1:] = np.nan

        speeds.append(sol[0])
        densities.append(sol[1])
        temperatures.append(sol[2])

    # Claude: geometric_mean() from process_swapi_old: skip sweeps whose fit
    # Claude: failed; if none succeeded, all three moments are NaN.
    speed_arr = np.asarray(speeds)
    density_arr = np.asarray(densities)
    temperature_arr = np.asarray(temperatures)
    valid = ~np.isnan(speed_arr) & ~np.isnan(density_arr) & ~np.isnan(temperature_arr)
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan")

    return (
        float(np.exp(np.mean(np.log(speed_arr[valid])))),
        float(np.exp(np.mean(np.log(density_arr[valid])))),
        float(np.exp(np.mean(np.log(temperature_arr[valid])))),
    )
