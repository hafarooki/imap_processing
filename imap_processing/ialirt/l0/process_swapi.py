"""Functions to support I-ALiRT SWAPI processing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from numba.typed.typedlist import List as TypedList
from scipy.optimize import OptimizeResult, least_squares

from imap_processing.ialirt.utils.grouping import (
    find_groups,
    populate_instrument_header_items,
)
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.swapi.constants import (
    BOLTZMANN_CONSTANT_J_PER_K,
    PROTON_CHARGE_C,
    Species,
)
from imap_processing.swapi.forward_model.maxwellian.integration import (
    maxwellian_count_rates,
)
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.response.central_effective_area import (
    CentralEffectiveAreaTable,
)
from imap_processing.swapi.forward_model.response.deadtime import deadtime_factor
from imap_processing.swapi.forward_model.response.measurement_species_response import (
    MeasurementSpeciesResponse,
    measurement_species_responses,
)
from imap_processing.swapi.forward_model.response.passband_fit import PassbandFitTable
from imap_processing.swapi.l1.swapi_l1 import process_sweep_data
from imap_processing.swapi.l2.swapi_l2 import SWAPI_LIVETIME

logger = logging.getLogger(__name__)

# TODO figure out where this belongs
SWEEP_CADENCE_SECONDS = 12.0


def process_swapi_ialirt(
    unpacked_data: xr.Dataset,
    calibration_lut_table: pd.DataFrame,
    *,
    passband_fit_table: PassbandFitTable,
    central_effective_area_table: CentralEffectiveAreaTable,
    azimuthal_transmission: AzimuthalTransmission,
) -> list[dict]:
    """
    Extract I-ALiRT variables and calculate coincidence count rate.

    Parameters
    ----------
    unpacked_data : xr.Dataset
        SWAPI I-ALiRT data that has been parsed from the spacecraft packet.
    calibration_lut_table : pd.DataFrame
        DataFrame containing the contents of the SWAPI esa-unit-conversion lookup table.
    passband_fit_table : PassbandFitTable
        Passband calibration data.
    central_effective_area_table : CentralEffectiveAreaTable
        Central effective area calibration data.
    azimuthal_transmission : AzimuthalTransmission
        Azimuthal transmission calibration data.

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

        # center of chunk time
        swapi_met = np.mean(chunk["sweep_start_met"].values) + SWEEP_CADENCE_SECONDS / 2
        swapi_epoch = int(met_to_ttj2000ns(swapi_met))

        esa_voltages = chunk["esa_energy"].values
        if not np.all(esa_voltages == esa_voltages[0]):
            # TODO check only the 62
            # TODO log the time, too
            logger.warning("Skipping chunk whose sweeps use different ESA voltages")
            continue

        speed, density, temperature = _fit_chunk(
            esa_voltages=esa_voltages[0],
            count_rate=chunk["count_rate"].values,
            epoch=swapi_epoch,
            passband_fit_table=passband_fit_table,
            central_effective_area_table=central_effective_area_table,
            azimuthal_transmission=azimuthal_transmission,
        )

        # TODO this matches the behavior of the previous code, and the other
        #   instruments, not sure if intended
        sc_met = np.array(
            [
                chunk["sweep_start_sc_met"].values[0],
                chunk["sweep_end_sc_met"].values[-1],
            ]
        )

        swapi_data.append(
            populate_instrument_header_items(sc_met)
            | {
                "instrument": "swapi",
                "swapi_epoch": swapi_epoch,
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


def _fit_chunk(
    *,
    esa_voltages: np.ndarray,
    count_rate: np.ndarray,
    epoch: int,
    passband_fit_table: PassbandFitTable,
    central_effective_area_table: CentralEffectiveAreaTable,
    azimuthal_transmission: AzimuthalTransmission,
) -> tuple[float, float, float]:
    """
    Fit solar wind moments for a chunk of 5 consecutive sweeps.

    Parameters
    ----------
    esa_voltages : np.ndarray of shape (esa_step=62)
        ESA voltages for the coarse steps in volts.
    count_rate : np.ndarray of shape (sweep=5, esa_step=62)
        Coincidence count rates for the coarse steps across the five sweeps in counts
        per second.
    epoch : int
        Time of the chunk in TT J2000 ns.
    passband_fit_table : PassbandFitTable
        Passband calibration data.
    central_effective_area_table : CentralEffectiveAreaTable
        Central effective area calibration data.
    azimuthal_transmission : AzimuthalTransmission
        Azimuthal transmission calibration data.

    Returns
    -------
    pseudo_moments : tuple[float, float, float]
        Pseudo proton speed [km/s], density [cm^-3] and temperature [K].
    """
    esa_voltages = np.asarray(esa_voltages, dtype=float)
    responses = measurement_species_responses(
        esa_voltages=esa_voltages,
        epochs=np.full(esa_voltages.shape, epoch),
        species=Species.PROTON,
        rtn_to_swapi=np.broadcast_to(np.eye(3), (esa_voltages.size, 3, 3)),
        passband_fit_table=passband_fit_table,
        central_effective_area_table=central_effective_area_table,
        azimuthal_transmission=azimuthal_transmission,
    )
    fit = _fit_single_maxwellian_no_spin(responses, count_rate)
    return float(fit["speed"]), float(fit["density"]), float(fit["temperature"])


def _fit_single_maxwellian_no_spin(
    responses: TypedList[MeasurementSpeciesResponse],
    count_rates: npt.ArrayLike,
) -> dict[str, float | npt.NDArray[np.float64]]:
    """
    Fit a chunk of sweeps with one proton Maxwellian aligned with the spin axis.

    Parameters
    ----------
    responses : numba.typed.List[MeasurementSpeciesResponse]
        Instrument response information for each measurement.
    count_rates : numpy.ndarray of shape (sweep=5, step=62)
        Measured coincidence count rates in counts/s.

    Returns
    -------
    dict[str, float | numpy.ndarray]
        Dictionary containing:
        - ``speed`` [km/s]
        - ``density`` [cm^-3]
        - ``temperature`` [K]
        - ``model_count_rates`` [counts/s] (shape matching ``count_rates``).
    """
    count_rates = np.asarray(count_rates, dtype=float)
    mean_rates = count_rates.mean(axis=0)

    def residuals_and_jacobian(
        x: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        log_density, log_temperature, log_speed = x
        speed = float(np.exp(log_speed))
        ideal, jacobian = maxwellian_count_rates(
            density=float(np.exp(log_density)),
            temperature=float(np.exp(log_temperature)),
            velocity_rtn_sc=np.array([0.0, -speed, 0.0]),
            responses=responses,
        )
        dead = deadtime_factor(ideal)
        # d/d(ln speed) = speed d/d(speed) = -speed d/d(v_y), and
        # measured = r / (1 + tau r), so d(measured)/dr = deadtime_factor**2
        jacobian = (
            np.stack([jacobian[:, 0], jacobian[:, 1], -speed * jacobian[:, 3]], axis=-1)
            * (dead**2)[:, np.newaxis]
        )
        return ideal * dead - mean_rates, jacobian

    # guess the speed as the central speed of the peak step
    speed = responses[int(np.argmax(mean_rates))].central_speed
    # temperature scales with speed squared (60,000 K at 400 km/s), >= 1 eV
    one_ev_in_k = PROTON_CHARGE_C / BOLTZMANN_CONSTANT_J_PER_K
    temperature = max(60_000.0 * (speed / 400.0) ** 2, one_ev_in_k)
    # density is the least-squares scale of the unit-density rates, >= 0.01
    unit_state = np.array([0.0, np.log(temperature), np.log(speed)])
    unit_rates = residuals_and_jacobian(unit_state)[0] + mean_rates
    density = max((mean_rates @ unit_rates) / (unit_rates @ unit_rates), 1)

    result = _least_squares_with_jacobian(
        residuals_and_jacobian,
        x0=np.array([np.log(density), np.log(temperature), np.log(speed)]),
        method="lm",
        xtol=1e-4,
        x_scale=1.0,
        max_nfev=1000,
    )

    log_density, log_temperature, log_speed = result.x
    return {
        "speed": float(np.exp(log_speed)),
        "density": float(np.exp(log_density)),
        "temperature": float(np.exp(log_temperature)),
        # result.fun holds the final residuals, model minus observed
        "model_count_rates": np.broadcast_to(
            result.fun + mean_rates, count_rates.shape
        ).copy(),
    }


def _least_squares_with_jacobian(
    residuals_and_jacobian: Callable[..., tuple[np.ndarray, np.ndarray]],
    x0: npt.NDArray[np.float64],
    **kwargs: Any,
) -> OptimizeResult:
    """
    Run ``scipy.optimize.least_squares`` for a function that provides Jacobians.

    The problem is that ``scipy.optimize.least_squares`` takes a separate function for
    the residuals and the Jacobian, but our forward model calculates them both in one
    go. The workaround is to cache the most recent evaluation and reuse it if the
    parameters remain the same.

    Parameters
    ----------
    residuals_and_jacobian : x -> (residuals, jacobians)
        The function that takes the model parameters ``x`` (n,) and provides the
        residuals (m,) and their Jacobians (m, n) w.r.t. ``x``.
    x0 : numpy.ndarray
        Initial state (n,).
    **kwargs : dict
        Extra arguments for ``scipy.optimize.least_squares``.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The result of ``scipy.optimize.least_squares``.
    """
    last_evaluation: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def evaluate(x: npt.NDArray[np.float64]) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the residuals and jacobian, reusing the last evaluation.

        Parameters
        ----------
        x : numpy.ndarray
            The model parameters.

        Returns
        -------
        residuals : numpy.ndarray
            Residuals for model parameters ``x``.
        jacobian : numpy.ndarray
            Derivatives of the residuals w.r.t. ``x``.
        """
        key = x.tobytes()
        if key not in last_evaluation:
            last_evaluation.clear()
            last_evaluation[key] = residuals_and_jacobian(x)
        return last_evaluation[key]

    return least_squares(
        lambda x: evaluate(x)[0], x0, jac=lambda x: evaluate(x)[1], **kwargs
    )
