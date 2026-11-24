"""Functions to support I-ALiRT SWAPI processing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from decimal import Decimal
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from numba.typed import List as NumbaList
from scipy.optimize import OptimizeResult, curve_fit, least_squares, nnls
from scipy.spatial.transform import Rotation

from imap_processing.codice.constants import SPIN_PERIOD_CONVERSION
from imap_processing.ialirt.utils.grouping import (
    find_groups,
    populate_instrument_header_items,
)
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.swapi.constants import (
    BOLTZMANN_CONSTANT_J_PER_K,
    PROTON_MASS_KG,
    SWEEP_CADENCE_SECONDS,
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
from imap_processing.swapi.forward_model.response.deadtime import (
    SWAPI_DEADTIME_S,
    deadtime_factor,
)
from imap_processing.swapi.forward_model.response.measurement_species_response import (
    MeasurementSpeciesResponse,
    measurement_species_responses,
    to_response_list,
)
from imap_processing.swapi.forward_model.response.measurement_times import (
    sweep_measurement_times,
)
from imap_processing.swapi.forward_model.response.passband_fit import PassbandFitTable
from imap_processing.swapi.l1.swapi_l1 import process_sweep_data
from imap_processing.swapi.l2.swapi_l2 import SWAPI_LIVETIME

logger = logging.getLogger(__name__)

# TODO inline these comments?
# Claude: floor on the density that starts the single-Maxwellian fits [cm^-3]
MIN_INITIAL_DENSITY = 0.01
# Claude: maximum number of refits of the spin fit from the mirrored flow
SPIN_FIT_MAX_RESTARTS = 6
# Claude: maximum number of model evaluations in each least-squares fit
MAX_FIT_EVALUATIONS = 1000

# Claude: no-spin fit temperature [K] above which the dual fit is attempted
DUAL_FIT_TEMPERATURE_THRESHOLD_K = 3e5

# TODO rename this
# Claude: single-Maxwellian fits use, and every fit is judged on, the steps
# Claude: within this many steps of the mean sweep's peak, as in SWAPI L3a
PEAK_HALF_WIDTH = 4
# Claude: R^2 on that window below which a fit is bad, as in SWAPI L3a
MIN_R_SQUARED = 0.9

# Claude: log-spaced trial temperatures for the dual fit's grid search, as
# Claude: multiples of the bulk energy: kT / (m v^2 / 2)
DUAL_GRID_NORMALIZED_TEMPERATURES: npt.NDArray[np.float64] = np.logspace(-4, 1, 126)
# Claude: ESA voltages [V] of the dense sweep the grid's spectra are computed
# Claude: on once; wider than the flight sweep so the flight steps stay covered
# Claude: after normalizing by any plausible bulk energy
DUAL_GRID_VOLTAGES: npt.NDArray[np.float64] = np.geomspace(10, 60_000, 2000)
# Claude: largest suprathermal/core temperature ratio the dual fit allows
DUAL_MAX_TEMPERATURE_RATIO = 20.0
# Claude: largest suprathermal/core density ratio of a grid cell that can start
# Claude: the dual fit; cells with comparable densities are spurious minima
DUAL_MAX_GRID_DENSITY_RATIO = 0.5
# Claude: floor on the dual fit's starting densities relative to their total,
# Claude: since NNLS can return zero and the fit's state is log density
DUAL_MIN_INITIAL_DENSITY_RATIO = 1e-3
# Claude: bounds on (ln n_core, ln T_core, ln speed, ln n_supra, ln T_ratio);
# Claude: the core temperature bounds keep trf away from NaN jacobians, and
# Claude: the suprathermal temperature is 1-20 times the core's
DUAL_LOWER_BOUNDS = np.array([-np.inf, np.log(1e3), -np.inf, -np.inf, 0.0])
DUAL_UPPER_BOUNDS = np.array(
    [np.inf, np.log(1e8), np.inf, np.inf, np.log(DUAL_MAX_TEMPERATURE_RATIO)]
)


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

    sweeps: xr.Dataset | None = _complete_sweeps(unpacked_data, calibration_lut_table)

    if sweeps is None:
        logger.info("no complete sweeps could be parsed, returning empty list")
        return []

    dual_grid = _precompute_dual_grid(
        int(met_to_ttj2000ns(sweeps["sweep_start_met"].values[0])),
        passband_fit_table=passband_fit_table,
        central_effective_area_table=central_effective_area_table,
        azimuthal_transmission=azimuthal_transmission,
    )

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
        sweep_start_epochs = met_to_ttj2000ns(chunk["sweep_start_met"].values).tolist()

        esa_voltages = chunk["esa_energy"].values
        if not np.all(esa_voltages == esa_voltages[0]):
            # TODO check only the 62
            # TODO log the time, too
            logger.warning("Skipping chunk whose sweeps use different ESA voltages")
            continue

        # has shape (sweep, esa_step), like the voltages and count rates
        measurement_epochs = np.array(
            [sweep_measurement_times(start)[1:63] for start in sweep_start_epochs]
        )
        # TODO change signature of _inertial_to_swapi
        inertial_to_swapi = _inertial_to_swapi(
            measurement_epochs.ravel(),
            reference_epoch=sweep_start_epochs[0],
            spin_period=chunk["spin_period"].median().item(),
        ).reshape(*measurement_epochs.shape, 3, 3)

        responses = [
            measurement_species_responses(
                esa_voltages=esa_voltages[sweep],
                epochs=measurement_epochs[sweep],
                inertial_to_swapi=inertial_to_swapi[sweep],
                species=Species.PROTON,
                passband_fit_table=passband_fit_table,
                central_effective_area_table=central_effective_area_table,
                azimuthal_transmission=azimuthal_transmission,
            )
            for sweep in range(measurement_epochs.shape[0])
        ]
        speed, density, temperature = _fit_chunk(
            responses=responses,
            count_rates=chunk["count_rate"].values,
            dual_grid=dual_grid,
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


def _complete_sweeps(
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
            * count_rate (sweep, esa_step=62) [ct/s]
            * esa_energy (sweep, esa_step=62) [eV/q]
            * spin_period (sweep) [s]
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
        [_voltages_for_table(calibration_lut_table, table_id) for table_id in table_ids]
    )

    sweep_start_met = first_packets["swapi_acq"].values.astype(float)

    spin_periods = (
        (complete_sweep_packets.sc_spin_period * SPIN_PERIOD_CONVERSION)
        # keep valid only
        .where(complete_sweep_packets.sc_spinperiodvalid)
        # take median of valid per sweep (group)
        .groupby("group")
        .median()
        # keep valid sweeps (groups) only
        .sel(group=complete_groups)
        .data
    )

    return xr.Dataset(
        data_vars={
            "count_rate": (["sweep", "esa_step"], count_rates, {"units": "ct/s"}),
            "esa_energy": (["sweep", "esa_step"], esa_energies, {"units": "eV/q"}),
            "spin_period": ("sweep", spin_periods, {"units": "s"}),
        },
        coords={
            "sweep_start_met": ("sweep", sweep_start_met),
            "sweep_start_sc_met": ("sweep", first_packets["met"].values),
            "sweep_end_sc_met": ("sweep", last_packets["met"].values),
        },
    )


def _voltages_for_table(
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


def _inertial_to_swapi(
    epochs: npt.NDArray[np.int64],
    *,
    reference_epoch: int,
    spin_period: float,
) -> npt.NDArray[np.float64]:
    # TODO ensure this is tested properly
    """
    Compute the rotations from an inertial frame to the SWAPI frame.

    The inertial frame is the SWAPI frame at ``reference_epoch``. The
    spacecraft spins about SWAPI's +Y (the spin axis) at one revolution per
    ``spin_period``, so the rotation at each epoch is about +Y by the spin
    phase accumulated since ``reference_epoch``.

    Parameters
    ----------
    epochs : numpy.ndarray of int64, shape (n,)
        Times of the rotations in TT J2000 ns.
    reference_epoch : int
        Time at which the inertial frame is the SWAPI frame in TT J2000 ns.
    spin_period : float
        The spin period of the spacecraft in seconds.

    Returns
    -------
    numpy.ndarray of float64, shape (n, 3, 3)
        Rotation matrices taking inertial vectors to SWAPI vectors.
    """
    # TODO double check the sign of the rotation
    spin_phase_degrees = 360 * ((epochs - reference_epoch) / 1e9) / spin_period
    rotations: npt.NDArray[np.float64] = Rotation.from_euler(
        # Claude: shape (N, 1); recent scipy rejects (N,) for a one-axis sequence
        "y",
        -spin_phase_degrees.reshape(-1, 1),
        degrees=True,
    ).as_matrix()
    return rotations.reshape(-1, 3, 3)


class SolarWindFit(NamedTuple):
    """Result of fitting a proton Maxwellian to a chunk of SWAPI sweeps."""

    density: float  # cm^-3
    temperature: float  # K
    speed: float  # km/s
    r_squared: float = np.nan  # TODO make it None instead?


class DualGrid(NamedTuple):
    """Unit-density spectra precomputed for the dual fit's grid search."""

    # Claude: ln(step energy / bulk energy) of each dense sweep step (n_steps,)
    log_energy_ratios: npt.NDArray[np.float64]
    # Claude: ideal rates at unit density for each of
    # Claude: DUAL_GRID_NORMALIZED_TEMPERATURES (n_temperatures, n_steps)
    unit_rates: npt.NDArray[np.float64]


def _temperature_k(normalized_temperature: float, speed: float) -> float:
    """
    Convert a temperature in units of the proton bulk energy to K.

    Parameters
    ----------
    normalized_temperature : float
        kT / (m v^2 / 2).
    speed : float
        Bulk speed in km/s.

    Returns
    -------
    float
        Temperature in K.
    """
    bulk_energy = 0.5 * PROTON_MASS_KG * (speed * 1e3) ** 2
    return float(normalized_temperature * bulk_energy / BOLTZMANN_CONSTANT_J_PER_K)


def _precompute_dual_grid(
    epoch: int,
    *,
    passband_fit_table: PassbandFitTable,
    central_effective_area_table: CentralEffectiveAreaTable,
    azimuthal_transmission: AzimuthalTransmission,
) -> DualGrid:
    """
    Compute the dual fit's grid spectra once, in bulk-energy-normalized units.

    Against step energy / bulk energy, a spectrum's shape depends on the
    temperature only through kT / bulk energy, to within a few percent over
    500-2000 km/s, so one set of spectra at a reference speed serves every
    chunk.

    Parameters
    ----------
    epoch : int
        Time at which to evaluate the efficiency in TT J2000 ns.
    passband_fit_table : PassbandFitTable
        Passband calibration data.
    central_effective_area_table : CentralEffectiveAreaTable
        Central effective area calibration data.
    azimuthal_transmission : AzimuthalTransmission
        Azimuthal transmission calibration data.

    Returns
    -------
    DualGrid
        The precomputed spectra.
    """
    voltages = DUAL_GRID_VOLTAGES
    # Claude: the flow is along the spin axis, so the spin phase doesn't matter
    responses = measurement_species_responses(
        esa_voltages=voltages,
        epochs=np.full(voltages.size, epoch),
        species=Species.PROTON,
        inertial_to_swapi=np.broadcast_to(np.eye(3), (voltages.size, 3, 3)),
        passband_fit_table=passband_fit_table,
        central_effective_area_table=central_effective_area_table,
        azimuthal_transmission=azimuthal_transmission,
    )
    central_speeds = np.array([response.central_speed for response in responses])
    # Claude: centered on the sweep, so the most normalized energies are covered
    speed = float(np.sqrt(central_speeds[0] * central_speeds[-1]))
    unit_rates = np.stack(
        [
            _boresight_aligned_ideal_rates_and_jacobian(
                np.array([0.0, np.log(_temperature_k(t, speed)), np.log(speed)]),
                responses=responses,
            )[0]
            for t in DUAL_GRID_NORMALIZED_TEMPERATURES
        ]
    )
    return DualGrid(2 * np.log(central_speeds / speed), unit_rates)


def _fit_chunk(
    *,
    responses: list[NumbaList[MeasurementSpeciesResponse]],
    count_rates: npt.NDArray[np.float64],
    dual_grid: DualGrid,
) -> tuple[float, float, float]:
    """
    Fit solar wind moments for a chunk of 5 consecutive sweeps.

    Parameters
    ----------
    responses : list[numba.typed.List[MeasurementSpeciesResponse]]
        Instrument response information for each measurement, indexed by
        [sweep][esa_step] like ``count_rates``.
    count_rates : numpy.ndarray of float64, shape (sweep=5, esa_step=62)
        Coincidence count rates in counts per second.
    dual_grid : DualGrid
        Precomputed spectra for the dual fit's grid search.

    Returns
    -------
    pseudo_moments : tuple[float, float, float]
        Pseudo proton speed [km/s], density [cm^-3] and temperature [K]. If there the
        fit fails, all are NaN.
    """
    no_spin_fit = _fit_single_maxwellian_no_spin(responses, count_rates)

    if no_spin_fit.temperature > DUAL_FIT_TEMPERATURE_THRESHOLD_K:
        dual_fit = _fit_dual_maxwellian_no_spin(responses, count_rates, dual_grid)
        if (
            dual_fit.r_squared >= MIN_R_SQUARED
            and dual_fit.density > no_spin_fit.density
        ):
            return dual_fit.speed, dual_fit.density, dual_fit.temperature

    spin_fit = _fit_single_maxwellian_with_spin(
        responses, count_rates, seed=no_spin_fit
    )
    if not spin_fit.r_squared >= MIN_R_SQUARED:
        return np.nan, np.nan, np.nan
    return spin_fit.speed, spin_fit.density, spin_fit.temperature


def _fit_single_maxwellian_no_spin(
    responses: list[NumbaList[MeasurementSpeciesResponse]],
    count_rates: npt.NDArray[np.float64],
) -> SolarWindFit:
    """
    Fit the peak of the mean sweep with one proton Maxwellian along the spin axis.

    Parameters
    ----------
    responses : list[numba.typed.List[MeasurementSpeciesResponse]]
        Instrument response information for each measurement, indexed by
        [sweep][esa_step] like ``count_rates``.
    count_rates : numpy.ndarray of float64, shape (sweep=5, esa_step=62)
        Coincidence count rates in counts per second.

    Returns
    -------
    result : SolarWindFit
        Fit result.
    """
    sweep_mean_rates = count_rates.mean(axis=0)
    first_sweep_responses = responses[0]
    window = _peak_window(count_rates)
    window_responses = first_sweep_responses[window]
    window_rates = sweep_mean_rates[window]

    # initial guess
    initial_speed = window_responses[int(np.argmax(window_rates))].central_speed
    initial_temperature = max(60_000.0 * (initial_speed / 400.0) ** 2, 10_000)
    unit_rates = _boresight_aligned_ideal_rates_and_jacobian(
        np.array([0.0, np.log(initial_temperature), np.log(initial_speed)]),
        responses=window_responses,
    )[0]
    unit_observed = unit_rates * deadtime_factor(unit_rates)
    initial_density = max(
        float(window_rates @ unit_observed) / float(unit_observed @ unit_observed),
        MIN_INITIAL_DENSITY,
    )

    # optimize
    x0 = np.log([initial_density, initial_temperature, initial_speed])
    fit = _least_squares_with_jacobian(
        lambda x: _observable_residuals_and_jacobian(
            *_boresight_aligned_ideal_rates_and_jacobian(x, responses=window_responses),
            window_rates,
        ),
        x0=x0,
        method="lm",
        xtol=1e-4,
        x_scale=1.0,
        max_nfev=MAX_FIT_EVALUATIONS,
    )
    log_density, log_temperature, log_speed = fit.x
    model_window_rates = fit.fun + window_rates

    result = SolarWindFit(
        density=float(np.exp(log_density)),
        temperature=float(np.exp(log_temperature)),
        speed=float(np.exp(log_speed)),
        r_squared=_r_squared(model_window_rates, window_rates),
    )
    return result


def _fit_dual_maxwellian_no_spin(
    responses: list[NumbaList[MeasurementSpeciesResponse]],
    count_rates: npt.NDArray[np.float64],
    dual_grid: DualGrid,
) -> SolarWindFit:
    """
    Fit the mean sweep with a core plus suprathermal proton Maxwellian.

    Parameters
    ----------
    responses : list[numba.typed.List[MeasurementSpeciesResponse]]
        Instrument response information for each measurement, indexed by
        [sweep][esa_step] like ``count_rates``.
    count_rates : numpy.ndarray of float64, shape (sweep=5, esa_step=62)
        Coincidence count rates in counts per second.
    dual_grid : DualGrid
        Precomputed spectra for the grid search that starts the fit.

    Returns
    -------
    result : SolarWindFit
        Fit result, with the parameters representing the core population. The
        suprathermal parameters are discarded.
    """
    # window data
    sweep_mean_rates = count_rates.mean(axis=0)
    first_sweep_responses = responses[0]
    window = _peak_window(count_rates)

    # initial state through grid search
    peak_speed = first_sweep_responses[int(np.argmax(sweep_mean_rates))].central_speed
    core_density, core_temperature, supra_density, supra_temperature = (
        _dual_maxwellian_grid_search(
            first_sweep_responses, sweep_mean_rates, peak_speed, dual_grid
        )
    )
    x0 = np.log(
        [
            core_density,
            core_temperature,
            peak_speed,
            supra_density,
            supra_temperature / core_temperature,
        ]
    )

    # optimize
    fit = _least_squares_with_jacobian(
        partial(
            _dual_residuals_and_jacobian,
            responses=first_sweep_responses,
            count_rates=sweep_mean_rates,
        ),
        x0=x0,
        method="trf",
        bounds=(DUAL_LOWER_BOUNDS, DUAL_UPPER_BOUNDS),
        xtol=1e-4,
        x_scale=1.0,
        max_nfev=MAX_FIT_EVALUATIONS,
    )
    log_n_core, log_t_core, log_speed, _, _ = fit.x
    model_rates = fit.fun + sweep_mean_rates

    result = SolarWindFit(
        density=float(np.exp(log_n_core)),
        temperature=float(np.exp(log_t_core)),
        speed=float(np.exp(log_speed)),
        # R^2 is only computed on the peak window even though all points are used to fit
        # TODO maybe extract this to a variable
        r_squared=_r_squared(model_rates[window], sweep_mean_rates[window]),
    )
    return result


def _fit_single_maxwellian_with_spin(
    responses: list[NumbaList[MeasurementSpeciesResponse]],
    count_rates: npt.NDArray[np.float64],
    *,
    seed: SolarWindFit,
) -> SolarWindFit:
    """
    Fit a single Maxwellian with an arbitrary velocity vector.

    Parameters
    ----------
    responses : list[numba.typed.List[MeasurementSpeciesResponse]]
        Instrument response information for each measurement, indexed by
        [sweep][esa_step] like ``count_rates``.
    count_rates : numpy.ndarray of float64, shape (sweep=5, esa_step=62)
        Coincidence count rates in counts per second.
    seed : SolarWindFit
        Starting point for the fit.

    Returns
    -------
    result : SolarWindFit
        Fit result.
    """
    # select the points within the fitting window
    window_slice = _peak_window(count_rates)
    window_responses = to_response_list(
        response for sweep in responses for response in sweep[window_slice]
    )
    window_rates = count_rates[:, window_slice]
    residuals_and_jacobian = partial(
        _spin_residuals_and_jacobian,
        responses=window_responses,
        count_rates=window_rates.flatten(),
    )

    def to_state(
        density: float, temperature: float, velocity: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.concatenate([np.log([density, temperature]), velocity])

    def fit_from_starting_point(
        density: float, temperature: float, velocity: npt.NDArray[np.float64]
    ) -> tuple[float, float, npt.NDArray[np.float64]]:
        fit = _least_squares_with_jacobian(
            residuals_and_jacobian,
            x0=to_state(density, temperature, velocity),
            method="lm",
            xtol=1e-4,
            x_scale=1.0,
            max_nfev=MAX_FIT_EVALUATIONS,
        )
        return float(np.exp(fit.x[0])), float(np.exp(fit.x[1])), fit.x[2:]

    # initial fit, starting with a spin-axis-aligned velocity
    current = fit_from_starting_point(
        seed.density, seed.temperature, np.array([0.0, -seed.speed, 0.0])
    )

    def residuals(
        density: float, temperature: float, velocity: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return residuals_and_jacobian(to_state(density, temperature, velocity))[0]

    def mse(
        density: float, temperature: float, velocity: npt.NDArray[np.float64]
    ) -> float:
        return float(np.mean(residuals(density, temperature, velocity) ** 2))

    # repeat the fit as needed to avoid local minima in velocity
    for _ in range(SPIN_FIT_MAX_RESTARTS):
        current_error = mse(*current)

        mirrored = _mirror_and_rescale_density(
            temperature=current[1],
            velocity=current[2],
            responses=window_responses,
            count_rates=window_rates.flatten(),
        )

        # skip if the mirrored flow fits much worse
        if mse(*mirrored) >= 100 * current_error:
            break

        candidate = fit_from_starting_point(*mirrored)
        # don't replace if it's not an improvement
        if mse(*candidate) > current_error:
            break
        current = candidate

    density, temperature, velocity = current

    # recover the model count rates for the final fit, use to compute goodness-of-fit
    model_rates = (
        residuals(density, temperature, velocity).reshape(window_rates.shape)
        + window_rates
    )
    r2 = _r_squared(
        model_rates=model_rates.mean(axis=0), observed_rates=window_rates.mean(axis=0)
    )

    result = SolarWindFit(
        density=density,
        temperature=temperature,
        speed=float(np.linalg.norm(velocity)),
        r_squared=r2,
    )
    return result


def _peak_window(count_rates: npt.NDArray[np.float64]) -> slice:
    """
    Find the steps within ``PEAK_HALF_WIDTH`` of the sweep-mean peak.

    Parameters
    ----------
    count_rates : numpy.ndarray of float64, shape (sweep=5, esa_step=62)
        Coincidence count rates in counts per second.

    Returns
    -------
    slice
        Slice for the target points.
    """
    sweep_mean_rates = count_rates.mean(axis=0)
    n_steps = sweep_mean_rates.shape[0]
    center = int(np.argmax(sweep_mean_rates))
    center = max(PEAK_HALF_WIDTH, min(center, n_steps - PEAK_HALF_WIDTH - 1))
    return slice(center - PEAK_HALF_WIDTH, center + PEAK_HALF_WIDTH + 1)


def _boresight_aligned_ideal_rates_and_jacobian(
    x: npt.NDArray[np.float64],
    *,
    responses: NumbaList[MeasurementSpeciesResponse],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the ideal rates of a Maxwellian flowing along the spin axis.

    Parameters
    ----------
    x : numpy.ndarray
        (ln density, ln temperature, ln speed) with density in cm^-3,
        temperature in K and speed in km/s (3,).
    responses : numba.typed.List[MeasurementSpeciesResponse]
        Instrument response information for each measurement.

    Returns
    -------
    rates : numpy.ndarray
        Ideal rates in counts/s (n,).
    jacobian : numpy.ndarray
        Derivatives of ``rates`` w.r.t. ``x`` (n, 3).
    """
    # TODO add a test for this using numerical derivative
    log_density, log_temperature, log_speed = x
    speed = float(np.exp(log_speed))
    # spin axis is aligned with SWAPI y, and -y is approximately (heliocentric) radial
    velocity_inertial_sc = np.array([0.0, -speed, 0.0])
    rates, jacobian = maxwellian_count_rates(
        density=float(np.exp(log_density)),
        temperature=float(np.exp(log_temperature)),
        velocity_inertial_sc=velocity_inertial_sc,
        responses=responses,
    )
    log_density_derivative = jacobian[:, 0]
    log_temperature_derivative = jacobian[:, 1]
    # d/d(ln speed) = speed d/d(speed) = -speed d/d(v_y)
    log_speed_derivative = -speed * jacobian[:, 3]
    return rates, np.stack(
        [log_density_derivative, log_temperature_derivative, log_speed_derivative],
        axis=-1,
    )


def _spin_residuals_and_jacobian(
    x: npt.NDArray[np.float64],
    *,
    responses: NumbaList[MeasurementSpeciesResponse],
    count_rates: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the residuals of a Maxwellian flowing in any direction.

    Parameters
    ----------
    x : numpy.ndarray
        (ln density, ln temperature, v_x, v_y, v_z) with density in cm^-3,
        temperature in K and the velocity in the responses' inertial frame in
        km/s (5,).
    responses : numba.typed.List[MeasurementSpeciesResponse]
        Instrument response information for each measurement.
    count_rates : numpy.ndarray of float64, shape (n,)
        Coincidence count rates in counts per second.

    Returns
    -------
    residuals : numpy.ndarray
        Post-deadtime model minus observed rates in counts/s (n,).
    jacobian : numpy.ndarray
        Derivatives of ``residuals`` w.r.t. ``x`` (n, 5).
    """
    # TODO add a test for this using numerical derivative
    rates, jacobian = maxwellian_count_rates(
        density=float(np.exp(x[0])),
        temperature=float(np.exp(x[1])),
        velocity_inertial_sc=np.ascontiguousarray(x[2:], dtype=float),
        responses=responses,
    )
    return _observable_residuals_and_jacobian(rates, jacobian, count_rates)


def _dual_residuals_and_jacobian(
    state: npt.NDArray[np.float64],
    *,
    responses: NumbaList[MeasurementSpeciesResponse],
    count_rates: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the residuals of a core plus suprathermal along the spin axis.

    Parameters
    ----------
    state : numpy.ndarray
        (ln n_core, ln T_core, ln speed, ln n_supra, ln T_supra/T_core) with
        densities in cm^-3, temperatures in K and speed in km/s (5,).
    responses : numba.typed.List[MeasurementSpeciesResponse]
        Instrument response information for each measurement.
    count_rates : numpy.ndarray of float64, shape (n,)
        Coincidence count rates in counts per second.

    Returns
    -------
    residuals : numpy.ndarray
        Post-deadtime model minus observed rates in counts/s (n,).
    jacobian : numpy.ndarray
        Derivatives of ``residuals`` w.r.t. ``state`` (n, 5).
    """
    # TODO add a test for this using numerical derivative
    log_n_core, log_t_core, log_speed, log_n_supra, log_t_ratio = state
    core_rates, core_jacobian = _boresight_aligned_ideal_rates_and_jacobian(
        np.array([log_n_core, log_t_core, log_speed]), responses=responses
    )
    supra_rates, supra_jacobian = _boresight_aligned_ideal_rates_and_jacobian(
        np.array([log_n_supra, log_t_core + log_t_ratio, log_speed]),
        responses=responses,
    )
    # R = R_c(ln(n_c), ln(T_c), ln(v)) + R_s(ln(n_s), ln(T_s), ln(v)),
    # ln(T_s) = ln(T_c) + ln(T_s/T_c)
    # (T_c and T_s/T_c are independent fitting parameters)
    jacobian = np.stack(
        [
            # dR/dln(n_c) = dR_c/dln(n_c)
            core_jacobian[:, 0],
            # dR/dln(T_c) = dR_c/dln(T_c)
            #     + dR_s/dln(T_s) * dln(T_s)/dln(T_c)
            #   = dR_c/dln(T_c) + dR_s/dln(T_s) [since dln(T_s)/dln(T_c) = 1]
            core_jacobian[:, 1] + supra_jacobian[:, 1],
            # dR/dln(v) = dR_c/dln(v) + dR_s/dln(v)
            core_jacobian[:, 2] + supra_jacobian[:, 2],
            # dR/dln(n_s) = dR_s/dln(n_s)
            supra_jacobian[:, 0],
            # dR/dln(T_s/T_c) = dR_s/dln(T_s) * dln(T_s)/dln(T_s/T_c)
            #   = dR_s/dln(T_s) [since dln(T_s)/dln(T_s/T_c) = 1]
            supra_jacobian[:, 1],
        ],
        axis=-1,
    )
    return _observable_residuals_and_jacobian(
        core_rates + supra_rates, jacobian, count_rates
    )


def _observable_residuals_and_jacobian(
    ideal: npt.NDArray[np.float64],
    jacobian: npt.NDArray[np.float64],
    count_rates: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Apply deadtime to ideal model rates and compute residuals against observations.

    Parameters
    ----------
    ideal : numpy.ndarray
        Ideal (pre-deadtime) model rates in counts/s (n,).
    jacobian : numpy.ndarray
        Derivatives of ``ideal`` w.r.t. the fit state (n, k).
    count_rates : numpy.ndarray of float64, shape (n,)
        Coincidence count rates in counts per second.

    Returns
    -------
    residuals : numpy.ndarray
        Post-deadtime model minus observed rates (n,).
    jacobian : numpy.ndarray
        Derivatives of ``residuals`` w.r.t. the fit state (n, k).
    """
    dead = deadtime_factor(ideal)
    # measured = r / (1 + tau r), so d(measured)/dr = deadtime_factor**2
    return ideal * dead - count_rates, jacobian * (dead**2)[:, np.newaxis]


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

    # TODO resolve pylance error here
    return least_squares(
        lambda x: evaluate(x)[0], x0, jac=lambda x: evaluate(x)[1], **kwargs
    )


def _r_squared(
    model_rates: npt.NDArray[np.float64], observed_rates: npt.NDArray[np.float64]
) -> float:
    """
    Compute the R^2 of model rates against observed rates.

    Parameters
    ----------
    model_rates : numpy.ndarray
        Model rates in counts/s (n,).
    observed_rates : numpy.ndarray
        Observed rates in counts/s (n,).

    Returns
    -------
    float
        R^2, NaN if the observed rates are constant.
    """
    total = float(np.sum((observed_rates - observed_rates.mean()) ** 2))
    if not total > 0:
        return np.nan
    return float(1.0 - np.sum((model_rates - observed_rates) ** 2) / total)


def _dual_maxwellian_grid_search(
    responses: NumbaList[MeasurementSpeciesResponse],
    count_rates: npt.NDArray[np.float64],
    speed: float,
    grid: DualGrid,
) -> tuple[float, float, float, float]:
    """
    Grid search the core and suprathermal temperatures of a dual Maxwellian.

    The grid's spectra are interpolated to this sweep's step energies relative
    to the bulk energy. For each pair of grid temperatures, the densities are
    solved by weighted NNLS and the pair is scored by the RMS residual after
    deadtime. Each core temperature's first local minimum in suprathermal
    temperature, with an acceptable density ratio, is a candidate; the best
    candidate wins, or the best pair if there are none. The winner's densities
    are then refit with exact spectra.

    Parameters
    ----------
    responses : numba.typed.List[MeasurementSpeciesResponse]
        Instrument response information for each measurement.
    count_rates : numpy.ndarray of float64, shape (n,)
        Coincidence count rates in counts per second.
    speed : float
        Speed of both populations in km/s.
    grid : DualGrid
        Precomputed spectra.

    Returns
    -------
    core_density : float
        Core density in cm^-3.
    core_temperature : float
        Core temperature in K.
    supra_density : float
        Suprathermal density in cm^-3.
    supra_temperature : float
        Suprathermal temperature in K.
    """
    temperatures = DUAL_GRID_NORMALIZED_TEMPERATURES
    n = len(temperatures)
    # Claude: the grid is log-spaced, so k steps up is a fixed temperature ratio
    n_ratios = int(
        np.sum(temperatures / temperatures[0] <= DUAL_MAX_TEMPERATURE_RATIO + 1e-9)
    )
    log_energy_ratios = 2 * np.log(
        np.array([response.central_speed for response in responses]) / speed
    )
    unit_rates = [
        np.interp(log_energy_ratios, grid.log_energy_ratios, rates, left=0, right=0)
        for rates in grid.unit_rates
    ]
    # Claude: NNLS on ideal rates, weighted by d(observed)/d(ideal) so the
    # Claude: residual matches the observed one to first order
    ideal_rates = count_rates / (1 - SWAPI_DEADTIME_S * count_rates)
    weights = deadtime_factor(ideal_rates) ** 2

    def solve_densities(basis: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        densities: npt.NDArray[np.float64] = nnls(
            basis * weights[:, None], ideal_rates * weights
        )[0]
        return densities

    # Claude: [core index i, steps up k]; off-grid pairs keep an infinite RMS
    densities: npt.NDArray[np.float64] = np.zeros((n, n_ratios, 2))
    rmse: npt.NDArray[np.float64] = np.full((n, n_ratios), np.inf)
    for i in range(n):
        for k in range(min(n_ratios, n - i)):
            basis = np.column_stack([unit_rates[i], unit_rates[i + k]])
            densities[i, k] = solve_densities(basis)
            modeled = basis @ densities[i, k]
            rmse[i, k] = np.sqrt(
                np.mean((modeled * deadtime_factor(modeled) - count_rates) ** 2)
            )
    rmse[~np.isfinite(rmse)] = np.inf

    padded = np.pad(rmse, ((0, 0), (1, 1)), constant_values=np.inf)
    is_candidate = (
        np.isfinite(rmse)
        & (rmse <= padded[:, :-2])
        & (rmse <= padded[:, 2:])
        & (densities[..., 1] <= DUAL_MAX_GRID_DENSITY_RATIO * densities[..., 0])
    )
    first_k = is_candidate.argmax(axis=1)
    candidate_rmse = np.where(is_candidate.any(axis=1), rmse[range(n), first_k], np.inf)
    if is_candidate.any():
        i = int(np.argmin(candidate_rmse))
        k = int(first_k[i])
    else:
        i, k = (int(x) for x in np.unravel_index(np.argmin(rmse), rmse.shape))

    # Claude: the grid's spectra are only right in shape, not amplitude, so
    # Claude: refit the winner's densities with exact spectra at this speed
    core_temperature = _temperature_k(temperatures[i], speed)
    supra_temperature = _temperature_k(temperatures[i + k], speed)
    best_densities = solve_densities(
        np.column_stack(
            [
                _boresight_aligned_ideal_rates_and_jacobian(
                    np.array([0.0, np.log(t), np.log(speed)]), responses=responses
                )[0]
                for t in (core_temperature, supra_temperature)
            ]
        )
    )
    # Claude: floor the densities, since the fit's state is log density
    core_density, supra_density = np.maximum(
        best_densities, DUAL_MIN_INITIAL_DENSITY_RATIO * best_densities.sum()
    )
    return (
        float(core_density),
        core_temperature,
        float(supra_density),
        supra_temperature,
    )


def _mirror_and_rescale_density(
    temperature: float,
    velocity: npt.NDArray[np.float64],
    *,
    responses: NumbaList[MeasurementSpeciesResponse],
    count_rates: npt.NDArray[np.float64],
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """
    Mirror a flow about the spin axis and refit the density for the new flow.

    Parameters
    ----------
    density : float
        Proton density in cm^-3. Ignored; it is replaced by the best-fit
        density for the mirrored flow.
    temperature : float
        Proton temperature in K.
    velocity : numpy.ndarray of float64, shape (3,)
        Bulk velocity in the responses' inertial frame in km/s.
    responses : numba.typed.List[MeasurementSpeciesResponse]
        Instrument response information for each measurement.
    count_rates : numpy.ndarray of float64, shape (n,)
        Coincidence count rates in counts per second.

    Returns
    -------
    density : float
        Best-fit density for the mirrored flow in cm^-3, at least
        ``MIN_INITIAL_DENSITY``.
    temperature : float
        The unchanged temperature in K.
    velocity : numpy.ndarray of float64, shape (3,)
        The mirrored velocity in km/s.
    """
    # the spin axis is +Y, so mirroring about it flips x and z
    # TODO make this not depend on that
    mirrored_velocity = velocity * [-1.0, 1.0, -1.0]
    # Claude: the ideal rates scale with density, so one evaluation suffices
    unit_rates = maxwellian_count_rates(
        density=1.0,
        temperature=temperature,
        velocity_inertial_sc=np.ascontiguousarray(mirrored_velocity, dtype=float),
        responses=responses,
    )[0]
    (mirrored_density,) = curve_fit(
        lambda unit_rate, density: (
            density * unit_rate * deadtime_factor(density * unit_rate)
        ),
        unit_rates,
        count_rates,
        p0=[1.0],
    )[0]
    mirrored_density = max(float(mirrored_density), MIN_INITIAL_DENSITY)
    return mirrored_density, temperature, mirrored_velocity
