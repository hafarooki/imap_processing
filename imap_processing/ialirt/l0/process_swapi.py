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
# the single Maxwellian is fit, and every fit is judged, on the steps within this
# many steps of the peak of the sweep-averaged rates, as SWAPI L3a judges its fits
PEAK_HALF_WIDTH = 4
# R^2 on that window below which a fit is considered bad, as in SWAPI L3a
MIN_R_SQUARED = 0.9
# when both fits are good, the dual fit is reported only if its density is at
# least this many times the single Maxwellian's
DUAL_DENSITY_RATIO = 2.0
# I-ALiRT packets each carry 6 consecutive steps of the sweep, one per step
# period
STEPS_PER_PACKET = 6
STEP_PERIOD_S = 1.0 / STEPS_PER_PACKET
# spacecraft spin phase counts to radians, as for I-ALiRT MAG
SPIN_PHASE_TO_RAD = 2 * np.pi / 65535.0


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

    # the packets carry every instrument's fields; keep only those used here, so
    # the per-sweep selections below don't copy the rest
    sci_dataset = unpacked_data[
        [
            "sc_sclk_sec",
            "sc_sclk_sub_sec",
            "src_seq_ctr",
            "swapi_seq_number",
            "swapi_acq",
            "swapi_version",
            "sc_spin_phase",
            *(f"swapi_coin_cnt{i}" for i in range(6)),
        ]
    ].sortby("epoch", ascending=True)

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

    # Extract ESA voltages from the calibration lookup table file
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

        # TODO extract constants
        coarse_step_indices = np.arange(1, 63)

        compressed_counts = process_sweep_data(grouped_subset, "swapi_coin_cnt")
        # I-ALiRT packets have counts compressed by a factor of 16.
        # Add 8 to avoid having counts truncated to 0 and to avoid
        # counts being systematically too low
        # Subset to only the relevant I-ALiRT energy steps
        count_rates = (
            compressed_counts * Consts.compression_factor + Consts.compression_offset
        ).reshape(72)[coarse_step_indices] / SWAPI_LIVETIME

        # start time (MET) of each step: each packet's steps follow its
        # acquisition time one step period apart
        step_met = (
            grouped_subset["swapi_acq"].values.astype(float)[:, np.newaxis]
            + STEP_PERIOD_S * np.arange(STEPS_PER_PACKET)
        ).reshape(72)[coarse_step_indices]

        sweep_id = int(grouped_subset["swapi_version"].values[0])
        subset_sweep = calibration_lut_table[
            calibration_lut_table["Sweep #"] == sweep_id
        ]

        # Find the sweep's ESA voltages for the latest time
        subset = subset_sweep[
            (subset_sweep["timestamp"] == subset_sweep["timestamp"].max())
        ]
        if subset.empty:
            raise ValueError(
                f"No esa unit conversion available for sweep {sweep_id}. "
                f"Check lookup table?"
            )

        subset = subset.sort_values(["timestamp", "ESA Step #"])
        # the forward model takes ESA voltages and applies its own k-factor,
        # so use the table's voltages rather than its energies, which were
        # computed with the table's (different) k-factor. Fine steps are
        # "Solve" (NaN here) but aren't among the coarse steps.
        esa_voltages = (
            pd.to_numeric(
                subset["Voltage"].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
            .to_numpy(dtype=float)
            .reshape(72)[coarse_step_indices]
        )

        sweeps.append(
            xr.Dataset(
                data_vars={
                    "count_rate": (["sweep", "esa_step"], [count_rates]),
                    "esa_voltage": (["sweep", "esa_step"], [esa_voltages]),
                    "step_met": (["sweep", "esa_step"], [step_met]),
                    # spacecraft clock and spin phase [rad] of each packet
                    "packet_sc_met": (
                        ["sweep", "packet"],
                        [grouped_subset["met"].values.astype(float)],
                    ),
                    "packet_spin_phase": (
                        ["sweep", "packet"],
                        [
                            grouped_subset["sc_spin_phase"].values.astype(float)
                            * SPIN_PHASE_TO_RAD
                        ],
                    ),
                    "sweep_center_met": (["sweep"], [mid_measurement]),
                    # spacecraft clock of the sweep's first and last packets,
                    # for the product header
                    "sweep_start_sc_met": (["sweep"], [float(met[0])]),
                    "sweep_end_sc_met": (["sweep"], [float(met[-1])]),
                },
                coords={
                    "esa_step": coarse_step_indices,
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

        logging.info(f"processing 5-sweep chunk with epoch {chunk_epoch}")

        fit_result = fit_chunk(
            response=response,
            esa_voltages=chunk.esa_voltage.values,
            count_rates=chunk.count_rate.values,
            spin_angles=spin_angles_at(
                chunk.step_met.values,
                chunk.packet_sc_met.values.ravel(),
                chunk.packet_spin_phase.values.ravel(),
            ),
            epoch=chunk_epoch,
        )

        if not fit_result["success"]:
            # TODO confirm this behavior
            fit_result = {"speed": np.nan, "density": np.nan, "temperature": np.nan}

        # TODO fit and convert to strings using Decimal (or f string?) as was
        # done before
        # the header time is the spacecraft clock at the center of the chunk
        # (first packet of the first sweep to last packet of the last sweep),
        # like swapi_epoch, rather than of the latest sweep only
        chunk_sc_met = np.array(
            [chunk.sweep_start_sc_met.values[0], chunk.sweep_end_sc_met.values[-1]]
        )
        swapi_data.append(
            _populate_instrument_header_items(chunk_sc_met)
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
    esa_voltages: NDArray[np.floating],
    count_rates: NDArray[np.floating],
    spin_angles: NDArray[np.floating],
    epoch: int,
) -> dict[str, float | bool]:
    """
    Fit proton Maxwellian(s) to a chunk of sweeps using the SWAPI forward model.

    A single Maxwellian flowing off the spin axis is fit to the steps around the
    peak of the sweep-averaged rates (see :func:`peak_window`), and judged by
    its R^2 on that window (see :func:`peak_r_squared`). If that R^2 is below
    ``MIN_R_SQUARED`` or the temperature exceeds
    ``DUAL_FIT_TEMPERATURE_THRESHOLD_K``, every step is also fit with a core plus
    suprathermal Maxwellian along the spin axis, judged by its R^2 on the same
    window. Then:

    * if neither fit is good, the chunk fails;
    * if only one is good, it is reported;
    * if both are good, the single fit is reported unless the dual fit's
      combined density is at least ``DUAL_DENSITY_RATIO`` times the single
      fit's, in which case the dual fit's combined density and temperature are.

    Parameters
    ----------
    response : SwapiResponse
        SWAPI instrument response tables.
    esa_voltages : numpy.ndarray
        ESA voltage magnitude of each step [V], with shape (sweep, step).
    count_rates : numpy.ndarray
        Decompressed observed coincidence rates [counts/s], with shape
        (sweep, step).
    spin_angles : numpy.ndarray
        Spacecraft spin angle [rad] at each step, with shape (sweep, step).
    epoch : int
        Center time of the chunk in TT J2000 ns, for the efficiency lookup.

    Returns
    -------
    dict[str, float | bool]
        ``success``, and ``speed`` [km/s], ``density`` [cm^-3] and
        ``temperature`` [K] when successful.
    """
    esa_voltages = np.asarray(esa_voltages, dtype=float)
    count_rates = np.asarray(count_rates, dtype=float)
    if not np.all(np.isfinite(esa_voltages) & (esa_voltages > 0)):
        logger.warning("SWAPI I-ALiRT chunk has invalid ESA voltages.")
        return {"success": False}
    if not np.all(esa_voltages == esa_voltages[0]):
        # the no-spin fits average the sweeps, so they must share voltages
        logger.warning("SWAPI I-ALiRT chunk's sweeps have different ESA voltages.")
        return {"success": False}

    window = peak_window(count_rates)
    # the single fit (and the no-spin fit it starts from) only sees the window
    peak_rates = np.full_like(count_rates, np.nan)
    peak_rates[:, window] = count_rates[:, window]
    single = fit_single_maxwellian_with_spin(
        esa_voltages,
        peak_rates,
        spin_angles=spin_angles,
        swapi_response=response,
        epoch=epoch,
    )
    single_good = _is_good(single, count_rates, window)
    fit = single
    if not single_good or single["temperature"] > DUAL_FIT_TEMPERATURE_THRESHOLD_K:
        dual = fit_dual_maxwellian_no_spin(
            esa_voltages, count_rates, swapi_response=response, epoch=epoch
        )
        dual_good = _is_good(dual, count_rates, window)
        if not single_good and not dual_good:
            logger.debug("SWAPI I-ALiRT single and dual fits both fit poorly.")
            return {"success": False}
        if not single_good or (
            dual_good and dual["density"] >= DUAL_DENSITY_RATIO * single["density"]
        ):
            fit = dual

    result = {key: float(fit[key]) for key in ("speed", "density", "temperature")}
    if not all(np.isfinite(value) for value in result.values()):
        logger.warning("SWAPI I-ALiRT fit gave non-finite moments.")
        return {"success": False}
    return {"success": True, **result}


def spin_angles_at(
    met: NDArray[np.floating],
    packet_met: NDArray[np.floating],
    packet_spin_phase: NDArray[np.floating],
) -> NDArray[np.float64]:
    """
    Compute the spacecraft spin angle at given times from packet spin phases.

    The spin phases reported in the spacecraft packets (as used for I-ALiRT
    MAG) are unwrapped and fit with a uniform spin, phase = a + b * MET, which
    is evaluated at ``met``. Unlike interpolation, the fit extends to steps
    acquired before the first packet's spacecraft time. The spin angle's zero
    doesn't matter to the fits, so no constant offset is applied.

    Parameters
    ----------
    met : numpy.ndarray
        Times [s MET] to compute the spin angle at, of any shape.
    packet_met : numpy.ndarray
        Spacecraft clock of each packet [s MET] (n,), n >= 2, at most half a
        spin apart.
    packet_spin_phase : numpy.ndarray
        Spin phase in each packet [rad] (n,), wrapped to [0, 2 pi).

    Returns
    -------
    numpy.ndarray
        Spin angle [rad] at ``met``, unwrapped, shaped like ``met``.
    """
    order = np.argsort(packet_met)
    times = np.asarray(packet_met, dtype=float)[order]
    phases = np.unwrap(np.asarray(packet_spin_phase, dtype=float)[order])
    rate, phase_at_start = np.polyfit(times - times[0], phases, 1)
    return np.asarray(phase_at_start + rate * (np.asarray(met) - times[0]))


def peak_window(count_rates: NDArray[np.floating]) -> slice:
    """
    Find the steps within ``PEAK_HALF_WIDTH`` of the peak of the mean sweep.

    The peak is that of the rates averaged over sweeps, and the window is
    shifted to stay inside the sweep, as in SWAPI L3a.

    Parameters
    ----------
    count_rates : numpy.ndarray
        Observed coincidence rates [counts/s] (sweep, step), NaN where invalid.

    Returns
    -------
    slice
        Steps of the window.
    """
    mean_rates = np.nanmean(count_rates, axis=0)
    n_steps = mean_rates.shape[0]
    center = int(np.nanargmax(mean_rates))
    center = max(PEAK_HALF_WIDTH, min(center, n_steps - PEAK_HALF_WIDTH - 1))
    return slice(center - PEAK_HALF_WIDTH, center + PEAK_HALF_WIDTH + 1)


def peak_r_squared(
    model_count_rates: NDArray[np.floating],
    count_rates: NDArray[np.floating],
    window: slice,
) -> float:
    """
    Compute the R^2 of model rates on a window of the mean sweep, as SWAPI L3a does.

    The residuals and the observed rates are each averaged over sweeps, and R^2
    is computed over the steps of the window.

    Parameters
    ----------
    model_count_rates : numpy.ndarray
        Model coincidence rates [counts/s] (sweep, step).
    count_rates : numpy.ndarray
        Observed coincidence rates [counts/s] (sweep, step), NaN where invalid.
    window : slice
        Steps to evaluate R^2 on.

    Returns
    -------
    float
        R^2, NaN if the observed rates are constant on the window.
    """
    residuals = np.nanmean(model_count_rates - count_rates, axis=0)[window]
    observed = np.nanmean(count_rates, axis=0)[window]
    total = float(np.sum((observed - observed.mean()) ** 2))
    if not total > 0:
        return np.nan
    return float(1.0 - np.sum(residuals**2) / total)


def _is_good(
    fit: dict[str, float | NDArray[np.float64]],
    count_rates: NDArray[np.floating],
    window: slice,
) -> bool:
    """
    Check whether a fit's R^2 on the window is at least ``MIN_R_SQUARED``.

    Parameters
    ----------
    fit : dict
        Fit result with ``model_count_rates`` [counts/s] (sweep, step).
    count_rates : numpy.ndarray
        Observed coincidence rates [counts/s] (sweep, step), NaN where invalid.
    window : slice
        Steps to evaluate R^2 on.

    Returns
    -------
    bool
        Whether the fit is good. NaN R^2 is not good.
    """
    r_squared = peak_r_squared(
        np.asarray(fit["model_count_rates"]), count_rates, window
    )
    return bool(r_squared >= MIN_R_SQUARED)
