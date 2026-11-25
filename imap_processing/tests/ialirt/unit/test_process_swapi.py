import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0 import process_swapi
from imap_processing.ialirt.l0.process_swapi import (
    DUAL_MAX_GRID_DENSITY_RATIO,
    DUAL_MAX_TEMPERATURE_RATIO,
    DUAL_MIN_GRID_DENSITY_RATIO,
    _complete_sweeps,
    _fit_chunk,
    _inertial_to_swapi,
    _score_temperature_pairs,
    _voltages_for_table,
    process_swapi_ialirt,
)
from imap_processing.spice.time import et_to_ttj2000ns, str_to_et
from imap_processing.swapi.constants import (
    NOMINAL_K_FACTOR,
    SWEEP_CADENCE_SECONDS,
    Species,
)
from imap_processing.swapi.forward_model.maxwellian.integration import (
    maxwellian_count_rates,
)
from imap_processing.swapi.forward_model.response.deadtime import (
    SWAPI_DEADTIME_S,
    deadtime_factor,
)
from imap_processing.swapi.forward_model.response.measurement_species_response import (
    measurement_species_responses,
)
from imap_processing.swapi.forward_model.response.measurement_times import (
    sweep_measurement_times,
)
from imap_processing.swapi.l2.swapi_l2 import SWAPI_LIVETIME
from imap_processing.swapi.swapi_utils import read_swapi_lut_table
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def esa_unit_conversion_table() -> pd.DataFrame:
    esa_file_path = (
        imap_module_directory
        / "tests/swapi/lut/imap_swapi_esa-unit-conversion_20251210_v002.csv"
    )
    df = read_swapi_lut_table(esa_file_path)
    return df


@pytest.fixture(scope="session")
def response_tables(
    passband_fit_table, central_effective_area_table, azimuthal_transmission
) -> dict:
    return {
        "passband_fit_table": passband_fit_table,
        "central_effective_area_table": central_effective_area_table,
        "azimuthal_transmission": azimuthal_transmission,
    }


@pytest.mark.parametrize(
    "table_id, expected_first, expected_last",
    [
        # Sweep 0 has two timestamps (2/11/2025 and 5/19/2025); the latest
        # one (622.52 ... 51.81) should be used, not the older (619 ... 57).
        (0, [622.52, 622.52], [56.5, 51.81]),
        (1, [2994.5, 2994.5], [56.5, 51.81]),
        # This is the one used in flight
        (2, [10240.0, 9389.94292], [56.5, 51.81]),
    ],
)
def test_voltages_for_table(
    esa_unit_conversion_table, table_id, expected_first, expected_last
):
    """Test that voltages_for_table() returns the absolute value of the 62
    coarse ESA step voltages from the most recent LUT entry for the sweep
    table."""
    voltages = _voltages_for_table(esa_unit_conversion_table, table_id)

    assert voltages.shape == (62,)
    assert voltages.dtype == np.float64
    assert np.all(voltages > 0)
    np.testing.assert_allclose(voltages[:2], expected_first)
    np.testing.assert_allclose(voltages[-2:], expected_last)


def test_voltages_for_table_unknown_sweep(esa_unit_conversion_table):
    """Test that voltages_for_table() raises for a sweep not in the LUT."""
    with pytest.raises(ValueError, match="No esa unit conversion available"):
        _voltages_for_table(esa_unit_conversion_table, 99)


@pytest.fixture
def sample_unpacked_data(sc_packet_path, furnish_kernels):
    """Flight I-ALiRT packets from 2026-03-31 (25 consecutive complete sweeps)."""
    _, xtce_path = sc_packet_path
    packet_files = sorted(
        (imap_module_directory / "tests/ialirt/data/l0").glob(
            "iois_1_packets_2026_090_*"
        )
    )
    with furnish_kernels(["naif0012.tls", "imap_sclk_0036.tsc"]):
        return xr.concat(
            [
                packet_file_to_datasets(f, xtce_path, use_derived_value=False)[478]
                for f in packet_files
            ],
            dim="epoch",
        )


@pytest.mark.external_test_data
def test_complete_sweeps_peak_matches_l2(
    sample_unpacked_data, esa_unit_conversion_table
):
    # TODO rewrite this
    """Spot check complete_sweeps() on flight I-ALiRT packets: the peak voltage and
    peak count rate of a sweep match the SWAPI L2 sweep with the same start time."""
    # Claude: from imap_swapi_l2_sci_20260331_v001.0003.cdf
    sweep_start_met = 512629392
    l2_peak_energy = 1038.004216
    l2_peak_count_rate = 10379.311

    sweeps = _complete_sweeps(sample_unpacked_data, esa_unit_conversion_table)
    assert sweeps is not None
    # Claude: all 25 sweeps in these files have all 12 packets.
    assert sweeps["count_rate"].shape == (25, 62)
    # Claude: every sweep in these packets uses sweep table 4.
    np.testing.assert_array_equal(
        sweeps["esa_energy"].values,
        np.broadcast_to(
            _voltages_for_table(esa_unit_conversion_table, 4),
            sweeps["esa_energy"].shape,
        ),
    )

    sweep = sweeps.isel(sweep=sweeps["sweep_start_met"].values == sweep_start_met)
    assert sweep.sizes["sweep"] == 1
    sweep = sweep.isel(sweep=0)

    peak = np.argmax(sweep["count_rate"].values)
    peak_voltage = sweep["esa_energy"].values[peak]
    # L2 esa_energy [eV/q] = voltage * K factor (1.93 in the LUT).
    np.testing.assert_allclose(peak_voltage, l2_peak_energy / 1.93, rtol=1e-3)

    # I-ALiRT counts are compressed by 16 (truncated), so the peak rate
    # is the L2 rate rounded down to a multiple of 16 counts.
    peak_count_rate = sweep["count_rate"].values[peak]
    l2_peak_counts = round(l2_peak_count_rate * SWAPI_LIVETIME)
    np.testing.assert_allclose(
        peak_count_rate, (l2_peak_counts // 16 * 16) / SWAPI_LIVETIME, rtol=1e-9
    )


@pytest.mark.external_test_data
def test_complete_sweeps_spin_period(sample_unpacked_data, esa_unit_conversion_table):
    # TODO rewrite this
    """Test the per-sweep spin period parsing behavior."""
    # the spin period is all ~14.889 s here
    sweeps = _complete_sweeps(sample_unpacked_data, esa_unit_conversion_table)
    np.testing.assert_allclose(sweeps["spin_period"].values, 14.889, rtol=1e-4)

    # if there are no valid values for the sweep, NaN spin period for just that sweep
    in_first_sweep = (acq >= first_start) & (acq < first_start + 12)
    modified["sc_spinperiodvalid"].values[in_first_sweep] = 0
    sweeps = _complete_sweeps(modified, esa_unit_conversion_table)
    assert np.isnan(sweeps["spin_period"].values[0])
    assert np.all(np.isfinite(sweeps["spin_period"].values[1:]))


@pytest.mark.external_test_data
def test_complete_sweeps_drops_incomplete(
    sample_unpacked_data, esa_unit_conversion_table
):
    # TODO rewrite this
    """Test that complete_sweeps() drops a sweep with a missing or duplicated
    packet, keeps the others, and returns None when no sweep is complete."""
    acq = sample_unpacked_data["swapi_acq"].values
    baseline = _complete_sweeps(sample_unpacked_data, esa_unit_conversion_table)
    assert baseline is not None
    all_starts = baseline["sweep_start_met"].values

    # drop the 6th packet (seq 5) of the second sweep.
    dropped_start = all_starts[1]
    missing = sample_unpacked_data.isel(epoch=acq != dropped_start + 5)
    sweeps = _complete_sweeps(missing, esa_unit_conversion_table)
    assert sweeps is not None
    expected = all_starts[all_starts != dropped_start]
    np.testing.assert_array_equal(sweeps["sweep_start_met"].values, expected)

    # repeat the 6th packet of the third sweep.
    repeated_start = all_starts[2]
    i_repeat = int(np.flatnonzero(acq == repeated_start + 5)[0])
    with_duplicate = xr.concat(
        [sample_unpacked_data, sample_unpacked_data.isel(epoch=[i_repeat])],
        dim="epoch",
    ).sortby("epoch")
    sweeps = _complete_sweeps(with_duplicate, esa_unit_conversion_table)
    assert sweeps is not None
    expected = all_starts[all_starts != repeated_start]
    np.testing.assert_array_equal(sweeps["sweep_start_met"].values, expected)

    # less than 12s of data -> no full sweeps
    assert (
        _complete_sweeps(
            sample_unpacked_data.isel(epoch=slice(0, 11)), esa_unit_conversion_table
        )
        is None
    )


@pytest.mark.external_test_data
def test_process_swapi_ialirt(
    sample_unpacked_data, esa_unit_conversion_table, furnish_kernels, response_tables
):
    """Test process_swapi_ialirt() on flight I-ALiRT packets: one record per
    window of 5 consecutive sweeps, with the header on the spacecraft clock and
    swapi_epoch on SWAPI's clock."""
    with furnish_kernels(["naif0012.tls", "imap_sclk_0036.tsc"]):
        swapi_data = process_swapi_ialirt(
            sample_unpacked_data, esa_unit_conversion_table, **response_tables
        )

    # TODO figure out how to deal with None here...
    sweeps = _complete_sweeps(sample_unpacked_data, esa_unit_conversion_table)
    assert sweeps is not None
    n_sweeps = sweeps.sizes["sweep"]

    assert len(swapi_data) == n_sweeps - 4

    # TODO add assertions on content


@pytest.mark.external_test_data
def test_process_swapi_ialirt_needs_5_consecutive_sweeps(
    sample_unpacked_data, esa_unit_conversion_table, response_tables
):
    """Test that process_swapi_ialirt() returns no records when there is no run
    of 5 consecutive sweeps."""
    acq = sample_unpacked_data["swapi_acq"].values
    baseline = _complete_sweeps(sample_unpacked_data, esa_unit_conversion_table)
    assert baseline is not None
    all_starts = baseline["sweep_start_met"].values

    # keep only the first 4 complete sweeps.
    four_sweeps = sample_unpacked_data.isel(epoch=acq < all_starts[4])
    sweeps = _complete_sweeps(four_sweeps, esa_unit_conversion_table)
    assert sweeps is not None
    assert sweeps.sizes["sweep"] == 4
    assert (
        process_swapi_ialirt(four_sweeps, esa_unit_conversion_table, **response_tables)
        == []
    )

    # drop one packet from every 5th sweep, leaving runs of 4.
    dropped_starts = all_starts[4::5]
    gapped = sample_unpacked_data.isel(epoch=~np.isin(acq, dropped_starts + 5))
    assert (
        process_swapi_ialirt(gapped, esa_unit_conversion_table, **response_tables) == []
    )


SPIN = "_fit_single_maxwellian_with_spin"
DUAL = "fit_dual_maxwellian_no_spin"


@pytest.mark.parametrize(
    ("params", "expected_calls"),
    [
        pytest.param(
            {
                "velocity": [0.0, -400.0, 0.0],
                "density": 5.0,
                "temperature": 80_000.0,
            },
            [SPIN],
            id="on_axis",
        ),
        pytest.param(
            {
                "velocity": [15.0, -450.0, -15.0],
                "density": 5.0,
                "temperature": 100_000.0,
            },
            [SPIN],
            id="deflected",
        ),
        pytest.param(
            {
                "velocity": [0.0, -400.0, 0.0],
                "density": 5.0,
                "temperature": 200_000.0,
                "suprathermal": (0.25, 3e6),
            },
            [DUAL],
            id="shocked",
        ),
        # Claude: the blended core and suprathermal put the peak step several
        # Claude: percent above the bulk speed, enough to start the fit in the
        # Claude: wrong pair without the grid search over speed
        pytest.param(
            {
                "velocity": [0.0, -350.0, 0.0],
                "density": 5.0,
                "temperature": 300_000.0,
                "suprathermal": (2.0, 1.2e6),
            },
            [DUAL],
            id="shocked_blended",
        ),
    ],
)
def test_fit_chunk(
    response_tables,
    monkeypatch,
    params,
    expected_calls,
):

    epoch = int(et_to_ttj2000ns(str_to_et("2026-01-01T00:00:00")))
    cadence_ns = int(SWEEP_CADENCE_SECONDS * 1e9)
    esa_voltages = np.geomspace(100, 20_000, 62) / NOMINAL_K_FACTOR
    responses = []
    for i in range(5):
        measurement_epochs = np.asarray(
            sweep_measurement_times(epoch + i * cadence_ns)[1:63]
        )
        responses.append(
            measurement_species_responses(
                esa_voltages=esa_voltages,
                epochs=measurement_epochs,
                inertial_to_swapi=_inertial_to_swapi(
                    measurement_epochs, reference_epoch=epoch, spin_period=14.889
                ),
                species=Species.PROTON,
                **response_tables,
            )
        )
    velocity = np.asarray(params["velocity"])
    populations = [(params["density"], params["temperature"])] + (
        [params["suprathermal"]] if "suprathermal" in params else []
    )
    ideal = np.array(
        [
            sum(
                maxwellian_count_rates(
                    density=n,
                    temperature=t,
                    velocity_inertial_sc=velocity,
                    responses=sweep_responses,
                )[0]
                for n, t in populations
            )
            for sweep_responses in responses
        ]
    )
    observable = ideal * deadtime_factor(ideal)

    # TODO is there a cleaner way to do this?
    calls = []
    for name in [DUAL, SPIN]:
        original = getattr(process_swapi, name)
        monkeypatch.setattr(
            process_swapi,
            name,
            lambda *a, _f=original, _n=name, **k: calls.append(_n) or _f(*a, **k),
        )

    fit_speed, fit_density, fit_temperature = _fit_chunk(
        responses=responses,
        count_rates=observable,
    )

    assert calls == expected_calls
    assert fit_speed == pytest.approx(np.linalg.norm(velocity), rel=0.01)
    assert fit_density == pytest.approx(params["density"], rel=0.05)
    assert fit_temperature == pytest.approx(params["temperature"], rel=0.05)


# TODO rewrite
def test_score_temperature_pairs():
    # Claude: a core and a hotter suprathermal population with Gaussian-shaped
    # Claude: spectra, observed after deadtime with 2% noise
    temperatures = np.geomspace(0.01, 10.0, 16)
    steps = np.linspace(-3, 3, 62)
    unit_rates = 1e5 * np.exp(-(steps**2) / temperatures[:, None])
    ideal = 3.0 * unit_rates[4] + 0.5 * unit_rates[8]
    noise = np.random.default_rng(0).normal(1, 0.02, steps.size)
    count_rates = ideal * deadtime_factor(ideal) * noise

    core_densities, supra_densities, rmse = _score_temperature_pairs(
        unit_rates, count_rates, temperatures
    )
    densities = np.stack([core_densities, supra_densities], axis=-1)

    # Claude: reference: unweighted least squares against the deadtime-corrected
    # Claude: rates, solving each allowed pair's normal equations with
    # Claude: np.linalg.solve and taking the residual from the fitted spectrum
    target = count_rates / (1 - SWAPI_DEADTIME_S * count_rates)
    expected_densities = np.zeros_like(densities)
    expected_rmse = np.full_like(rmse, np.inf)
    n_allowed = 0
    for i, j in np.ndindex(rmse.shape):
        if not 1 < temperatures[j] / temperatures[i] <= DUAL_MAX_TEMPERATURE_RATIO:
            continue
        n_allowed += 1
        design = np.column_stack([unit_rates[i], unit_rates[j]])
        pair_densities = np.linalg.solve(design.T @ design, design.T @ target)
        core, supra = pair_densities
        if (
            core >= 0
            and supra >= 0
            and supra <= DUAL_MAX_GRID_DENSITY_RATIO * core
            and supra >= DUAL_MIN_GRID_DENSITY_RATIO * core
        ):
            expected_densities[i, j] = pair_densities
            residual = design @ pair_densities - target
            expected_rmse[i, j] = np.sqrt(np.mean(residual**2))

    # Claude: some allowed pairs are kept and some rejected for their densities
    assert 0 < np.isfinite(expected_rmse).sum() < n_allowed
    np.testing.assert_allclose(densities, expected_densities, rtol=1e-6)
    np.testing.assert_allclose(rmse, expected_rmse, rtol=1e-6)
