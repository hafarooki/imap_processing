import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_swapi import (
    complete_sweeps,
    process_swapi_ialirt,
    voltages_for_table,
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
    voltages = voltages_for_table(esa_unit_conversion_table, table_id)

    assert voltages.shape == (62,)
    assert voltages.dtype == np.float64
    assert np.all(voltages > 0)
    np.testing.assert_allclose(voltages[:2], expected_first)
    np.testing.assert_allclose(voltages[-2:], expected_last)


def test_voltages_for_table_unknown_sweep(esa_unit_conversion_table):
    """Test that voltages_for_table() raises for a sweep not in the LUT."""
    with pytest.raises(ValueError, match="No esa unit conversion available"):
        voltages_for_table(esa_unit_conversion_table, 99)


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
    """Spot check complete_sweeps() on flight I-ALiRT packets: the peak voltage and
    peak count rate of a sweep match the SWAPI L2 sweep with the same start time."""
    # Claude: from imap_swapi_l2_sci_20260331_v001.0003.cdf
    sweep_start_met = 512629392
    l2_peak_energy = 1038.004216
    l2_peak_count_rate = 10379.311

    sweeps = complete_sweeps(sample_unpacked_data, esa_unit_conversion_table)
    assert sweeps is not None
    # Claude: all 25 sweeps in these files have all 12 packets.
    assert sweeps["count_rate"].shape == (25, 62)
    # Claude: every sweep in these packets uses sweep table 4.
    np.testing.assert_array_equal(
        sweeps["esa_energy"].values,
        np.broadcast_to(
            voltages_for_table(esa_unit_conversion_table, 4),
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
def test_complete_sweeps_drops_incomplete(
    sample_unpacked_data, esa_unit_conversion_table
):
    """Test that complete_sweeps() drops a sweep with a missing or duplicated
    packet, keeps the others, and returns None when no sweep is complete."""
    acq = sample_unpacked_data["swapi_acq"].values
    baseline = complete_sweeps(sample_unpacked_data, esa_unit_conversion_table)
    assert baseline is not None
    all_starts = baseline["sweep_start_met"].values

    # drop the 6th packet (seq 5) of the second sweep.
    dropped_start = all_starts[1]
    missing = sample_unpacked_data.isel(epoch=acq != dropped_start + 5)
    sweeps = complete_sweeps(missing, esa_unit_conversion_table)
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
    sweeps = complete_sweeps(with_duplicate, esa_unit_conversion_table)
    assert sweeps is not None
    expected = all_starts[all_starts != repeated_start]
    np.testing.assert_array_equal(sweeps["sweep_start_met"].values, expected)

    # less than 12s of data -> no full sweeps
    assert (
        complete_sweeps(
            sample_unpacked_data.isel(epoch=slice(0, 11)), esa_unit_conversion_table
        )
        is None
    )


@pytest.mark.external_test_data
def test_process_swapi_ialirt(
    sample_unpacked_data, esa_unit_conversion_table, furnish_kernels
):
    """Test process_swapi_ialirt() on flight I-ALiRT packets: one record per
    window of 5 consecutive sweeps, with the header on the spacecraft clock and
    swapi_epoch on SWAPI's clock."""
    with furnish_kernels(["naif0012.tls", "imap_sclk_0036.tsc"]):
        swapi_data = process_swapi_ialirt(
            sample_unpacked_data, esa_unit_conversion_table
        )

    # TODO figure out how to deal with None here...
    n_sweeps = len(complete_sweeps(sample_unpacked_data, esa_unit_conversion_table))

    assert len(swapi_data) == n_sweeps - 4

    # TODO add assertions on content


@pytest.mark.external_test_data
def test_process_swapi_ialirt_needs_5_consecutive_sweeps(
    sample_unpacked_data, esa_unit_conversion_table
):
    """Test that process_swapi_ialirt() returns no records when there is no run
    of 5 consecutive sweeps."""
    acq = sample_unpacked_data["swapi_acq"].values
    baseline = complete_sweeps(sample_unpacked_data, esa_unit_conversion_table)
    assert baseline is not None
    all_starts = baseline["sweep_start_met"].values

    # keep only the first 4 complete sweeps.
    four_sweeps = sample_unpacked_data.isel(epoch=acq < all_starts[4])
    sweeps = complete_sweeps(four_sweeps, esa_unit_conversion_table)
    assert sweeps is not None
    assert sweeps.sizes["sweep"] == 4
    assert process_swapi_ialirt(four_sweeps, esa_unit_conversion_table) == []

    # drop one packet from every 5th sweep, leaving runs of 4.
    dropped_starts = all_starts[4::5]
    gapped = sample_unpacked_data.isel(epoch=~np.isin(acq, dropped_starts + 5))
    assert process_swapi_ialirt(gapped, esa_unit_conversion_table) == []
