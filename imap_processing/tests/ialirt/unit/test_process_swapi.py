import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_swapi import complete_sweeps, voltages_for_table
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
def flight_packets_20251210(swapi_postlaunch_sc_packet_path, furnish_kernels):
    """Flight I-ALiRT packets from 2025-12-10 (two files with a gap between)."""
    packet_files, xtce_path = swapi_postlaunch_sc_packet_path
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
    flight_packets_20251210, esa_unit_conversion_table
):
    """Spot check complete_sweeps() on flight I-ALiRT packets: the peak voltage and
    peak count rate of a sweep match the SWAPI L2 sweep with the same start time."""
    # from imap_swapi_l2_sci_20251210_v001.0004.cdf
    sweep_start_met = 503042280
    l2_peak_energy = 800.3650292
    l2_peak_count_rate = 41324.14

    sweeps = complete_sweeps(flight_packets_20251210, esa_unit_conversion_table)
    assert sweeps is not None
    # 8 L2 sweeps have all 12 packets in these files.
    assert sweeps["count_rate"].shape == (8, 62)
    # every sweep in these packets uses sweep table 3.
    np.testing.assert_array_equal(
        sweeps["esa_energy"].values,
        np.broadcast_to(
            voltages_for_table(esa_unit_conversion_table, 3),
            sweeps["esa_energy"].shape,
        ),
    )

    sweep = sweeps.isel(sweep=sweeps["sweep_start_time"].values == sweep_start_met)
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
    flight_packets_20251210, esa_unit_conversion_table
):
    """Test that complete_sweeps() drops a sweep with a missing or duplicated
    packet, keeps the others, and returns None when no sweep is complete."""
    acq = flight_packets_20251210["swapi_acq"].values
    baseline = complete_sweeps(flight_packets_20251210, esa_unit_conversion_table)
    assert baseline is not None
    all_starts = baseline["sweep_start_time"].values

    # drop the 6th packet (seq 5) of the sweep starting at 503042292.
    missing = flight_packets_20251210.isel(epoch=acq != 503042292 + 5)
    sweeps = complete_sweeps(missing, esa_unit_conversion_table)
    assert sweeps is not None
    assert 503042292 in all_starts
    expected = all_starts[all_starts != 503042292]
    np.testing.assert_array_equal(sweeps["sweep_start_time"].values, expected)

    # repeat the 6th packet of the sweep starting at 503042304.
    i_repeat = int(np.flatnonzero(acq == 503042304 + 5)[0])
    duplicated = xr.concat(
        [flight_packets_20251210, flight_packets_20251210.isel(epoch=[i_repeat])],
        dim="epoch",
    )
    sweeps = complete_sweeps(duplicated, esa_unit_conversion_table)
    assert sweeps is not None
    assert 503042304 in all_starts
    expected = all_starts[all_starts != 503042304]
    np.testing.assert_array_equal(sweeps["sweep_start_time"].values, expected)

    # 11 packets can't make a full sweep.
    assert (
        complete_sweeps(
            flight_packets_20251210.isel(epoch=slice(0, 11)), esa_unit_conversion_table
        )
        is None
    )
