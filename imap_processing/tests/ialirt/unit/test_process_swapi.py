from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_swapi import (
    DUAL_DENSITY_RATIO,
    DUAL_FIT_TEMPERATURE_THRESHOLD_K,
    MIN_R_SQUARED,
    PEAK_HALF_WIDTH,
    SPIN_PHASE_TO_RAD,
    fit_chunk,
    peak_r_squared,
    peak_window,
    process_swapi_ialirt,
    spin_angles_at,
)
from imap_processing.spice.time import met_to_ttj2000ns, ttj2000ns_to_met
from imap_processing.swapi.constants import NUM_ENERGY_STEPS, SIMION_K_FACTOR, Species
from imap_processing.swapi.fitting.fit_single_maxwellian_with_spin import (
    spin_rates_and_jacobian,
    spin_rotations,
)
from imap_processing.swapi.forward_model.deadtime import deadtime_factor
from imap_processing.swapi.forward_model.response_tables import SwapiResponse
from imap_processing.swapi.swapi_utils import read_swapi_lut_table
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def xtce_swapi_path():
    """Returns the xtce auxiliary directory."""
    return imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_swapi.xml"


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the binary packet path."""
    return (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "BinLog CCSDS_FRAG_TLM_20240826_152323Z_IALIRT_data_for_SDC.bin"
    )


@pytest.fixture(scope="session")
def esa_unit_conversion_table() -> pd.DataFrame:
    """
    Read the ESA unit conversion table.

    Returns
    -------
    esa_unit_conversion_table : pandas.DataFrame
        The ESA unit conversion table.
    """
    esa_file_path = (
        imap_module_directory
        / "tests/swapi/lut/imap_swapi_esa-unit-conversion_20250626_v001.csv"
    )
    df = read_swapi_lut_table(esa_file_path)
    return df


@pytest.fixture(scope="session")
def swapi_response() -> SwapiResponse:
    """SWAPI instrument response tables for the forward-model fits."""
    lut = imap_module_directory / "tests/swapi/lut"
    return SwapiResponse.from_files(
        azimuthal_transmission_path=lut / "azimuthal-transmission.csv",
        central_effective_area_path=lut / "central-effective-area.csv",
        passband_fit_coefficients_path=lut / "passband-fit-coefficients.csv",
        efficiency_path=lut / "efficiency-lut.dat",
    )


@pytest.fixture(scope="session")
def swapi_test_data():
    """Returns the l0 validation dataframe."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "eu_SWP_IAL_20240826_152033.csv"
    )
    data = pd.read_csv(data_path)

    return data


@pytest.fixture
def xarray_data(binary_packet_path, xtce_swapi_path):
    """Create SWAPI xarray dataset for testing."""

    xarray_data = packet_file_to_datasets(
        binary_packet_path, xtce_swapi_path, use_derived_value=True
    )[1187]
    return xarray_data


@pytest.fixture
def sc_xarray_data(sc_packet_path):
    """Extract spacecraft packet for testing."""

    packet_path, xtce_ialirt_path = sc_packet_path
    sc_xarray_data = packet_file_to_datasets(
        packet_path, xtce_ialirt_path, use_derived_value=False
    )[478]
    return sc_xarray_data


def test_decom_packets(xarray_data, swapi_test_data):
    """Check that all instrument parameters are accounted for after decom."""

    # TODO: confirm w/ SWAPI team validity_enum flag can be
    #  consistent with other instruments.
    fields_to_test = {
        "swapi_flag": "I_ALIRT_STATUS",
        "swapi_reserved": "INST_RES_ST",
        "swapi_seq_number": "SEQ_NUMBER",
        "swapi_version": "SWEEP_TABLE",
        "swapi_coin_cnt0": "COIN_CNT0",
        "swapi_coin_cnt1": "COIN_CNT1",
        "swapi_coin_cnt2": "COIN_CNT2",
        "swapi_coin_cnt3": "COIN_CNT3",
        "swapi_coin_cnt4": "COIN_CNT4",
        "swapi_coin_cnt5": "COIN_CNT5",
        "swapi_spare": "SPARE",
        "swapi_shcoarse": "SHCOARSE",
    }
    _, index, test_index = np.intersect1d(
        xarray_data["swapi_acq"], swapi_test_data["ACQ_TIME"], return_indices=True
    )

    for xarray_field, test_field in fields_to_test.items():
        actual_values = xarray_data[xarray_field].values[index]
        expected_values = swapi_test_data[test_field].values[test_index]

        # Assert that all values match
        assert np.all(actual_values == expected_values), (
            f"Mismatch found in {xarray_field}: "
            f"actual {actual_values}, expected {expected_values}"
        )


@pytest.mark.external_test_data
@mock.patch("imap_processing.ialirt.l0.process_swapi.fit_chunk")
def test_process_swapi_ialirt(
    mock_fit_chunk,
    xarray_data,
    sc_xarray_data,
    esa_unit_conversion_table,
    swapi_response,
):
    """Test that process_swapi_ialirt() reports the fit of each 5-sweep chunk."""

    mock_fit_chunk.return_value = {
        "success": True,
        "speed": 450.0,
        "density": 5.0,
        "temperature": 1e5,
    }

    # Adding necessary time variables from spacecraft packet
    xarray_data = xarray_data.assign(sc_sclk_sec=sc_xarray_data["sc_sclk_sec"])
    xarray_data["sc_sclk_sec"].data = sc_xarray_data["sc_sclk_sec"][
        0 : xarray_data["swapi_flag"].shape[0]
    ].data
    xarray_data = xarray_data.assign(sc_sclk_sub_sec=sc_xarray_data["sc_sclk_sub_sec"])
    xarray_data["sc_sclk_sub_sec"].data = sc_xarray_data["sc_sclk_sub_sec"][
        0 : xarray_data["swapi_flag"].shape[0]
    ].data
    xarray_data = xarray_data.assign(sc_spin_phase=sc_xarray_data["sc_spin_phase"])
    xarray_data["sc_spin_phase"].data = sc_xarray_data["sc_spin_phase"][
        0 : xarray_data["swapi_flag"].shape[0]
    ].data

    swapi_result = process_swapi_ialirt(
        xarray_data, esa_unit_conversion_table, swapi_response
    )

    assert len(swapi_result) == mock_fit_chunk.call_count
    for call in mock_fit_chunk.call_args_list:
        assert call.kwargs["response"] is swapi_response
        assert call.kwargs["esa_voltages"].shape == (5, 62)
        assert call.kwargs["count_rates"].shape == (5, 62)
        assert call.kwargs["spin_angles"].shape == (5, 62)
    for product in swapi_result:
        assert product["instrument"] == "swapi"
        assert product["swapi_pseudo_proton_speed"] == 450.0
        assert product["swapi_pseudo_proton_density"] == 5.0
        assert product["swapi_pseudo_proton_temperature"] == 1e5


@mock.patch("imap_processing.ialirt.l0.process_swapi.fit_chunk")
def test_process_swapi_ialirt_header_met_is_chunk_center(
    mock_fit_chunk, xarray_data, esa_unit_conversion_table, swapi_response
):
    """The header time is the spacecraft clock at the center of the chunk."""
    mock_fit_chunk.return_value = {
        "success": True,
        "speed": 450.0,
        "density": 5.0,
        "temperature": 1e5,
    }
    # synthetic spacecraft clock, offset from SWAPI's acquisition time
    sc_offset_s = 3.0
    xarray_data = xarray_data.assign(
        sc_sclk_sec=xarray_data["swapi_acq"] + int(sc_offset_s),
        sc_sclk_sub_sec=xr.zeros_like(xarray_data["swapi_acq"]),
        sc_spin_phase=xr.zeros_like(xarray_data["swapi_acq"]),
    )

    swapi_result = process_swapi_ialirt(
        xarray_data, esa_unit_conversion_table, swapi_response
    )

    assert swapi_result
    for product, call in zip(swapi_result, mock_fit_chunk.call_args_list, strict=True):
        # the chunk spans 5 sweeps of 12 one-second packets
        epoch_met = ttj2000ns_to_met(product["swapi_epoch"])
        chunk_start_acq = epoch_met - 29
        expected_met = (2 * (chunk_start_acq + sc_offset_s) + 59) // 2
        assert product["met"] == expected_met
        assert product["ttj2000ns"] == met_to_ttj2000ns(expected_met)
        assert call.kwargs["epoch"] == product["swapi_epoch"]


@mock.patch("imap_processing.ialirt.l0.process_swapi.fit_chunk")
def test_process_swapi_ialirt_fits_esa_voltages(
    mock_fit_chunk, xarray_data, esa_unit_conversion_table, swapi_response
):
    """The fit gets the table's ESA voltages, not its k-factor-dependent energies."""
    mock_fit_chunk.return_value = {"success": False}
    xarray_data = xarray_data.assign(
        sc_sclk_sec=xarray_data["swapi_acq"],
        sc_sclk_sub_sec=xr.zeros_like(xarray_data["swapi_acq"]),
        sc_spin_phase=xr.zeros_like(xarray_data["swapi_acq"]),
    )

    process_swapi_ialirt(xarray_data, esa_unit_conversion_table, swapi_response)

    assert mock_fit_chunk.call_count > 0
    table = esa_unit_conversion_table
    for call in mock_fit_chunk.call_args_list:
        voltages = call.kwargs["esa_voltages"]
        sweep_id = int(xarray_data["swapi_version"].values[0])
        rows = table[table["Sweep #"] == sweep_id]
        rows = rows[rows["timestamp"] == rows["timestamp"].max()].sort_values(
            "ESA Step #"
        )
        expected = rows["Voltage"].to_numpy()[1:63].astype(float)
        np.testing.assert_array_equal(voltages, np.tile(expected, (5, 1)))


@pytest.mark.external_test_data
def test_process_spacecraft_packet(
    esa_unit_conversion_table, swapi_response, swapi_postlaunch_sc_packet_path
):
    """Tests spacecraft packet processing."""

    packet_path, xtce_ialirt_path = swapi_postlaunch_sc_packet_path
    xarray_data = tuple(
        packet_file_to_datasets(packet, xtce_ialirt_path, use_derived_value=False)[478]
        for packet in packet_path
    )
    postlaunch_sc_xarray_data = xr.concat(xarray_data, dim="epoch")

    postlaunch_sc_xarray_data["swapi_version"].data = np.full_like(
        postlaunch_sc_xarray_data["swapi_version"].data, 2
    )
    swapi_product = process_swapi_ialirt(
        postlaunch_sc_xarray_data, esa_unit_conversion_table, swapi_response
    )

    assert len(swapi_product) == 0


# ~2026-01-01 in TT J2000 ns, for the relative efficiency lookup
FIT_EPOCH = 820_497_669_184_000_000
FIT_VOLTAGES = np.tile(
    np.geomspace(100.0, 20000.0, NUM_ENERGY_STEPS) / SIMION_K_FACTOR, (5, 1)
)
# spin angles of consecutive 12 s sweeps of 1/6 s steps with a 14.89 s spin
FIT_SPIN_ANGLES = (
    2
    * np.pi
    * np.add.outer(12.0 * np.arange(5), np.arange(NUM_ENERGY_STEPS) / 6.0)
    / 14.89
)


def _single_maxwellian_rates(response, density, temperature, velocity):
    """Post-deadtime rates of a 5-sweep chunk seeing one spinning Maxwellian."""
    responses = response.step_responses(FIT_VOLTAGES, FIT_EPOCH, Species.PROTON)
    rotations = spin_rotations(FIT_SPIN_ANGLES).reshape(-1, 3, 3)
    state = np.array([np.log(density), np.log(temperature), *velocity])
    ideal = spin_rates_and_jacobian(responses, rotations, state)[0]
    return (ideal * deadtime_factor(ideal)).reshape(FIT_VOLTAGES.shape)


@mock.patch(
    "imap_processing.ialirt.l0.process_swapi.fit_dual_maxwellian_no_spin",
)
def test_fit_chunk_single_maxwellian(mock_dual_fit, swapi_response):
    """A cool plasma is fit with the single Maxwellian with spin only."""
    velocity = np.array([2.0, -450.0, -2.0])
    count_rates = _single_maxwellian_rates(swapi_response, 5.0, 1e5, velocity)

    result = fit_chunk(
        response=swapi_response,
        esa_voltages=FIT_VOLTAGES,
        count_rates=count_rates,
        spin_angles=FIT_SPIN_ANGLES,
        epoch=FIT_EPOCH,
    )

    mock_dual_fit.assert_not_called()
    assert result["success"]
    assert result["speed"] == pytest.approx(np.linalg.norm(velocity), rel=0.01)
    assert result["density"] == pytest.approx(5.0, rel=0.05)
    assert result["temperature"] == pytest.approx(1e5, rel=0.05)


# a peaked chunk of rates, peaking at step 30, for the fit-selection tests
PEAKED_RATES = np.tile(
    1e4 * np.exp(-0.5 * ((np.arange(NUM_ENERGY_STEPS) - 30) / 3.0) ** 2) + 10.0,
    (5, 1),
)
PEAK_STEPS = slice(30 - PEAK_HALF_WIDTH, 30 + PEAK_HALF_WIDTH + 1)


def _mock_fit(temperature, good, density=5.0):
    """A fit result whose model is the observed rates (good) or flat (bad)."""
    return {
        "speed": 400.0,
        "density": density,
        "temperature": temperature,
        "model_count_rates": PEAKED_RATES if good else np.zeros_like(PEAKED_RATES),
    }


def _fit_chunk_with_mocks(swapi_response, single_fit, dual_fit):
    """Run fit_chunk on PEAKED_RATES with mocked single and dual fits."""
    with (
        mock.patch(
            "imap_processing.ialirt.l0.process_swapi.fit_single_maxwellian_with_spin",
            return_value=single_fit,
        ) as mock_single_fit,
        mock.patch(
            "imap_processing.ialirt.l0.process_swapi.fit_dual_maxwellian_no_spin",
            return_value=dual_fit,
        ) as mock_dual_fit,
    ):
        result = fit_chunk(
            response=swapi_response,
            esa_voltages=FIT_VOLTAGES,
            count_rates=PEAKED_RATES,
            spin_angles=FIT_SPIN_ANGLES,
            epoch=FIT_EPOCH,
        )
    return result, mock_single_fit, mock_dual_fit


def _moments(fit):
    return {key: fit[key] for key in ("speed", "density", "temperature")}


def test_fit_chunk_hot_plasma_uses_dual_fit(swapi_response):
    """A plasma hotter than the threshold is refit with the dual Maxwellian."""
    temperature = 2 * DUAL_FIT_TEMPERATURE_THRESHOLD_K
    count_rates = _single_maxwellian_rates(
        swapi_response, 5.0, temperature, [0.0, -450.0, 0.0]
    )
    # dense enough that the dual fit is reported whether or not the single fit
    # is good
    dual_fit = {
        "speed": 451.0,
        "density": 4 * DUAL_DENSITY_RATIO * 5.0,
        "temperature": 6.1e5,
        "model_count_rates": count_rates,
    }

    with mock.patch(
        "imap_processing.ialirt.l0.process_swapi.fit_dual_maxwellian_no_spin",
        return_value=dual_fit,
    ) as mock_dual_fit:
        result = fit_chunk(
            response=swapi_response,
            esa_voltages=FIT_VOLTAGES,
            count_rates=count_rates,
            spin_angles=FIT_SPIN_ANGLES,
            epoch=FIT_EPOCH,
        )

    mock_dual_fit.assert_called_once()
    voltages, rates = mock_dual_fit.call_args.args
    np.testing.assert_array_equal(voltages, FIT_VOLTAGES)
    # the dual fit sees every step
    np.testing.assert_array_equal(rates, count_rates)
    assert mock_dual_fit.call_args.kwargs == {
        "swapi_response": swapi_response,
        "epoch": FIT_EPOCH,
    }
    assert result == {"success": True, **_moments(dual_fit)}


def test_fit_chunk_single_fit_sees_peak_window(swapi_response):
    """The single fit is given only the steps around the peak."""
    single_fit = _mock_fit(1e5, good=True)
    _, mock_single_fit, mock_dual_fit = _fit_chunk_with_mocks(
        swapi_response, single_fit, None
    )

    mock_dual_fit.assert_not_called()
    voltages, rates = mock_single_fit.call_args.args
    np.testing.assert_array_equal(voltages, FIT_VOLTAGES)
    assert mock_single_fit.call_args.kwargs["spin_angles"] is FIT_SPIN_ANGLES
    expected = np.full_like(PEAKED_RATES, np.nan)
    expected[:, PEAK_STEPS] = PEAKED_RATES[:, PEAK_STEPS]
    np.testing.assert_array_equal(rates, expected)


@pytest.mark.parametrize(
    ("single_temperature", "expect_dual"),
    [
        (DUAL_FIT_TEMPERATURE_THRESHOLD_K, False),
        (np.nextafter(DUAL_FIT_TEMPERATURE_THRESHOLD_K, np.inf), True),
    ],
)
def test_fit_chunk_threshold(swapi_response, single_temperature, expect_dual):
    """A good single fit is refit with the dual only above the threshold."""
    single_fit = _mock_fit(single_temperature, good=True)
    # dense enough that the dual fit is reported whenever it is tried
    dual_fit = _mock_fit(7e5, good=True, density=10 * DUAL_DENSITY_RATIO * 5.0)
    result, _, mock_dual_fit = _fit_chunk_with_mocks(
        swapi_response, single_fit, dual_fit
    )

    assert mock_dual_fit.called == expect_dual
    expected = dual_fit if expect_dual else single_fit
    assert result == {"success": True, **_moments(expected)}


@pytest.mark.parametrize(
    ("single_good", "dual_good", "density_ratio", "expected"),
    [
        # neither fit is good
        (False, False, 10.0, None),
        # only one fit is good
        (False, True, 1.0, "dual"),
        (True, False, 10.0, "single"),
        # both are good: the dual only with a density at least 2x higher
        (True, True, DUAL_DENSITY_RATIO, "dual"),
        (True, True, np.nextafter(DUAL_DENSITY_RATIO, 0), "single"),
    ],
)
def test_fit_chunk_selection(
    swapi_response, single_good, dual_good, density_ratio, expected
):
    """The reported fit depends on the fits' R^2 and their densities."""
    single_density = 4.0
    single_fit = _mock_fit(4e5, good=single_good, density=single_density)
    dual_fit = _mock_fit(3e5, good=dual_good, density=single_density * density_ratio)
    result, _, mock_dual_fit = _fit_chunk_with_mocks(
        swapi_response, single_fit, dual_fit
    )

    mock_dual_fit.assert_called_once()
    if expected is None:
        assert result == {"success": False}
    else:
        fit = dual_fit if expected == "dual" else single_fit
        assert result == {"success": True, **_moments(fit)}


def test_fit_chunk_bad_cool_single_fit_tries_dual(swapi_response):
    """A cool single fit with a bad R^2 is refit with the dual Maxwellian."""
    single_fit = _mock_fit(1e5, good=False)
    dual_fit = _mock_fit(1.2e5, good=True)
    result, _, mock_dual_fit = _fit_chunk_with_mocks(
        swapi_response, single_fit, dual_fit
    )

    mock_dual_fit.assert_called_once()
    assert result == {"success": True, **_moments(dual_fit)}


def test_fit_chunk_dual_judged_on_peak_window(swapi_response):
    """The dual fit's R^2 ignores misfit steps outside the peak window."""
    model = PEAKED_RATES.copy()
    model[:, : PEAK_STEPS.start] = 0.0
    model[:, PEAK_STEPS.stop :] = 1e6
    dual_fit = _mock_fit(1e5, good=True)
    dual_fit["model_count_rates"] = model
    result, _, _ = _fit_chunk_with_mocks(
        swapi_response, _mock_fit(1e5, good=False), dual_fit
    )

    assert result == {"success": True, **_moments(dual_fit)}


@pytest.mark.parametrize(
    ("peak", "expected"),
    [
        (30, slice(30 - PEAK_HALF_WIDTH, 30 + PEAK_HALF_WIDTH + 1)),
        # windows near the ends are shifted to stay inside the sweep
        (1, slice(0, 2 * PEAK_HALF_WIDTH + 1)),
        (60, slice(62 - 2 * PEAK_HALF_WIDTH - 1, 62)),
    ],
)
def test_peak_window(peak, expected):
    """The window is centered on the peak of the mean sweep."""
    rates = np.ones((5, 62))
    rates[:, peak] = 10.0
    # an invalid sample doesn't stop its step being the peak
    rates[0, peak] = np.nan
    assert peak_window(rates) == expected


def test_peak_r_squared():
    """R^2 uses the sweep-averaged residuals and rates on the window."""
    rates = np.array([[1.0, 2.0, 5.0, 2.0], [3.0, 4.0, 7.0, np.nan]])
    model = np.array([[2.0, 3.0, 6.0, 0.0], [3.0, 4.0, 5.0, 3.0]])
    window = slice(0, 3)
    # mean rates (2, 3, 6) and mean residuals (0.5, 0.5, -0.5) on the window
    observed = np.array([2.0, 3.0, 6.0])
    expected = 1.0 - 0.75 / np.sum((observed - observed.mean()) ** 2)
    assert peak_r_squared(model, rates, window) == pytest.approx(expected)
    assert peak_r_squared(model, rates, window) >= MIN_R_SQUARED


def test_peak_r_squared_constant_rates():
    """R^2 is NaN where the observed rates don't vary."""
    rates = np.ones((2, 4))
    assert np.isnan(peak_r_squared(rates, rates, slice(0, 4)))


def test_fit_chunk_invalid_voltages(swapi_response):
    """Non-positive ESA voltages make the fit fail without raising."""
    voltages = FIT_VOLTAGES.copy()
    voltages[0, 0] = 0.0
    with mock.patch(
        "imap_processing.ialirt.l0.process_swapi.fit_single_maxwellian_with_spin"
    ) as mock_single_fit:
        result = fit_chunk(
            response=swapi_response,
            esa_voltages=voltages,
            count_rates=np.ones(voltages.shape),
            spin_angles=FIT_SPIN_ANGLES,
            epoch=FIT_EPOCH,
        )

    mock_single_fit.assert_not_called()
    assert result == {"success": False}


def test_fit_chunk_mismatched_sweep_voltages(swapi_response):
    """Sweeps with different ESA voltages make the fit fail without raising."""
    voltages = FIT_VOLTAGES.copy()
    voltages[1] *= 1.01
    with mock.patch(
        "imap_processing.ialirt.l0.process_swapi.fit_single_maxwellian_with_spin"
    ) as mock_single_fit:
        result = fit_chunk(
            response=swapi_response,
            esa_voltages=voltages,
            count_rates=np.ones(voltages.shape),
            spin_angles=FIT_SPIN_ANGLES,
            epoch=FIT_EPOCH,
        )

    mock_single_fit.assert_not_called()
    assert result == {"success": False}


def test_fit_chunk_non_finite_fit(swapi_response):
    """A fit with non-finite moments is reported as failed."""
    with mock.patch(
        "imap_processing.ialirt.l0.process_swapi.fit_single_maxwellian_with_spin",
        return_value={**_mock_fit(1e5, good=True), "density": np.nan},
    ):
        result = fit_chunk(
            response=swapi_response,
            esa_voltages=FIT_VOLTAGES,
            count_rates=PEAKED_RATES,
            spin_angles=FIT_SPIN_ANGLES,
            epoch=FIT_EPOCH,
        )

    assert result == {"success": False}


def test_spin_angles_at():
    """Spin phases are unwrapped and fit with a uniform spin, extrapolated."""
    period_s = 14.89
    packet_met = 1000.0 + np.arange(60.0)
    true_angle = 0.3 + 2 * np.pi * (packet_met - 1000.0) / period_s
    # telemetry counts, wrapped and truncated as the spacecraft reports them
    counts = np.floor((true_angle % (2 * np.pi)) / SPIN_PHASE_TO_RAD)
    shuffled = np.random.default_rng(0).permutation(60)

    met = np.array([[997.0, 1030.5], [1059.0, 1062.0]])
    angles = spin_angles_at(
        met, packet_met[shuffled], counts[shuffled] * SPIN_PHASE_TO_RAD
    )

    expected = 0.3 + 2 * np.pi * (met - 1000.0) / period_s
    np.testing.assert_allclose(angles, expected, atol=1e-3)


@mock.patch("imap_processing.ialirt.l0.process_swapi.fit_chunk")
def test_process_swapi_ialirt_spin_angles(
    mock_fit_chunk, xarray_data, esa_unit_conversion_table, swapi_response
):
    """The fit gets the spin angle of each step from the spacecraft spin phase."""
    mock_fit_chunk.return_value = {"success": False}
    period_s = 14.89
    sc_offset_s = 3.0
    acq = xarray_data["swapi_acq"].astype(float)
    # spin phase counts at each packet's spacecraft time
    sc_angle = 2 * np.pi * (acq + sc_offset_s) / period_s
    xarray_data = xarray_data.assign(
        sc_sclk_sec=xarray_data["swapi_acq"] + int(sc_offset_s),
        sc_sclk_sub_sec=xr.zeros_like(xarray_data["swapi_acq"]),
        sc_spin_phase=np.floor((sc_angle % (2 * np.pi)) / SPIN_PHASE_TO_RAD),
    )

    process_swapi_ialirt(xarray_data, esa_unit_conversion_table, swapi_response)

    assert mock_fit_chunk.call_count > 0
    for call in mock_fit_chunk.call_args_list:
        angles = call.kwargs["spin_angles"]
        # steps 1/6 s apart within a sweep, sweeps 12 s apart
        np.testing.assert_allclose(
            np.diff(angles, axis=1), 2 * np.pi / 6 / period_s, atol=1e-4
        )
        np.testing.assert_allclose(
            np.diff(angles, axis=0), 2 * np.pi * 12 / period_s, atol=1e-4
        )
        # the angle at a step is that of the spacecraft at the step's MET
        epoch_met = ttj2000ns_to_met(call.kwargs["epoch"])
        first_step_met = epoch_met - 29 + 1 / 6
        assert np.cos(angles[0, 0]) == pytest.approx(
            np.cos(2 * np.pi * first_step_met / period_s), abs=1e-3
        )
        assert np.sin(angles[0, 0]) == pytest.approx(
            np.sin(2 * np.pi * first_step_met / period_s), abs=1e-3
        )
