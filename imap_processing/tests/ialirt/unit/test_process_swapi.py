from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_swapi import (
    DUAL_FIT_TEMPERATURE_THRESHOLD_K,
    fit_chunk,
    process_swapi_ialirt,
)
from imap_processing.swapi.constants import NUM_ENERGY_STEPS, Species
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
        transmission_path=lut / "azimuthal-transmission.csv",
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

    swapi_result = process_swapi_ialirt(
        xarray_data, esa_unit_conversion_table, swapi_response
    )

    assert len(swapi_result) == mock_fit_chunk.call_count
    for call in mock_fit_chunk.call_args_list:
        assert call.kwargs["response"] is swapi_response
        assert call.kwargs["esa_energies"].shape == (5, NUM_ENERGY_STEPS)
        assert call.kwargs["count_rates"].shape == (5, NUM_ENERGY_STEPS)
    for product in swapi_result:
        assert product["instrument"] == "swapi"
        assert product["swapi_pseudo_proton_speed"] == 450.0
        assert product["swapi_pseudo_proton_density"] == 5.0
        assert product["swapi_pseudo_proton_temperature"] == 1e5


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
FIT_ENERGIES = np.tile(np.geomspace(100.0, 20000.0, NUM_ENERGY_STEPS), (5, 1))


def _single_maxwellian_rates(response, density, temperature, velocity):
    """Post-deadtime rates of a 5-sweep chunk seeing one spinning Maxwellian."""
    passbands = response.passbands(FIT_ENERGIES)
    responses = response.step_responses(
        passbands, FIT_ENERGIES, FIT_EPOCH, Species.PROTON
    )
    rotations = spin_rotations(*FIT_ENERGIES.shape).reshape(-1, 3, 3)
    state = np.array([np.log(density), np.log(temperature), *velocity])
    ideal = spin_rates_and_jacobian(responses, rotations, state)[0]
    return (ideal * deadtime_factor(ideal)).reshape(FIT_ENERGIES.shape)


@mock.patch(
    "imap_processing.ialirt.l0.process_swapi.fit_dual_maxwellian_no_spin",
)
def test_fit_chunk_single_maxwellian(mock_dual_fit, swapi_response):
    """A cool plasma is fit with the single Maxwellian with spin only."""
    velocity = np.array([2.0, -450.0, -2.0])
    count_rates = _single_maxwellian_rates(swapi_response, 5.0, 1e5, velocity)

    result = fit_chunk(
        response=swapi_response,
        esa_energies=FIT_ENERGIES,
        count_rates=count_rates,
        epoch=FIT_EPOCH,
    )

    mock_dual_fit.assert_not_called()
    assert result["success"]
    assert result["speed"] == pytest.approx(np.linalg.norm(velocity), rel=0.01)
    assert result["density"] == pytest.approx(5.0, rel=0.05)
    assert result["temperature"] == pytest.approx(1e5, rel=0.05)


def test_fit_chunk_hot_plasma_uses_dual_fit(swapi_response):
    """A plasma hotter than the threshold is refit with the dual Maxwellian."""
    temperature = 2 * DUAL_FIT_TEMPERATURE_THRESHOLD_K
    count_rates = _single_maxwellian_rates(
        swapi_response, 5.0, temperature, [0.0, -450.0, 0.0]
    )
    dual_fit = {"speed": 451.0, "density": 4.9, "temperature": 6.1e5}

    with mock.patch(
        "imap_processing.ialirt.l0.process_swapi.fit_dual_maxwellian_no_spin",
        return_value=dual_fit,
    ) as mock_dual_fit:
        result = fit_chunk(
            response=swapi_response,
            esa_energies=FIT_ENERGIES,
            count_rates=count_rates,
            epoch=FIT_EPOCH,
        )

    mock_dual_fit.assert_called_once()
    responses, rates = mock_dual_fit.call_args.args
    assert responses.central_speed.shape == (FIT_ENERGIES.size,)
    np.testing.assert_array_equal(rates, count_rates)
    assert result == {"success": True, **dual_fit}


@pytest.mark.parametrize(
    ("single_temperature", "expect_dual"),
    [
        (DUAL_FIT_TEMPERATURE_THRESHOLD_K, False),
        (np.nextafter(DUAL_FIT_TEMPERATURE_THRESHOLD_K, np.inf), True),
    ],
)
def test_fit_chunk_threshold(swapi_response, single_temperature, expect_dual):
    """The dual fit is used only above the temperature threshold."""
    single_fit = {"speed": 400.0, "density": 5.0, "temperature": single_temperature}
    dual_fit = {"speed": 410.0, "density": 6.0, "temperature": 7e5}
    with (
        mock.patch(
            "imap_processing.ialirt.l0.process_swapi.fit_single_maxwellian_with_spin",
            return_value=single_fit,
        ),
        mock.patch(
            "imap_processing.ialirt.l0.process_swapi.fit_dual_maxwellian_no_spin",
            return_value=dual_fit,
        ) as mock_dual_fit,
    ):
        result = fit_chunk(
            response=swapi_response,
            esa_energies=FIT_ENERGIES,
            count_rates=np.ones(FIT_ENERGIES.shape),
            epoch=FIT_EPOCH,
        )

    assert mock_dual_fit.called == expect_dual
    assert result == {"success": True, **(dual_fit if expect_dual else single_fit)}


def test_fit_chunk_invalid_energies(swapi_response):
    """Non-positive ESA energies make the fit fail without raising."""
    energies = FIT_ENERGIES.copy()
    energies[0, 0] = 0.0
    with mock.patch(
        "imap_processing.ialirt.l0.process_swapi.fit_single_maxwellian_with_spin"
    ) as mock_single_fit:
        result = fit_chunk(
            response=swapi_response,
            esa_energies=energies,
            count_rates=np.ones(energies.shape),
            epoch=FIT_EPOCH,
        )

    mock_single_fit.assert_not_called()
    assert result == {"success": False}


def test_fit_chunk_non_finite_fit(swapi_response):
    """A fit with non-finite moments is reported as failed."""
    with mock.patch(
        "imap_processing.ialirt.l0.process_swapi.fit_single_maxwellian_with_spin",
        return_value={"speed": 400.0, "density": np.nan, "temperature": 1e5},
    ):
        result = fit_chunk(
            response=swapi_response,
            esa_energies=FIT_ENERGIES,
            count_rates=np.ones(FIT_ENERGIES.shape),
            epoch=FIT_EPOCH,
        )

    assert result == {"success": False}
