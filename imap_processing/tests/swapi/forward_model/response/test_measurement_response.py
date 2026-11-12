import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from imap_processing.swapi.constants import (
    PROTON_CHARGE_C,
    PROTON_MASS_KG,
    SIMION_K_FACTOR,
    Species,
)
from imap_processing.swapi.forward_model.response import passband as passband_module
from imap_processing.swapi.forward_model.response.measurement_species_response import (
    MeasurementSpeciesResponse,
    measurement_species_responses,
)
from imap_processing.swapi.forward_model.response.passband import Aperture
from imap_processing.swapi.forward_model.response.response_tables import (
    SwapiResponseTables,
)

# Placeholder: ESA voltage magnitude in V
ESA_VOLTAGE = 1_000.0


@pytest.fixture
def response_tables(swapi_response):
    return swapi_response


def _responses(
    response_tables, epoch, esa_voltages, species=Species.PROTON, rtn_to_swapi=None
):
    # Claude: one epoch or rotation is repeated for every sample
    n_samples = np.size(esa_voltages)
    epochs = np.broadcast_to(epoch, (n_samples,))
    if rtn_to_swapi is None:
        rtn_to_swapi = np.eye(3)
    rotations = np.broadcast_to(rtn_to_swapi, (n_samples, 3, 3))
    return measurement_species_responses(
        response_tables, esa_voltages, epochs, species, rotations
    )


def _expected_central_speed(esa_voltage, species):
    esa_energy = SIMION_K_FACTOR * esa_voltage
    proton_speed = np.sqrt(2 * esa_energy * PROTON_CHARGE_C / PROTON_MASS_KG) / 1e3
    return proton_speed / np.sqrt(species.mass_per_charge)


def _assert_passbands_equal(actual, expected):
    for field in ("min_elevation", "elevation_spacing"):
        assert getattr(actual, field) == getattr(expected, field)
    for field in ("min_speed_ratio", "speed_ratio_spacing"):
        assert getattr(actual, field) == getattr(expected, field)
    for field in ("values", "min_boundary", "max_boundary", "elevation_range"):
        assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_central_speed(epoch, response_tables):
    esa_energy = SIMION_K_FACTOR * ESA_VOLTAGE
    expected_km_per_s = np.sqrt(2 * esa_energy * PROTON_CHARGE_C / PROTON_MASS_KG) / 1e3

    [proton] = _responses(response_tables, epoch, [ESA_VOLTAGE], Species.PROTON)
    [alpha] = _responses(response_tables, epoch, [ESA_VOLTAGE], Species.ALPHA)

    assert proton.central_speed == pytest.approx(expected_km_per_s)
    assert alpha.central_speed == pytest.approx(
        expected_km_per_s / np.sqrt(Species.ALPHA.mass_per_charge)
    )


@pytest.mark.parametrize("species", [Species.PROTON, Species.ALPHA])
def test_responses_per_sample_scalars(epoch, response_tables, species):
    esa_voltages = [ESA_VOLTAGE, 2 * ESA_VOLTAGE]
    responses = _responses(response_tables, epoch, esa_voltages, species)

    assert [r.mass_kg for r in responses] == [species.mass_kg] * len(esa_voltages)
    assert_allclose(
        [r.central_speed for r in responses],
        [_expected_central_speed(esa_voltage, species) for esa_voltage in esa_voltages],
    )
    assert_allclose(
        [r.central_effective_area for r in responses],
        response_tables.central_effective_area_at(esa_voltages, epoch, species),
    )


def test_responses_hold_the_passbands_of_each_sample(epoch, response_tables):
    esa_voltages = [2 * ESA_VOLTAGE, ESA_VOLTAGE, 2 * ESA_VOLTAGE]
    responses = _responses(response_tables, epoch, esa_voltages)

    assert len(responses) == len(esa_voltages)
    for response, esa_voltage in zip(responses, esa_voltages, strict=True):
        assert isinstance(response, MeasurementSpeciesResponse)
        _assert_passbands_equal(
            response.sg_passband,
            response_tables.passband_fits.passband(esa_voltage, Aperture.SUNGLASSES),
        )
        _assert_passbands_equal(
            response.oa_passband,
            response_tables.passband_fits.passband(esa_voltage, Aperture.OPEN_APERTURE),
        )
    # Placeholder: samples at the same voltage share their passbands
    assert np.shares_memory(
        responses[0].sg_passband.values, responses[2].sg_passband.values
    )
    assert not np.shares_memory(
        responses[0].sg_passband.values, responses[1].sg_passband.values
    )


def test_responses_flatten_the_voltages(epoch, response_tables):
    esa_voltages = np.array(
        [[ESA_VOLTAGE, 2 * ESA_VOLTAGE], [3 * ESA_VOLTAGE, ESA_VOLTAGE]]
    )
    responses = _responses(response_tables, epoch, esa_voltages)

    assert_allclose(
        [r.central_speed for r in responses],
        [_expected_central_speed(e, Species.PROTON) for e in esa_voltages.ravel()],
    )


@pytest.mark.parametrize("esa_voltage", [0.0, -ESA_VOLTAGE, np.nan, np.inf])
def test_responses_reject_nonpositive_or_nonfinite_voltages(
    epoch, response_tables, esa_voltage
):
    with pytest.raises(ValueError, match="positive and finite"):
        _responses(response_tables, epoch, [ESA_VOLTAGE, esa_voltage])


def test_responses_reject_no_voltages(epoch, response_tables):
    with pytest.raises(ValueError, match="No ESA voltages"):
        _responses(response_tables, epoch, [])


def _sg_values(responses):
    return responses[0].sg_passband.values


def test_passbands_are_cached(epoch, response_table_paths):
    response_tables = SwapiResponseTables.from_files(**response_table_paths)
    voltages = np.array(
        [[ESA_VOLTAGE, 2 * ESA_VOLTAGE], [ESA_VOLTAGE, 2 * ESA_VOLTAGE]]
    )
    first = _sg_values(_responses(response_tables, epoch, voltages))

    # Claude: the same voltages reuse the passbands, whatever the species, epoch
    # or rotations
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for responses in (
        _responses(response_tables, epoch + 1, voltages.copy()),
        _responses(response_tables, epoch, voltages, Species.ALPHA),
        _responses(response_tables, epoch, voltages, rtn_to_swapi=rotation),
    ):
        assert _sg_values(responses) is first
    # Placeholder: other voltages are a new entry
    assert _sg_values(_responses(response_tables, epoch, voltages[:1])) is not first


def test_passbands_cache_is_bounded(epoch, response_table_paths, monkeypatch):
    monkeypatch.setattr(passband_module, "PASSBAND_CACHE_SIZE", 2)
    response_tables = SwapiResponseTables.from_files(**response_table_paths)

    def sg_values(esa_voltage):
        return _sg_values(_responses(response_tables, epoch, [esa_voltage]))

    first = sg_values(ESA_VOLTAGE)
    sg_values(2 * ESA_VOLTAGE)
    # Placeholder: touching the first entry keeps it over the second
    assert sg_values(ESA_VOLTAGE) is first
    sg_values(3 * ESA_VOLTAGE)

    assert sg_values(ESA_VOLTAGE) is first
    assert len(response_tables.passband_fits._cache) == 2


def test_responses_take_a_rotation_per_sample(epoch, response_tables):
    esa_voltages = np.array(
        [[ESA_VOLTAGE, 2 * ESA_VOLTAGE], [ESA_VOLTAGE, ESA_VOLTAGE]]
    )
    rotations = np.arange(4 * 9, dtype=float).reshape(4, 3, 3)
    responses = _responses(response_tables, epoch, esa_voltages, rtn_to_swapi=rotations)

    for response, rotation in zip(responses, rotations.copy(), strict=True):
        assert_array_equal(response.rtn_to_swapi, rotation)
    # Claude: copied, so changing the caller's matrices leaves the responses alone
    rotations[0, 0, 0] = 5.0
    assert responses[0].rtn_to_swapi[0, 0] == 0.0


@pytest.mark.parametrize("shape", [(3, 3), (3,), (2, 3, 3), (3, 3, 3), (3, 2)])
def test_responses_reject_mis_shaped_rotations(epoch, response_tables, shape):
    with pytest.raises(ValueError, match="rotations must have shape"):
        measurement_species_responses(
            response_tables,
            [ESA_VOLTAGE, ESA_VOLTAGE, ESA_VOLTAGE, ESA_VOLTAGE],
            np.full(4, epoch),
            Species.PROTON,
            np.zeros(shape),
        )


def test_responses_share_the_azimuthal_transmission(epoch, response_tables):
    responses = _responses(response_tables, epoch, [ESA_VOLTAGE, 2 * ESA_VOLTAGE])

    for response in responses:
        assert np.shares_memory(
            response.azimuthal_transmission.values,
            response_tables.azimuthal_transmission.values,
        )


def test_responses_identify_their_measurement(epoch, response_tables):
    esa_voltages = np.array([[ESA_VOLTAGE, 2 * ESA_VOLTAGE], [3 * ESA_VOLTAGE, 1.5]])
    responses = _responses(response_tables, epoch, esa_voltages)

    assert [r.esa_voltage for r in responses] == list(esa_voltages.ravel())
    assert [r.epoch for r in responses] == [epoch] * esa_voltages.size
    assert all(isinstance(r.epoch, int) for r in responses)


def test_responses_take_an_epoch_per_sample(response_tables):
    # Claude: epochs between different pairs of efficiency table rows, so the
    # samples get different efficiencies
    row_epochs = response_tables.relative_efficiency.efficiency_epoch.values
    epochs = np.array([row_epochs[1] + 1, row_epochs[1] + 2, row_epochs[2] + 1])
    responses = _responses(
        response_tables, epochs, [ESA_VOLTAGE, ESA_VOLTAGE, ESA_VOLTAGE]
    )

    assert [r.epoch for r in responses] == list(epochs)
    expected = response_tables.central_effective_area_at(
        ESA_VOLTAGE, epochs, Species.PROTON
    )
    assert_allclose([r.central_effective_area for r in responses], expected)
    assert expected[0] != expected[2]


@pytest.mark.parametrize("shape", [(), (1,), (3,), (2, 2)])
def test_responses_reject_mis_shaped_epochs(epoch, response_tables, shape):
    with pytest.raises(ValueError, match="Epochs must have shape"):
        measurement_species_responses(
            response_tables,
            [ESA_VOLTAGE, ESA_VOLTAGE, ESA_VOLTAGE, ESA_VOLTAGE],
            np.full(shape, epoch),
            Species.PROTON,
            np.broadcast_to(np.eye(3), (4, 3, 3)),
        )


def test_responses_require_the_jit(epoch, response_tables, monkeypatch):
    monkeypatch.setattr(numba.config, "DISABLE_JIT", True)
    with pytest.raises(RuntimeError, match="NUMBA_DISABLE_JIT"):
        _responses(response_tables, epoch, [ESA_VOLTAGE])
