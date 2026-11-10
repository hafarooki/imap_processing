import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from imap_processing.spice.time import et_to_ttj2000ns, str_to_et
from imap_processing.swapi.constants import (
    OA_PLATEAU_AZIMUTH_MAX_DEG,
    OA_PLATEAU_AZIMUTH_MIN_DEG,
    OA_PLATEAU_TRANSMISSION,
    PASSBAND_BOUNDARY_THRESHOLD,
    PROTON_CHARGE_C,
    PROTON_MASS_KG,
    SG_PLATEAU_AZIMUTH_MAX_DEG,
    SG_PLATEAU_TRANSMISSION,
    SIMION_K_FACTOR,
    Species,
)
from imap_processing.swapi.forward_model.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.instrument_response import (
    StepResponse,
    central_speed,
    step_response_at,
)
from imap_processing.swapi.forward_model.passband import Aperture, passband_at
from imap_processing.swapi.forward_model.response_tables import SwapiResponse

# ESA energy per charge at 1 kV, in eV/q
ESA_ENERGY = SIMION_K_FACTOR * 1_000.0


def _ttj2000ns(utc):
    return int(et_to_ttj2000ns(str_to_et(utc)))


def _responses(response_tables, epoch, esa_energies, species=Species.PROTON):
    passbands = response_tables.passbands(esa_energies)
    return response_tables.step_responses(passbands, esa_energies, epoch, species)


def _assert_passbands_equal(actual, expected):
    for field in ("min_elevation", "elevation_spacing"):
        assert getattr(actual, field) == getattr(expected, field)
    for field in ("min_speed_ratio", "speed_ratio_spacing"):
        assert getattr(actual, field) == getattr(expected, field)
    for field in ("values", "min_boundary", "max_boundary", "elevation_range"):
        assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_passband_fits(response_tables, response_table_paths):
    table = pd.read_csv(response_table_paths["passband_fit_coefficients_path"])
    assert set(response_tables.passband_fits) == set(Aperture)
    for aperture, fit in response_tables.passband_fits.items():
        assert fit.coefficients.dims == ("power", "energy_ratio", "elevation")
        assert list(fit.coefficients.power.values) == [2, 1, 0]
        min_energy = fit.coefficients.min_esa_energy
        max_energy = fit.coefficients.max_esa_energy
        assert min_energy.dims == max_energy.dims == ("energy_ratio", "elevation")
        assert min_energy.attrs["units"] == max_energy.attrs["units"] == "eV/q"
        has_fit = fit.coefficients.notnull().all("power").values
        assert np.all(min_energy.values[has_fit] > 0)
        assert np.all(min_energy.values[has_fit] < max_energy.values[has_fit])
        assert np.all(min_energy.isnull().values == ~has_fit)

        # the file's voltage bounds, converted with the k factor
        rows = table[table["region"] == aperture.value]
        assert float(min_energy.min()) == pytest.approx(
            SIMION_K_FACTOR * rows["min_esa_voltage"].min()
        )
        assert float(max_energy.max()) == pytest.approx(
            SIMION_K_FACTOR * rows["max_esa_voltage"].max()
        )


def _plateau_table():
    azimuths = np.arange(0.0, 150.5, 0.5)
    transmission = np.full(azimuths.size, 0.5)
    transmission[azimuths <= SG_PLATEAU_AZIMUTH_MAX_DEG] = SG_PLATEAU_TRANSMISSION
    in_oa = (azimuths >= OA_PLATEAU_AZIMUTH_MIN_DEG) & (
        azimuths <= OA_PLATEAU_AZIMUTH_MAX_DEG
    )
    transmission[in_oa] = OA_PLATEAU_TRANSMISSION
    return azimuths, transmission


def test_transmission_accepts_matching_plateaus():
    azimuths, transmission = _plateau_table()
    table = AzimuthalTransmission.from_table(azimuths, transmission)
    assert_array_equal(table.values, transmission)
    assert table.spacing == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("azimuth", "plateau_name"),
    [
        (0.0, "sunglasses"),
        (SG_PLATEAU_AZIMUTH_MAX_DEG, "sunglasses"),
        (OA_PLATEAU_AZIMUTH_MIN_DEG, "open-aperture"),
        (OA_PLATEAU_AZIMUTH_MAX_DEG, "open-aperture"),
    ],
)
def test_transmission_rejects_a_wrong_plateau(azimuth, plateau_name):
    azimuths, transmission = _plateau_table()
    transmission[np.isclose(azimuths, azimuth)] = 0.5
    with pytest.raises(ValueError, match=f"{plateau_name} plateau"):
        AzimuthalTransmission.from_table(azimuths, transmission)


def test_transmission_rejects_a_size_mismatch():
    azimuths, transmission = _plateau_table()
    with pytest.raises(ValueError, match="one value per azimuth"):
        AzimuthalTransmission.from_table(azimuths, transmission[:-1])


def test_transmission_spacing_comes_from_the_table(response_tables):
    assert response_tables.azimuthal_transmission.spacing == pytest.approx(0.1)


def test_transmission_rejects_a_nonuniform_grid():
    with pytest.raises(ValueError, match="uniform grid"):
        AzimuthalTransmission.from_table([0.0, 0.1, 0.3], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="uniform grid"):
        AzimuthalTransmission.from_table([0.5, 0.6, 0.7], [1.0, 1.0, 1.0])


def _passband_value(band, elevation, speed_ratio):
    i = round((elevation - band.min_elevation) / band.elevation_spacing)
    j = round((speed_ratio - band.min_speed_ratio) / band.speed_ratio_spacing)
    return band.values[i, j]


def _grid(band):
    n_elevations, n_speed_ratios = band.values.shape
    elevations = band.min_elevation + band.elevation_spacing * np.arange(n_elevations)
    speed_ratios = band.min_speed_ratio + band.speed_ratio_spacing * np.arange(
        n_speed_ratios
    )
    return elevations, speed_ratios


@pytest.mark.parametrize("aperture", list(Aperture))
def test_passband_is_normalized_at_center(response_tables, aperture):
    band = response_tables.passband(ESA_ENERGY, aperture)

    assert _passband_value(band, 0.0, 1.0) == pytest.approx(1.0)
    assert band.values.min() >= 0.0
    assert band.values.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("aperture", list(Aperture))
def test_passband_boundaries_bracket_the_response(response_tables, aperture):
    band = response_tables.passband(ESA_ENERGY, aperture)
    elevations, speed_ratios = _grid(band)
    cutoff = PASSBAND_BOUNDARY_THRESHOLD * band.values.max()

    i_center = int(np.flatnonzero(np.isclose(elevations, 0.0))[0])
    assert band.min_boundary[i_center] < 1.0 < band.max_boundary[i_center]
    elevation_lower, elevation_upper = band.elevation_range
    assert elevation_lower < 0.0 < elevation_upper

    outside_speed = (speed_ratios[None, :] < band.min_boundary[:, None]) | (
        speed_ratios[None, :] > band.max_boundary[:, None]
    )
    assert np.all(band.values[outside_speed] < cutoff)

    outside_elevation = (elevations < elevation_lower) | (elevations > elevation_upper)
    assert np.all(band.values[outside_elevation] < cutoff)


def test_passband_clamps_energy_to_the_fit_range(response_tables):
    coefficients = response_tables.passband_fits[Aperture.SUNGLASSES].coefficients
    min_energy = float(coefficients.min_esa_energy.min())
    max_energy = float(coefficients.max_esa_energy.max())

    def values(esa_energy):
        return response_tables.passband(esa_energy, Aperture.SUNGLASSES).values

    np.testing.assert_allclose(values(10 * max_energy), values(max_energy))
    np.testing.assert_allclose(values(min_energy / 10), values(min_energy))


def test_passband_clamps_each_fit_to_its_own_range(response_table_paths, tmp_path):
    """Lowering one elevation's upper bound clamps only that row."""
    elevation, lowered_max_voltage = 4.0, 500.0
    lowered_max = SIMION_K_FACTOR * lowered_max_voltage
    table = pd.read_csv(response_table_paths["passband_fit_coefficients_path"])
    rows = (table["region"] == Aperture.SUNGLASSES.value) & (
        table["elevation"] == elevation
    )
    table.loc[rows, "max_esa_voltage"] = lowered_max_voltage
    lowered = tmp_path / "passband-fit-coefficients.csv"
    table.to_csv(lowered, index=False)

    original = SwapiResponse.from_files(**response_table_paths)
    modified = SwapiResponse.from_files(
        **{**response_table_paths, "passband_fit_coefficients_path": lowered}
    )
    band = modified.passband(ESA_ENERGY, Aperture.SUNGLASSES)
    elevations, _ = _grid(band)
    i_row = int(np.flatnonzero(np.isclose(elevations, elevation))[0])
    other_rows = np.arange(len(elevations)) != i_row

    np.testing.assert_allclose(
        band.values[other_rows],
        original.passband(ESA_ENERGY, Aperture.SUNGLASSES).values[other_rows],
    )

    # the row is the fit at the lowered bound, up to the normalization, which
    # is set by the unchanged center row
    def shape(row):
        return row / row.max()

    np.testing.assert_allclose(
        shape(band.values[i_row]),
        shape(original.passband(lowered_max, Aperture.SUNGLASSES).values[i_row]),
    )
    assert not np.allclose(
        shape(band.values[i_row]),
        shape(original.passband(ESA_ENERGY, Aperture.SUNGLASSES).values[i_row]),
    )


def test_passband_powers_come_from_the_file(response_table_paths, tmp_path):
    """An extra zero cubic term is picked up and leaves the passband unchanged."""
    table = pd.read_csv(response_table_paths["passband_fit_coefficients_path"])
    table.insert(table.columns.get_loc("2"), "3", 0.0)
    cubic = tmp_path / "passband-fit-coefficients.csv"
    table.to_csv(cubic, index=False)

    quadratic_tables = SwapiResponse.from_files(**response_table_paths)
    cubic_tables = SwapiResponse.from_files(
        **{**response_table_paths, "passband_fit_coefficients_path": cubic}
    )

    fit = cubic_tables.passband_fits[Aperture.SUNGLASSES]
    assert list(fit.coefficients.power.values) == [3, 2, 1, 0]
    np.testing.assert_allclose(
        cubic_tables.passband(ESA_ENERGY, Aperture.SUNGLASSES).values,
        quadratic_tables.passband(ESA_ENERGY, Aperture.SUNGLASSES).values,
    )


def test_passbands_stack_each_unique_energy(response_tables):
    passbands = response_tables.passbands([2 * ESA_ENERGY, ESA_ENERGY, 2 * ESA_ENERGY])

    assert_array_equal(passbands.esa_energy, [ESA_ENERGY, 2 * ESA_ENERGY])
    for aperture, stacked in [
        (Aperture.SUNGLASSES, passbands.sunglasses),
        (Aperture.OPEN_APERTURE, passbands.open_aperture),
    ]:
        assert stacked.values.shape[0] == 2
        for k, esa_energy in enumerate(passbands.esa_energy):
            _assert_passbands_equal(
                passband_at(stacked, k), response_tables.passband(esa_energy, aperture)
            )


@pytest.mark.parametrize("esa_energy", [0.0, -ESA_ENERGY, np.nan, np.inf])
def test_passbands_reject_nonpositive_or_nonfinite_energies(
    response_tables, esa_energy
):
    with pytest.raises(ValueError, match="positive and finite"):
        response_tables.passbands([ESA_ENERGY, esa_energy])


def test_passband_differs_between_apertures(response_tables):
    passbands = response_tables.passbands([ESA_ENERGY])

    assert not np.allclose(passbands.sunglasses.values, passbands.open_aperture.values)


def test_central_speed():
    expected_km_per_s = np.sqrt(2 * ESA_ENERGY * PROTON_CHARGE_C / PROTON_MASS_KG) / 1e3

    assert central_speed(ESA_ENERGY, Species.PROTON) == pytest.approx(expected_km_per_s)
    assert central_speed(ESA_ENERGY, Species.ALPHA) == pytest.approx(
        expected_km_per_s / np.sqrt(Species.ALPHA.mass_per_charge)
    )


def test_central_effective_area(response_tables):
    table = response_tables.central_effective_area
    assert table.esa_energy.attrs["units"] == "eV/q"
    esa_energy = float(table.esa_energy[0] + table.esa_energy[1]) / 2
    expected = float(table[0] + table[1]) / 2

    assert response_tables.central_effective_area_at(esa_energy) == pytest.approx(
        expected
    )
    # held at the end values outside the table
    assert response_tables.central_effective_area_at(
        float(table.esa_energy[0]) / 2
    ) == pytest.approx(float(table[0]))
    assert response_tables.central_effective_area_at(
        float(table.esa_energy[-1]) * 2
    ) == pytest.approx(float(table[-1]))


def test_central_effective_area_table_is_in_energy(
    response_tables, response_table_paths
):
    table = pd.read_csv(response_table_paths["central_effective_area_path"])

    assert_allclose(
        response_tables.central_effective_area.esa_energy.values,
        SIMION_K_FACTOR * table["esa_voltage"].to_numpy(),
    )


@pytest.mark.parametrize(
    ("utc", "species", "efficiency"),
    [
        ("2020-01-01T00:00:00", Species.PROTON, 0.09),
        ("2020-01-01T00:00:00", Species.ALPHA, 0.95),
        ("2020-01-01T00:00:00", Species.HELIUM_PLUS, 0.95),
        ("2025-01-01T00:00:00", Species.PROTON, 0.0882),
        ("2026-01-01T00:00:00", Species.ALPHA, 1.05),
    ],
)
def test_relative_efficiency_is_the_latest_row_before_epoch(
    response_tables, utc, species, efficiency
):
    assert response_tables.relative_efficiency_at(
        _ttj2000ns(utc), species
    ) == pytest.approx(efficiency)


@pytest.mark.parametrize("species", [Species.PROTON, Species.ALPHA])
def test_responses_per_sample_scalars(epoch, response_tables, species):
    esa_energies = [ESA_ENERGY, 2 * ESA_ENERGY]
    responses = _responses(response_tables, epoch, esa_energies, species)
    efficiency = response_tables.relative_efficiency_at(epoch, species)

    assert responses.mass_kg == species.mass_kg
    assert_allclose(
        responses.central_speed,
        [central_speed(esa_energy, species) for esa_energy in esa_energies],
    )
    assert_allclose(
        responses.central_effective_area,
        [
            response_tables.central_effective_area_at(esa_energy) * efficiency
            for esa_energy in esa_energies
        ],
    )


def test_responses_index_the_passband_of_each_sample(epoch, response_tables):
    esa_energies = [2 * ESA_ENERGY, ESA_ENERGY, 2 * ESA_ENERGY]
    passbands = response_tables.passbands(esa_energies)
    responses = response_tables.step_responses(
        passbands, esa_energies, epoch, Species.PROTON
    )

    assert_array_equal(responses.passband_index, [1, 0, 1])
    assert responses.sg_passbands is passbands.sunglasses
    assert responses.oa_passbands is passbands.open_aperture
    for i, esa_energy in enumerate(esa_energies):
        response = step_response_at(responses, i)
        assert isinstance(response, StepResponse)
        assert response.mass_kg == Species.PROTON.mass_kg
        assert response.central_speed == responses.central_speed[i]
        assert response.central_effective_area == responses.central_effective_area[i]
        _assert_passbands_equal(
            response.sg_passband,
            response_tables.passband(esa_energy, Aperture.SUNGLASSES),
        )
        _assert_passbands_equal(
            response.oa_passband,
            response_tables.passband(esa_energy, Aperture.OPEN_APERTURE),
        )


def test_responses_flatten_the_energies(epoch, response_tables):
    esa_energies = np.array(
        [[ESA_ENERGY, 2 * ESA_ENERGY], [3 * ESA_ENERGY, ESA_ENERGY]]
    )
    responses = _responses(response_tables, epoch, esa_energies)

    assert_allclose(
        responses.central_speed,
        [central_speed(e, Species.PROTON) for e in esa_energies.ravel()],
    )


def test_responses_reject_an_energy_without_a_passband(epoch, response_tables):
    passbands = response_tables.passbands([ESA_ENERGY])

    with pytest.raises(ValueError, match="passband"):
        response_tables.step_responses(
            passbands, [ESA_ENERGY, 2 * ESA_ENERGY], epoch, Species.PROTON
        )


def test_responses_subset(epoch, response_tables):
    responses = _responses(
        response_tables, epoch, [ESA_ENERGY, 2 * ESA_ENERGY, 3 * ESA_ENERGY]
    )
    subset = responses.subset([2, 0])

    assert_array_equal(subset.central_speed, responses.central_speed[[2, 0]])
    assert_array_equal(
        subset.central_effective_area, responses.central_effective_area[[2, 0]]
    )
    assert_array_equal(subset.passband_index, responses.passband_index[[2, 0]])
    assert subset.sg_passbands is responses.sg_passbands
    assert subset.oa_passbands is responses.oa_passbands


def test_no_efficiency_before_the_first_row(response_tables):
    with pytest.raises(ValueError, match="efficiency"):
        _responses(response_tables, _ttj2000ns("1999-12-31T00:00:00"), [ESA_ENERGY])


def test_responses_share_the_azimuthal_transmission(epoch, response_tables):
    responses = _responses(response_tables, epoch, [ESA_ENERGY, 2 * ESA_ENERGY])

    assert responses.azimuthal_transmission is response_tables.azimuthal_transmission
