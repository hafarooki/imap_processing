import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from imap_processing.swapi.constants import (
    PASSBAND_BOUNDARY_THRESHOLD,
)
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalTransmission,
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


def test_passband_fits(response_tables, response_table_paths):
    table = pd.read_csv(response_table_paths["passband_fit_coefficients_path"])
    assert set(response_tables.passband_fits.fits) == set(Aperture)
    for aperture, fit in response_tables.passband_fits.fits.items():
        assert fit.coefficients.dims == ("power", "energy_ratio", "elevation")
        assert list(fit.coefficients.power.values) == [2, 1, 0]
        min_voltage = fit.coefficients.min_esa_voltage
        max_voltage = fit.coefficients.max_esa_voltage
        assert min_voltage.dims == max_voltage.dims == ("energy_ratio", "elevation")
        assert min_voltage.attrs["units"] == max_voltage.attrs["units"] == "V"
        has_fit = fit.coefficients.notnull().all("power").values
        assert np.all(min_voltage.values[has_fit] > 0)
        assert np.all(min_voltage.values[has_fit] < max_voltage.values[has_fit])
        assert np.all(min_voltage.isnull().values == ~has_fit)

        # Placeholder: the file's voltage bounds, converted with the k factor
        rows = table[table["region"] == aperture.value]
        assert float(min_voltage.min()) == pytest.approx(rows["min_esa_voltage"].min())
        assert float(max_voltage.max()) == pytest.approx(rows["max_esa_voltage"].max())


def _plateau_table():
    azimuths = np.arange(0.0, 150.5, 0.5)
    transmission = np.full(azimuths.size, 0.5)
    transmission[azimuths <= 9.0] = 1e-3
    in_oa = (azimuths >= 31.0) & (azimuths <= 115.0)
    transmission[in_oa] = 1.0
    return azimuths, transmission


def _write_table(tmp_path, azimuths, transmission):
    path = tmp_path / "azimuthal-transmission.csv"
    pd.DataFrame({"abs_azimuth": azimuths, "transmission": transmission}).to_csv(
        path, index=False
    )
    return path


def test_transmission_accepts_matching_plateaus(tmp_path):
    azimuths, transmission = _plateau_table()
    table = AzimuthalTransmission.from_csv(
        _write_table(tmp_path, azimuths, transmission)
    )
    assert_array_equal(table.values, transmission)
    assert table.spacing == pytest.approx(0.5)


def test_transmission_treats_missing_values_as_zero(tmp_path):
    azimuths, transmission = _plateau_table()
    transmission[-1] = np.nan
    table = AzimuthalTransmission.from_csv(
        _write_table(tmp_path, azimuths, transmission)
    )
    assert table.values[-1] == 0.0


@pytest.mark.parametrize(
    ("azimuth", "plateau_name"),
    [
        (0.0, "sunglasses"),
        (9.0, "sunglasses"),
        (31.0, "open-aperture"),
        (115.0, "open-aperture"),
    ],
)
def test_transmission_rejects_a_wrong_plateau(tmp_path, azimuth, plateau_name):
    azimuths, transmission = _plateau_table()
    transmission[np.isclose(azimuths, azimuth)] = 0.5
    with pytest.raises(ValueError, match=f"{plateau_name} plateau"):
        AzimuthalTransmission.from_csv(_write_table(tmp_path, azimuths, transmission))


def test_transmission_spacing_comes_from_the_table(response_tables):
    assert response_tables.azimuthal_transmission.spacing == pytest.approx(0.1)


@pytest.mark.parametrize("azimuths", [[0.0, 0.1, 0.3], [0.5, 0.6, 0.7]])
def test_transmission_rejects_a_nonuniform_grid(tmp_path, azimuths):
    with pytest.raises(ValueError, match="uniform grid"):
        AzimuthalTransmission.from_csv(
            _write_table(tmp_path, azimuths, [1.0, 1.0, 1.0])
        )


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
    band = response_tables.passband_fits.passband(ESA_VOLTAGE, aperture)

    assert _passband_value(band, 0.0, 1.0) == pytest.approx(1.0)
    assert band.values.min() >= 0.0
    assert band.values.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("aperture", list(Aperture))
def test_passband_boundaries_bracket_the_response(response_tables, aperture):
    band = response_tables.passband_fits.passband(ESA_VOLTAGE, aperture)
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
    coefficients = response_tables.passband_fits.fits[Aperture.SUNGLASSES].coefficients
    min_voltage = float(coefficients.min_esa_voltage.min())
    max_voltage = float(coefficients.max_esa_voltage.max())

    def values(esa_voltage):
        return response_tables.passband_fits.passband(
            esa_voltage, Aperture.SUNGLASSES
        ).values

    np.testing.assert_allclose(values(10 * max_voltage), values(max_voltage))
    np.testing.assert_allclose(values(min_voltage / 10), values(min_voltage))


def test_passband_clamps_each_fit_to_its_own_range(response_table_paths, tmp_path):
    """Placeholder: Lowering one elevation's upper bound clamps only that row."""
    elevation, lowered_max = 4.0, 500.0
    table = pd.read_csv(response_table_paths["passband_fit_coefficients_path"])
    rows = (table["region"] == Aperture.SUNGLASSES.value) & (
        table["elevation"] == elevation
    )
    table.loc[rows, "max_esa_voltage"] = lowered_max
    lowered = tmp_path / "passband-fit-coefficients.csv"
    table.to_csv(lowered, index=False)

    original = SwapiResponseTables.from_files(**response_table_paths)
    modified = SwapiResponseTables.from_files(
        **{**response_table_paths, "passband_fit_coefficients_path": lowered}
    )
    band = modified.passband_fits.passband(ESA_VOLTAGE, Aperture.SUNGLASSES)
    elevations, _ = _grid(band)
    i_row = int(np.flatnonzero(np.isclose(elevations, elevation))[0])
    other_rows = np.arange(len(elevations)) != i_row

    np.testing.assert_allclose(
        band.values[other_rows],
        original.passband_fits.passband(ESA_VOLTAGE, Aperture.SUNGLASSES).values[
            other_rows
        ],
    )

    # Placeholder: the row is the fit at the lowered bound, up to the normalization,
    # which is set by the unchanged center row
    def shape(row):
        return row / row.max()

    np.testing.assert_allclose(
        shape(band.values[i_row]),
        shape(
            original.passband_fits.passband(lowered_max, Aperture.SUNGLASSES).values[
                i_row
            ]
        ),
    )
    assert not np.allclose(
        shape(band.values[i_row]),
        shape(
            original.passband_fits.passband(ESA_VOLTAGE, Aperture.SUNGLASSES).values[
                i_row
            ]
        ),
    )


def test_passband_powers_come_from_the_file(response_table_paths, tmp_path):
    """
    Placeholder: An extra zero cubic term is picked up and leaves the passband
    unchanged.
    """
    table = pd.read_csv(response_table_paths["passband_fit_coefficients_path"])
    table.insert(table.columns.get_loc("2"), "3", 0.0)
    cubic = tmp_path / "passband-fit-coefficients.csv"
    table.to_csv(cubic, index=False)

    quadratic_tables = SwapiResponseTables.from_files(**response_table_paths)
    cubic_tables = SwapiResponseTables.from_files(
        **{**response_table_paths, "passband_fit_coefficients_path": cubic}
    )

    fit = cubic_tables.passband_fits.fits[Aperture.SUNGLASSES]
    assert list(fit.coefficients.power.values) == [3, 2, 1, 0]
    np.testing.assert_allclose(
        cubic_tables.passband_fits.passband(ESA_VOLTAGE, Aperture.SUNGLASSES).values,
        quadratic_tables.passband_fits.passband(
            ESA_VOLTAGE, Aperture.SUNGLASSES
        ).values,
    )


def test_passband_differs_between_apertures(response_tables):
    assert not np.allclose(
        response_tables.passband_fits.passband(ESA_VOLTAGE, Aperture.SUNGLASSES).values,
        response_tables.passband_fits.passband(
            ESA_VOLTAGE, Aperture.OPEN_APERTURE
        ).values,
    )
