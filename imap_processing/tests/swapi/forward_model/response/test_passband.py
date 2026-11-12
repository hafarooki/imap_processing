"""Placeholder: Ported from L3a's ``tests/swapi/response/test_passband_grid.py``, run on
the passband fitted at 1 kV instead of a synthetic Gaussian.
"""

import numpy as np
import pandas as pd
import pytest

from imap_processing.swapi.constants import (
    PASSBAND_BOUNDARY_THRESHOLD,
    PASSBAND_SPEED_RATIOS,
)
from imap_processing.swapi.forward_model.response.passband import (
    Aperture,
    PassbandFits,
    interpolate_passband,
    speed_ratio_range_at_elevation,
)

# Placeholder: ESA voltage magnitude in V
ESA_VOLTAGE = 1_000.0


@pytest.fixture(params=list(Aperture))
def passband(request, passband_fits):
    return passband_fits.passband(ESA_VOLTAGE, request.param)


def row_index(passband, elevation):
    return round((elevation - passband.min_elevation) / passband.elevation_spacing)


@pytest.mark.parametrize(
    ("elevation", "speed_ratio"),
    [
        # Placeholder: on a node
        (0.0, 1.0),
        # Placeholder: at a cell center
        (1.25, 1.005),
        # Placeholder: off center
        (1.1, 1.0014),
    ],
)
def test_interpolation_is_the_corner_weighted_sum(passband, elevation, speed_ratio):
    i_float = (elevation - passband.min_elevation) / passband.elevation_spacing
    j_float = (speed_ratio - passband.min_speed_ratio) / passband.speed_ratio_spacing
    i_lo, j_lo = int(i_float), int(j_float)
    i_weight, j_weight = i_float - i_lo, j_float - j_lo
    corners = passband.values[i_lo : i_lo + 2, j_lo : j_lo + 2]
    weights = np.array(
        [
            [(1 - i_weight) * (1 - j_weight), (1 - i_weight) * j_weight],
            [i_weight * (1 - j_weight), i_weight * j_weight],
        ]
    )

    np.testing.assert_allclose(
        interpolate_passband(passband, elevation, speed_ratio),
        (corners * weights).sum(),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("elevation", "speed_ratio"),
    [(-100.0, 1.0), (100.0, 1.0), (0.0, 0.5), (0.0, 1.5)],
)
def test_interpolation_is_zero_outside_the_grid(passband, elevation, speed_ratio):
    assert interpolate_passband(passband, elevation, speed_ratio) == 0.0


def test_speed_ratio_range_marks_cutoff_crossing_on_grid(passband):
    cutoff = PASSBAND_BOUNDARY_THRESHOLD * passband.values.max()
    rows = np.flatnonzero(~np.isnan(passband.min_boundary))
    for i in rows:
        elevation = passband.min_elevation + passband.elevation_spacing * i
        # Placeholder: the bounds of an on-grid row and its upper neighbor are combined,
        # so only the row's own crossing is bracketed
        lo, hi = speed_ratio_range_at_elevation(passband, float(elevation))
        row = passband.values[i]
        assert np.interp(lo, PASSBAND_SPEED_RATIOS, row) <= cutoff + 1e-12
        assert np.interp(hi, PASSBAND_SPEED_RATIOS, row) <= cutoff + 1e-12
        assert lo <= passband.min_boundary[i]
        assert hi >= passband.max_boundary[i]


def test_speed_ratio_range_is_the_union_of_bracketing_rows(passband):
    lower, upper = row_index(passband, 0.0), row_index(passband, 0.5)
    lo, hi = speed_ratio_range_at_elevation(passband, 0.25)

    assert lo == pytest.approx(
        min(passband.min_boundary[lower], passband.min_boundary[upper])
    )
    assert hi == pytest.approx(
        max(passband.max_boundary[lower], passband.max_boundary[upper])
    )


def test_speed_ratio_range_min_is_no_greater_than_max(passband):
    active_lo, active_hi = passband.elevation_range
    for elevation in np.linspace(active_lo, active_hi, 7):
        lo, hi = speed_ratio_range_at_elevation(passband, float(elevation))
        assert lo <= hi


def test_speed_ratio_range_at_the_edges_of_the_active_rows(passband):
    """Placeholder: Past the active rows both bracketing rows are NaN, which raises; at
    the edge only one is, and its bounds are returned."""
    valid = np.flatnonzero(~np.isnan(passband.min_boundary))
    first, last = int(valid[0]), int(valid[-1])
    n_rows = passband.min_boundary.shape[0]
    spacing, min_elevation = passband.elevation_spacing, passband.min_elevation

    # Placeholder: the fitted passbands fall below the cutoff before the grid's edges
    assert first > 1
    assert last < n_rows - 1

    with pytest.raises(ValueError, match="Elevation out of range"):
        speed_ratio_range_at_elevation(passband, min_elevation - 100.0)
    with pytest.raises(ValueError, match="Elevation out of range"):
        speed_ratio_range_at_elevation(
            passband, min_elevation + spacing * n_rows + 100.0
        )
    for row, elevation_offset in [(first, -0.5), (last, 0.5)]:
        lo, hi = speed_ratio_range_at_elevation(
            passband, min_elevation + spacing * (row + elevation_offset)
        )
        assert lo == pytest.approx(passband.min_boundary[row])
        assert hi == pytest.approx(passband.max_boundary[row])


def test_passband_fits(passband_fits, response_table_paths):
    table = pd.read_csv(response_table_paths["passband_fit_coefficients_path"])
    assert set(passband_fits.fits) == set(Aperture)
    for aperture, fit in passband_fits.fits.items():
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
def test_passband_is_normalized_at_center(passband_fits, aperture):
    band = passband_fits.passband(ESA_VOLTAGE, aperture)

    assert _passband_value(band, 0.0, 1.0) == pytest.approx(1.0)
    assert band.values.min() >= 0.0
    assert band.values.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("aperture", list(Aperture))
def test_passband_boundaries_bracket_the_response(passband_fits, aperture):
    band = passband_fits.passband(ESA_VOLTAGE, aperture)
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


def test_passband_clamps_energy_to_the_fit_range(passband_fits):
    coefficients = passband_fits.fits[Aperture.SUNGLASSES].coefficients
    min_voltage = float(coefficients.min_esa_voltage.min())
    max_voltage = float(coefficients.max_esa_voltage.max())

    def values(esa_voltage):
        return passband_fits.passband(esa_voltage, Aperture.SUNGLASSES).values

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

    original = PassbandFits.from_csv(
        response_table_paths["passband_fit_coefficients_path"]
    )
    modified = PassbandFits.from_csv(lowered)
    band = modified.passband(ESA_VOLTAGE, Aperture.SUNGLASSES)
    elevations, _ = _grid(band)
    i_row = int(np.flatnonzero(np.isclose(elevations, elevation))[0])
    other_rows = np.arange(len(elevations)) != i_row

    np.testing.assert_allclose(
        band.values[other_rows],
        original.passband(ESA_VOLTAGE, Aperture.SUNGLASSES).values[other_rows],
    )

    # Placeholder: the row is the fit at the lowered bound, up to the normalization,
    # which is set by the unchanged center row
    def shape(row):
        return row / row.max()

    np.testing.assert_allclose(
        shape(band.values[i_row]),
        shape(original.passband(lowered_max, Aperture.SUNGLASSES).values[i_row]),
    )
    assert not np.allclose(
        shape(band.values[i_row]),
        shape(original.passband(ESA_VOLTAGE, Aperture.SUNGLASSES).values[i_row]),
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

    quadratic = PassbandFits.from_csv(
        response_table_paths["passband_fit_coefficients_path"]
    )
    cubic_fits = PassbandFits.from_csv(cubic)

    fit = cubic_fits.fits[Aperture.SUNGLASSES]
    assert list(fit.coefficients.power.values) == [3, 2, 1, 0]
    np.testing.assert_allclose(
        cubic_fits.passband(ESA_VOLTAGE, Aperture.SUNGLASSES).values,
        quadratic.passband(ESA_VOLTAGE, Aperture.SUNGLASSES).values,
    )


def test_passband_differs_between_apertures(passband_fits):
    assert not np.allclose(
        passband_fits.passband(ESA_VOLTAGE, Aperture.SUNGLASSES).values,
        passband_fits.passband(ESA_VOLTAGE, Aperture.OPEN_APERTURE).values,
    )
