"""Placeholder: Ported from L3a's ``tests/swapi/response/test_passband_grid.py``, run on
the passband fitted at 1 kV instead of a synthetic Gaussian.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.swapi.forward_model.response.passband import (
    PassbandGrid,
    interpolate_passband,
    speed_ratio_range_at_elevation,
)
from imap_processing.swapi.forward_model.response.passband_fit import (
    APERTURES,
    PassbandFitTable,
)

# Placeholder: ESA voltage magnitude in V
ESA_VOLTAGE = 1_000.0

# Claude: the default grid of PassbandGrid.from_values; synthetic passbands given on
# Claude: it are carried over unchanged by the interpolation
PASSBAND_ELEVATIONS_DEG = np.linspace(-15.0, 15.0, 61)
PASSBAND_SPEED_RATIOS = np.linspace(0.9, 1.1, 101)
# Claude: the default contour_threshold of PassbandGrid.from_values
PASSBAND_BOUNDARY_THRESHOLD = 1e-2


def as_passband(
    values, elevations=PASSBAND_ELEVATIONS_DEG, speed_ratios=PASSBAND_SPEED_RATIOS
):
    return xr.DataArray(
        values,
        dims=("elevation", "speed_ratio"),
        coords={"elevation": elevations, "speed_ratio": speed_ratios},
    )


def from_grid_values(values, **kwargs):
    return PassbandGrid.from_values(as_passband(values), **kwargs)


@pytest.fixture(params=list(APERTURES))
def passband(request, passband_fit_table):
    return passband_fit_table.evaluate_passband(ESA_VOLTAGE, request.param)


def row_index(passband, elevation):
    return round((elevation - passband.min_elevation) / passband.elevation_spacing)


def boundaries(passband):
    # Claude: per-row (min, max) arrays, NaN where a row has no span, for vectorized
    # Claude: checks against the grid
    n_rows = passband.values.shape[0]
    spans = [passband.speed_ratio_spans.get(i, (np.nan, np.nan)) for i in range(n_rows)]
    spans = np.array(spans)
    return spans[:, 0], spans[:, 1]


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
    rows = np.flatnonzero(passband.values.max(axis=1) >= cutoff)
    for i in rows:
        elevation = passband.min_elevation + passband.elevation_spacing * i
        # Placeholder: the bounds of an on-grid row and its upper neighbor are combined,
        # so only the row's own crossing is bracketed
        lo, hi = speed_ratio_range_at_elevation(passband, float(elevation))
        row = passband.values[i]
        assert np.interp(lo, PASSBAND_SPEED_RATIOS, row) <= cutoff + 1e-12
        assert np.interp(hi, PASSBAND_SPEED_RATIOS, row) <= cutoff + 1e-12
        row_lo, row_hi = passband.speed_ratio_spans[i]
        assert lo <= row_lo
        assert hi >= row_hi


def test_speed_ratio_range_is_the_union_of_bracketing_rows(passband):
    lower, upper = row_index(passband, 0.0), row_index(passband, 0.5)
    lo, hi = speed_ratio_range_at_elevation(passband, 0.25)
    lower_lo, lower_hi = passband.speed_ratio_spans[lower]
    upper_lo, upper_hi = passband.speed_ratio_spans[upper]

    assert lo == pytest.approx(min(lower_lo, upper_lo))
    assert hi == pytest.approx(max(lower_hi, upper_hi))


def test_speed_ratio_range_min_is_no_greater_than_max(passband):
    active_lo, active_hi = passband.elevation_range
    for elevation in np.linspace(active_lo, active_hi, 7):
        lo, hi = speed_ratio_range_at_elevation(passband, float(elevation))
        assert lo <= hi


def test_speed_ratio_range_at_the_edges_of_the_active_rows(passband):
    """Claude: Outside the elevation range there is no span. Just inside it, one
    bracketing row is below the cutoff and the active edge row's span is returned."""
    cutoff = PASSBAND_BOUNDARY_THRESHOLD * passband.values.max()
    active = np.flatnonzero(passband.values.max(axis=1) >= cutoff)
    first, last = int(active[0]), int(active[-1])
    n_rows = passband.values.shape[0]
    min_range, max_range = passband.elevation_range

    # Claude: the fitted passbands fall below the cutoff before the grid's edges
    assert first > 1
    assert last < n_rows - 1

    for elevation in [
        min_range - 1e-9,
        max_range + 1e-9,
        passband.min_elevation - 100.0,
        passband.min_elevation + passband.elevation_spacing * n_rows + 100.0,
    ]:
        assert speed_ratio_range_at_elevation(passband, elevation) is None

    for row, elevation in [(first, min_range), (last, max_range)]:
        lo, hi = speed_ratio_range_at_elevation(passband, elevation)
        row_lo, row_hi = passband.speed_ratio_spans[row]
        assert lo == pytest.approx(row_lo)
        assert hi == pytest.approx(row_hi)


@pytest.mark.parametrize("center_value", [0.0, -1.0, np.nan])
def test_passband_without_a_positive_center_raises(center_value):
    values = np.ones((len(PASSBAND_ELEVATIONS_DEG), len(PASSBAND_SPEED_RATIOS)))
    values[len(PASSBAND_ELEVATIONS_DEG) // 2] = center_value
    with pytest.raises(ValueError, match="positive and finite at its center"):
        from_grid_values(values)


def test_passband_without_a_contour_raises():
    # Claude: the center is fine, but a NaN elsewhere makes the cutoff NaN
    values = np.ones((len(PASSBAND_ELEVATIONS_DEG), len(PASSBAND_SPEED_RATIOS)))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="never reaches its contour level"):
        from_grid_values(values)


def test_from_values_ignores_input_order():
    # Claude: a Gaussian bump, so every grid value depends on both axes
    elevations = PASSBAND_ELEVATIONS_DEG[:, None]
    speed_ratios = PASSBAND_SPEED_RATIOS[None, :]
    values = np.exp(-(elevations**2) / 8 - (speed_ratios - 1) ** 2 / 0.002)
    passband = as_passband(values)
    reference = PassbandGrid.from_values(passband)

    rng = np.random.default_rng(0)
    shuffled = passband.isel(
        elevation=rng.permutation(passband.sizes["elevation"]),
        speed_ratio=rng.permutation(passband.sizes["speed_ratio"]),
    ).transpose("speed_ratio", "elevation")
    band = PassbandGrid.from_values(shuffled)

    np.testing.assert_allclose(band.values, reference.values)
    assert band.elevation_range == reference.elevation_range


def test_from_values_default_grid():
    band = from_grid_values(
        np.ones((len(PASSBAND_ELEVATIONS_DEG), len(PASSBAND_SPEED_RATIOS)))
    )

    assert band.values.shape == (61, 101)
    assert band.min_elevation == -15.0
    assert band.elevation_spacing == pytest.approx(0.5)
    assert band.min_speed_ratio == pytest.approx(0.9)
    assert band.speed_ratio_spacing == pytest.approx(0.002)


def test_from_values_contour_threshold_sets_the_contour():
    # Claude: rows ramp linearly in speed ratio, so the lower contour crossing sits
    # Claude: at the threshold's fraction of the way up the ramp
    ramp_start = 0.95
    ramp = np.clip(PASSBAND_SPEED_RATIOS - ramp_start, 0.0, None)
    values = np.tile(ramp, (len(PASSBAND_ELEVATIONS_DEG), 1))
    ramp_height = PASSBAND_SPEED_RATIOS[-1] - ramp_start

    for threshold in [0.01, 0.5]:
        band = from_grid_values(values, contour_threshold=threshold)
        min_boundary, _ = boundaries(band)
        np.testing.assert_allclose(min_boundary, ramp_start + threshold * ramp_height)


def test_from_values_interpolates_onto_the_grid():
    # Claude: a passband linear in both axes, on a coarse, nonuniform input grid,
    # Claude: is reproduced exactly inside the input and is zero outside it
    elevations = np.array([-2.0, 0.0, 3.0])
    speed_ratios = np.array([0.95, 1.0, 1.08])
    values = 10.0 + elevations[:, None] + 100.0 * (speed_ratios[None, :] - 1.0)
    band = PassbandGrid.from_values(
        as_passband(values, elevations, speed_ratios),
        min_elevation=-4.0,
        max_elevation=4.0,
        elevation_spacing=1.0,
        min_speed_ratio=0.9,
        max_speed_ratio=1.1,
        speed_ratio_spacing=0.01,
    )

    assert band.values.shape == (9, 21)
    grid_elevations = -4.0 + np.arange(9)
    grid_speed_ratios = 0.9 + 0.01 * np.arange(21)
    inside = (
        (grid_elevations[:, None] >= -2.0)
        & (grid_elevations[:, None] <= 3.0)
        & (grid_speed_ratios[None, :] >= 0.95 - 1e-12)
        & (grid_speed_ratios[None, :] <= 1.08 + 1e-12)
    )
    expected = (
        10.0 + grid_elevations[:, None] + 100.0 * (grid_speed_ratios[None, :] - 1.0)
    ) / 10.0
    np.testing.assert_allclose(band.values[inside], expected[inside])
    np.testing.assert_array_equal(band.values[~inside], 0.0)


def test_from_values_normalizes_at_center_without_mutating_input():
    values = np.full((len(PASSBAND_ELEVATIONS_DEG), len(PASSBAND_SPEED_RATIOS)), 4.0)
    band = from_grid_values(values)

    np.testing.assert_allclose(band.values, 1.0)
    np.testing.assert_allclose(values, 4.0)


def test_only_active_rows_have_spans(passband):
    cutoff = PASSBAND_BOUNDARY_THRESHOLD * passband.values.max()
    active = np.flatnonzero(passband.values.max(axis=1) >= cutoff)

    assert sorted(passband.speed_ratio_spans.keys()) == list(active)
    assert all(np.all(np.isfinite(s)) for s in passband.speed_ratio_spans.values())


def test_speed_ratio_range_is_none_between_two_inactive_rows():
    # Claude: only the first, center (needed for normalization), and last rows are
    # Claude: active, so elevation_range spans the whole grid but the rows 1 and 2
    # Claude: bracketing an elevation between them are both empty
    values = np.zeros((len(PASSBAND_ELEVATIONS_DEG), len(PASSBAND_SPEED_RATIOS)))
    mid = len(PASSBAND_ELEVATIONS_DEG) // 2
    values[[0, mid, -1], len(PASSBAND_SPEED_RATIOS) // 2] = 1.0
    band = from_grid_values(values)

    elevation = float(PASSBAND_ELEVATIONS_DEG[1]) + 0.5 * band.elevation_spacing
    assert speed_ratio_range_at_elevation(band, elevation) is None


def test_passband_fit_table(passband_fit_table, response_table_paths):
    table = pd.read_csv(response_table_paths["passband_fit_coefficients_path"])
    all_coefficients = passband_fit_table.coefficients
    assert all_coefficients.dims == ("aperture", "power", "energy_ratio", "elevation")
    assert list(all_coefficients.aperture.values) == ["SG", "OA"]
    assert list(all_coefficients.power.values) == [2, 1, 0]
    for aperture in APERTURES:
        coefficients = all_coefficients.sel(aperture=aperture)
        min_voltage = coefficients.min_esa_voltage
        max_voltage = coefficients.max_esa_voltage
        assert min_voltage.dims == max_voltage.dims == ("energy_ratio", "elevation")
        assert min_voltage.attrs["units"] == max_voltage.attrs["units"] == "V"
        has_fit = coefficients.notnull().all("power").values
        assert np.all(min_voltage.values[has_fit] > 0)
        assert np.all(min_voltage.values[has_fit] < max_voltage.values[has_fit])
        assert np.all(min_voltage.isnull().values == ~has_fit)

        # Placeholder: the file's voltage bounds, converted with the k factor
        rows = table[table["region"] == aperture]
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


@pytest.mark.parametrize("aperture", list(APERTURES))
def test_passband_is_normalized_at_center(passband_fit_table, aperture):
    band = passband_fit_table.evaluate_passband(ESA_VOLTAGE, aperture)

    assert _passband_value(band, 0.0, 1.0) == pytest.approx(1.0)
    assert band.values.min() >= 0.0
    assert band.values.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("aperture", list(APERTURES))
def test_passband_boundaries_bracket_the_response(passband_fit_table, aperture):
    band = passband_fit_table.evaluate_passband(ESA_VOLTAGE, aperture)
    elevations, speed_ratios = _grid(band)
    cutoff = PASSBAND_BOUNDARY_THRESHOLD * band.values.max()

    min_boundary, max_boundary = boundaries(band)

    i_center = int(np.flatnonzero(np.isclose(elevations, 0.0))[0])
    assert min_boundary[i_center] < 1.0 < max_boundary[i_center]
    elevation_lower, elevation_upper = band.elevation_range
    assert elevation_lower < 0.0 < elevation_upper

    outside_speed = (speed_ratios[None, :] < min_boundary[:, None]) | (
        speed_ratios[None, :] > max_boundary[:, None]
    )
    assert np.all(band.values[outside_speed] < cutoff)

    outside_elevation = (elevations < elevation_lower) | (elevations > elevation_upper)
    assert np.all(band.values[outside_elevation] < cutoff)


def test_passband_clamps_energy_to_the_fit_range(passband_fit_table):
    coefficients = passband_fit_table.coefficients.sel(aperture="SG")
    min_voltage = float(coefficients.min_esa_voltage.min())
    max_voltage = float(coefficients.max_esa_voltage.max())

    def values(esa_voltage):
        return passband_fit_table.evaluate_passband(esa_voltage, "SG").values

    np.testing.assert_allclose(values(10 * max_voltage), values(max_voltage))
    np.testing.assert_allclose(values(min_voltage / 10), values(min_voltage))


def test_passband_clamps_each_fit_to_its_own_range(response_table_paths, tmp_path):
    """Placeholder: Lowering one elevation's upper bound clamps only that row."""
    elevation, lowered_max = 4.0, 500.0
    table = pd.read_csv(response_table_paths["passband_fit_coefficients_path"])
    rows = (table["region"] == "SG") & (table["elevation"] == elevation)
    table.loc[rows, "max_esa_voltage"] = lowered_max
    lowered = tmp_path / "passband-fit-coefficients.csv"
    table.to_csv(lowered, index=False)

    original = PassbandFitTable(response_table_paths["passband_fit_coefficients_path"])
    modified = PassbandFitTable(lowered)

    def sg_values(fit_table, esa_voltage):
        return fit_table.evaluate_passband(esa_voltage, "SG").values

    band = modified.evaluate_passband(ESA_VOLTAGE, "SG")
    elevations, _ = _grid(band)
    i_row = int(np.flatnonzero(np.isclose(elevations, elevation))[0])
    other_rows = np.arange(len(elevations)) != i_row

    np.testing.assert_allclose(
        band.values[other_rows], sg_values(original, ESA_VOLTAGE)[other_rows]
    )

    # Placeholder: the row is the fit at the lowered bound, up to the normalization,
    # which is set by the unchanged center row
    def shape(row):
        return row / row.max()

    np.testing.assert_allclose(
        shape(band.values[i_row]),
        shape(sg_values(original, lowered_max)[i_row]),
    )
    assert not np.allclose(
        shape(band.values[i_row]),
        shape(sg_values(original, ESA_VOLTAGE)[i_row]),
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

    quadratic = PassbandFitTable(response_table_paths["passband_fit_coefficients_path"])
    cubic_fits = PassbandFitTable(cubic)

    assert list(cubic_fits.coefficients.power.values) == [3, 2, 1, 0]
    np.testing.assert_allclose(
        cubic_fits.evaluate_passband(ESA_VOLTAGE, "SG").values,
        quadratic.evaluate_passband(ESA_VOLTAGE, "SG").values,
    )


def test_passband_differs_between_apertures(passband_fit_table):
    assert not np.allclose(
        passband_fit_table.evaluate_passband(ESA_VOLTAGE, "SG").values,
        passband_fit_table.evaluate_passband(ESA_VOLTAGE, "OA").values,
    )


def test_passband_cache_rounds_the_voltage_to_a_millivolt(passband_fit_table):
    band = passband_fit_table.evaluate_passband(ESA_VOLTAGE, "SG")

    assert passband_fit_table.evaluate_passband(ESA_VOLTAGE + 1e-4, "SG") is band
    assert passband_fit_table.evaluate_passband(ESA_VOLTAGE + 1e-2, "SG") is not band
    assert passband_fit_table.evaluate_passband(ESA_VOLTAGE, "OA") is not band


def test_passband_fit_table_requires_every_aperture(response_table_paths, tmp_path):
    table = pd.read_csv(response_table_paths["passband_fit_coefficients_path"])
    sg_only = tmp_path / "passband-fit-coefficients.csv"
    table[table["region"] == "SG"].to_csv(sg_only, index=False)

    with pytest.raises(ValueError, match="No passband fit for aperture OA"):
        PassbandFitTable(sg_only)


def test_contour_is_clipped_at_the_grid_edges():
    # Claude: rows ramp up in speed ratio from 0.95 (so the center at speed ratio 1
    # Claude: is nonzero), so the left crossing is interior and the right edge is
    # Claude: still above the cutoff; every elevation row is active
    ramp_start = 0.95
    ramp = np.clip(PASSBAND_SPEED_RATIOS - ramp_start, 0.0, None)
    values = np.tile(ramp, (len(PASSBAND_ELEVATIONS_DEG), 1))

    band = from_grid_values(values)

    cutoff_ratio = ramp_start + PASSBAND_BOUNDARY_THRESHOLD * (
        PASSBAND_SPEED_RATIOS[-1] - ramp_start
    )
    min_boundary, max_boundary = boundaries(band)
    np.testing.assert_allclose(min_boundary, cutoff_ratio)
    np.testing.assert_allclose(max_boundary, PASSBAND_SPEED_RATIOS[-1])
    assert band.elevation_range == (
        PASSBAND_ELEVATIONS_DEG[0],
        PASSBAND_ELEVATIONS_DEG[-1],
    )
