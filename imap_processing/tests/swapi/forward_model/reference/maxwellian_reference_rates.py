"""Generate maxwellian-reference-rates.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from astropy import constants as const
from astropy import units as u

LUT_PATH = Path(__file__).parents[2] / "lut"
OUTPUT_PATH = Path(__file__).parent / "maxwellian-reference-rates.csv"

K_FACTOR = 1.89

CASES = [
    # typical slow solar wind, boresight-aligned
    dict(
        density=5.0 * u.cm**-3,
        temperature=1e5 * u.K,
        bulk_speed=450.0 * u.km / u.s,
        bulk_azimuth=0.0 * u.deg,
        bulk_elevation=0.0 * u.deg,
    ),
    # hot, fast wind, with moderate bulk velocity deflection
    dict(
        density=3.0 * u.cm**-3,
        temperature=3e5 * u.K,
        bulk_speed=650.0 * u.km / u.s,
        bulk_azimuth=5.0 * u.deg,
        bulk_elevation=4.0 * u.deg,
    ),
    # very high temperature
    dict(
        density=10.0 * u.cm**-3,
        temperature=1e7 * u.K,
        bulk_speed=500.0 * u.km / u.s,
        bulk_azimuth=-3.0 * u.deg,
        bulk_elevation=-5.0 * u.deg,
    ),
    # very low temperature
    dict(
        density=10.0 * u.cm**-3,
        temperature=1e4 * u.K,
        bulk_speed=500.0 * u.km / u.s,
        bulk_azimuth=-3.0 * u.deg,
        bulk_elevation=-5.0 * u.deg,
    ),
    # flow at huge deflection angle
    dict(
        density=5.0 * u.cm**-3,
        temperature=1e5 * u.K,
        bulk_speed=400.0 * u.km / u.s,
        bulk_azimuth=20.0 * u.deg,
        bulk_elevation=-3.0 * u.deg,
    ),
]


def main() -> None:
    """Compute all cases and write the CSV."""
    area_grid = effective_area_grid(LUT_PATH)

    # rounded to the 4 significant digits written in the CSV header for reproducibility
    esa_voltages = np.array([float(f"{v:.3e}") for v in np.geomspace(10240, 51.81, 62)])

    voltage_columns = [f"{voltage:.3e}" for voltage in esa_voltages]
    rows = []
    for case in CASES:
        rates = reference(
            area_grid,
            esa_voltage=esa_voltages * u.V,
            **case,
        )
        params = {name: quantity.value for name, quantity in case.items()}
        rows.append(
            params | dict(zip(voltage_columns, rates.to_value(1 / u.s), strict=True))
        )
    table = pd.DataFrame(rows)
    table[voltage_columns] = table[voltage_columns].round(3)
    for angle in ["bulk_azimuth", "bulk_elevation"]:
        table[angle] = table[angle].map("{:+.3e}".format)
    for param in ["density", "temperature", "bulk_speed"]:
        table[param] = table[param].map("{:.1e}".format)
    table.to_csv(OUTPUT_PATH, index=False, float_format="%.3e")


def effective_area_grid(lut_path: Path) -> xr.DataArray:
    """Reference effective area [cm^2] over ESA voltage, angles and speed ratio."""
    central_effective_area = (
        pd.read_csv(lut_path / "central-effective-area.csv")
        .set_index("esa_voltage")["effective_area"]
        .to_xarray()
    )

    target_azimuth_deg = np.arange(-180, 180 + 0.5, 0.5)
    transmission = (
        pd.read_csv(lut_path / "azimuthal-transmission.csv")
        .set_index("abs_azimuth")["transmission"]
        .dropna()
    )
    azimuthal_transmission = xr.DataArray(
        np.interp(
            np.abs(target_azimuth_deg), transmission.index.values, transmission.values
        ),
        dims="azimuth",
        coords={"azimuth": target_azimuth_deg},
    )

    # raw data is stored as coefficients of voltage-angle passbands
    # as a function of log beam energy
    # note that the x-axis is energy_ratio=E/|V|
    passband_table = (
        pd.read_csv(lut_path / "passband-fit-coefficients.csv")
        .set_index(["region", "energy_ratio", "elevation"])
        .to_xarray()
    )
    passband_coefficients = (
        passband_table[["0", "1", "2"]]
        .to_dataarray("degree")
        .pipe(lambda da: da.assign_coords(degree=da["degree"].astype(int)))
    )

    # we define log beam energies as a function of ESA voltage,
    # effectively associating each energy-angle passband's
    # to the corresponding voltage-angle passband
    # using E = k*V (V in volts is numerically equal to E in eV,
    # since k* is dimensionless)
    # the fits are only valid within [min_esa_voltage, max_esa_voltage],
    # so voltages outside that range reuse the passband at the nearest bound
    clamped_voltage = np.abs(central_effective_area.esa_voltage).clip(
        passband_table.min_esa_voltage, passband_table.max_esa_voltage
    )
    log_beam_energy = np.log(K_FACTOR * clamped_voltage)
    passband_unnormalized = np.exp(
        xr.polyval(log_beam_energy, passband_coefficients, degree_dim="degree")
    ).fillna(0)

    # finally, we normalize the passband so
    # that it equals 1 at elevation = 0, energy ratio = k*
    norm = passband_unnormalized.interp(elevation=0.0, energy_ratio=K_FACTOR)
    passband = passband_unnormalized / norm

    effective_area_grid = (
        central_effective_area
        * azimuthal_transmission
        * xr.where(
            np.abs(azimuthal_transmission.azimuth) <= 20.0,
            passband.sel(region="SG"),
            passband.sel(region="OA"),
        )
    )
    return effective_area_grid.assign_coords(
        speed_ratio=(effective_area_grid.energy_ratio / K_FACTOR) ** 0.5
    )


def reference(
    effective_area_grid: xr.DataArray,
    esa_voltage: u.Quantity[u.V],
    density: u.Quantity[u.cm**-3],
    bulk_elevation: u.Quantity[u.deg],
    bulk_azimuth: u.Quantity[u.deg],
    bulk_speed: u.Quantity[u.km / u.s],
    temperature: u.Quantity[u.K],
) -> u.Quantity[1 / u.s]:
    """Reference Maxwellian model count rates, one per ESA voltage step."""
    proton_mass = const.m_p
    deadtime = u.Quantity(183.7, u.ns)

    thermal_speed = np.sqrt(const.k_B * temperature / proton_mass)
    # passband-center speed, from E = k* q |V| = m v^2 / 2
    central_speed = np.sqrt(
        2 * K_FACTOR * const.e.si * np.abs(esa_voltage) / proton_mass
    )
    normalized_central_speed = (central_speed / thermal_speed).to_value(u.one)
    normalized_bulk_speed = (bulk_speed / thermal_speed).to_value(u.one)
    flux_prefactor = density * thermal_speed * u.cm**2 * u.deg**2 / (2 * np.pi) ** 1.5
    table_voltage = effective_area_grid.esa_voltage
    interpolated_area_grid = effective_area_grid.interp(
        esa_voltage=xr.DataArray(
            np.abs(esa_voltage.to_value(u.V)), dims="esa_step"
        ).clip(table_voltage.min(), table_voltage.max())
    )

    n_sigma = 10.0
    table_speed_ratio = effective_area_grid.speed_ratio.values
    min_speed_ratio, max_speed_ratio = table_speed_ratio.min(), table_speed_ratio.max()
    min_speed = max(
        normalized_bulk_speed - n_sigma,
        normalized_central_speed.min() * min_speed_ratio,
    )
    max_speed = min(
        normalized_bulk_speed + n_sigma,
        normalized_central_speed.max() * max_speed_ratio,
    )
    log_spacing = min(
        0.5 * np.diff(np.log(np.sort(table_speed_ratio))).min(),
        0.1 / max_speed,
    )
    speed_grid = np.exp(
        np.arange(np.log(min_speed), np.log(max_speed) + log_spacing, log_spacing)
    )

    # higher angular resolution for sunglasses
    sg_angles = {
        "elevation": np.arange(-15, 15, 0.1),
        "azimuth": np.arange(-20, 20, 0.1),
    }
    oa_angles = {
        "elevation": np.arange(-15, 15, 0.5),
        "azimuth": np.concatenate([np.arange(-150, -20, 0.5), np.arange(20, 150, 0.5)]),
    }

    integral = np.zeros(esa_voltage.size)
    for step in range(esa_voltage.size):
        step_central_speed = normalized_central_speed[step]
        in_passband = (speed_grid >= step_central_speed * min_speed_ratio) & (
            speed_grid <= step_central_speed * max_speed_ratio
        )
        if in_passband.sum() < 2:
            continue
        step_speed = speed_grid[in_passband]
        step_area_grid = (
            interpolated_area_grid.isel(esa_step=step)
            .drop_vars("speed_ratio")
            .interp(
                energy_ratio=xr.DataArray(
                    K_FACTOR * (step_speed / step_central_speed) ** 2,
                    dims="speed",
                    coords={"speed": step_speed},
                )
            )
            .drop_vars("energy_ratio")
        )
        speed = step_area_grid.speed
        for angles in (sg_angles, oa_angles):
            region_effective_area = step_area_grid.interp(angles).fillna(0)
            elevation = np.deg2rad(region_effective_area.elevation)
            azimuth = np.deg2rad(region_effective_area.azimuth)
            cos_angle = np.sin(elevation) * np.sin(
                bulk_elevation.to_value(u.rad)
            ) + np.cos(elevation) * np.cos(bulk_elevation.to_value(u.rad)) * np.cos(
                azimuth - bulk_azimuth.to_value(u.rad)
            )
            vdf = np.exp(
                -(
                    speed**2
                    + normalized_bulk_speed**2
                    - 2 * speed * normalized_bulk_speed * cos_angle
                )
                / 2
            )
            # the v^3 includes a v^2 from spherical integration and a v from n*v (flux)
            integrand = region_effective_area * np.cos(elevation) * speed**3 * vdf
            integral[step] += float(
                integrand.integrate(["azimuth", "elevation", "speed"])
            )

    ideal_rate = (flux_prefactor * integral).to(
        1 / u.s, equivalencies=u.dimensionless_angles()
    )
    return ideal_rate / (1 + deadtime * ideal_rate)


if __name__ == "__main__":
    main()
