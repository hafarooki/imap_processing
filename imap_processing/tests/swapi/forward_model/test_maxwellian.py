from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imap_processing.swapi.constants import Species
from imap_processing.swapi.forward_model.look_angles import look_angles_to_velocity
from imap_processing.swapi.forward_model.maxwellian.integration import (
    maxwellian_count_rates,
)
from imap_processing.swapi.forward_model.response.deadtime import deadtime_factor
from imap_processing.swapi.forward_model.response.measurement_species_response import (
    measurement_species_responses,
)

REFERENCE_RATES = pd.read_csv(
    Path(__file__).parent / "reference" / "maxwellian-reference-rates.csv"
).set_index(["density", "temperature", "bulk_speed", "bulk_azimuth", "bulk_elevation"])
ESA_VOLTAGES = REFERENCE_RATES.columns.astype(float).to_numpy()


@pytest.mark.parametrize(
    ("density", "temperature", "bulk_speed", "bulk_azimuth", "bulk_elevation"),
    REFERENCE_RATES.index,
    ids=[
        f"n{n:g}_T{t:.0f}_v{v:g}_az{az:g}_el{el:g}"
        for n, t, v, az, el in REFERENCE_RATES.index
    ],
)
def test_maxwellian_model_matches_reference(
    passband_fit_table,
    central_effective_area_table,
    azimuthal_transmission,
    epoch,
    density,
    temperature,
    bulk_speed,
    bulk_azimuth,
    bulk_elevation,
):
    esa_voltages = ESA_VOLTAGES

    measurement_times = np.full(esa_voltages.size, epoch)
    # TODO use non-constant rotation matrices or add a separate test for that
    rtn_to_swapi = np.broadcast_to(np.eye(3), (esa_voltages.size, 3, 3))
    responses = measurement_species_responses(
        esa_voltages=esa_voltages,
        epochs=measurement_times,
        species=Species.PROTON,
        rtn_to_swapi=rtn_to_swapi,
        passband_fit_table=passband_fit_table,
        central_effective_area_table=central_effective_area_table,
        azimuthal_transmission=azimuthal_transmission,
    )
    velocity_xyz = look_angles_to_velocity(
        speed=bulk_speed,
        azimuth=bulk_azimuth,
        elevation=bulk_elevation
    )
    ideal_rates, _ = maxwellian_count_rates(
        density=density,
        temperature=temperature,
        velocity_rtn_sc=velocity_xyz,
        responses=responses,
    )
    observable_rates = ideal_rates * deadtime_factor(ideal_rates)

    reference_rates = REFERENCE_RATES.loc[
        (density, temperature, bulk_speed, bulk_azimuth, bulk_elevation)
    ].to_numpy()

    rtol, atol = 0.01, 1e-4 * reference_rates.max()
    np.testing.assert_allclose(
        observable_rates,
        reference_rates,
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"density={density} cm^-3, temperature={temperature} K, "
            f"bulk_speed={bulk_speed} km/s, bulk_azimuth={bulk_azimuth} deg, "
            f"bulk_elevation={bulk_elevation} deg\n"
        ),
    )
