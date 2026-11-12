"""Constants for SWAPI processing."""

from enum import Enum

import numpy as np

NUM_PACKETS_PER_SWEEP = 12
NUM_ENERGY_STEPS = 72

# Revised SWAPI ESA k-factor (E_beam = k |V|) from high-resolution SIMION
# simulations (Q ~ 1.89 eV/V at theta = 0). Used by the forward model for
# ESA energies, passband normalization and central-speed conversions.
SIMION_K_FACTOR = 1.89
# Nominal k-factor; L2's esa_energy was written with it and has to be divided
# by it to recover the ESA voltage.
NOMINAL_K_FACTOR = 1.93

# CODATA 2022, as used by the L3 pipeline
PROTON_CHARGE_C = 1.602176634e-19
PROTON_MASS_KG = 1.67262192595e-27
ALPHA_PARTICLE_MASS_KG = 6.6446573450e-27
HELIUM_PLUS_MASS_KG = 6.64556828337e-27
BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23
METERS_PER_KILOMETER = 1_000


class Species(Enum):
    """
    Ion species whose SWAPI response is modeled.

    Parameters
    ----------
    mass_kg : float
        Ion mass in kg.
    charge_e : int
        Ion charge in elementary charges.

    Attributes
    ----------
    mass_kg : float
        Ion mass in kg.
    mass_per_charge : float
        Ion mass per charge in proton masses per elementary charge.
    """

    mass_kg: float
    mass_per_charge: float

    PROTON = (PROTON_MASS_KG, 1)
    ALPHA = (ALPHA_PARTICLE_MASS_KG, 2)
    HELIUM_PLUS = (HELIUM_PLUS_MASS_KG, 1)

    def __init__(self, mass_kg: float, charge_e: int) -> None:
        self.mass_kg = mass_kg
        self.mass_per_charge = mass_kg / PROTON_MASS_KG / charge_e


SWAPI_DEADTIME_S = 183.7e-9


# Grid and cutoff of the energy-elevation passbands
PASSBAND_ELEVATIONS_DEG = np.arange(-15, 15 + 0.5, 0.5)
PASSBAND_SPEED_RATIOS = np.linspace(0.9, 1.1, 101)
PASSBAND_BOUNDARY_THRESHOLD = 1e-2
