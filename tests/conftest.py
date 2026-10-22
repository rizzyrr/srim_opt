import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import srim_core as core  # noqa: E402


@pytest.fixture
def synthetic_profile():
    # Smooth heavy-ion-like profile in Cu: damage rising to a peak at 1500 nm, ions implanted around 1700 nm
    depth = np.linspace(10, 2500, 100)
    damage = np.exp(-0.5 * ((depth - 1500) / 450) ** 2) + 0.15   # eV/A/ion, arbitrary scale
    damage[depth > 1500] = np.exp(-0.5 * ((depth[depth > 1500] - 1500) / 200) ** 2) + 0.15 * np.exp(
        -(depth[depth > 1500] - 1500) / 100)
    implanted = 1e4 * np.exp(-0.5 * ((depth - 1700) / 250) ** 2)   # 1/cm per ion
    material = core.replace(core.MATERIALS['Cu'], width_A=25000.0)
    Ed = material.nrt_Ed
    return core.DamageProfile(material=material, ion_symbol='Cu', energy_eV=5e6, number_ions=1000,
                              depth_nm=depth, Tdam_density=damage, disp_density=0.8 * damage / (2 * Ed),
                              implanted=implanted, Tdam_eV=float(np.sum(damage) * 249), N_NRT=1.0,
                              Tdam_balance_eV=float(np.sum(damage) * 249))
