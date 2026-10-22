import os

import numpy as np
import pytest

import srim_core as core


def test_materials_registry():
    assert set(core.MATERIALS) >= {'Fe', 'W', 'CuCrZr', 'Fe-9Cr', 'SiC'}
    cucrzr = core.MATERIALS['CuCrZr']
    assert [c.E_d for c in cucrzr.constituents] == [30.0, 40.0, 40.0]   # ASTM E521
    assert abs(core.MATERIALS['SiC'].density - 3.21) < 1e-9
    fe9cr = core.MATERIALS['Fe-9Cr']   # 9 wt% Cr is about 9.6 at%
    assert 9.4 < fe9cr.fractions[1] * 100 < 9.8


V2_ALLOYS = [name for name in core.MATERIALS if name not in core.V1_MATERIALS]


def test_v2_alloys_registered():
    assert len(V2_ALLOYS) == 12 and {'EUROFER97', 'W-5Re', '316L', 'Zircaloy-4', 'CoCrFeMnNi'} <= set(V2_ALLOYS)


@pytest.mark.parametrize('name', V2_ALLOYS)
def test_alloy_compositions_and_densities(name):
    m = core.MATERIALS[name]
    assert not m.validate() and core.material_name(m) == name
    assert np.all(m.fractions > 0) and abs(m.weight_pct.sum() - 100) < 1e-9
    # The set density is within 5 % of ideal mixing of the metallic constituents (pysrim lists gas densities
    # for N and O, and interstitial C adds no volume)
    metals = core.Material(tuple(c for c in m.constituents if c.symbol not in ('C', 'N', 'O')))
    assert m.density_override is not None and abs(m.density / metals.estimated_density - 1) < 0.05


def test_assumed_displacement_energies_are_flagged():
    note = core.assumed_Ed_note(core.MATERIALS['W-5Re'])
    assert note and 'Re 90 eV' in note
    assert core.assumed_Ed_note(core.MATERIALS['Fe']) is None
    assert 'O 40 eV' in core.assumed_Ed_note(core.MATERIALS['Zircaloy-4'])
    with pytest.raises(ValueError, match='Xe'):
        core.named_material({'Fe': 99.0, 'Xe': 1.0})


def test_energy_balance_check_ignores_round_off_for_light_ions(synthetic_profile):
    p = synthetic_profile                                      # a fresh copy for each test
    p.energy_eV, p.Tdam_eV = 5e6, 1.5e6                      # heavy ion: damage energy 30 % of the beam's
    p.Tdam_balance_eV = 1.5e6 * 1.06
    assert core.energy_balance_off(p)
    p.Tdam_eV, p.Tdam_balance_eV = 2e3, 2e3 * 1.10            # H: damage energy 0.04 %, a 10 % gap is round-off
    assert not core.energy_balance_off(p)


def test_unreadable_cache_file_is_a_miss(tmp_path):
    bad = tmp_path / 'abc.pkl'
    bad.write_bytes(b'')   # what an interrupted run used to leave behind
    assert core._load_pickle(str(bad)) is None
    assert core.cached_profiles(cache_dir=str(tmp_path)) == []


def test_atomic_save_round_trip(tmp_path):
    path = str(tmp_path / 'sub' / 'x.pkl')
    core._save_pickle({'a': 1}, path)
    assert core._load_pickle(path) == {'a': 1}
    assert not [f for f in os.listdir(tmp_path / 'sub') if f.endswith('.tmp')]


def test_range_table_interpolation():
    E = np.geomspace(1e4, 6e7, 50)
    t = core.RangeTable('Cu', E, E * 0, E * 0, 1e-3 * E ** 0.8, 3e-4 * E ** 0.8)
    rp, drp = t.at(5e6)
    assert abs(rp - 1e-3 * 5e6 ** 0.8) / rp < 1e-6
    assert abs(t.energy_for_range(rp) - 5e6) / 5e6 < 1e-6
    assert t.energy_for_range(1e12) == float('inf')
    assert core.RangeTable.from_json(t.to_json()).at(5e6)[0] == rp


def test_roi_distance_mode_unchanged_by_new_options(synthetic_profile):
    p = synthetic_profile
    fluence = p.fluence_for_peak_dpa(3.0) * 1.2
    target = float(np.mean(p.dpa(fluence)[(p.depth_nm > 800) & (p.depth_nm < 1000)]))
    old = core.roi_options(p, target, 0.1, 100, 100, 100, 'width', None)
    new = core.roi_options(p, target, 0.1, 100, 100, 100, 'width', None, implant_pct=None, max_fluence=None)
    assert old == new and old is not None
    assert old['roi_end_nm'] <= p.implant_peak_nm - 100


def test_roi_percent_mode_respects_concentration_limit(synthetic_profile):
    p = synthetic_profile
    r = core.roi_options(p, 2.0, 0.1, 100, 100, 0, 'width', None, implant_pct=10)
    assert r is not None
    imp = core.smooth(p.implanted_appm(1.0))
    in_roi = (p.depth_nm >= r['roi_start_nm']) & (p.depth_nm <= r['roi_end_nm'])
    assert imp[in_roi].max() <= 0.1 * imp.max() + 1e-12
    # A tighter limit can only shorten the usable depth
    r1 = core.roi_options(p, 2.0, 0.1, 100, 100, 0, 'width', None, implant_pct=1)
    assert r1 is None or r1['roi_end_nm'] <= r['roi_end_nm']


def test_width_slack_trades_a_little_width_for_lower_fluence(synthetic_profile):
    p = synthetic_profile
    strict = core.roi_options(p, 2.0, 0.1, 100, 100, 100, 'width', 0.10)
    slack = core.roi_options(p, 2.0, 0.1, 100, 100, 100, 'width', 0.10, width_slack=0.05)
    assert strict is not None and slack is not None
    assert slack['roi_width_nm'] >= 0.95 * strict['roi_width_nm']
    assert slack['fluence'] <= strict['fluence']


def test_max_fluence_cap(synthetic_profile):
    p = synthetic_profile
    r = core.roi_options(p, 2.0, 0.1, 100, 100, 100, 'width', None)
    assert r is not None
    assert core.roi_options(p, 2.0, 0.1, 100, 100, 100, 'width', None, max_fluence=r['fluence'] * 0.01) is None
