import copy
import json
import math

import numpy as np
import pytest

import srim_dataset as dataset
import srim_optimiser as opt


class FakeRuns:
    # Stands in for _Runs: 'TRIM' results come from a formula of the energy instead of real runs
    def __init__(self, objective, fn):
        self.spec = opt.BeamSpec(target_dpa=3.0, objective=objective)
        self.points, self.fn, self.calls = {}, fn, 0

    def run(self, E, n, stage):
        self.calls += 1
        r = self.fn(E)
        self.points[E] = (None, r)
        return None, r

    def affordable(self, E, confirm_E):
        return True


def window(width, fluence, implant=0.0):
    return {'roi_width_nm': width, 'fluence': fluence, 'max_implanted_appm_in_roi': implant}


def test_search_widest_roi_goes_to_top_of_window():
    runs = FakeRuns('width', lambda E: window(100 * E, 1e15))
    assert opt._search(runs, 1.0, 10.0) == 10.0
    assert runs.calls == 4   # the grid only: the optimum is at the boundary


def test_search_lowest_fluence_finds_feasibility_edge():
    # Feasible only from 2.6 MeV; fluence rises with energy, so the best is just above 2.6 MeV
    runs = FakeRuns('fluence', lambda E: window(200, 1e15 * E) if E >= 2.6 else None)
    best = opt._search(runs, 1.0, 10.0)
    assert 2.6 <= best <= 2.6 * 1.12


def test_search_interior_optimum():
    runs = FakeRuns('width', lambda E: window(1000 - 300 * math.log(E / 4.0) ** 2, 1e15 * E))
    assert 3.0 <= opt._search(runs, 1.0, 10.0) <= 5.5


def test_pick_best_takes_the_widest_energy():
    # A slightly narrower but cheaper energy does not win: widths are not known to 5 % between energies
    spec = opt.BeamSpec(target_dpa=1.0)
    pts = [(5.0, window(340, 1.8e15)), (10.0, window(345, 1.0e16)), (2.0, window(200, 1e15)), (3.0, None)]
    assert opt.pick_best(pts, spec) == 10.0
    pts = [(5.0, window(345, 1.0e16)), (10.0, window(345, 1.8e15))]   # equal widths: the lower fluence
    assert opt.pick_best(pts, spec) == 10.0
    assert opt.pick_best([(5.0, None)], spec) is None


def test_pick_predicted_breaks_width_ties_towards_higher_energy():
    spec = opt.BeamSpec(target_dpa=1.0)
    pts = [(4.15, window(294, 2e15)), (5.92, window(282, 3e15)), (8.45, window(280, 5e15)), (2.0, window(150, 1e15))]
    assert opt.pick_predicted(pts, spec) == 8.45          # within 15 % of the widest: the highest energy
    pts = [(4.15, window(400, 2e15)), (8.45, window(300, 5e15))]
    assert opt.pick_predicted(pts, spec) == 4.15          # clearly wider wins
    fluence_spec = opt.BeamSpec(target_dpa=1.0, objective='fluence')
    assert opt.pick_predicted(pts, fluence_spec) == opt.pick_best(pts, fluence_spec)


def test_search_nothing_feasible():
    assert opt._search(FakeRuns('width', lambda E: None), 1.0, 10.0) is None


def test_facility_window_and_charge_state():
    f = opt.Facility('test 3 MV', 'tandem', 3.0, 0.6, 3)
    lo, hi, _ = opt.energy_window(f, 'Cu')
    assert (lo, hi) == (1.2, 12.0)                       # (1 + 3) * 3 MV
    assert opt.beam_setting(f, 12.0, 'Cu') == {'charge_state': 3, 'terminal_MV': 3.0}
    assert opt.beam_setting(f, 9.0, 'Cu') == {'charge_state': 2, 'terminal_MV': 3.0}
    assert opt.beam_setting(f, 5.0, 'Cu') == {'charge_state': 1, 'terminal_MV': 2.5}
    assert opt.beam_setting(f, 13.0, 'Cu') is None       # beyond the highest charge state
    _, _, notes = opt.energy_window(f, 'Kr')
    assert any('negative ions' in n for n in notes)
    lo, hi, _ = opt.energy_window('DCF 5 MV tandem (Dalton, UK)', 'Au')
    assert hi == 35.0                                    # published cap


def test_light_ions_run_with_more_ions():
    spec = opt.BeamSpec(target_dpa=1.0)
    h = opt._with_ion_floors(spec, 'H')
    assert (h.ions_search, h.ions_confirm) == opt.LIGHT_MIN_IONS['H']
    assert opt._with_ion_floors(spec, 'Cu') == spec                       # heavy ions: as asked
    more = opt._with_ion_floors(opt.BeamSpec(target_dpa=1.0, ions_confirm=50000), 'He')
    assert more.ions_confirm == 50000 and more.ions_search == opt.LIGHT_MIN_IONS['He'][0]


def test_noisy_confirmation_is_flagged(synthetic_profile):
    # The fixture is smooth; with 40 % bin-to-bin noise, the ROI's mean dpa is uncertain by about 40 % / sqrt(28)
    p = synthetic_profile
    r = {'roi_start_nm': 500.0, 'roi_end_nm': 1200.0, 'roi_width_nm': 700.0}
    assert not any('TRIM noise' in f for f in opt._flags(p, r, 5.0, 1.0, 10.0, None, 'width'))
    noisy = copy.copy(p)
    noisy.disp_density = p.disp_density * (1 + 0.4 * np.random.default_rng(0).standard_normal(len(p.depth_nm)))
    assert any('TRIM noise' in f and '1 sigma' in f for f in opt._flags(noisy, r, 5.0, 1.0, 10.0, None, 'width'))


def test_beam_time():
    t = opt.beam_time(1e15, current_nA=100, charge_state=3, area_cm2=1.0, roi_mean_dpa=3.0)
    assert abs(t['seconds'] - 1e15 * 3 * opt.E_CHARGE / 100e-9) < 1e-6
    assert abs(t['dpa_rate_roi_per_s'] - 3.0 / t['seconds']) < 1e-12


def test_spec_validation_and_round_trip():
    assert opt.BeamSpec(target_dpa=3.0).validate() == []
    assert opt.BeamSpec(target_dpa=-1, objective='nope').validate()
    spec = opt.BeamSpec(target_dpa=3.0, facility='Helsinki 5 MV tandem')
    assert opt.BeamSpec.from_dict({**spec.__dict__, 'unknown_key': 1}) == spec


def test_plan_result_is_json_safe():
    import numpy as np
    r = opt.PlanResult(ok=True, message='x', material='Cu', ion='Cu', spec={}, energy_MeV=np.float64(5.0),
                       fluence=float('inf'), curve=[{'energy_MeV': np.float32(1.5), 'objective': None}])
    d = json.loads(json.dumps(r.to_dict()))
    assert d['energy_MeV'] == 5.0 and d['fluence'] is None


def test_dataset_plan_is_deterministic_and_complete():
    jobs = dataset.plan_jobs()
    assert jobs == dataset.plan_jobs()
    assert len({dataset.job_id(j) for j in jobs}) == len(jobs)
    v1 = [j for j in jobs if j[0] in ('train', 'bench')]
    assert len({(j[1], j[2]) for j in v1}) == 123          # Part 1's set, unchanged
    assert sum(j[0] == 'train' for j in jobs) == 1230
    assert all(0.5e6 <= j[3] <= 50e6 for j in jobs if j[1] not in dataset.LIGHT_IONS)
    for ion, (lo, hi) in dataset.LIGHT_RANGES_MEV.items():
        assert all(lo * 1e6 <= j[3] <= hi * 1e6 for j in jobs if j[1] == ion)
    assert sum(j[0] == 'alloy' for j in jobs) == len(dataset.ALLOYS) * 9 * 3 + 2 * 3   # + self-ions V and Zr
    assert sum(j[0] == 'light' for j in jobs) == 25 * 2 * dataset.LIGHT_ENERGIES
    kinds = [j[0] for j in jobs]
    assert kinds[:kinds.count('test')] == ['test'] * kinds.count('test')   # the test set runs first
    assert dataset.cache_dir_for('test') != dataset.cache_dir_for('alloy')


def test_part1_runs_stay_done():
    # The v1 job ids are what progress.csv marks done; a change to them would run Part 1 again
    progress = dataset.read_progress()
    done = set(progress.loc[progress['ok'] == True, 'job_id'])   # noqa: E712
    assert {dataset.job_id(j) for j in dataset.v1_jobs()} <= done


@pytest.mark.parametrize('target', dataset.TARGETS + dataset.ALLOYS)
def test_every_target_has_its_self_ion(target):
    assert dataset.SELF_ION.get(target, target) in dataset.ions_for(target)
