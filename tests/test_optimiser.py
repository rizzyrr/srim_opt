import json
import math

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
    # Widths within 5 % of the widest (2.66-6.0 MeV here) tie, and the lowest fluence wins
    runs = FakeRuns('width', lambda E: window(1000 - 300 * math.log(E / 4.0) ** 2, 1e15 * E))
    best = opt._search(runs, 1.0, 10.0)
    assert 2.6 <= best <= 6.0
    runs = FakeRuns('width', lambda E: window(1000 - 300 * math.log(E / 4.0) ** 2, 1e15 * E))
    runs.spec.width_slack = 0.0
    assert 3.0 <= opt._search(runs, 1.0, 10.0) <= 5.5


def test_pick_best_prefers_lower_fluence_among_near_widest():
    spec = opt.BeamSpec(target_dpa=1.0)
    pts = [(5.0, window(340, 1.0e16)), (10.0, window(345, 1.8e15)), (2.0, window(200, 1e15)), (3.0, None)]
    assert opt.pick_best(pts, spec) == 10.0
    pts = [(5.0, window(345, 1.8e15)), (10.0, window(340, 1.0e16))]
    assert opt.pick_best(pts, spec) == 5.0


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
    assert len({(j[1], j[2]) for j in jobs}) == 123
    assert sum(j[0] == 'train' for j in jobs) == 1230
    assert all(0.5e6 <= j[3] <= 50e6 for j in jobs)


@pytest.mark.parametrize('target', dataset.TARGETS)
def test_every_target_has_its_self_ion(target):
    assert dataset.SELF_ION.get(target, target) in dataset.ions_for(target)
