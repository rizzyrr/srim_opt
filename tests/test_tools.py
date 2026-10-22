import json

import pytest

import srim_optimiser as opt
import srim_tools as tools


def call(name, **args):
    return json.loads(tools.call_tool(name, args))


def test_schemas_cover_every_tool():
    names = [t['function']['name'] for t in tools.TOOLS]
    assert sorted(names) == sorted(tools.FUNCTIONS)
    for t in tools.TOOLS:
        params = t['function']['parameters']
        assert params['type'] == 'object' and set(params['required']) <= set(params['properties'])


def test_list_options_is_json_and_names_materials_and_facilities():
    out = call('list_options')
    assert 'CuCrZr' in out['materials'] and 'HZDR 3 MV Tandetron' in out['facilities']


def test_energy_window_matches_optimiser():
    out = call('energy_window', facility='HZDR 3 MV Tandetron', ion='Cu')
    e_min, e_max, _ = opt.energy_window('HZDR 3 MV Tandetron', 'Cu')
    assert (out['e_min_MeV'], out['e_max_MeV']) == (e_min, e_max)


def test_energy_window_without_facility_lists_all():
    out = call('energy_window', ion='Fe')
    assert [f['facility'] for f in out['facilities']] == list(opt.FACILITIES)


def test_screen_is_labelled_a_prediction_and_reports_defaults():
    # Cu into CuCrZr is in the dataset, so the SR table is cached and no SRIM is needed
    out = call('screen_beam', material='CuCrZr', ion='Cu', target_dpa=3.0, facility='HZDR 3 MV Tandetron')
    assert out['ok'] and out['status'].startswith('PREDICTED')
    assert out['prediction']['fluence'] > 0 and out['prediction']['beam_setting']['charge_state'] >= 1
    assert out['defaults_used']['min_width_nm'] == 100.0


def test_missing_essentials_ask_the_user():
    out = call('screen_beam', material='Fe', ion='Fe', facility='HZDR 3 MV Tandetron')
    assert not out['ok'] and 'target_dpa' in out['message']
    out = call('screen_beam', material='Fe', ion='Fe', target_dpa=1.0)
    assert not out['ok'] and 'facility' in out['message']


def test_bad_names_come_back_as_messages():
    assert 'Unknown material' in call('screen_beam', material='Unobtainium', ion='Fe', target_dpa=1,
                                      e_min_MeV=1, e_max_MeV=10)['message']
    assert 'No displacement energy' in call('screen_beam', composition={'Xx': 100}, ion='Fe', target_dpa=1,
                                            e_min_MeV=1, e_max_MeV=10)['message']
    assert 'Unknown tool' in call('rm_rf')['message']


def test_beam_time_matches_optimiser():
    out = call('beam_time', fluence=1e16, current_nA=50, charge_state=3)
    assert out['hours'] == pytest.approx(opt.beam_time(1e16, 50, 3, 1.0)['hours'], rel=1e-3)   # results are rounded


def test_plan_is_labelled_confirmed(monkeypatch):
    # plan() itself needs SRIM; check only the wrapping
    def fake_plan(material, ion, spec, srim_dir=None, progress=None, minimised=False):
        r = opt.PlanResult(ok=True, message='done', material=material, ion=ion, spec={}, energy_MeV=12.0,
                           fluence=2e15, runs=[{'key': 'a'}], provenance={'code': 'SRIM-2013'})
        if progress:
            progress('confirm', 1, 1, 'TRIM 1000 ions')
        return r

    monkeypatch.setattr(opt, 'plan', fake_plan)
    seen = []
    out = json.loads(tools.call_tool('plan_beam', {'material': 'W', 'ion': 'W', 'target_dpa': 1,
                                                   'facility': 'Helsinki 5 MV tandem'},
                                     progress=lambda *a: seen.append(a)))
    assert out['status'].startswith('CONFIRMED') and out['result']['energy_MeV'] == 12.0
    assert out['trim_runs'] == 1 and seen
