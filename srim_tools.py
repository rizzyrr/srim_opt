# Tools for the LLM interface (Part 2): the physics core's API as plain functions that take and return
# JSON-safe values, plus their schemas in the OpenAI function-calling format, which Groq, LangGraph and
# most other LLM clients accept. No LLM code lives here, so the provider can change without touching it.
#
#   TOOLS                         schemas to hand to the model
#   call_tool(name, args)         run one tool call; always returns a JSON string, errors included
#
# Results are trimmed before the model sees them (no energy curves or run lists), to keep each chat turn
# well under the provider's token limits. Every number the model may quote is in these results.

import json
from typing import Callable, Dict, List, Optional

import srim_core as core
import srim_optimiser as opt

# BeamSpec fields the model may set, and what the experimenter gets if it doesn't
SPEC_FIELDS = {
    'target_dpa': 'Target mean NRT dpa over the region of interest (ROI). Required.',
    'facility': 'Accelerator name from list_options; sets the energy window. Or give e_min_MeV and e_max_MeV.',
    'e_min_MeV': 'Lowest beam energy, MeV (only without a facility).',
    'e_max_MeV': 'Highest beam energy, MeV (only without a facility).',
    'min_depth_nm': 'The ROI starts at or below this depth, nm. Default 100.',
    'min_width_nm': 'The ROI is at least this wide, nm. Default 100.',
    'implant_mode': "'distance' (ROI ends implant_margin_nm before the implant peak) or 'percent'. Default 'distance'.",
    'implant_margin_nm': 'Distance mode: gap between the ROI end and the implant peak, nm. Default 100.',
    'implant_pct': 'Percent mode: implanted ions in the ROI stay below this % of their peak. Default 10.',
    'max_spread': 'dpa flatness across the ROI as a fraction, e.g. 0.1 for +/-10 %. Default 0.1.',
    'objective': "'width' (widest ROI, default), 'fluence' (shortest beam time) or 'implant' (fewest implanted ions).",
    'max_fluence': 'Optional cap on fluence, ions/cm2.',
}
_NUMBER_FIELDS = {'target_dpa', 'e_min_MeV', 'e_max_MeV', 'min_depth_nm', 'min_width_nm', 'implant_margin_nm',
                  'implant_pct', 'max_spread', 'max_fluence'}

_MATERIAL_PARAMS = {
    'material': {'type': 'string',
                 'description': 'Target material name from list_options (e.g. Fe, W, CuCrZr, Fe-9Cr, SiC). '
                                'Leave empty and give composition for anything else.'},
    'composition': {'type': 'object', 'additionalProperties': {'type': 'number'},
                    'description': 'Custom target as {element symbol: percent}, e.g. {"Fe": 86, "Cr": 14}.'},
    'composition_unit': {'type': 'string', 'enum': ['at', 'wt'],
                         'description': "Percent unit of composition: 'at' (atomic, default) or 'wt' (weight)."},
    'ion': {'type': 'string', 'description': 'Ion element symbol, e.g. Fe, W, Cu.'},
}


def _spec_params() -> dict:
    props = {}
    for name, text in SPEC_FIELDS.items():
        if name == 'facility':
            props[name] = {'type': 'string', 'enum': list(opt.FACILITIES), 'description': text}
        elif name == 'implant_mode':
            props[name] = {'type': 'string', 'enum': ['distance', 'percent'], 'description': text}
        elif name == 'objective':
            props[name] = {'type': 'string', 'enum': list(core.OBJECTIVES), 'description': text}
        else:
            props[name] = {'type': 'number', 'description': text}
    return props


def _function(name: str, description: str, properties: dict, required: List[str]) -> dict:
    return {'type': 'function', 'function': {
        'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': required}}}


TOOLS = [
    _function('list_options',
              'Materials, accelerator facilities and optimisation objectives the tools know about. Call it before '
              'guessing a material or facility name.', {}, []),
    _function('energy_window',
              'Energies (MeV) an accelerator can deliver for an ion, with caveats. Instant. Leave out facility '
              'to get every facility: use that for "which facilities can..." questions.',
              {'facility': {'type': 'string', 'enum': list(opt.FACILITIES)}, 'ion': _MATERIAL_PARAMS['ion']},
              ['ion']),
    _function('screen_beam',
              'Instant PREDICTION (surrogate model, no TRIM) of the best beam energy, fluence and ROI for a target '
              'dpa. Numbers are predictions, not yet verified. Use first, before plan_beam.',
              {**_MATERIAL_PARAMS, **_spec_params()}, ['ion', 'target_dpa']),
    _function('plan_beam',
              'Best beam energy and fluence CONFIRMED by a real TRIM simulation. Takes 2-15 minutes. Use after '
              'screen_beam when the user wants verified numbers.',
              {**_MATERIAL_PARAMS, **_spec_params()}, ['ion', 'target_dpa']),
    _function('beam_time',
              'Irradiation time and dpa rate for a fluence at a given beam current. Instant.',
              {'fluence': {'type': 'number', 'description': 'ions/cm2, from screen_beam or plan_beam'},
               'current_nA': {'type': 'number', 'description': 'Electrical beam current on target, nA'},
               'charge_state': {'type': 'integer', 'description': 'Ion charge state, from beam_setting in a result'},
               'area_cm2': {'type': 'number', 'description': 'Irradiated area, cm2. Default 1.'},
               'roi_mean_dpa': {'type': 'number', 'description': 'Target mean dpa, for the dpa rate'},
               'peak_dpa': {'type': 'number', 'description': 'Peak dpa from the result, for the peak dpa rate'}},
              ['fluence', 'current_nA', 'charge_state']),
]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _material(args: dict):
    # A MATERIALS name, or a custom composition with ASTM E521 displacement energies
    if args.get('composition'):
        comp = {str(k): float(v) for k, v in args['composition'].items()}
        unknown = sorted(set(comp) - set(core.E_D_ASTM))
        if unknown:
            raise ValueError(f'No displacement energy for {", ".join(unknown)}. Elements with one: '
                             f'{", ".join(sorted(core.E_D_ASTM))}.')
        return core.named_material(comp, unit=args.get('composition_unit') or 'at')
    name = args.get('material')
    if name not in core.MATERIALS:
        raise ValueError(f'Unknown material "{name}". Known: {", ".join(core.MATERIALS)}; '
                         f'or give a composition.')
    return name


def _spec(args: dict) -> opt.BeamSpec:
    d = {k: args[k] for k in SPEC_FIELDS if args.get(k) is not None}
    for k in _NUMBER_FIELDS & set(d):
        d[k] = float(d[k])
    if 'target_dpa' not in d:
        raise ValueError('target_dpa is required: ask the user for the dpa they need.')
    if not d.get('facility') and ('e_min_MeV' not in d or 'e_max_MeV' not in d):
        raise ValueError('Give a facility, or both e_min_MeV and e_max_MeV: ask the user which accelerator '
                         'or energy range they have.')
    return opt.BeamSpec.from_dict(d)


def _defaults_used(args: dict) -> Dict[str, object]:
    # Spec fields the model left out, with the default the result is based on, so it can say so
    default = opt.BeamSpec(target_dpa=1.0)
    skip = {'target_dpa', 'facility', 'e_min_MeV', 'e_max_MeV', 'max_fluence'}
    skip.add('implant_margin_nm' if args.get('implant_mode') == 'percent' else 'implant_pct')
    return {k: getattr(default, k) for k in SPEC_FIELDS if k not in skip and args.get(k) is None}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def list_options() -> dict:
    return {
        'materials': {name: {c.symbol: round(float(x), 3) for c, x in zip(m.constituents, m.fractions)}
                      for name, m in core.MATERIALS.items()},
        'materials_note': 'Compositions in atomic fraction. The surrogate was trained on these 13 materials '
                          'with heavy ions (C to Au); predictions outside that range are less reliable.',
        'facilities': {name: {'type': f.kind, 'terminal_MV_max': f.v_max_MV, 'charge_state_max': f.q_max,
                              'energy_cap_MeV': f.e_cap_MeV, 'note': f.note or None}
                       for name, f in opt.FACILITIES.items()},
        'objectives': core.OBJECTIVES,
    }


def energy_window(ion: str, facility: Optional[str] = None) -> dict:
    # One facility, or all of them (so "which facilities can..." needs no guessing which ones to check)
    if facility is None:
        return {'ion': ion, 'facilities': [energy_window(ion, name) for name in opt.FACILITIES]}
    e_min, e_max, notes = opt.energy_window(facility, ion)
    return {'facility': facility, 'ion': ion, 'e_min_MeV': e_min, 'e_max_MeV': e_max, 'notes': notes}


_RESULT_FIELDS = ('energy_MeV', 'fluence', 'roi_start_nm', 'roi_end_nm', 'roi_width_nm', 'dpa_min_in_roi',
                  'dpa_max_in_roi', 'peak_dpa', 'max_implanted_appm_in_roi', 'beam_setting')


def _numbers(r: dict, spec: opt.BeamSpec) -> dict:
    # The result fields the model may quote. fluence_min/max are renamed: they are the fluences that keep the
    # ROI mean within the dpa tolerance, not an uncertainty (a model read them as "model min/max")
    out = {k: r.get(k) for k in _RESULT_FIELDS}
    if r.get('fluence_min') is not None:
        out['fluence_range_for_dpa_tolerance'] = {'tolerance_dpa': spec.tol_dpa,
                                                  'fluence': [r['fluence_min'], r['fluence_max']]}
    return out


def _keep(raw: Optional[dict], material, spec: opt.BeamSpec, result) -> None:
    # The untrimmed result for the caller (the app draws charts from it); never sent to the model
    if raw is not None:
        raw.update(material=material, spec=spec, result=result)


def screen_beam(srim_dir: Optional[str] = None, raw: Optional[dict] = None, **args) -> dict:
    spec, material = _spec(args), _material(args)
    result = opt.screen(material, args['ion'], spec, srim_dir=srim_dir or core.DEFAULT_SRIM_DIR)
    _keep(raw, material, spec, result)
    r = result.to_dict()
    out = {'status': ('NOT FEASIBLE with this energy window (from SR range tables, not the surrogate model)'
                      if not r['ok']
                      else 'PREDICTED (surrogate model, not verified by TRIM)' if r['source'] == 'surrogate'
                      else 'ENERGY WINDOW ONLY (SR ranges; no surrogate prediction for this case)'),
           'ok': r['ok'], 'message': r['message'], 'material': r['material'], 'ion': r['ion'],
           'energy_window_MeV': r['energy_window_MeV'], 'feasible_from_MeV': r['feasible_from_MeV'],
           'flags': r['flags'], 'defaults_used': _defaults_used(args)}
    if r['best']:
        best = r['best']
        out['prediction'] = _numbers(best, spec)
        out['prediction']['fluence_80pct_uncertainty_band'] = best['band'].get('fluence')
        out['prediction']['uncertainty'] = best['uncertainty']
    return out


def plan_beam(progress: Optional[Callable[[str, int, int, str], None]] = None, srim_dir: Optional[str] = None,
              raw: Optional[dict] = None, **args) -> dict:
    spec, material = _spec(args), _material(args)
    result = opt.plan(material, args['ion'], spec, srim_dir=srim_dir or core.DEFAULT_SRIM_DIR,
                      progress=progress)   # TRIM's window stays visible: the run was approved by a person
    _keep(raw, material, spec, result)
    r = result.to_dict()
    out = {'status': 'CONFIRMED by TRIM simulation' if r['ok'] else 'FAILED',
           'ok': r['ok'], 'message': r['message'], 'material': r['material'], 'ion': r['ion'],
           'energy_window_MeV': r['energy_window_MeV'], 'flags': r['flags'],
           'defaults_used': _defaults_used(args), 'trim_runs': len(r['runs']), 'elapsed_s': r['elapsed_s']}
    if r['ok']:
        out['result'] = _numbers(r, spec)
        if r['prediction']:
            out['surrogate_predicted'] = {k: r['prediction'].get(k) for k in ('energy_MeV', 'fluence')}
        out['method'] = r['provenance']
    return out


def beam_time(fluence: float, current_nA: float, charge_state: int, area_cm2: float = 1.0,
              roi_mean_dpa: Optional[float] = None, peak_dpa: Optional[float] = None) -> dict:
    out = opt.beam_time(float(fluence), float(current_nA), int(charge_state), float(area_cm2),
                        roi_mean_dpa, peak_dpa)
    return {'inputs': {'fluence': fluence, 'current_nA': current_nA, 'charge_state': charge_state,
                       'area_cm2': area_cm2}, **out}


FUNCTIONS: Dict[str, Callable[..., dict]] = {
    'list_options': list_options, 'energy_window': energy_window, 'screen_beam': screen_beam,
    'plan_beam': plan_beam, 'beam_time': beam_time,
}


def call_tool(name: str, args: Optional[dict] = None, progress=None, srim_dir: Optional[str] = None,
              raw: Optional[dict] = None) -> str:
    # One tool call from the model. Failures come back as {"ok": false, "message": ...} so the model can tell
    # the user or fix its arguments, instead of the chat crashing. srim_dir is the caller's, never the model's.
    # Pass a dict as raw to receive the untrimmed screen/plan result alongside (for charts).
    args = {k: v for k, v in (args or {}).items() if k not in ('srim_dir', 'progress', 'raw')}
    try:
        if name not in FUNCTIONS:
            raise ValueError(f'Unknown tool "{name}". Tools: {", ".join(FUNCTIONS)}.')
        if name == 'plan_beam':
            out = plan_beam(progress=progress, srim_dir=srim_dir, raw=raw, **args)
        elif name == 'screen_beam':
            out = screen_beam(srim_dir=srim_dir, raw=raw, **args)
        else:
            out = FUNCTIONS[name](**args)
    except (ValueError, KeyError, TypeError) as e:
        out = {'ok': False, 'message': str(e).strip('"\'')}
    except Exception as e:   # SRIM missing, TRIM failure: report, don't crash the chat
        out = {'ok': False, 'message': f'{type(e).__name__}: {e}'}
    return json.dumps(_rounded(opt._jsonable(out)))


def _rounded(x, digits: int = 4):
    # 4 significant figures: more than TRIM's statistics support, and short enough for the model to quote
    # exactly instead of re-rounding (which would defeat the check that answers only quote tool numbers)
    if isinstance(x, dict):
        return {k: _rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [_rounded(v, digits) for v in x]
    if isinstance(x, float) and x != 0:
        return float(f'{x:.{digits}g}')
    return x
