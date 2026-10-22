# One-click beam optimiser: the ion energy and fluence that give a target mean dpa over a region of
# interest (ROI), with as few TRIM runs as possible. OPTIMISER_PROBLEM.md describes the problem.
#
# dpa(x) = fluence * k(x; E), so for any energy and depth window the fluence is exact; only the energy
# needs TRIM runs. Without a surrogate the search is: a feasibility bracket from SR ranges, a short log
# grid of 300-ion runs, a refinement around the best point (most optima sit on a boundary: the widest
# ROI at the highest energy, the lowest fluence at the lowest feasible energy), then a >= 1000-ion
# confirmation run that the reported numbers come from.
#
# API used by the app and the LLM interface:
#   BeamSpec                              what the experimenter needs
#   plan(material, ion, spec)             one-click answer, verified with TRIM        -> PlanResult
#   screen(material, ion, spec)           instant answer without TRIM (surrogate)    -> ScreenResult
#   energy_window(facility, ion)          energies a facility can deliver for an ion
#   beam_setting(facility, E, ion)        charge state and terminal voltage for an energy
#   beam_time(fluence, current, q, area)  irradiation time and dpa rate

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import srim_core as core

E_CHARGE = 1.602176634e-19   # C
OBJECTIVES = core.OBJECTIVES

ProgressFn = Callable[[str, int, int, str], None]   # (stage, step, total steps, message)


# ---------------------------------------------------------------------------
# Facilities: the accelerator sets the energy window
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Facility:
    name: str
    kind: str                 # 'tandem': E = (1 + q) V;  'single' (single-ended or ECR source): E = q V
    v_max_MV: float
    v_min_MV: float           # lowest stable terminal voltage (typical value)
    q_max: int                # highest charge state with useful current (typical value)
    e_cap_MeV: Optional[float] = None   # published ceiling for heavy ions, if any
    note: str = ''


def _tandem(name: str, v_max: float, q_max: int, e_cap: Optional[float] = None, note: str = '') -> Facility:
    return Facility(name, 'tandem', v_max, round(0.2 * v_max, 2), q_max, e_cap, note)


# Terminal voltages and energy caps follow the facilities' published specifications. The lowest voltage
# and the highest useful charge state are typical values: check both with the facility.
FACILITIES: Dict[str, Facility] = {f.name: f for f in [
    _tandem('MIBL 3 MV Pelletron (Michigan)', 3.0, 5),
    _tandem('MIBL 1.7 MV Tandetron (Michigan)', 1.7, 4),
    _tandem('Helsinki 5 MV tandem', 5.0, 6),
    _tandem('IPP Garching 3 MV tandem', 3.0, 6, note='W6+ gives about 20 MeV.'),
    _tandem('DCF 5 MV tandem (Dalton, UK)', 5.0, 6, 35.0, 'Heavy ions up to about 35 MeV.'),
    Facility('JANNuS-Saclay Epimethee 3 MV (ECR)', 'single', 3.0, 0.6, 12, 36.0,
             'ECR ion source: high charge states, W up to 36 MeV.'),
    _tandem('JANNuS-Saclay Japet 2 MV tandem', 2.0, 2, 6.0, 'Up to 6 MeV with 2+ ions.'),
    _tandem('HZDR 6 MV Tandetron', 6.0, 6),
    _tandem('HZDR 3 MV Tandetron', 3.0, 5),
    _tandem('Uppsala 5 MV tandem', 5.0, 9, 50.0, 'Heavy ions up to about 50 MeV.'),
    _tandem('ANSTO SIRIUS 6 MV tandem', 6.0, 6),
]}

# Tandems accelerate negative ions, which noble gases barely form (He is the usual exception, by charge exchange)
_NO_NEGATIVE_IONS = {'Ne', 'Ar', 'Kr', 'Xe', 'Rn'}


def _facility(facility: Union[str, Facility]) -> Facility:
    return facility if isinstance(facility, Facility) else FACILITIES[facility]


def energy_window(facility: Union[str, Facility], ion_symbol: str,
                  q_max: Optional[int] = None) -> Tuple[float, float, List[str]]:
    # Lowest and highest energy (MeV) the facility can deliver for this ion, and any caveats
    f = _facility(facility)
    q = min(q_max or f.q_max, core.element_data(ion_symbol)['z'])
    notes = []
    if f.kind == 'tandem':
        e_min, e_max = 2 * f.v_min_MV, (1 + q) * f.v_max_MV
        if ion_symbol in _NO_NEGATIVE_IONS:
            notes.append(f'{ion_symbol} forms almost no negative ions, so a tandem usually cannot deliver it.')
    else:
        e_min, e_max = f.v_min_MV, q * f.v_max_MV
    if f.e_cap_MeV:
        e_max = min(e_max, f.e_cap_MeV)
    if f.note:
        notes.append(f.note)
    return e_min, e_max, notes


def beam_setting(facility: Union[str, Facility], energy_MeV: float, ion_symbol: str,
                 q_max: Optional[int] = None) -> Optional[dict]:
    # Lowest charge state that reaches energy_MeV (it gives the most beam current) and its terminal voltage
    f = _facility(facility)
    for q in range(1, min(q_max or f.q_max, core.element_data(ion_symbol)['z']) + 1):
        v = energy_MeV / (1 + q) if f.kind == 'tandem' else energy_MeV / q
        if v <= f.v_max_MV:
            return {'charge_state': q, 'terminal_MV': round(v, 3)} if v >= f.v_min_MV else None
    return None


def beam_time(fluence: float, current_nA: float, charge_state: int, area_cm2: float,
              roi_mean_dpa: Optional[float] = None, peak_dpa: Optional[float] = None) -> dict:
    # Irradiation time for a fluence. Particle flux = electrical current / (charge state * e * area).
    flux = current_nA * 1e-9 / (charge_state * E_CHARGE * area_cm2)   # ions / cm2 / s
    seconds = fluence / flux
    out = {'flux_ions_per_cm2_s': flux, 'seconds': seconds, 'hours': seconds / 3600}
    if roi_mean_dpa is not None:
        out['dpa_rate_roi_per_s'] = roi_mean_dpa / seconds
    if peak_dpa is not None:
        out['dpa_rate_peak_per_s'] = peak_dpa / seconds
    return out


# ---------------------------------------------------------------------------
# What the experimenter asks for, and what comes back
# ---------------------------------------------------------------------------

@dataclass
class BeamSpec:
    target_dpa: float                       # mean NRT dpa over the ROI
    tol_dpa: float = 0.15                   # +/- on the mean: sets the acceptable fluence range
    min_depth_nm: float = 100.0             # the ROI starts at or below this depth
    min_width_nm: float = 100.0             # and is at least this wide
    implant_mode: str = 'distance'          # 'distance' or 'percent'
    implant_margin_nm: float = 100.0        # distance mode: the ROI ends this far before the implant peak
    implant_pct: float = 10.0               # percent mode: implanted ions <= this % of their peak in the ROI
    max_spread: Optional[float] = 0.10      # every point within target * (1 +/- max_spread); None = mean only
    e_min_MeV: float = 1.0                  # accelerator limits (replaced by the facility window if set)
    e_max_MeV: float = 10.0
    objective: str = 'width'                # 'width', 'fluence' or 'implant' (see core.OBJECTIVES)
    width_slack: float = 0.05               # 'width': within one profile, ROIs within 5 % of the widest tie
                                            # and the lowest fluence wins
    max_fluence: Optional[float] = None     # optional beam-time cap, ions/cm2
    ions_search: int = 300
    ions_confirm: int = 1000
    time_budget_s: float = 300.0            # soft limit for the whole plan, confirmation included
    facility: Optional[str] = None          # a FACILITIES name
    q_max: Optional[int] = None             # highest charge state to use (default: the facility's)

    def validate(self) -> List[str]:
        problems = []
        if self.target_dpa <= 0:
            problems.append('Target dpa must be greater than zero.')
        if self.min_width_nm <= 0:
            problems.append('Minimum ROI width must be greater than zero.')
        if self.implant_mode not in ('distance', 'percent'):
            problems.append("implant_mode must be 'distance' or 'percent'.")
        if self.objective not in OBJECTIVES:
            problems.append(f'objective must be one of {", ".join(OBJECTIVES)}.')
        if self.facility is None and not 0 < self.e_min_MeV < self.e_max_MeV:
            problems.append('Energy limits must satisfy 0 < e_min_MeV < e_max_MeV.')
        if self.facility is not None and self.facility not in FACILITIES:
            problems.append(f'Unknown facility "{self.facility}". Known: {", ".join(FACILITIES)}.')
        if self.max_spread is not None and not 0 < self.max_spread < 1:
            problems.append('max_spread is a fraction between 0 and 1, e.g. 0.1 for +/-10 %.')
        return problems

    def energy_limits(self, ion_symbol: str) -> Tuple[float, float, List[str]]:
        if self.facility:
            return energy_window(self.facility, ion_symbol, self.q_max)
        return self.e_min_MeV, self.e_max_MeV, []

    @classmethod
    def from_dict(cls, d: dict) -> 'BeamSpec':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _jsonable(x):
    # Plain Python types throughout, so results can go straight to json.dumps
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


@dataclass
class PlanResult:
    ok: bool
    message: str                       # one-line answer, or why there is none
    material: str
    ion: str
    spec: dict
    energy_MeV: Optional[float] = None
    fluence: Optional[float] = None    # ions/cm2 for the target mean dpa
    fluence_min: Optional[float] = None
    fluence_max: Optional[float] = None
    roi_start_nm: Optional[float] = None
    roi_end_nm: Optional[float] = None
    roi_width_nm: Optional[float] = None
    dpa_min_in_roi: Optional[float] = None
    dpa_max_in_roi: Optional[float] = None
    peak_dpa: Optional[float] = None
    max_implanted_appm_in_roi: Optional[float] = None
    implant_peak_nm: Optional[float] = None
    roi_limit_nm: Optional[float] = None
    bin_nm: Optional[float] = None
    beam_setting: Optional[dict] = None          # charge state and terminal voltage, if a facility is set
    energy_window_MeV: Optional[Tuple[float, float]] = None
    prediction: Optional[dict] = None            # the surrogate's numbers before TRIM confirmed them
    curve: List[dict] = field(default_factory=list)    # objective vs energy, every evaluation
    runs: List[dict] = field(default_factory=list)     # every TRIM run used
    flags: List[str] = field(default_factory=list)     # quality warnings
    provenance: dict = field(default_factory=dict)
    confirm_cache_key: Optional[str] = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


@dataclass
class ScreenResult:
    ok: bool
    message: str
    material: str
    ion: str
    spec: dict
    energy_window_MeV: Optional[Tuple[float, float]] = None
    feasible_from_MeV: Optional[float] = None    # lowest energy the SR ranges allow
    best: Optional[dict] = None                  # predicted best window (surrogate), with an uncertainty band
    curve: List[dict] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    source: str = 'sr'                           # 'sr' (ranges only) or 'surrogate'

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


# ---------------------------------------------------------------------------
# Scoring one profile
# ---------------------------------------------------------------------------

def evaluate(p: core.DamageProfile, spec: BeamSpec) -> Optional[dict]:
    # Best ROI window for one profile under the spec (None if nothing fits)
    return core.roi_options(p, spec.target_dpa, spec.tol_dpa, spec.min_width_nm, spec.min_depth_nm,
                            spec.implant_margin_nm, spec.objective, spec.max_spread,
                            implant_pct=spec.implant_pct if spec.implant_mode == 'percent' else None,
                            max_fluence=spec.max_fluence, width_slack=spec.width_slack)


def pick_best(points: List[Tuple[float, Optional[dict]]], spec: BeamSpec) -> Optional[float]:
    # Best energy among (energy, window) pairs; ties go to the lower fluence. width_slack is not applied
    # between energies: widths from 300-ion runs or the surrogate are uncertain by more than 5 %, so such a
    # tie would choose the energy on noise. (It still applies within one profile, where both windows come
    # from the same run.)
    feas = [(E, r) for E, r in points if r is not None]
    if not feas:
        return None
    return min(feas, key=lambda t: (score(t[1], spec.objective), t[0]))[0]


SURROGATE_WIDTH_TIE = 0.15   # predicted ROI widths this close to the widest count as tied (see pick_predicted)


def pick_predicted(points: List[Tuple[float, Optional[dict]]], spec: BeamSpec) -> Optional[float]:
    # Best energy from surrogate predictions. Its ROI widths are uncertain by 20-25 % (srim_surrogate.py
    # validate), more than the differences between the top energies, so for the widest-ROI objective the
    # energies within SURROGATE_WIDTH_TIE of the widest tie and the highest wins: more energy leaves more room
    # before the implant limit, and in TRIM the widest flat window is nearly always at the top of the window.
    # The TRIM confirmation run then has the last word.
    if spec.objective != 'width':
        return pick_best(points, spec)
    feas = [(E, r) for E, r in points if r is not None]
    if not feas:
        return None
    w_max = max(r['roi_width_nm'] for _, r in feas)
    return max(E for E, r in feas if r['roi_width_nm'] >= (1 - SURROGATE_WIDTH_TIE) * w_max)


def score(r: Optional[dict], objective: str) -> Tuple[float, float]:
    # Lower is better; ties go to the lower fluence. Infeasible windows score infinity.
    if r is None:
        return (math.inf, math.inf)
    primary = {'width': -r['roi_width_nm'], 'fluence': r['fluence'],
               'implant': r['max_implanted_appm_in_roi']}[objective]
    return (primary, r['fluence'])


def objective_value(r: Optional[dict], objective: str) -> Optional[float]:
    if r is None:
        return None
    return {'width': r['roi_width_nm'], 'fluence': r['fluence'], 'implant': r['max_implanted_appm_in_roi']}[objective]


def _resolve_material(material: Union[str, core.Material]) -> Tuple[core.Material, str]:
    if isinstance(material, str):
        if material not in core.MATERIALS:
            raise KeyError(f'Unknown material "{material}". Known: {", ".join(core.MATERIALS)}.')
        return core.MATERIALS[material], material
    return material, material.label


def feasible_from_MeV(table: core.RangeTable, spec: BeamSpec, e_lo: float, e_hi: float) -> Optional[float]:
    # Lowest energy at which the implant rule can leave room for the ROI, from SR ranges. A necessary
    # condition only (the ROI must also sit before the dpa peak), with a margin for the approximation.
    need = spec.min_depth_nm + spec.min_width_nm
    E = np.geomspace(e_lo, e_hi, 400)
    Rp, dRp = np.array([table.at(e * 1e6) for e in E]).T
    if spec.implant_mode == 'percent':
        reach = Rp - dRp * math.sqrt(2 * math.log(100 / spec.implant_pct))
    else:
        reach = Rp - spec.implant_margin_nm
    ok = reach >= 0.85 * need
    return float(E[np.argmax(ok)]) if ok.any() else None


def _infeasible_message(table: core.RangeTable, spec: BeamSpec, ion: str, mat_name: str, e_hi: float) -> str:
    need = spec.min_depth_nm + spec.min_width_nm
    Rp, dRp = table.at(e_hi * 1e6)
    if spec.implant_mode == 'percent':
        extra = dRp * math.sqrt(2 * math.log(100 / spec.implant_pct))
        rule = f'the {spec.implant_pct:g} % implant limit sits about {extra:.0f} nm before the range'
    else:
        extra = spec.implant_margin_nm
        rule = f'the ROI must end {spec.implant_margin_nm:g} nm before the implant peak'
    e_needed = table.energy_for_range(need + extra)
    fix = (f'Raise the maximum energy to about {e_needed / 1e6:.3g} MeV' if math.isfinite(e_needed)
           else 'No energy up to 60 MeV is enough')
    return (f'E_max = {e_hi:g} MeV is too low: {ion} ions in {mat_name} reach about {Rp:.0f} nm, but the ROI '
            f'needs d_min + w = {need:g} nm and {rule}. {fix}, or reduce the ROI depth, width or implant limit.')


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

class _Runs:
    # The TRIM runs of one plan: timing, a time model, and a record of every evaluation
    def __init__(self, material, ion, spec, srim_dir, cache_dir, table, progress, minimised=False):
        self.material, self.ion, self.spec = material, ion, spec
        self.srim_dir, self.cache_dir, self.table, self.minimised = srim_dir, cache_dir, table, minimised
        self.progress = progress or (lambda *a: None)
        self.started = time.time()
        self.runs: List[dict] = []
        self.curve: List[dict] = []
        self.points: Dict[float, Tuple[core.DamageProfile, Optional[dict]]] = {}   # search runs by energy
        self._ratios: List[float] = []

    def elapsed(self) -> float:
        return time.time() - self.started

    def estimate_s(self, E_MeV: float, n: int) -> float:
        # Measured on this machine: ~0.15 s per ion at 5 MeV, slower at higher energy. Rescaled by
        # how fast this plan's own runs have been.
        prior = 3.0 + n * 0.15 * (E_MeV / 5.0) ** 0.5
        return prior * (float(np.median(self._ratios)) if self._ratios else 1.0)

    def run(self, E_MeV: float, n: int, stage: str) -> Tuple[core.DamageProfile, Optional[dict]]:
        E_MeV = float(f'{E_MeV:.3g}')
        self.progress(stage, len(self.runs) + 1, 0, f'TRIM {self.ion} {E_MeV:g} MeV, {n} ions')
        estimate = self.estimate_s(E_MeV, n)
        p = core.run_sized(self.material, self.ion, E_MeV * 1e6, n, self.srim_dir, self.cache_dir, self.table,
                           minimised=self.minimised)
        r = evaluate(p, self.spec)
        if not p.meta.get('from_cache'):
            self._ratios.append(p.meta.get('wall_s', estimate) / estimate)
        self.runs.append({'stage': stage, 'energy_MeV': E_MeV, 'ions': n, 'cache_key': p.meta.get('cache_key'),
                          'wall_s': p.meta.get('wall_s'), 'from_cache': bool(p.meta.get('from_cache')),
                          'too_thin': bool(p.meta.get('too_thin', False))})
        self.curve.append({'energy_MeV': E_MeV, 'objective': objective_value(r, self.spec.objective),
                           'feasible': r is not None, 'source': f'trim{n}'})
        if stage == 'search':
            self.points[E_MeV] = (p, r)
        return p, r

    def affordable(self, E_MeV: float, confirm_E: float) -> bool:
        s = self.spec
        return (self.elapsed() + self.estimate_s(E_MeV, s.ions_search)
                + self.estimate_s(confirm_E, s.ions_confirm) <= s.time_budget_s)


def _round(E: float) -> float:
    return float(f'{E:.3g}')


def _search(runs: _Runs, e_lo: float, e_hi: float) -> Optional[float]:
    # Best energy from 300-ion runs: a 4-point log grid from the top down, then a refinement.
    spec = runs.spec

    def best_point():
        return pick_best([(E, r) for E, (_, r) in runs.points.items()], spec)

    grid = [_round(E) for E in np.geomspace(e_lo, e_hi, 4)][::-1]
    for k, E in enumerate(grid):
        if k == 0 or runs.affordable(E, e_hi):   # always try the top of the window
            runs.run(E, spec.ions_search, 'search')

    for _ in range(4):   # refinement steps, while the budget allows
        best = best_point()
        if best is None:
            return None
        Es = sorted(runs.points)
        i = Es.index(best)
        lower = Es[i - 1] if i > 0 else None
        upper = Es[i + 1] if i + 1 < len(Es) else None
        if upper is None:
            return best          # best at the top of the window: the hardware limit binds
        if lower is None:
            if best <= e_lo * 1.02:
                return best      # best at the bottom of the window
            lower = e_lo         # best is the lowest point tried: look below it
        if lower in runs.points and runs.points[lower][1] is None:
            # Feasibility edge between an infeasible lower point and the best one: bisect towards it
            if best / lower < 1.05:
                return best
            candidates = [math.sqrt(lower * best)]
        else:
            # Interior optimum: try the midpoints on both sides
            if upper / lower < 1.05:
                return best
            candidates = [math.sqrt(lower * best), math.sqrt(best * upper)]
        candidates = [E for E in map(_round, candidates) if E not in runs.points]
        if not candidates:
            return best
        for E in candidates:
            if not runs.affordable(E, best):
                return best
            runs.run(E, spec.ions_search, 'search')
    return best_point()


def _flags(p: core.DamageProfile, r: dict, E: float, e_lo: float, e_hi: float,
           search_r: Optional[dict], objective: str) -> List[str]:
    flags = []
    if p.bin_nm > r['roi_width_nm'] / 10:
        flags.append(f'Coarse depth bins: {p.bin_nm:.0f} nm bins across a {r["roi_width_nm"]:.0f} nm ROI.')
    if p.damaged_bins < 20:
        flags.append(f'Only {p.damaged_bins} depth bins contain damage; the profile is coarse.')
    if p.Tdam_balance_eV > 0 and abs(p.Tdam_eV - p.Tdam_balance_eV) / p.Tdam_balance_eV > 0.05:
        flags.append('Damage energy from PHONON.txt and the energy balance differ by more than 5 %.')
    if E >= e_hi * 0.97:
        flags.append(f'The optimum is at the highest energy allowed ({e_hi:g} MeV): the accelerator limit binds.')
    elif E <= e_lo * 1.03:
        flags.append(f'The optimum is at the lowest energy allowed ({e_lo:g} MeV).')
    if p.meta.get('too_thin'):
        flags.append('Ions reached the back of the target even after making it thicker.')
    if search_r is not None:
        a, b = objective_value(search_r, objective), objective_value(r, objective)
        if a and b and abs(a - b) / max(abs(b), 1e-30) > 0.15:
            flags.append(f'The {objective} at {E:g} MeV changed by {abs(a - b) / abs(b):.0%} between the '
                         f'300-ion search run and the confirmation run: TRIM noise is significant here.')
    return flags


def plan(material: Union[str, core.Material], ion: str, spec: BeamSpec,
         srim_dir: str = core.DEFAULT_SRIM_DIR, cache_dir: str = core.CACHE_DIR,
         progress: Optional[ProgressFn] = None, surrogate=None, minimised: bool = False) -> PlanResult:
    """One-click optimiser: best energy and fluence for the spec, confirmed with a >= 1000-ion TRIM run.

    `material` is a Material or a MATERIALS name. `progress(stage, step, total, message)` is called before
    every TRIM run. If a trained surrogate is available (passed in, or saved by srim_surrogate.py) its
    proposal is confirmed directly; otherwise the energy is searched with 300-ion runs.
    `minimised=True` runs TRIM in a minimised, low-priority window (for background use).
    """
    m, name = _resolve_material(material)
    result = PlanResult(ok=False, message='', material=name, ion=ion, spec=asdict(spec))
    problems = spec.validate() + m.validate()
    if problems:
        result.message = ' '.join(problems)
        return result

    e_lo, e_hi, notes = spec.energy_limits(ion)
    result.energy_window_MeV = (e_lo, e_hi)
    result.flags += notes
    table = core.sr_table(m, ion, srim_dir=srim_dir)
    e_feas = feasible_from_MeV(table, spec, e_lo, e_hi)
    if e_feas is None:
        result.message = _infeasible_message(table, spec, ion, name, e_hi)
        return result

    lo_E = max(e_lo, e_feas)
    n_confirm = max(spec.ions_confirm, spec.ions_search)
    runs = _Runs(m, ion, spec, srim_dir, cache_dir, table, progress, minimised)
    proposal = _surrogate_proposal(surrogate, m, name, ion, spec, lo_E, e_hi)
    if proposal is not None:
        E_best, result.curve = proposal['energy_MeV'], proposal['curve']
        result.prediction = {k: proposal['predicted'][k] for k in ('fluence', 'roi_start_nm', 'roi_end_nm',
                                                                  'roi_width_nm')}
        result.prediction.update(energy_MeV=E_best, band=proposal['band'], uncertainty=proposal['uncertainty'])
    else:
        E_best = _search(runs, lo_E, e_hi)
    if E_best is None:
        result.curve += runs.curve
        result.runs = runs.runs
        result.message = ('No energy between {:g} and {:g} MeV gives a valid ROI. Try a smaller minimum width, '
                          'a shallower start depth, a looser implant limit or flatness limit, or a higher '
                          'maximum energy.').format(lo_E, e_hi)
        result.elapsed_s = runs.elapsed()
        return result

    search_r = runs.points.get(_round(E_best), (None, None))[1]
    p, r = runs.run(E_best, n_confirm, 'confirm')
    if r is None and proposal is not None:
        # The surrogate was wrong here: search a narrow bracket around its proposal with TRIM
        E_local = _search(runs, max(lo_E, _round(E_best / 1.5)), min(e_hi, _round(E_best * 1.5)))
        if E_local is not None and E_local != E_best:
            E_best, search_r = E_local, runs.points.get(E_local, (None, None))[1]
            p, r = runs.run(E_best, n_confirm, 'confirm')
    if r is None:
        # The edge moved with better statistics: step up to the next feasible energy tried, or 5 % higher
        higher = sorted(E for E, (_, rr) in runs.points.items() if rr is not None and E > E_best)
        E_next = higher[0] if higher else min(_round(E_best * 1.05), e_hi)
        if E_next != E_best:
            E_best, search_r = E_next, runs.points.get(E_next, (None, None))[1]
            p, r = runs.run(E_best, max(spec.ions_confirm, spec.ions_search), 'confirm')
    result.curve += runs.curve
    result.runs = runs.runs
    result.elapsed_s = runs.elapsed()
    if r is None:
        result.message = (f'The best search energy ({E_best:g} MeV) did not give a valid ROI in the '
                          f'{spec.ions_confirm}-ion confirmation run; loosen the limits slightly and run again.')
        return result

    result.ok = True
    for k in ('fluence', 'fluence_min', 'fluence_max', 'roi_start_nm', 'roi_end_nm', 'roi_width_nm',
              'dpa_min_in_roi', 'dpa_max_in_roi', 'peak_dpa', 'max_implanted_appm_in_roi', 'implant_peak_nm',
              'roi_limit_nm', 'bin_nm'):
        setattr(result, k, r[k])
    result.energy_MeV = E_best
    result.confirm_cache_key = p.meta.get('cache_key')
    if result.prediction is not None:
        # How well the surrogate knew the dose: its fluence for the confirmed energy and ROI against TRIM's.
        # The confirmation can choose a different ROI when two windows are nearly equally good, and then the
        # predicted and confirmed fluences belong to different windows.
        pred = result.prediction
        q = _load_surrogate(surrogate).predict_profiles(m, ion, [E_best * 1e6], table=table)[0]
        x = np.linspace(r['roi_start_nm'], r['roi_end_nm'], 50)
        f_same = spec.target_dpa / float(np.interp(x, q.depth_nm, q.dpa(1.0)).mean())
        overlap = min(r['roi_end_nm'], pred['roi_end_nm']) - max(r['roi_start_nm'], pred['roi_start_nm'])
        pred.update(fluence_on_confirmed_roi=f_same, fluence_error=r['fluence'] / f_same - 1,
                    roi_moved=bool(overlap < 0.5 * r['roi_width_nm']))
    if spec.facility:
        result.beam_setting = beam_setting(spec.facility, E_best, ion, spec.q_max)
    result.flags += _flags(p, r, E_best, max(e_lo, e_feas), e_hi, search_r, spec.objective)
    result.provenance = {
        'damage_model': 'NRT dpa from SRIM Quick Kinchin-Pease damage energy (Stoller et al. 2013, erratum 2019)',
        'code': 'SRIM-2013 TRIM via pysrim',
        'displacement_energies_eV': {c.symbol: c.E_d for c in p.material.constituents},
        'nrt_Ed_eV': p.material.nrt_Ed,
        'target_thickness_nm': p.material.width_A / 10,
        'ions_confirm': p.number_ions,
        'proposal': 'surrogate' if proposal is not None else 'search',
    }
    setting = result.beam_setting
    result.message = (f'{ion} at {E_best:g} MeV, {r["fluence"]:.3g} ions/cm2: mean {spec.target_dpa:g} dpa over '
                      f'{r["roi_start_nm"]:.0f}-{r["roi_end_nm"]:.0f} nm ({r["roi_width_nm"]:.0f} nm wide), '
                      f'{r["dpa_min_in_roi"]:.2f}-{r["dpa_max_in_roi"]:.2f} dpa across it'
                      + (f'; {ion}{setting["charge_state"]}+ at {setting["terminal_MV"]:g} MV terminal'
                         if setting else '') + '.')
    return result


MAX_UNCERTAINTY = 0.5   # surrogate spread (relative to the training spread) above which TRIM searches instead


def _load_surrogate(surrogate):
    # surrogate=None: the saved model if there is one; False: never; otherwise the model passed in
    if surrogate is False:
        return None
    if surrogate is not None:
        return surrogate
    try:
        import srim_surrogate
        return srim_surrogate.load_latest()
    except Exception:   # no scikit-learn, or an unreadable model file: search with TRIM instead
        return None


def _surrogate_proposal(surrogate, material, name, ion, spec, e_lo, e_hi, n_energies: int = 60,
                        n_samples: int = 30) -> Optional[dict]:
    # Best energy from the surrogate's predicted profiles, or None if no model is trained or it is unsure
    model = _load_surrogate(surrogate)
    if model is None:
        return None
    table = core.sr_table(material, ion)
    energies = np.geomspace(e_lo, e_hi, n_energies)
    profiles = model.predict_profiles(material, ion, energies * 1e6, table=table)
    results = [evaluate(p, spec) for p in profiles]
    best_E = pick_predicted(list(zip(range(len(results)), results)), spec)   # indices stand in for energies
    if best_E is None:
        return None   # the surrogate sees no valid ROI: let TRIM check that
    best = int(best_E)
    unc = profiles[best].meta['uncertainty']
    if not profiles[best].meta['in_domain'] or unc > MAX_UNCERTAINTY:
        return None
    # Uncertainty from draws of the predictive distribution at the chosen energy: the fluence for the chosen
    # window itself, and how much the best window's width moves when each draw is optimised on its own
    r_best = results[best]
    samples = model.predict_profiles(material, ion, [energies[best] * 1e6], table=table, n_samples=n_samples)
    x = np.linspace(r_best['roi_start_nm'], r_best['roi_end_nm'], 50)
    fluences = [spec.target_dpa / np.interp(x, p.depth_nm, p.dpa(1.0)).mean() for p in samples]
    redone = [evaluate(p, spec) for p in samples]
    ok = [r for r in redone if r is not None]
    band = {'fluence': [float(v) for v in np.percentile(fluences, [10, 90])],
            'feasible_fraction': len(ok) / len(redone)}
    q = model.info.get('fluence_log_q80')
    if q:
        # The model's own spread is too narrow: widen it to what held-out ion-target pairs showed (80 % of
        # them were inside this factor of the prediction)
        f = r_best['fluence']
        band['fluence'] = [min(band['fluence'][0], f * math.exp(-q)), max(band['fluence'][1], f * math.exp(q))]
        band['calibrated_on_heldout_pairs'] = True
    if ok:
        band['roi_width_nm'] = [float(v) for v in np.percentile([r['roi_width_nm'] for r in ok], [10, 90])]
    curve = [{'energy_MeV': float(E), 'objective': objective_value(r, spec.objective), 'feasible': r is not None,
              'source': 'surrogate'} for E, r in zip(energies, results)]
    return {'energy_MeV': _round(energies[best]), 'curve': curve, 'predicted': results[best], 'band': band,
            'uncertainty': unc}


def screen(material: Union[str, core.Material], ion: str, spec: BeamSpec,
           srim_dir: str = core.DEFAULT_SRIM_DIR, surrogate=None) -> ScreenResult:
    # Instant answer without TRIM: the feasible energy window from SR ranges (and, once trained, the
    # surrogate's predicted best energy, fluence and ROI)
    m, name = _resolve_material(material)
    result = ScreenResult(ok=False, message='', material=name, ion=ion, spec=asdict(spec))
    problems = spec.validate() + m.validate()
    if problems:
        result.message = ' '.join(problems)
        return result
    e_lo, e_hi, notes = spec.energy_limits(ion)
    result.energy_window_MeV, result.flags = (e_lo, e_hi), notes
    table = core.sr_table(m, ion, srim_dir=srim_dir)
    e_feas = feasible_from_MeV(table, spec, e_lo, e_hi)
    if e_feas is None:
        result.message = _infeasible_message(table, spec, ion, name, e_hi)
        return result
    result.ok, result.feasible_from_MeV = True, max(e_lo, e_feas)
    proposal = _surrogate_proposal(surrogate, m, name, ion, spec, result.feasible_from_MeV, e_hi)
    if proposal is None:
        result.message = (f'Energies from about {result.feasible_from_MeV:.3g} to {e_hi:g} MeV can leave room for '
                          f'the ROI; run plan() to find the best one with TRIM.')
        return result
    r, band = proposal['predicted'], proposal['band']
    result.source, result.curve = 'surrogate', proposal['curve']
    result.best = {**{k: r[k] for k in ('fluence', 'fluence_min', 'fluence_max', 'roi_start_nm', 'roi_end_nm',
                                        'roi_width_nm', 'dpa_min_in_roi', 'dpa_max_in_roi', 'peak_dpa',
                                        'max_implanted_appm_in_roi')},
                   'energy_MeV': proposal['energy_MeV'], 'band': band, 'uncertainty': proposal['uncertainty']}
    if spec.facility:
        result.best['beam_setting'] = beam_setting(spec.facility, proposal['energy_MeV'], ion, spec.q_max)
    fl = band.get('fluence')
    result.message = (f'Predicted (not yet verified): {ion} at {proposal["energy_MeV"]:g} MeV, {r["fluence"]:.3g} '
                      f'ions/cm2' + (f' (80 % band {fl[0]:.2g}-{fl[1]:.2g})' if fl else '')
                      + f', ROI {r["roi_start_nm"]:.0f}-{r["roi_end_nm"]:.0f} nm. Run plan() to confirm with TRIM.')
    return result
