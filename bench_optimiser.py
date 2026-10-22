# Optimiser benchmark: how many TRIM runs (and how much TRIM time) each strategy needs to choose the beam
# energy, and how good its choice is, for five realistic cases.
#
# Each case has dense reference sweeps (12 energies, at 300 and at 1000 ions, made by srim_dataset.py). Every
# strategy "runs TRIM" by looking up the nearest reference energy inside the case's energy window, from the
# sweep with the ion count it asked for, and pays the TRIM time measured for that run, so the comparison
# costs no new TRIM time. Choices are scored on the 1000-ion sweep. The surrogate is retrained without the
# five benchmark ion-target pairs, so it predicts combinations it has never seen.
#
#   python bench_optimiser.py             comparison table, saved to srim_outputs/benchmark/
#   python bench_optimiser.py --confirm   also run the real one-click plan() for each case

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import srim_core as core
import srim_dataset as dataset
import srim_optimiser as opt
import srim_surrogate as sur

OUT_DIR = os.path.join(core.OUTPUT_DIR, 'benchmark')

CASES: List[Tuple[str, str, opt.BeamSpec]] = [
    ('Cu', 'CuCrZr', opt.BeamSpec(target_dpa=3.0, min_depth_nm=100, min_width_nm=100, implant_margin_nm=100,
                                  e_min_MeV=1, e_max_MeV=10)),
    ('Fe', 'Fe-9Cr', opt.BeamSpec(target_dpa=10.0, min_depth_nm=200, min_width_nm=300, implant_margin_nm=200,
                                  facility='HZDR 3 MV Tandetron')),
    ('W', 'W', opt.BeamSpec(target_dpa=1.0, min_depth_nm=200, min_width_nm=300, implant_margin_nm=200,
                            facility='Helsinki 5 MV tandem')),
    ('Ni', 'Ni', opt.BeamSpec(target_dpa=5.0, min_depth_nm=150, min_width_nm=200, implant_margin_nm=150,
                              objective='fluence', e_min_MeV=1, e_max_MeV=20)),
    ('Si', 'SiC', opt.BeamSpec(target_dpa=1.0, min_depth_nm=200, min_width_nm=300, implant_margin_nm=200,
                               e_min_MeV=1, e_max_MeV=30)),
]


TRUTH_IONS = 1000


class Oracle:
    # Stands in for TRIM: answers from the reference sweeps at the nearest energy in the window (the 300-ion
    # sweep for runs of fewer than 1000 ions, the 1000-ion sweep otherwise), and keeps the bill
    def __init__(self, sweeps: Dict[int, Dict[float, core.DamageProfile]], spec: opt.BeamSpec,
                 e_lo: float, e_hi: float):
        self.sweeps = {n: {E: p for E, p in s.items() if e_lo * 0.999 <= E <= e_hi * 1.001}
                       for n, s in sweeps.items()}
        self.spec = spec
        self.energies = np.array(sorted(self.sweeps[TRUTH_IONS]))
        self.points: Dict[float, tuple] = {}
        self.bill: Dict[Tuple[float, int], int] = {}   # (energy, ions) of every run paid for
        self.path = ''                                 # how the strategy chose (for the surrogate: used, or fell back)

    def snap(self, E: float) -> float:
        return float(self.energies[np.argmin(np.abs(np.log(self.energies / E)))])

    def profile(self, E: float, n: int) -> core.DamageProfile:
        return self.sweeps[dataset.TRAIN_IONS if n < TRUTH_IONS else TRUTH_IONS][E]

    def truth(self, E: float) -> Optional[dict]:
        return opt.evaluate(self.sweeps[TRUTH_IONS][E], self.spec)

    def run(self, E: float, n: int, stage: str):
        E = self.snap(E)
        p = self.profile(E, n)
        r = opt.evaluate(p, self.spec)
        self.bill[(E, n)] = n
        if stage == 'search':
            self.points[E] = (p, r)
        return p, r

    def affordable(self, E: float, confirm_E: float) -> bool:
        return True

    def cost(self) -> Tuple[int, int, float]:
        # (TRIM runs, ion histories, minutes) using the measured time of the reference run at each energy
        minutes = sum(self.profile(E, n).meta.get('wall_s', 0.0) * n / self.profile(E, n).number_ions / 60
                      for (E, n) in self.bill)
        return len(self.bill), sum(self.bill.values()), minutes


def grid_strategy(o: Oracle, e_lo: float, e_hi: float) -> Optional[float]:
    # The app's manual optimiser: 6 log-spaced energies at 1000 ions, then 2 more either side of the best
    grid = sorted({o.snap(E) for E in np.geomspace(e_lo, e_hi, 6)})
    pts = [(E, o.run(E, 1000, 'grid')[1]) for E in grid]
    best = opt.pick_best(pts, o.spec)
    if best is None:
        return None
    k = grid.index(best)
    extra = [o.snap(math.sqrt(best * grid[n])) for n in (k - 1, k + 1) if 0 <= n < len(grid)]
    pts += [(E, o.run(E, 1000, 'grid')[1]) for E in extra if E not in grid]
    return opt.pick_best(pts, o.spec)


def search_strategy(o: Oracle, e_lo: float, e_hi: float) -> Optional[float]:
    # One-click without a surrogate: 300-ion search (srim_optimiser._search), then a 1000-ion confirmation
    E = opt._search(o, e_lo, e_hi)
    if E is not None:
        o.run(E, 1000, 'confirm')
    return E if E is None else o.snap(E)


def surrogate_strategy(o: Oracle, model: sur.Surrogate, material: core.Material, ion: str,
                       e_lo: float, e_hi: float) -> Optional[float]:
    # One-click with the surrogate: choose among the reference energies from predictions alone, confirm once,
    # and search around the proposal with TRIM only if the confirmation finds no valid ROI. Like plan(), it
    # searches with TRIM instead when the surrogate sees no valid ROI or is unsure.
    window = [float(E) for E in o.energies]
    preds = model.predict_profiles(material, ion, np.array(window) * 1e6)
    E = opt.pick_predicted([(E, opt.evaluate(p, o.spec)) for E, p in zip(window, preds)], o.spec)
    if E is None:
        o.path = 'search (surrogate sees no valid ROI)'
        return search_strategy(o, e_lo, e_hi)
    p = preds[window.index(E)]
    if not p.meta['in_domain'] or p.meta['uncertainty'] > opt.MAX_UNCERTAINTY:
        o.path = f'search (surrogate unsure: {p.meta["uncertainty"]:.2f})'
        return search_strategy(o, e_lo, e_hi)
    o.path = 'surrogate'
    _, r = o.run(E, 1000, 'confirm')
    if r is None:
        E2 = opt._search(o, max(e_lo, E / 1.5), min(e_hi, E * 1.5))
        if E2 is not None and o.snap(E2) != E:
            E = o.snap(E2)
            o.run(E, 1000, 'confirm')
    return E


def reference_sweep(ion: str, target: str, n_ions: int, runs=None) -> Dict[float, core.DamageProfile]:
    wanted = {float(f'{E:.3g}') for E in dataset.BENCHMARK_ENERGIES_MEV}
    refs = {}
    for name, p, _ in runs if runs is not None else sur.load_runs():
        E = float(f'{p.energy_eV / 1e6:.3g}')
        if name == target and p.ion_symbol == ion and E in wanted and p.number_ions == n_ions:
            refs[E] = p
    return refs


def benchmark() -> pd.DataFrame:
    runs = sur.load_runs()
    held_pairs = {(t, ion) for ion, t, _ in CASES}
    print(f'Training a surrogate without the {len(held_pairs)} benchmark ion-target pairs...', flush=True)
    model = sur.fit_family([r for r in runs if (r[0], r[1].ion_symbol) not in held_pairs], 'heavy')
    rows = []
    for ion, target, spec in CASES:
        sweeps = {n: reference_sweep(ion, target, n, runs) for n in dataset.BENCHMARK_IONS}
        e_lo, e_hi, _ = spec.energy_limits(ion)
        needed = sum(e_lo * 0.999 <= float(f'{E:.3g}') <= e_hi * 1.001 for E in dataset.BENCHMARK_ENERGIES_MEV)
        have = {n: sum(e_lo * 0.999 <= E <= e_hi * 1.001 for E in s) for n, s in sweeps.items()}
        if min(have.values()) < needed:
            print(f'{ion} -> {target}: reference sweeps not finished yet ({have} of {needed} energies); skipped')
            continue
        truth = Oracle(sweeps, spec, e_lo, e_hi)
        in_window = [float(E) for E in truth.energies]
        best_true = opt.pick_best([(E, truth.truth(E)) for E in in_window], spec)
        best_val = opt.objective_value(truth.truth(best_true), spec.objective) if best_true else None
        material = core.MATERIALS[target]
        for label, fn in [('grid: 6+2 runs at 1000 ions (app today)', lambda o: grid_strategy(o, e_lo, e_hi)),
                          ('search: 300-ion runs + 1 confirmation', lambda o: search_strategy(o, e_lo, e_hi)),
                          ('surrogate: predicted, then 1 confirmation',
                           lambda o: surrogate_strategy(o, model, material, ion, e_lo, e_hi))]:
            o = Oracle(sweeps, spec, e_lo, e_hi)
            chosen = fn(o)
            n_runs, histories, minutes = o.cost()
            val = opt.objective_value(o.truth(chosen), spec.objective) if chosen else None
            regret = (None if val is None or best_val is None else
                      (best_val - val) / best_val if spec.objective == 'width' else (val - best_val) / best_val)
            rows.append({'case': f'{ion} -> {target}', 'objective': spec.objective, 'strategy': label,
                         'path': o.path, 'trim_runs': n_runs, 'ion_histories': histories,
                         'trim_minutes': round(minutes, 1),
                         'chosen_MeV': chosen, 'best_MeV': best_true, 'objective_value': val,
                         'best_value': best_val, 'shortfall_vs_best': regret})
    return pd.DataFrame(rows)


def confirm_cases() -> pd.DataFrame:
    rows = []
    for ion, target, spec in CASES:
        print(f'plan(): {ion} -> {target} ...', flush=True)
        r = opt.plan(target, ion, spec, minimised=True)
        pred = r.prediction or {}
        rows.append({'case': f'{ion} -> {target}', 'ok': r.ok, 'energy_MeV': r.energy_MeV, 'fluence': r.fluence,
                     'roi_nm': f'{r.roi_start_nm:.0f}-{r.roi_end_nm:.0f}' if r.ok else '',
                     'trim_runs': len(r.runs), 'seconds': round(r.elapsed_s), 'proposal': r.provenance.get('proposal'),
                     'surrogate_fluence_error': pred.get('fluence_error'), 'roi_moved': pred.get('roi_moved'),
                     'message': r.message})
        print('   ' + r.message, flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Benchmark the beam-energy search strategies')
    parser.add_argument('--confirm', action='store_true', help='also run the real plan() for each case')
    args = parser.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    df = benchmark()
    if len(df):
        df.to_csv(os.path.join(OUT_DIR, 'benchmark.csv'), index=False)
        with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.max_colwidth', 45):
            print(df[['case', 'strategy', 'path', 'trim_runs', 'ion_histories', 'trim_minutes', 'chosen_MeV',
                      'best_MeV', 'shortfall_vs_best']].to_string(index=False))
        summary = df.groupby('strategy')[['trim_runs', 'ion_histories', 'trim_minutes']].sum()
        print('\nTotals over the cases:\n' + summary.to_string())
    if args.confirm:
        c = confirm_cases()
        c.to_csv(os.path.join(OUT_DIR, 'confirmed_plans.csv'), index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
