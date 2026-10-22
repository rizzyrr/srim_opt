# Surrogate for TRIM: the damage-energy and implanted-ion depth profiles for a heavy ion, target and
# energy in milliseconds, so the optimiser can scan energies before it runs TRIM once to confirm.
#
# What the physics already fixes is taken out before anything is learnt:
#   fluence    dpa is linear in it                    -> applied afterwards
#   E_d        NRT dpa = 0.8 T_dam / (2 E_d)          -> the model predicts damage energy, not dpa
#   depth      SR gives the projected range Rp(E)     -> profiles live on u = depth / Rp
#   magnitude  totals are split from shapes           -> damage-energy fraction and stopped fraction
#   partition  SR's nuclear share of the stopping     -> the model learns TRIM's damage energy relative to it
# The shapes left on the u grid are compressed with PCA, and a Gaussian process learns the PCA weights
# and the two totals from four physics inputs: ZBL reduced energy, mass ratio M1/M2, Z1 and Z2.
#
#   python srim_surrogate.py fit              train on every run so far, print held-out errors, save
#   python srim_surrogate.py validate         unseen ion-target pairs, then whole materials left out
#                                             (Cu + CuCrZr, W, SiC)
#   python srim_surrogate.py learning-curve   held-out error against the number of training runs
#   python srim_surrogate.py few-shot         a new material: error after 0, 1, 2, 4, 8 of its runs

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import srim_core as core

MODEL_DIR = os.path.join(core.OUTPUT_DIR, 'surrogate')
MODEL_PATH = os.path.join(MODEL_DIR, 'surrogate.npz')
U_EDGES = np.linspace(0.0, 2.0, 81)          # depth / projected range
DU = float(U_EDGES[1] - U_EDGES[0])
OUT_EDGES = np.linspace(0.0, 2.0, 161)       # predicted profiles come out on a finer grid, so an ROI edge at
DU_OUT = float(OUT_EDGES[1] - OUT_EDGES[0])  # high energy is not rounded to whole 1/40-of-Rp bins
MIN_IONS = 300
HOLDOUT_GROUPS = {'Cu + CuCrZr': ['Cu', 'CuCrZr'], 'W': ['W'], 'SiC': ['SiC']}
GROUPS = ('totals', 'damage', 'implant')   # one Gaussian process each


# ---------------------------------------------------------------------------
# Physics inputs
# ---------------------------------------------------------------------------

def target_ZM(material: core.Material) -> Tuple[float, float]:
    # Atom-fraction mean atomic number and mass of the target
    z = np.array([core.element_data(c.symbol)['z'] for c in material.constituents], dtype=float)
    return float(np.sum(material.fractions * z)), float(np.sum(material.fractions * material.masses))


def reduced_energy(Z1: float, M1: float, Z2: float, M2: float, energy_eV) -> np.ndarray:
    # ZBL reduced energy (dimensionless) with the universal screening length
    return 32.53 * M2 * np.asarray(energy_eV) / 1e3 / (Z1 * Z2 * (M1 + M2) * (Z1 ** 0.23 + Z2 ** 0.23))


def features(material: core.Material, ion_symbol: str, energies_eV) -> np.ndarray:
    ion = core.element_data(ion_symbol)
    Z1, M1 = float(ion['z']), float(ion['mass'])
    Z2, M2 = target_ZM(material)
    eps = reduced_energy(Z1, M1, Z2, M2, np.atleast_1d(energies_eV))
    n = eps.size
    return np.column_stack([np.log10(eps), np.full(n, np.log10(M1 / M2)), np.full(n, Z1), np.full(n, Z2)])


def lindhard_eta(Z1: float, M1: float, Z2: float, M2: float, energy_eV) -> np.ndarray:
    # Damage-energy fraction T_dam / E from the Lindhard partition with Robinson's fit (as in the NRT
    # formula), applied to the ion itself. TRIM's value differs from it by a smooth factor of order one.
    z23 = Z1 ** (2 / 3) + Z2 ** (2 / 3)
    E_L = 30.724 * Z1 * Z2 * math.sqrt(z23) * (M1 + M2) / M2           # eV
    k = 0.0793 * Z1 ** (2 / 3) * Z2 ** 0.5 * (M1 + M2) ** 1.5 / (z23 ** 0.75 * M1 ** 1.5 * M2 ** 0.5)
    eps = np.asarray(energy_eV, dtype=float) / E_L
    return 1.0 / (1.0 + k * (3.4008 * eps ** (1 / 6) + 0.40244 * eps ** 0.75 + eps))


def sr_nuclear_fraction(table: core.RangeTable, energies_eV) -> np.ndarray:
    # Fraction of its energy the ion loses in nuclear collisions while slowing from E to rest, from SR's
    # stopping powers. They carry the Z1 oscillations of electronic stopping that Lindhard's smooth formula
    # misses (Kr and Xe sit off its trend); what is left for the model is the share of recoil energy that
    # becomes damage energy (about 0.5-0.7, smooth in the target).
    e = table.energy_eV
    f = table.Sn_eV_per_A / (table.Sn_eV_per_A + table.Se_eV_per_A)
    cum = e[0] * f[0] + np.concatenate(([0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(e))))
    E = np.atleast_1d(np.asarray(energies_eV, dtype=float))
    return np.interp(E, e, cum) / E


ETA_BASELINES = ('sr', 'lindhard', None)


def log_eta_baseline(material: core.Material, ion_symbol: str, energies_eV, kind: Optional[str] = 'sr',
                     table: Optional[core.RangeTable] = None) -> np.ndarray:
    # log of the damage-energy fraction the model is measured against: 'sr' (nuclear stopping share from
    # SR), 'lindhard' (Lindhard-Robinson for the ion), or None (the model learns log(T_dam / E) directly)
    E = np.atleast_1d(np.asarray(energies_eV, dtype=float))
    kind = {'lindhard-robinson': 'lindhard'}.get(kind, kind)     # name used by models saved earlier
    if kind is None:
        return np.zeros(E.size)
    if kind == 'sr':
        return np.log(sr_nuclear_fraction(table or core.sr_table(material, ion_symbol), E))
    ion = core.element_data(ion_symbol)
    Z2, M2 = target_ZM(material)
    return np.log(lindhard_eta(float(ion['z']), float(ion['mass']), Z2, M2, E))


# ---------------------------------------------------------------------------
# Training data: TRIM profiles -> totals and shapes on the u grid
# ---------------------------------------------------------------------------

def shape_on_u(depth_hi_nm: np.ndarray, density: np.ndarray, Rp_nm: float) -> Tuple[np.ndarray, float]:
    # Area-preserving resampling of a binned depth density onto u = depth / Rp. TRIM depths are the
    # bins' upper edges. Returns the shape (area 1 over all depths) and the total area (density x nm).
    dx = float(np.mean(np.diff(depth_hi_nm)))
    edges = np.concatenate(([depth_hi_nm[0] - dx], depth_hi_nm))
    cum = np.concatenate(([0.0], np.cumsum(np.clip(density, 0, None) * dx)))
    total = float(cum[-1])
    if total <= 0:
        return np.zeros(len(U_EDGES) - 1), 0.0
    return np.diff(np.interp(U_EDGES * Rp_nm, edges, cum)) / total / DU, total


def _refine(shape: np.ndarray) -> np.ndarray:
    # A shape on the U_EDGES bins, linearly interpolated onto the OUT_EDGES bins with its area kept
    centres = lambda e: (e[:-1] + e[1:]) / 2      # noqa: E731
    fine = np.interp(centres(OUT_EDGES), centres(U_EDGES), shape)
    area = float(np.sum(fine)) * DU_OUT
    return fine * (float(np.sum(shape)) * DU / area) if area > 0 else fine


@dataclass
class TrainingSet:
    targets: np.ndarray       # material names
    ions: np.ndarray
    energy_eV: np.ndarray
    X: np.ndarray             # physics inputs (n, 4)
    log_eta: np.ndarray       # log(damage energy / beam energy)
    stopped: np.ndarray       # fraction of ions stopped in the target
    S: np.ndarray             # damage-energy shapes on the u grid (n, 80)
    G: np.ndarray             # implanted-ion shapes on the u grid (n, 80)
    order: np.ndarray         # file time of each run (for learning curves)

    def subset(self, mask: np.ndarray) -> 'TrainingSet':
        return TrainingSet(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))

    def __len__(self) -> int:
        return len(self.energy_eV)


def load_runs(min_ions: int = MIN_IONS) -> List[Tuple[str, core.DamageProfile, float]]:
    # (material name, profile, file time) for every run on a named material: the dataset plus the
    # optimiser's confirmation runs, so each confirmation run also improves the model
    out = []
    for cache in (core.DATASET_CACHE_DIR, core.CACHE_DIR):
        if not os.path.isdir(cache):
            continue
        times = {f[:-4]: os.path.getmtime(os.path.join(cache, f)) for f in os.listdir(cache) if f.endswith('.pkl')}
        for p in core.cached_profiles(min_ions=min_ions, cache_dir=cache):
            name = core.material_name(p.material)
            if name and not p.meta.get('too_thin'):
                out.append((name, p, times.get(p.meta.get('cache_key', ''), 0.0)))
    return out


def build_training_set(runs: Sequence[Tuple[str, core.DamageProfile, float]]) -> TrainingSet:
    rows = {f: [] for f in TrainingSet.__dataclass_fields__}
    for name, p, t in runs:
        Rp, _ = core.sr_table(p.material, p.ion_symbol).at(p.energy_eV)
        s, _ = shape_on_u(p.depth_nm, p.Tdam_density, Rp)
        g, imp_total = shape_on_u(p.depth_nm, p.implanted, Rp)
        for f, v in [('targets', name), ('ions', p.ion_symbol), ('energy_eV', p.energy_eV),
                     ('X', features(p.material, p.ion_symbol, p.energy_eV)[0]),
                     ('log_eta', math.log(max(p.Tdam_eV, 1e-9) / p.energy_eV)), ('stopped', imp_total * 1e-7),
                     ('S', s), ('G', g), ('order', t)]:
            rows[f].append(v)
    return TrainingSet(**{f: np.array(v) for f, v in rows.items()})


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def _kernel():
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    return (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e3),
                                                      nu=2.5)
            + WhiteKernel(1e-3, (1e-8, 1.0)))


def _gp(kernel, optimise: bool, restarts: int = 0):
    from sklearn.gaussian_process import GaussianProcessRegressor
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer='fmin_l_bfgs_b' if optimise else None,
                                    n_restarts_optimizer=restarts, random_state=0)


class _PerOutputGP:
    # One Gaussian process per output, so each PCA weight gets its own length scales and noise level (the
    # minor components are mostly TRIM noise; with a shared kernel their noise would smooth the main ones)

    def __init__(self, gps: list):
        self.gps = gps

    @classmethod
    def fit(cls, xs: np.ndarray, Y: np.ndarray, thetas: np.ndarray) -> '_PerOutputGP':
        return cls([_gp(_kernel().clone_with_theta(th), False).fit(xs, Y[:, j]) for j, th in enumerate(thetas)])

    def predict(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Mean and spread of the underlying profile: the fitted white noise is the Monte-Carlo scatter between
        # TRIM runs, not uncertainty about the profile, so it is taken out of the predictive spread
        means, stds = [], []
        for gp in self.gps:
            m, s = gp.predict(xs, return_std=True)
            noise = gp.kernel_.k2.noise_level * float(np.squeeze(getattr(gp, '_y_train_std', 1.0))) ** 2
            means.append(m)
            stds.append(np.sqrt(np.clip(s ** 2 - noise, 0, None)))
        return np.column_stack(means), np.column_stack(stds)


def _outputs(a: Dict[str, np.ndarray], ts: TrainingSet, eta_baseline: Optional[str]) -> Dict[str, np.ndarray]:
    # What the Gaussian processes learn for each run: log damage-energy fraction against the baseline and
    # the stopped fraction, then the PCA weights of the damage and implant shapes (basis in `a`)
    base = np.zeros(len(ts))
    if eta_baseline is not None:
        for t, i in set(zip(ts.targets, ts.ions)):
            rows = (ts.targets == t) & (ts.ions == i)
            base[rows] = log_eta_baseline(core.MATERIALS[t], i, ts.energy_eV[rows], eta_baseline)
    return {'totals': np.column_stack([ts.log_eta - base, ts.stopped]),
            'damage': (ts.S - a['damage_mean']) @ a['damage_comp'].T,
            'implant': (ts.G - a['implant_mean']) @ a['implant_comp'].T}


def _n_components(sv: np.ndarray, cap: int, keep: float = 0.995) -> int:
    frac = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    return int(min(cap, np.searchsorted(frac, keep) + 1))


class Surrogate:
    # PCA of the profile shapes + one Gaussian process per output, on standardised physics inputs

    def __init__(self, arrays: Dict[str, np.ndarray], info: dict, gps: Optional[dict] = None):
        self.a, self.info = arrays, info
        if gps is None:
            gps = {}
            for g in GROUPS:
                Y, thetas = arrays[f'Y_{g}'], np.atleast_2d(arrays[f'theta_{g}'])
                if len(thetas) == 1:    # a model saved with one kernel shared by the outputs
                    thetas = np.repeat(thetas, Y.shape[1], axis=0)
                gps[g] = _PerOutputGP.fit(self._xs(arrays['X']), Y, thetas)
        self.gps = gps

    # --- training --------------------------------------------------------
    @classmethod
    def fit(cls, ts: TrainingSet, max_hyper_points: int = 500, verbose: bool = False,
            eta_baseline: Optional[str] = 'sr', restarts: int = 0) -> 'Surrogate':
        # eta_baseline: what the damage energy is learnt relative to (see log_eta_baseline; the others are
        # for the ablation)
        if eta_baseline not in ETA_BASELINES:
            raise ValueError(f'eta_baseline must be one of {ETA_BASELINES}')
        started = time.time()
        a = {'X': ts.X, 'x_mean': ts.X.mean(0), 'x_std': ts.X.std(0) + 1e-12}
        for key, M, cap in (('damage', ts.S, 8), ('implant', ts.G, 6)):
            mean = M.mean(0)
            _, sv, vt = np.linalg.svd(M - mean, full_matrices=False)
            k = _n_components(sv, cap)
            a[f'{key}_mean'], a[f'{key}_comp'] = mean, vt[:k]
            a[f'{key}_var'] = np.array(float(np.sum(sv[:k] ** 2) / np.sum(sv ** 2)))
        for g, Y in _outputs(a, ts, eta_baseline).items():
            a[f'Y_{g}'] = Y
        xs = (ts.X - a['x_mean']) / a['x_std']
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xs), min(len(xs), max_hyper_points), replace=False)   # hyperparameters on a subset
        gps = {}
        for g in GROUPS:
            Y = a[f'Y_{g}']
            thetas = []
            for j in range(Y.shape[1]):
                with warnings.catch_warnings():   # hyperparameters at a bound are fine (an input that barely matters)
                    warnings.simplefilter('ignore')
                    thetas.append(_gp(_kernel(), True, restarts).fit(xs[idx], Y[idx, j]).kernel_.theta)
            a[f'theta_{g}'] = np.array(thetas)
            gps[g] = _PerOutputGP.fit(xs, Y, a[f'theta_{g}'])
            a[f'y_std_{g}'] = Y.std(0) + 1e-12
            if verbose:
                noise = [f'{gp.kernel_.k2.noise_level:.2g}' for gp in gps[g].gps]
                print(f'  {g:7}: {Y.shape[1]} outputs, noise share {", ".join(noise)}', flush=True)
        info = {'n_runs': len(ts), 'targets': sorted(set(ts.targets.tolist())), 'ions': sorted(set(ts.ions.tolist())),
                'energy_MeV': [float(ts.energy_eV.min() / 1e6), float(ts.energy_eV.max() / 1e6)],
                'x_min': ts.X.min(0).tolist(), 'x_max': ts.X.max(0).tolist(),
                'eta_baseline': eta_baseline,
                'components': {'damage': int(len(a['damage_comp'])), 'implant': int(len(a['implant_comp']))},
                'variance_kept': {'damage': float(a['damage_var']), 'implant': float(a['implant_var'])},
                'fit_seconds': round(time.time() - started, 1), 'trained': time.strftime('%Y-%m-%d %H:%M')}
        return cls(a, info, gps)

    def update(self, ts: TrainingSet) -> 'Surrogate':
        """This model refitted with extra runs, keeping its PCA basis, input scaling and kernels.

        Takes seconds rather than a full fit, so a confirmation run on a new material can be fed back
        straight away; a full fit (which also retunes the kernels) can follow later.
        """
        a = dict(self.a)
        a['X'] = np.vstack([a['X'], ts.X])
        for g, Y in _outputs(a, ts, self.info.get('eta_baseline')).items():
            a[f'Y_{g}'] = np.vstack([a[f'Y_{g}'], Y])
        info = dict(self.info, n_runs=self.info['n_runs'] + len(ts),
                    targets=sorted(set(self.info['targets']) | set(ts.targets.tolist())),
                    ions=sorted(set(self.info['ions']) | set(ts.ions.tolist())),
                    energy_MeV=[min(self.info['energy_MeV'][0], float(ts.energy_eV.min() / 1e6)),
                                max(self.info['energy_MeV'][1], float(ts.energy_eV.max() / 1e6))],
                    x_min=np.minimum(self.info['x_min'], ts.X.min(0)).tolist(),
                    x_max=np.maximum(self.info['x_max'], ts.X.max(0)).tolist(),
                    updated_with_runs=self.info.get('updated_with_runs', 0) + len(ts))
        return Surrogate(a, info)

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **self.a)
        with open(path.replace('.npz', '.json'), 'w') as f:
            json.dump(self.info, f, indent=1)

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> Optional['Surrogate']:
        # The saved model, refitted with its stored hyperparameters (portable: no pickled sklearn objects)
        if not os.path.exists(path):
            return None
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        with open(path.replace('.npz', '.json')) as f:
            info = json.load(f)
        return cls(arrays, info)

    # --- prediction ------------------------------------------------------
    def _xs(self, X: np.ndarray) -> np.ndarray:
        return (X - self.a['x_mean']) / self.a['x_std']

    def in_domain(self, X: np.ndarray, slack: float = 0.05) -> np.ndarray:
        # Inside the box of training inputs (plus a small slack): outside it the model extrapolates
        lo, hi = np.array(self.info['x_min']), np.array(self.info['x_max'])
        pad = slack * (hi - lo)
        return np.all((X >= lo - pad) & (X <= hi + pad), axis=1)

    def predict_outputs(self, X: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        means, stds = {}, {}
        for g in GROUPS:
            means[g], stds[g] = self.gps[g].predict(self._xs(X))
        return means, stds

    def uncertainty(self, stds: Dict[str, np.ndarray]) -> np.ndarray:
        # Predictive spread of the shape weights relative to their spread in the training set (0 = sure)
        return np.mean(np.hstack([stds[g] / self.a[f'y_std_{g}'] for g in ('damage', 'implant')]), axis=1)

    def predict_profiles(self, material: core.Material, ion_symbol: str, energies_eV: Sequence[float],
                         table: Optional[core.RangeTable] = None, n_samples: int = 0,
                         seed: int = 0) -> List[core.DamageProfile]:
        """Predicted DamageProfiles (per incident ion, like TRIM's) at each energy.

        With n_samples > 0, returns n_samples draws per energy from the predictive distribution instead of
        the mean (energy-major order), for uncertainty bands.
        """
        energies_eV = np.atleast_1d(np.asarray(energies_eV, dtype=float))
        table = table or core.sr_table(material, ion_symbol)
        X = features(material, ion_symbol, energies_eV)
        means, stds = self.predict_outputs(X)
        unc, inside = self.uncertainty(stds), self.in_domain(X)
        base = log_eta_baseline(material, ion_symbol, energies_eV, self.info.get('eta_baseline'), table)
        rng = np.random.default_rng(seed)
        profiles = []
        for i, E in enumerate(energies_eV):
            draws = [{g: means[g][i] for g in GROUPS}] if n_samples == 0 else [
                {g: rng.normal(means[g][i], stds[g][i]) for g in GROUPS} for _ in range(n_samples)]
            Rp, _ = table.at(E)
            for d in draws:
                profiles.append(self._profile(material, ion_symbol, E, Rp, d, float(base[i]), float(unc[i]),
                                              bool(inside[i])))
        return profiles

    def _profile(self, material, ion_symbol, energy_eV, Rp_nm, out, log_eta_base, unc, inside) -> core.DamageProfile:
        s = _refine(np.clip(self.a['damage_mean'] + out['damage'] @ self.a['damage_comp'], 0, None))
        g = _refine(np.clip(self.a['implant_mean'] + out['implant'] @ self.a['implant_comp'], 0, None))
        T_total = math.exp(out['totals'][0] + log_eta_base) * energy_eV
        stopped = float(np.clip(out['totals'][1], 0, 1))
        Tdam_density = T_total * s / (Rp_nm * 10)          # eV / A / ion (s has unit area over u)
        implanted = stopped * g / (Rp_nm * 1e-7)           # 1/cm per ion
        T_in_grid = float(np.sum(Tdam_density) * DU_OUT * Rp_nm * 10)
        return core.DamageProfile(
            material=material, ion_symbol=ion_symbol, energy_eV=float(energy_eV), number_ions=0,
            depth_nm=OUT_EDGES[1:] * Rp_nm, Tdam_density=Tdam_density,
            disp_density=0.8 * Tdam_density / (2 * material.nrt_Ed), implanted=implanted,
            Tdam_eV=T_in_grid, N_NRT=0.8 * T_in_grid / (2 * material.nrt_Ed), Tdam_balance_eV=T_in_grid,
            meta={'surrogate': True, 'uncertainty': unc, 'in_domain': inside, 'Rp_nm': Rp_nm})


_cached_model: Dict[str, Tuple[float, Surrogate]] = {}


def load_latest(path: str = MODEL_PATH) -> Optional[Surrogate]:
    # The saved model, reloaded only when the file changes
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    if path not in _cached_model or _cached_model[path][0] != mtime:
        _cached_model[path] = (mtime, Surrogate.load(path))
    return _cached_model[path][1]


# ---------------------------------------------------------------------------
# Checking it against TRIM
# ---------------------------------------------------------------------------

def _relative_window_rules(Rp_nm: float) -> dict:
    # A scale-free ROI test for any energy: depth limits as fractions of the range, +/-10 % flatness
    return dict(target_dpa=1.0, tol=0.05, min_width_nm=0.15 * Rp_nm, min_depth_nm=0.1 * Rp_nm,
                implant_margin_nm=0.1 * Rp_nm, objective='width', max_spread=0.10)


def compare(p_true: core.DamageProfile, p_pred: core.DamageProfile) -> dict:
    # Errors that matter for planning: the fluence for the same window, and the ROI width the rules allow
    Rp = p_pred.meta['Rp_nm']
    rules = _relative_window_rules(Rp)
    r_true, r_pred = core.roi_options(p_true, **rules), core.roi_options(p_pred, **rules)
    s_true, _ = shape_on_u(p_true.depth_nm, p_true.Tdam_density, Rp)
    s_pred, _ = shape_on_u(p_pred.depth_nm, p_pred.Tdam_density, Rp)
    out = {'shape_error': 0.5 * float(np.sum(np.abs(s_true - s_pred)) * DU),
           'damage_energy_error': p_pred.Tdam_eV / p_true.Tdam_eV - 1, 'fluence_error': np.nan,
           'width_error': np.nan, 'fluence_1dpa_trim': np.nan, 'fluence_1dpa_pred': np.nan}
    if r_true is not None:
        lo, hi = r_true['roi_start_nm'], r_true['roi_end_nm']
        x = np.linspace(lo, hi, 50)
        k_true = np.interp(x, p_true.depth_nm, p_true.dpa(1.0)).mean()
        k_pred = np.interp(x, p_pred.depth_nm, p_pred.dpa(1.0)).mean()
        out['fluence_error'] = k_true / k_pred - 1        # fluence ~ 1 / mean dpa per ion in the window
        out['fluence_1dpa_trim'], out['fluence_1dpa_pred'] = 1 / k_true, 1 / k_pred
        out['width_error'] = ((r_pred['roi_width_nm'] if r_pred else 0.0) - r_true['roi_width_nm']) / r_true['roi_width_nm']
    return out


def evaluate_on(model: Surrogate, runs: Sequence[Tuple[str, core.DamageProfile, float]]) -> List[dict]:
    rows = []
    for name, p, _ in runs:
        pred = model.predict_profiles(p.material, p.ion_symbol, [p.energy_eV])[0]
        rows.append({'target': name, 'ion': p.ion_symbol, 'energy_MeV': p.energy_eV / 1e6,
                     'uncertainty': pred.meta['uncertainty'], **compare(p, pred)})
    return rows


def summarise(rows: List[dict]) -> str:
    def pct(key, q):
        v = np.abs([r[key] for r in rows if np.isfinite(r[key])])
        return f'{np.percentile(v, q) * 100:5.1f}%' if v.size else '   --'
    with_window = sum(np.isfinite(r['fluence_error']) for r in rows)
    return (f'fluence error median {pct("fluence_error", 50)}, p90 {pct("fluence_error", 90)} | '
            f'ROI width error median {pct("width_error", 50)}, p90 {pct("width_error", 90)} | '
            f'damage energy error median {pct("damage_energy_error", 50)} | '
            f'shape error median {pct("shape_error", 50)} | n = {len(rows)} ({with_window} with a TRIM window)')


def split_pairs(runs, test_fraction: float = 0.15, seed: int = 1):
    # Whole ion-target pairs held out, so the test asks about combinations the model never saw
    pairs = sorted({(n, p.ion_symbol) for n, p, _ in runs})
    rng = np.random.default_rng(seed)
    test = {pairs[i] for i in rng.choice(len(pairs), max(1, int(test_fraction * len(pairs))), replace=False)}
    train = [r for r in runs if (r[0], r[1].ion_symbol) not in test]
    held = [r for r in runs if (r[0], r[1].ion_symbol) in test]
    return train, held


def band_quantile(rows: List[dict], q: float = 80) -> Optional[float]:
    # Split-conformal band: the q-th percentile of |log(true / predicted fluence)| on held-out pairs
    v = np.abs(np.log1p([r['fluence_error'] for r in rows if np.isfinite(r['fluence_error'])]))
    return float(np.percentile(v, q)) if v.size >= 5 else None


def cmd_fit() -> None:
    runs = load_runs()
    print(f'{len(runs)} training runs on {len({r[0] for r in runs})} targets.')
    if len(runs) < 30:
        print('Not enough runs yet; wait for srim_dataset.py run to make more.')
        return
    train, held = split_pairs(runs)
    t = time.time()
    model = Surrogate.fit(build_training_set(train))
    rows = evaluate_on(model, held)
    print(f'Held-out ion-target pairs ({len(held)} runs, model on {len(train)} runs, '
          f'{model.info["components"]} PCA components, fit {time.time() - t:.0f} s):')
    print('  ' + summarise(rows))
    final = Surrogate.fit(build_training_set(runs))
    q = band_quantile(rows)
    final.info['fluence_log_q80'] = q
    final.save()
    if q:
        print(f'80 % of held-out fluences were within x{math.exp(q):.2f} of the prediction: '
              f'the optimiser\'s fluence band is at least this wide.')
    print(f'Saved the model trained on all {len(runs)} runs to {MODEL_PATH}')


def cmd_validate() -> None:
    # Unseen ion-target pairs, then whole materials left out
    runs = load_runs()
    rows_all = []
    splits = {'unseen pairs': split_pairs(runs)}
    for group, names in HOLDOUT_GROUPS.items():
        splits[group] = ([r for r in runs if r[0] not in names], [r for r in runs if r[0] in names])
    for group, (train, held) in splits.items():
        if not held:
            print(f'{group:12}: no runs yet')
            continue
        model = Surrogate.fit(build_training_set(train))
        rows = evaluate_on(model, held)
        rows_all += [{'holdout': group, **r} for r in rows]
        print(f'{group:12} held out: ' + summarise(rows), flush=True)
    if rows_all:
        import pandas as pd
        path = os.path.join(MODEL_DIR, 'validation.csv')
        os.makedirs(MODEL_DIR, exist_ok=True)
        pd.DataFrame(rows_all).to_csv(path, index=False)
        print(f'Per-run errors: {path}')


def cmd_learning_curve() -> None:
    import pandas as pd
    runs = load_runs()
    train, held = split_pairs(runs)
    train = sorted(train, key=lambda r: r[2])     # in the order the runs finished
    rows = []
    for n in [n for n in (50, 100, 200, 400, 800, 1200, 1600) if n < len(train)] + [len(train)]:
        model = Surrogate.fit(build_training_set(train[:n]))
        res = evaluate_on(model, held)
        med = lambda k, q=50: float(np.nanpercentile(np.abs([r[k] for r in res]), q))   # noqa: E731
        rows.append({'training_runs': n, 'fluence_error_median': med('fluence_error'),
                     'fluence_error_p90': med('fluence_error', 90), 'width_error_median': med('width_error'),
                     'shape_error_median': med('shape_error'),
                     'damage_energy_error_median': med('damage_energy_error')})
        print(f'{n:5d} runs: ' + summarise(res), flush=True)
    path = os.path.join(MODEL_DIR, 'learning_curve.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'Saved {path}')


def cmd_few_shot(ks: Sequence[int] = (0, 1, 2, 4, 8), seeds: int = 4) -> None:
    # A material the model has never seen, then k of its TRIM runs fed back with update(): how quickly the
    # error falls. In use, each confirmation run on a new material is such a run.
    import pandas as pd
    runs = load_runs()
    rows = []
    for group, names in HOLDOUT_GROUPS.items():
        held = [r for r in runs if r[0] in names]
        rng = np.random.default_rng(0)
        order = rng.permutation(len(held))
        pool, test = [held[i] for i in order[:max(ks)]], [held[i] for i in order[max(ks):]]
        if len(test) < 10:
            print(f'{group}: only {len(held)} runs yet')
            continue
        model0 = Surrogate.fit(build_training_set([r for r in runs if r[0] not in names]))
        for k in ks:
            for seed in range(seeds if k else 1):
                add = [pool[i] for i in rng.choice(len(pool), k, replace=False)] if k else []
                model = model0.update(build_training_set(add)) if add else model0
                res = evaluate_on(model, test)
                pct = lambda key, q=50: float(np.nanpercentile(np.abs([r[key] for r in res]), q))   # noqa: E731
                rows.append({'holdout': group, 'runs_added': k, 'seed': seed, 'n_test': len(test),
                             'fluence_error_median': pct('fluence_error'), 'fluence_error_p90': pct('fluence_error', 90),
                             'damage_energy_error_median': pct('damage_energy_error'),
                             'shape_error_median': pct('shape_error')})
            sel = [r for r in rows if r['holdout'] == group and r['runs_added'] == k]
            print(f'{group:12} +{k} runs: fluence error median '
                  f'{100 * np.median([r["fluence_error_median"] for r in sel]):4.1f}% '
                  f'(seeds {100 * min(r["fluence_error_median"] for r in sel):.1f}-'
                  f'{100 * max(r["fluence_error_median"] for r in sel):.1f}%), '
                  f'damage energy {100 * np.median([r["damage_energy_error_median"] for r in sel]):4.1f}%, '
                  f'n_test {len(test)}', flush=True)
    path = os.path.join(MODEL_DIR, 'few_shot.csv')
    os.makedirs(MODEL_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'Saved {path}')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='TRIM surrogate: fit, validate, learning curve, few-shot')
    parser.add_argument('cmd', choices=['fit', 'validate', 'learning-curve', 'few-shot'])
    args = parser.parse_args(argv)
    {'fit': cmd_fit, 'validate': cmd_validate, 'learning-curve': cmd_learning_curve,
     'few-shot': cmd_few_shot}[args.cmd]()


if __name__ == '__main__':
    main(sys.argv[1:])
