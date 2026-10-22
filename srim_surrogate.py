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
# and the two totals from physics inputs: ZBL reduced energy, mass ratio M1/M2, Z1 and Z2 (v1), plus the
# relative straggling and the nuclear stopping share from SR (features 'sr').
#
# Two model families, each its own file: heavy ions (C to Au) and light ions (H and He), whose profiles (a long
# plateau, then a sharp end-of-range peak) would share a PCA basis badly with heavy ions'. Each model stores its
# own u grid; the light one has bins half the size, and works on the sqrt of the shapes.
#
#   python srim_surrogate.py fit [--family light]   train on every run so far, print held-out errors, save
#   python srim_surrogate.py validate         unseen ion-target pairs, then whole materials left out
#                                             (Cu + CuCrZr, W, SiC)
#   python srim_surrogate.py learning-curve   held-out error against the number of training runs
#   python srim_surrogate.py few-shot         a new material: error after 0, 1, 2, 4, 8 of its runs
#   python srim_surrogate.py evaluate-test    a saved model against the high-statistics test runs

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import srim_core as core

MODEL_DIR = os.path.join(core.OUTPUT_DIR, 'surrogate')
MODEL_PATH = os.path.join(MODEL_DIR, 'surrogate.npz')                # heavy ions
LIGHT_MODEL_PATH = os.path.join(MODEL_DIR, 'surrogate_light.npz')    # H and He
MODEL_PATHS = {'heavy': MODEL_PATH, 'light': LIGHT_MODEL_PATH}
U_EDGES = np.linspace(0.0, 2.0, 81)          # depth / projected range (v1's grid; a model file may hold its own)
DU = float(U_EDGES[1] - U_EDGES[0])
OUT_EDGES = np.linspace(0.0, 2.0, 161)       # predicted profiles come out on a finer grid, so an ROI edge at
DU_OUT = float(OUT_EDGES[1] - OUT_EDGES[0])  # high energy is not rounded to whole 1/40-of-Rp bins
LIGHT_U_EDGES = np.linspace(0.0, 2.0, 161)   # H and He: bins of TRIM's own size (0.016-0.04 Rp), twice v1's
MIN_IONS = 300
HOLDOUT_GROUPS = {'Cu + CuCrZr': ['Cu', 'CuCrZr'], 'W': ['W'], 'SiC': ['SiC']}
GROUPS = ('totals', 'damage', 'implant')   # one Gaussian process each
FEATURES = {'v1': 4, 'sr': 6}               # number of inputs of each feature set


def ion_family(ion_symbol: str) -> str:
    # 'light' for H and He, 'heavy' otherwise
    return 'light' if core.element_data(ion_symbol)['z'] <= 2 else 'heavy'


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


def features(material: core.Material, ion_symbol: str, energies_eV, kind: str = 'v1',
             table: Optional[core.RangeTable] = None) -> np.ndarray:
    # 'v1': log reduced energy, log M1/M2, Z1, Z2. 'sr' adds log10 of the relative straggling dRp/Rp and of the
    # nuclear stopping share from SR's tables, which encode SRIM's own stopping for the exact ion and target
    ion = core.element_data(ion_symbol)
    Z1, M1 = float(ion['z']), float(ion['mass'])
    Z2, M2 = target_ZM(material)
    E = np.atleast_1d(np.asarray(energies_eV, dtype=float))
    eps = reduced_energy(Z1, M1, Z2, M2, E)
    n = eps.size
    X = np.column_stack([np.log10(eps), np.full(n, np.log10(M1 / M2)), np.full(n, Z1), np.full(n, Z2)])
    if kind == 'v1':
        return X
    table = table or core.sr_table(material, ion_symbol)
    straggling = np.array([dRp / Rp for Rp, dRp in (table.at(e) for e in E)])
    return np.column_stack([X, np.log10(straggling), np.log10(sr_nuclear_fraction(table, E))])


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

def out_edges(u_edges: np.ndarray) -> np.ndarray:
    # The output grid: each bin of u_edges split in two
    return np.sort(np.concatenate([u_edges, (u_edges[:-1] + u_edges[1:]) / 2]))


def shape_on_u(depth_hi_nm: np.ndarray, density: np.ndarray, Rp_nm: float,
               u_edges: np.ndarray = U_EDGES) -> Tuple[np.ndarray, float]:
    # Area-preserving resampling of a binned depth density onto u = depth / Rp. TRIM depths are the
    # bins' upper edges. Returns the shape (area 1 over all depths) and the total area (density x nm).
    dx = float(np.mean(np.diff(depth_hi_nm)))
    edges = np.concatenate(([depth_hi_nm[0] - dx], depth_hi_nm))
    cum = np.concatenate(([0.0], np.cumsum(np.clip(density, 0, None) * dx)))
    total = float(cum[-1])
    if total <= 0:
        return np.zeros(len(u_edges) - 1), 0.0
    return np.diff(np.interp(u_edges * Rp_nm, edges, cum)) / total / np.diff(u_edges), total


def _refine(shape: np.ndarray, u_edges: np.ndarray = U_EDGES) -> np.ndarray:
    # A shape on the u_edges bins, linearly interpolated onto the output grid's bins with its area kept
    centres = lambda e: (e[:-1] + e[1:]) / 2      # noqa: E731
    fine_edges = out_edges(u_edges)
    fine = np.interp(centres(fine_edges), centres(u_edges), shape)
    area = float(np.sum(fine * np.diff(fine_edges)))
    return fine * (float(np.sum(shape * np.diff(u_edges))) / area) if area > 0 else fine


PER_RUN = ('targets', 'ions', 'energy_eV', 'X', 'log_eta', 'stopped', 'S', 'G', 'order', 'n_ions')


@dataclass
class TrainingSet:
    targets: np.ndarray       # material names
    ions: np.ndarray
    energy_eV: np.ndarray
    X: np.ndarray             # physics inputs (n, 6): the 'sr' features; the 'v1' ones are the first 4
    log_eta: np.ndarray       # log(damage energy / beam energy)
    stopped: np.ndarray       # fraction of ions stopped in the target
    S: np.ndarray             # damage-energy shapes on the u grid (n, bins)
    G: np.ndarray             # implanted-ion shapes on the u grid (n, bins)
    order: np.ndarray         # file time of each run (for learning curves)
    n_ions: Optional[np.ndarray] = None   # ions per run (for weighting runs by their Monte Carlo noise)
    u_edges: np.ndarray = field(default_factory=lambda: U_EDGES)

    def subset(self, mask: np.ndarray) -> 'TrainingSet':
        return TrainingSet(**{f: (v[mask] if f in PER_RUN and v is not None else v)
                              for f, v in ((f, getattr(self, f)) for f in self.__dataclass_fields__)})

    def __len__(self) -> int:
        return len(self.energy_eV)


def load_runs(min_ions: int = MIN_IONS, family: str = 'heavy', targets: Optional[Sequence[str]] = None,
              caches: Sequence[str] = (core.DATASET_CACHE_DIR, core.CACHE_DIR)) -> List[Tuple[str, core.DamageProfile, float]]:
    # (material name, profile, file time) for every run of the ion family on a named material (or on `targets`):
    # the dataset plus the optimiser's confirmation runs, so each confirmation run also improves the model.
    # The high-statistics test runs live in core.TEST_CACHE_DIR, which only load_test_runs reads.
    out = []
    for cache in caches:
        if not os.path.isdir(cache):
            continue
        times = {f[:-4]: os.path.getmtime(os.path.join(cache, f)) for f in os.listdir(cache) if f.endswith('.pkl')}
        for p in core.cached_profiles(min_ions=min_ions, cache_dir=cache):
            name = core.material_name(p.material)
            if not name or ion_family(p.ion_symbol) != family or (targets is not None and name not in targets):
                continue
            # too thin: ions went through the back (a first attempt run_sized redid with a thicker target)
            if not p.meta.get('too_thin') and not core.target_too_thin(p):
                out.append((name, p, times.get(p.meta.get('cache_key', ''), 0.0)))
    return out


def load_test_runs(family: str = 'heavy', targets: Optional[Sequence[str]] = None):
    return load_runs(min_ions=1000, family=family, targets=targets, caches=(core.TEST_CACHE_DIR,))


def build_training_set(runs: Sequence[Tuple[str, core.DamageProfile, float]],
                       u_edges: np.ndarray = U_EDGES) -> TrainingSet:
    rows = {f: [] for f in PER_RUN}
    for name, p, t in runs:
        table = core.sr_table(p.material, p.ion_symbol)
        Rp, _ = table.at(p.energy_eV)
        s, _ = shape_on_u(p.depth_nm, p.Tdam_density, Rp, u_edges)
        g, imp_total = shape_on_u(p.depth_nm, p.implanted, Rp, u_edges)
        for f, v in [('targets', name), ('ions', p.ion_symbol), ('energy_eV', p.energy_eV),
                     ('X', features(p.material, p.ion_symbol, p.energy_eV, 'sr', table)[0]),
                     ('log_eta', math.log(max(p.Tdam_eV, 1e-9) / p.energy_eV)), ('stopped', imp_total * 1e-7),
                     ('S', s), ('G', g), ('order', t), ('n_ions', p.number_ions)]:
            rows[f].append(v)
    n_bins = len(u_edges) - 1
    arrays = {f: np.array(v) for f, v in rows.items()}
    if not runs:   # an empty set still has the right widths
        arrays.update(X=np.zeros((0, FEATURES['sr'])), S=np.zeros((0, n_bins)), G=np.zeros((0, n_bins)))
    return TrainingSet(**arrays, u_edges=u_edges)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def _kernel(n_inputs: int = 4):
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    return (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(n_inputs), length_scale_bounds=(1e-2, 1e3),
                                                      nu=2.5)
            + WhiteKernel(1e-3, (1e-8, 1.0)))


def _gp(kernel, optimise: bool, restarts: int = 0, alpha=1e-10):
    from sklearn.gaussian_process import GaussianProcessRegressor
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer='fmin_l_bfgs_b' if optimise else None,
                                    n_restarts_optimizer=restarts, random_state=0, alpha=alpha)


class _PerOutputGP:
    # One Gaussian process per output, so each PCA weight gets its own length scales and noise level (the
    # minor components are mostly TRIM noise; with a shared kernel their noise would smooth the main ones)

    def __init__(self, gps: list):
        self.gps = gps

    @classmethod
    def fit(cls, xs: np.ndarray, Y: np.ndarray, thetas: np.ndarray,
            noise_scale: Optional[np.ndarray] = None) -> '_PerOutputGP':
        # noise_scale (one per run, 1 at the typical ion count): the fitted white noise becomes a per-run noise
        # of noise_level * noise_scale, so a run with more ions counts for more
        gps = []
        for j, th in enumerate(thetas):
            kernel = _kernel(xs.shape[1]).clone_with_theta(th)
            if noise_scale is None:
                gps.append(_gp(kernel, False).fit(xs, Y[:, j]))
            else:
                alpha = kernel.k2.noise_level * noise_scale + 1e-10
                gps.append(_gp(kernel.k1, False, alpha=alpha).fit(xs, Y[:, j]))
        return cls(gps)

    def predict(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Mean and spread of the underlying profile: the fitted white noise is the Monte-Carlo scatter between
        # TRIM runs, not uncertainty about the profile, so it is taken out of the predictive spread (with
        # per-run noise it never entered it)
        from sklearn.gaussian_process.kernels import WhiteKernel
        means, stds = [], []
        for gp in self.gps:
            m, s = gp.predict(xs, return_std=True)
            white = getattr(gp.kernel_, 'k2', None)
            noise = (white.noise_level * float(np.squeeze(getattr(gp, '_y_train_std', 1.0))) ** 2
                     if isinstance(white, WhiteKernel) else 0.0)
            means.append(m)
            stds.append(np.sqrt(np.clip(s ** 2 - noise, 0, None)))
        return np.column_stack(means), np.column_stack(stds)


SHAPE_TRANSFORMS = ('none', 'sqrt')


def _to_pca(M: np.ndarray, transform: str) -> np.ndarray:
    # Shapes as the PCA sees them. 'sqrt' (a Hellinger-like scale) gives a profile's low-damage parts, such as the
    # near-surface region at high energy, more weight than the peak has over them on the plain scale
    return np.sqrt(np.clip(M, 0, None)) if transform == 'sqrt' else M


def _from_pca(v: np.ndarray, transform: str) -> np.ndarray:
    v = np.clip(v, 0, None)
    return v ** 2 if transform == 'sqrt' else v


def _outputs(a: Dict[str, np.ndarray], ts: TrainingSet, eta_baseline: Optional[str],
             transform: str = 'none') -> Dict[str, np.ndarray]:
    # What the Gaussian processes learn for each run: log damage-energy fraction against the baseline and
    # the stopped fraction, then the PCA weights of the damage and implant shapes (basis in `a`)
    base = np.zeros(len(ts))
    if eta_baseline is not None:
        for t, i in set(zip(ts.targets, ts.ions)):
            rows = (ts.targets == t) & (ts.ions == i)
            base[rows] = log_eta_baseline(core.MATERIALS[t], i, ts.energy_eV[rows], eta_baseline)
    return {'totals': np.column_stack([ts.log_eta - base, ts.stopped]),
            'damage': (_to_pca(ts.S, transform) - a['damage_mean']) @ a['damage_comp'].T,
            'implant': (_to_pca(ts.G, transform) - a['implant_mean']) @ a['implant_comp'].T}


def _n_components(sv: np.ndarray, cap: int, keep: float = 0.995) -> int:
    frac = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    return int(min(cap, np.searchsorted(frac, keep) + 1))


class Surrogate:
    # PCA of the profile shapes + one Gaussian process per output, on standardised physics inputs

    def __init__(self, arrays: Dict[str, np.ndarray], info: dict, gps: Optional[dict] = None):
        self.a, self.info = arrays, info
        self.feature_kind = info.get('features', 'v1')      # models saved before v2 have the v1 inputs
        self.u_edges = np.asarray(arrays['u_edges']) if 'u_edges' in arrays else U_EDGES
        if gps is None:
            gps = {}
            scale = self._noise_scale(arrays.get('n_ions'))
            for g in GROUPS:
                Y, thetas = arrays[f'Y_{g}'], np.atleast_2d(arrays[f'theta_{g}'])
                if len(thetas) == 1:    # a model saved with one kernel shared by the outputs
                    thetas = np.repeat(thetas, Y.shape[1], axis=0)
                gps[g] = _PerOutputGP.fit(self._xs(arrays['X']), Y, thetas, scale)
        self.gps = gps

    def _noise_scale(self, n_ions: Optional[np.ndarray]) -> Optional[np.ndarray]:
        # Each run's Monte Carlo noise relative to a run of the typical ion count (variance ~ 1 / ions), when the
        # model weights runs that way
        if not self.info.get('ion_weighting') or n_ions is None:
            return None
        return float(self.info['n_ref']) / np.asarray(n_ions, dtype=float)

    # --- training --------------------------------------------------------
    @classmethod
    def fit(cls, ts: TrainingSet, max_hyper_points: int = 500, verbose: bool = False,
            eta_baseline: Optional[str] = 'sr', restarts: int = 0, features: str = 'v1',
            ion_weighting: bool = False, caps: Tuple[int, int] = (8, 6), family: str = 'heavy',
            shape_transform: str = 'none') -> 'Surrogate':
        # eta_baseline: what the damage energy is learnt relative to (see log_eta_baseline; the others are
        # for the ablation). features: 'v1' or 'sr' (see features()). ion_weighting: a run's noise ~ 1 / its ions.
        # caps: the most PCA components kept for the damage and the implant shapes. shape_transform: the scale the
        # PCA works on ('none' or 'sqrt', see _to_pca).
        if eta_baseline not in ETA_BASELINES:
            raise ValueError(f'eta_baseline must be one of {ETA_BASELINES}')
        if features not in FEATURES:
            raise ValueError(f'features must be one of {", ".join(FEATURES)}')
        if shape_transform not in SHAPE_TRANSFORMS:
            raise ValueError(f'shape_transform must be one of {", ".join(SHAPE_TRANSFORMS)}')
        started = time.time()
        X = ts.X[:, :FEATURES[features]]
        a = {'X': X, 'x_mean': X.mean(0), 'x_std': X.std(0) + 1e-12, 'u_edges': np.asarray(ts.u_edges)}
        for (key, M), cap in zip((('damage', ts.S), ('implant', ts.G)), caps):
            M = _to_pca(M, shape_transform)
            mean = M.mean(0)
            _, sv, vt = np.linalg.svd(M - mean, full_matrices=False)
            k = _n_components(sv, cap)
            a[f'{key}_mean'], a[f'{key}_comp'] = mean, vt[:k]
            a[f'{key}_var'] = np.array(float(np.sum(sv[:k] ** 2) / np.sum(sv ** 2)))
        for g, Y in _outputs(a, ts, eta_baseline, shape_transform).items():
            a[f'Y_{g}'] = Y
        xs = (X - a['x_mean']) / a['x_std']
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xs), min(len(xs), max_hyper_points), replace=False)   # hyperparameters on a subset
        n_ref, scale = None, None
        if ion_weighting:
            if ts.n_ions is None:
                raise ValueError('ion_weighting needs the number of ions of each run')
            a['n_ions'] = np.asarray(ts.n_ions, dtype=float)
            n_ref = float(np.median(a['n_ions'][idx]))     # the count the fitted white noise stands for
            scale = n_ref / a['n_ions']
        gps = {}
        for g in GROUPS:
            Y = a[f'Y_{g}']
            thetas = []
            for j in range(Y.shape[1]):
                with warnings.catch_warnings():   # hyperparameters at a bound are fine (an input that barely matters)
                    warnings.simplefilter('ignore')
                    thetas.append(_gp(_kernel(X.shape[1]), True, restarts).fit(xs[idx], Y[idx, j]).kernel_.theta)
            a[f'theta_{g}'] = np.array(thetas)
            gps[g] = _PerOutputGP.fit(xs, Y, a[f'theta_{g}'], scale)
            a[f'y_std_{g}'] = Y.std(0) + 1e-12
            if verbose:
                noise = [f'{math.exp(th[-1]):.2g}' for th in thetas]
                print(f'  {g:7}: {Y.shape[1]} outputs, noise share {", ".join(noise)}', flush=True)
        info = {'n_runs': len(ts), 'family': family, 'features': features, 'ion_weighting': ion_weighting,
                'n_ref': n_ref, 'shape_transform': shape_transform, 'hyper_runs': int(len(idx)),
                'restarts': restarts, 'targets': sorted(set(ts.targets.tolist())), 'ions': sorted(set(ts.ions.tolist())),
                'pairs': [list(p) for p in sorted({(str(t), str(i)) for t, i in zip(ts.targets, ts.ions)})],
                'energy_MeV': [float(ts.energy_eV.min() / 1e6), float(ts.energy_eV.max() / 1e6)],
                'x_min': X.min(0).tolist(), 'x_max': X.max(0).tolist(),
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
        if len(ts.u_edges) != len(self.u_edges) or not np.allclose(ts.u_edges, self.u_edges):
            raise ValueError("the runs' shapes are on another u grid than this model's")
        a = dict(self.a)
        X = ts.X[:, :FEATURES[self.feature_kind]]
        a['X'] = np.vstack([a['X'], X])
        if self.info.get('ion_weighting'):
            a['n_ions'] = np.concatenate([a['n_ions'], np.asarray(ts.n_ions, dtype=float)])
        for g, Y in _outputs(a, ts, self.info.get('eta_baseline'), self.info.get('shape_transform', 'none')).items():
            a[f'Y_{g}'] = np.vstack([a[f'Y_{g}'], Y])
        pairs = {tuple(p) for p in self.info.get('pairs', [])} | {(str(t), str(i)) for t, i in zip(ts.targets, ts.ions)}
        info = dict(self.info, n_runs=self.info['n_runs'] + len(ts),
                    targets=sorted(set(self.info['targets']) | set(ts.targets.tolist())),
                    ions=sorted(set(self.info['ions']) | set(ts.ions.tolist())), pairs=[list(p) for p in sorted(pairs)],
                    energy_MeV=[min(self.info['energy_MeV'][0], float(ts.energy_eV.min() / 1e6)),
                                max(self.info['energy_MeV'][1], float(ts.energy_eV.max() / 1e6))],
                    x_min=np.minimum(self.info['x_min'], X.min(0)).tolist(),
                    x_max=np.maximum(self.info['x_max'], X.max(0)).tolist(),
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
        # X may hold more inputs than the model uses (a TrainingSet's 'sr' set): v1's are the first four
        X = np.asarray(X)[:, :len(self.a['x_mean'])]
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
        X = features(material, ion_symbol, energies_eV, self.feature_kind, table)
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

    def _shape(self, key: str, weights: np.ndarray) -> np.ndarray:
        # A shape on the model's u grid from its PCA weights; with the sqrt scale, squared and brought back to area 1
        transform = self.info.get('shape_transform', 'none')
        s = _from_pca(self.a[f'{key}_mean'] + weights @ self.a[f'{key}_comp'], transform)
        if transform == 'sqrt':
            area = float(np.sum(s * np.diff(self.u_edges)))
            s = s / area if area > 0 else s
        return s

    def _profile(self, material, ion_symbol, energy_eV, Rp_nm, out, log_eta_base, unc, inside) -> core.DamageProfile:
        fine = out_edges(self.u_edges)
        s = _refine(self._shape('damage', out['damage']), self.u_edges)
        g = _refine(self._shape('implant', out['implant']), self.u_edges)
        T_total = math.exp(out['totals'][0] + log_eta_base) * energy_eV
        stopped = float(np.clip(out['totals'][1], 0, 1))
        Tdam_density = T_total * s / (Rp_nm * 10)          # eV / A / ion (s has unit area over u)
        implanted = stopped * g / (Rp_nm * 1e-7)           # 1/cm per ion
        T_in_grid = float(np.sum(Tdam_density * np.diff(fine)) * Rp_nm * 10)
        return core.DamageProfile(
            material=material, ion_symbol=ion_symbol, energy_eV=float(energy_eV), number_ions=0,
            depth_nm=fine[1:] * Rp_nm, Tdam_density=Tdam_density,
            disp_density=0.8 * Tdam_density / (2 * material.nrt_Ed), implanted=implanted,
            Tdam_eV=T_in_grid, N_NRT=0.8 * T_in_grid / (2 * material.nrt_Ed), Tdam_balance_eV=T_in_grid,
            meta={'surrogate': True, 'uncertainty': unc, 'in_domain': inside, 'Rp_nm': Rp_nm})


_cached_model: Dict[str, Tuple[float, Surrogate]] = {}


def load_latest(ion: Optional[str] = None, path: Optional[str] = None) -> Optional[Surrogate]:
    # The saved model for the ion's family (the heavy-ion model when no ion is given), reloaded only when the
    # file changes; None if that family has no model
    if ion is not None and ion.endswith('.npz'):   # the old signature, load_latest(path)
        ion, path = None, ion
    path = path or MODEL_PATHS[ion_family(ion) if ion else 'heavy']
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


EVAL_U_EDGES = {'heavy': U_EDGES, 'light': LIGHT_U_EDGES}   # the shape error is measured on its family's grid


window_noise = core.mean_dpa_noise   # the truth's own noise, reported beside every error


def _window_fluence(p: core.DamageProfile, lo: float, hi: float) -> float:
    # Fluence for 1 dpa mean over [lo, hi]
    with np.errstate(divide='ignore'):
        return 1 / np.interp(np.linspace(lo, hi, 50), p.depth_nm, p.dpa(1.0)).mean()


def compare(p_true: core.DamageProfile, p_pred: core.DamageProfile, u_edges: np.ndarray = U_EDGES) -> dict:
    # Errors that matter for planning: the fluence for the same window, and the ROI width the rules allow
    Rp = p_pred.meta['Rp_nm']
    rules = _relative_window_rules(Rp)
    r_true, r_pred = core.roi_options(p_true, **rules), core.roi_options(p_pred, **rules)
    s_true, _ = shape_on_u(p_true.depth_nm, p_true.Tdam_density, Rp, u_edges)
    s_pred, _ = shape_on_u(p_pred.depth_nm, p_pred.Tdam_density, Rp, u_edges)
    # The fixed window (0.2-0.6 Rp: the rising side for heavy ions, the plateau for H and He) is defined for every
    # run, including those where TRIM's own profile has no window flat to +-10 %
    fixed = _window_fluence(p_pred, 0.2 * Rp, 0.6 * Rp) / _window_fluence(p_true, 0.2 * Rp, 0.6 * Rp) - 1
    out = {'shape_error': 0.5 * float(np.sum(np.abs(s_true - s_pred) * np.diff(u_edges))),
           'damage_energy_error': p_pred.Tdam_eV / p_true.Tdam_eV - 1, 'fixed_window_error': fixed,
           'truth_noise_fixed': window_noise(p_true, 0.2 * Rp, 0.6 * Rp),
           'fluence_error': np.nan, 'width_error': np.nan, 'fluence_1dpa_trim': np.nan, 'fluence_1dpa_pred': np.nan,
           'roi_start_nm': np.nan, 'roi_end_nm': np.nan}
    if r_true is not None:
        lo, hi = r_true['roi_start_nm'], r_true['roi_end_nm']
        f_true, f_pred = _window_fluence(p_true, lo, hi), _window_fluence(p_pred, lo, hi)
        # fluence ~ 1 / mean dpa per ion in the window; no predicted damage there counts as an infinite error
        out.update(fluence_error=f_pred / f_true - 1, fluence_1dpa_trim=f_true, fluence_1dpa_pred=f_pred,
                   roi_start_nm=lo, roi_end_nm=hi)
        out['width_error'] = ((r_pred['roi_width_nm'] if r_pred else 0.0) - r_true['roi_width_nm']) / r_true['roi_width_nm']
    return out


def evaluate_on(model: Surrogate, runs: Sequence[Tuple[str, core.DamageProfile, float]],
                u_edges: np.ndarray = U_EDGES, posterior_draws: int = 0) -> List[dict]:
    # One row per run. With posterior_draws, also whether TRIM's fluence lies inside the 10-90 % band of that
    # many draws from the model's predictive distribution (for the same window)
    pairs = {tuple(p) for p in model.info.get('pairs', [])}
    rows = []
    for name, p, _ in runs:
        pred = model.predict_profiles(p.material, p.ion_symbol, [p.energy_eV])[0]
        row = {'target': name, 'ion': p.ion_symbol, 'energy_MeV': p.energy_eV / 1e6,
               'number_ions': p.number_ions, 'uncertainty': pred.meta['uncertainty'],
               'in_domain': bool(pred.meta['in_domain']), 'pair_in_training': (name, p.ion_symbol) in pairs,
               **compare(p, pred, u_edges)}
        if posterior_draws and np.isfinite(row['fluence_error']):
            draws = model.predict_profiles(p.material, p.ion_symbol, [p.energy_eV], n_samples=posterior_draws)
            f = [_window_fluence(d, row['roi_start_nm'], row['roi_end_nm']) for d in draws]
            lo, hi = np.percentile(f, [10, 90])
            row['posterior_covers'] = bool(lo <= row['fluence_1dpa_trim'] <= hi)
        rows.append(row)
    return rows


def band_coverage(rows: List[dict], q: Optional[float]) -> Optional[float]:
    # Share of runs whose TRIM fluence lies within x exp(+-q) of the prediction (the optimiser's calibrated band)
    v = np.abs(np.log1p([r['fluence_error'] for r in rows if np.isfinite(r['fluence_error'])]))
    return float(np.mean(v <= q)) if q and v.size else None


def summarise(rows: List[dict]) -> str:
    # Percentiles over the runs where TRIM has an ROI window; a prediction with no damage in that window
    # counts as an infinite error, not as a missing one
    def pct(key, q):
        v = np.abs([r[key] for r in rows if not np.isnan(r[key])])
        return f'{np.percentile(v, q) * 100:5.1f}%' if v.size else '   --'
    with_window = sum(not np.isnan(r['fluence_error']) for r in rows)
    no_damage = sum(np.isinf(r['fluence_error']) for r in rows)
    outside = sum(not r.get('in_domain', True) for r in rows)
    fixed = (f' | 0.2-0.6 Rp dose error median {pct("fixed_window_error", 50)}, p90 {pct("fixed_window_error", 90)}'
             if rows and 'fixed_window_error' in rows[0] else '')
    return (f'fluence error median {pct("fluence_error", 50)}, p90 {pct("fluence_error", 90)} | '
            f'ROI width error median {pct("width_error", 50)}, p90 {pct("width_error", 90)} | '
            f'damage energy error median {pct("damage_energy_error", 50)} | '
            f'shape error median {pct("shape_error", 50)}{fixed} | n = {len(rows)} ({with_window} with a TRIM window'
            + (f', {no_damage} predicted with no damage there' if no_damage else '')
            + (f'; {outside} flagged out of domain' if outside else '') + ')')


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


# What each family is trained with (cmd_fit); cmd_ablation compares the alternatives on the high-statistics test runs
# (docs/surrogate_v2_report.md).
# Heavy ions: v1's configuration, with the kernels tuned on every run instead of a random 500. No change beat v1 with
# significance; this one was not worse, kept the optimiser benchmark's choices, and makes the fit reproducible: the
# random subset alone moved predictions by up to 4 % between two fits on the same runs.
# Light ions: their plateau, where an ROI sits, is 1-10 % of the end-of-range peak, so the shapes are compressed on
# the sqrt scale, where the plateau carries its share of the variance. v1's inputs: the SR ones were worse on
# ion-target pairs left out of training, as they were for heavy ions.
V1_CONFIG = {'features': 'v1', 'ion_weighting': False, 'caps': (8, 6)}
CONFIGS = {'heavy': {**V1_CONFIG, 'max_hyper_points': 100000},
           'light': {'features': 'v1', 'ion_weighting': False, 'caps': (10, 8), 'shape_transform': 'sqrt'}}
GRIDS = {'heavy': U_EDGES, 'light': LIGHT_U_EDGES}


def fit_family(runs, family: str, **overrides) -> Surrogate:
    # A model of the family's configuration (with any overrides) on these runs
    return Surrogate.fit(build_training_set(runs, GRIDS[family]), family=family, **{**CONFIGS[family], **overrides})


def test_group(target: str, ion: str, family: str) -> str:
    if family == 'light':
        return f'{ion} beams'
    return 'v1 targets' if target in core.V1_MATERIALS else 'v2 alloys'


def cmd_fit(family: str = 'heavy', out: Optional[str] = None) -> None:
    runs = load_runs(family=family)
    print(f'{len(runs)} {family}-ion training runs on {len({r[0] for r in runs})} targets.')
    if len(runs) < 30:
        print('Not enough runs yet; wait for srim_dataset.py run to make more.')
        return
    train, held = split_pairs(runs)
    t = time.time()
    model = fit_family(train, family)
    rows = evaluate_on(model, held, EVAL_U_EDGES[family])
    print(f'Held-out ion-target pairs ({len(held)} runs, model on {len(train)} runs, '
          f'{model.info["components"]} PCA components, fit {time.time() - t:.0f} s):')
    print('  ' + summarise(rows))
    final = fit_family(runs, family)
    q = band_quantile(rows)
    final.info['fluence_log_q80'] = q
    final.info['band_calibrated_on'] = f'{len(held)} held-out training runs (mostly 300 ions)'
    # With high-statistics test runs, calibrate on them instead: a heavy-ion test run's dose noise is about 1 %, a
    # 300-ion training run's 2-3 %. Unseen pairs: a model without the test pairs; seen pairs: the final one.
    test = load_test_runs(family)
    if len(test) >= 10:
        pairs = {(n, p.ion_symbol) for n, p, _ in test}
        unseen = fit_family([r for r in runs if (r[0], r[1].ion_symbol) not in pairs], family)
        q_unseen = band_quantile(evaluate_on(unseen, test, EVAL_U_EDGES[family]))
        q_seen = band_quantile(evaluate_on(final, test, EVAL_U_EDGES[family]))
        if q_unseen and q_seen:   # None: too few test runs with an ROI window to calibrate on
            final.info.update(fluence_log_q80=q_unseen, fluence_log_q80_seen_pairs=q_seen,
                              band_calibrated_on=f'{len(test)} high-statistics test runs')
            print(f'Band from {len(test)} high-statistics test runs: x{math.exp(q_unseen):.3f} for an ion-target '
                  f'pair not in training, x{math.exp(q_seen):.3f} for one in training (at another energy).')
    out = out or MODEL_PATHS[family]
    final.save(out)
    q = final.info['fluence_log_q80']
    if q:
        print(f'80 % of held-out fluences were within x{math.exp(q):.3f} of the prediction: '
              f'the optimiser\'s fluence band is at least this wide.')
    print(f'Saved the model trained on all {len(runs)} runs to {out}')


def loo_coverage(rows: List[dict], q: float = 80) -> Optional[float]:
    # Leave-one-out conformal coverage: each run judged against the band the other runs would calibrate. A fair
    # check of a band calibrated on these same runs (cmd_fit's high-statistics calibration).
    v = np.abs(np.log1p([r['fluence_error'] for r in rows if np.isfinite(r['fluence_error'])]))
    if v.size < 6:
        return None
    return float(np.mean([v[i] <= np.percentile(np.delete(v, i), q) for i in range(v.size)]))


def _coverage_text(rows: List[dict], model: Surrogate) -> str:
    q = model.info.get('fluence_log_q80')
    cov, loo = band_coverage(rows, q), loo_coverage(rows)
    post = [r['posterior_covers'] for r in rows if 'posterior_covers' in r]
    return ((f'80 % band x{math.exp(q):.3f} covers {cov:.0%}' if cov is not None else 'no calibrated band')
            + (f'; a band calibrated on the other test runs covers {loo:.0%} (leave-one-out)' if loo is not None else '')
            + (f'; posterior 10-90 % covers {np.mean(post):.0%}' if post else ''))


def cmd_evaluate_test(family: Optional[str] = None, path: Optional[str] = None, label: Optional[str] = None) -> None:
    # A saved model (each family's own by default) against the high-statistics test runs, which no model trains on
    import pandas as pd
    for fam in [family] if family else list(MODEL_PATHS):
        model_path = path or MODEL_PATHS[fam]
        model = Surrogate.load(model_path)
        runs = load_test_runs(fam)
        if model is None or not runs:
            print(f'{fam}: ' + ('no model at ' + model_path if model is None else 'no test runs yet'))
            continue
        rows = evaluate_on(model, runs, EVAL_U_EDGES[fam], posterior_draws=30)
        for r in rows:
            r['group'] = test_group(r['target'], r['ion'], fam)
        print(f'{fam} model {model_path} ({model.info["n_runs"]} runs, inputs {model.feature_kind}, '
              f'ion weighting {"on" if model.info.get("ion_weighting") else "off"}), {len(rows)} test runs:')
        for group in sorted({r['group'] for r in rows}):
            sel = [r for r in rows if r['group'] == group]
            print(f'  {group:11} {summarise(sel)}\n  {"":11} {_coverage_text(sel, model)}')
        out = os.path.join(MODEL_DIR, f'test_scores_{label or fam}.csv')
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'  per-run errors: {out}')


ABLATION = {   # each change on its own (and the two input-side changes together), against v1's configuration
    'v1': {},
    'sr': {'features': 'sr'},                                   # log(dRp/Rp) and the SR nuclear share as inputs
    'weight': {'ion_weighting': True},                          # a run's noise ~ 1 / its ions
    'sr+weight': {'features': 'sr', 'ion_weighting': True},
    'pca12': {'caps': (12, 10)},                                # more PCA components
    'restarts': {'restarts': 2},                                # kernel fits from 2 more random starts
    'hyper1000': {'max_hyper_points': 1000},                    # kernels tuned on 1000 runs instead of 500
    'hyper_all': {'max_hyper_points': 100000},                  # kernels tuned on every run (no random subset)
    'sqrt': {'shape_transform': 'sqrt'},                        # PCA of the square root of the shapes
    'sqrt+hyper_all': {'shape_transform': 'sqrt', 'max_hyper_points': 100000},
    'sqrt+sr': {'shape_transform': 'sqrt', 'features': 'sr'},
}
# The light model's first configuration, which its ablation starts from ('v1 inputs' won: CONFIGS['light'])
LIGHT_FIRST_CONFIG = {'features': 'sr', 'ion_weighting': False, 'caps': (10, 8), 'shape_transform': 'sqrt'}
LIGHT_ABLATION = {
    'light': {},
    'plain': {'shape_transform': 'none'},                       # PCA of the shapes themselves
    'v1 inputs': {'features': 'v1'},                            # without the SR inputs
    'pca8': {'caps': (8, 6)},                                   # v1's number of PCA components
    'pca14': {'caps': (14, 10)},
    'hyper_all': {'max_hyper_points': 100000},
    'v1 inputs + plain': {'features': 'v1', 'shape_transform': 'none'},
}


def cmd_ablation(variants: Optional[Sequence[str]] = None, family: str = 'heavy') -> None:
    """Each change trained on the same runs and scored on the family's test runs.

    Heavy ions: trained on v1's 13 targets, against v1's configuration. seen pairs: the test runs' ion-target pairs
    are in training at other energies (a named material's case); unseen pairs: their training runs left out; alloys:
    the v2 alloys, which no variant has seen (zero-shot). The saved v1 model is scored too.
    Light ions: trained on every H and He run, against LIGHT_FIRST_CONFIG; seen and unseen pairs as for heavy ions.
    """
    import pandas as pd
    import srim_dataset
    if family == 'heavy':
        runs, table, base_cfg = load_runs(family='heavy', targets=core.V1_MATERIALS), ABLATION, V1_CONFIG
        test = load_test_runs('heavy')
        held = {'seen pairs': [r for r in test if r[0] in core.V1_MATERIALS],
                'alloys (zero-shot)': [r for r in test if r[0] not in core.V1_MATERIALS]}
        # The planned test pairs, so the models can be fitted (and cached) before every test run has finished
        test_pairs = ({(t, i) for i, t in srim_dataset.TEST_PAIRS} |
                      {(t, i) for i, t in srim_dataset.TEST_BENCH_WINDOWS_MEV})
    else:
        runs, table, base_cfg = load_runs(family='light'), LIGHT_ABLATION, LIGHT_FIRST_CONFIG
        held = {'seen pairs': load_test_runs('light')}
        test_pairs = {(t, i) for i, t in srim_dataset.TEST_LIGHT_PAIRS}
    held['unseen pairs'] = held['seen pairs']
    baseline = next(iter(table))
    grid, eval_grid = GRIDS[family], EVAL_U_EDGES[family]
    ts = build_training_set(runs, grid)
    ts_unseen = ts.subset(np.array([(t, i) not in test_pairs for t, i in zip(ts.targets, ts.ions)]))
    print(f'{len(runs)} {family}-ion training runs ({len(ts_unseen)} without the test pairs); test: '
          + ', '.join(f'{k} {len(v)}' for k, v in held.items()))
    cache = os.path.join(MODEL_DIR, 'ablation_models')

    def fitted(name: str, split: str, data: TrainingSet, cfg: dict) -> Surrogate:
        # Fitted once and kept: the test runs change while the queue runs, the training runs do not
        prefix = '' if family == 'heavy' else 'light_'
        path = os.path.join(cache, f'{prefix}{name}__{split}.npz'.replace(' ', '_'))
        model = Surrogate.load(path)
        if model is None or model.info['n_runs'] != len(data):
            model = Surrogate.fit(data, family=family, **cfg)
            model.save(path)
        return model
    per_run, summary = [], []

    def score(name: str, split: str, model: Surrogate) -> None:
        if not held.get(split):
            return
        rows = evaluate_on(model, held[split], eval_grid)
        per_run.extend({'variant': name, 'split': split, **r} for r in rows)
        pct = lambda k, q=50: float(np.nanpercentile(np.abs([r[k] for r in rows]), q))   # noqa: E731
        summary.append({'variant': name, 'split': split, 'n': len(rows), 'fluence_median': pct('fluence_error'),
                        'fluence_p90': pct('fluence_error', 90), 'fixed_median': pct('fixed_window_error'),
                        'fixed_p90': pct('fixed_window_error', 90), 'width_median': pct('width_error'),
                        'shape_median': pct('shape_error'), 'damage_energy_median': pct('damage_energy_error')})
        print(f'  {name:30} {split:19} {summarise(rows)}', flush=True)

    saved = Surrogate.load(os.path.join(MODEL_DIR, 'v1', 'surrogate.npz')) if family == 'heavy' else None
    if saved is not None:
        for split in ('seen pairs', 'alloys (zero-shot)'):
            score('v1 (saved file)', split, saved)
    for name in variants or table:
        cfg = {**base_cfg, **table[name]}
        t = time.time()
        model = fitted(name, 'all', ts, cfg)
        for split in ('seen pairs', 'alloys (zero-shot)'):
            score(name, split, model)
        score(name, 'unseen pairs', fitted(name, 'unseen pairs', ts_unseen, cfg))
        print(f'  ({name}: {time.time() - t:.0f} s)', flush=True)

    # Paired against the baseline on the same test runs: how often each variant's error is the smaller, and a
    # bootstrap 90 % interval for the change in the median error (an interval below zero: the variant is better)
    df = pd.DataFrame(per_run)
    keys = ['split', 'target', 'ion', 'energy_MeV']
    out = pd.DataFrame(summary)
    rng = np.random.default_rng(0)
    for i, s in out.iterrows():
        for metric in ('fluence_error', 'fixed_window_error', 'width_error', 'shape_error'):
            base = df[df.variant == baseline].set_index(keys)[metric].abs()
            mine = df[(df.variant == s.variant) & (df.split == s.split)].set_index(keys)[metric].abs()
            both = pd.concat([mine, base], axis=1, keys=['mine', 'base']).dropna()
            both = both[np.isfinite(both).all(axis=1)].to_numpy()
            if not len(both):
                continue
            idx = rng.integers(0, len(both), (2000, len(both)))
            change = np.median(both[idx, 0], axis=1) - np.median(both[idx, 1], axis=1)
            name = metric.split('_')[0]
            out.loc[i, f'{name}_better'] = float((both[:, 0] < both[:, 1]).mean())
            out.loc[i, f'{name}_median_change_ci90'] = f'{100 * np.percentile(change, 5):+.2f} to {100 * np.percentile(change, 95):+.2f}'
    stem = 'ablation' if family == 'heavy' else 'ablation_light'
    out.to_csv(os.path.join(MODEL_DIR, f'{stem}.csv'), index=False)
    df.to_csv(os.path.join(MODEL_DIR, f'{stem}_runs.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(out.assign(**{c: (100 * out[c]).round(2) for c in out.columns if c.endswith(('median', 'p90'))})
              .to_string(index=False))
    print(f'Saved {os.path.join(MODEL_DIR, stem + ".csv")} (errors in %; *_better: share of test runs with a smaller '
          f'error than {baseline}; *_median_change_ci90: bootstrap 90 % interval of the change in median error, '
          'percentage points)')


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
        model = fit_family(train, 'heavy')
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
        model = fit_family(train[:n], 'heavy')
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
        model0 = fit_family([r for r in runs if r[0] not in names], 'heavy')
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
    parser = argparse.ArgumentParser(description='TRIM surrogate: fit, validate, learning curve, few-shot, '
                                                 'evaluate-test, ablation')
    parser.add_argument('cmd', choices=['fit', 'validate', 'learning-curve', 'few-shot', 'evaluate-test', 'ablation'])
    parser.add_argument('--family', choices=list(MODEL_PATHS), help='fit, ablation: which model (default heavy); '
                                                                     'evaluate-test: only this one')
    parser.add_argument('--model', help='evaluate-test: a model file other than the family\'s saved one; '
                                        'fit: where to save the model (default: the family\'s file)')
    parser.add_argument('--label', help='evaluate-test: name of the per-run CSV (test_scores_<label>.csv)')
    parser.add_argument('--variants', nargs='+', choices=list(ABLATION) + list(LIGHT_ABLATION), help='ablation: only these')
    args = parser.parse_args(argv)
    if args.cmd == 'fit':
        cmd_fit(args.family or 'heavy', args.model)
    elif args.cmd == 'evaluate-test':
        cmd_evaluate_test(args.family, args.model, args.label)
    elif args.cmd == 'ablation':
        cmd_ablation(args.variants, args.family or 'heavy')
    else:
        {'validate': cmd_validate, 'learning-curve': cmd_learning_curve, 'few-shot': cmd_few_shot}[args.cmd]()


if __name__ == '__main__':
    main(sys.argv[1:])
