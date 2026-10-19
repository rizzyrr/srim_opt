# Core SRIM/TRIM radiation-damage calculations for SRIM app by Rizqullah Rasyiddin 01/10/2026

# Follows the Quick Kinchin-Pease approach of Stoller et al. (2013, erratum 2019) 

#    damage energy density   T_dam(x) = PHONONS by recoils - PHONONS by ions   [eV / A / ion]
#    NRT displacements       N_d(x)   = 0.8 * T_dam(x) / (2 * E_d)            [disp / A / ion]
#    dpa                     dpa(x)   = N_d(x) * fluence / (atomic density * 1e-8 cm/A)

# Initiialise packages
from email.policy import default
from email.policy import default
import hashlib
import json
import os
import pickle
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from srim import TRIM, Ion, Layer, Target
from srim.core.elementdb import ElementDB
from srim.output import Results

NA = 6.02214076e23  # atoms/mol
DEFAULT_SRIM_DIR = r'C:\Users\Riz\Documents\SRIM'
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'srim_outputs', 'cache')

# TRIM reads and writes fixed files inside the SRIM folder, so only one run may happen at a time per machine
_trim_lock = threading.Lock()


def element_data(symbol: str) -> dict:
    # Mass (amu), elemental density (g/cm3) and name from pysrim's element table
    return ElementDB.lookup(symbol)


def all_element_symbols() -> List[str]:
    # Element symbols ordered by atomic number (SRIM supports Z <= 92)
    db = ElementDB._db
    return sorted((s for s in db if 0 < db[s]['z'] <= 92), key=lambda s: db[s]['z'])


# Material definition


def convert_composition(symbols: List[str], values, to: str) -> np.ndarray:
    """Convert a composition between weight % and atomic % (result normalised to 100).

    to='at': `values` are weight %, returns atomic %   x_i = (w_i/M_i) / sum(w/M)
    to='wt': `values` are atomic %, returns weight %   w_i = x_i M_i / sum(x M)
    """
    v = np.asarray(values, dtype=float)
    m = np.array([element_data(s)['mass'] for s in symbols])
    if to == 'at':
        n = v / m
    elif to == 'wt':
        n = v * m
    else:
        raise ValueError("to must be 'at' or 'wt'")
    return n / n.sum() * 100


def _srim_nonzero(energy_eV: float) -> float:
    # TRIM silently replaces a binding energy of exactly 0 eV throught pysrim with its own non-zero
    # default (seen for alloying elements), so we pass an small value instead (1e-50)   
    return energy_eV if energy_eV > 0 else 1e-50


@dataclass(frozen=True)
class Constituent:
    symbol: str
    at_pct: float            # atomic percent (normalised automatically)
    E_d: float = 25.0        # displacement energy, eV
    lattice: float = 0.0     # lattice binding energy, eV (0 recommended for Quick K-P)
    surface: float = 0.0     # surface binding energy, eV


@dataclass(frozen=True)
class Material:
    constituents: Tuple[Constituent, ...]
    density_override: Optional[float] = None   # g/cm3; None -> estimated from composition
    width_A: float = 50000.0                   # target thickness, Angstrom

    def validate(self) -> List[str]:
        # Return a list of human-readable problems (empty if the material is valid)
        problems = []
        if not self.constituents:
            problems.append('Add at least one element to the target.')
        symbols = [c.symbol for c in self.constituents]
        if len(set(symbols)) != len(symbols):
            problems.append('Each element may only appear once in the target.')
        for c in self.constituents:
            try:
                element_data(c.symbol)
            except Exception:
                problems.append(f'Unknown element symbol "{c.symbol}".')
            if c.at_pct <= 0:
                problems.append(f'{c.symbol}: atomic % must be greater than zero.')
            if c.E_d <= 0:
                problems.append(f'{c.symbol}: displacement energy must be greater than zero.')
        if self.density_override is not None and self.density_override <= 0:
            problems.append('Density must be greater than zero.')
        if self.width_A <= 0:
            problems.append('Target thickness must be greater than zero.')
        return problems

    @property
    def fractions(self) -> np.ndarray:
        x = np.array([c.at_pct for c in self.constituents], dtype=float)
        return x / x.sum()

    @property
    def weight_pct(self) -> np.ndarray:
        return convert_composition([c.symbol for c in self.constituents], self.fractions, to='wt')

    @property
    def masses(self) -> np.ndarray:
        return np.array([element_data(c.symbol)['mass'] for c in self.constituents])

    @property
    def mean_mass(self) -> float:
        # Atom-fraction-weighted mean atomic mass, g/mol
        return float(np.sum(self.fractions * self.masses))

    @property
    def estimated_density(self) -> float:
        # Ideal-mixing (additive atomic volume) density estimate, g/cm3
        rho = np.array([element_data(c.symbol)['density'] for c in self.constituents])
        x, m = self.fractions, self.masses
        return float(np.sum(x * m) / np.sum(x * m / rho))

    @property
    def density(self) -> float:
        # Mass density used everywhere (TRIM input and dpa conversion), g/cm3
        return self.density_override if self.density_override else self.estimated_density

    @property
    def atomic_density(self) -> float:
        # atoms/cm3, consistent with `density`
        return self.density * NA / self.mean_mass

    @property
    def nrt_Ed(self) -> float:
        # Atom-fraction-weighted displacement energy used in the NRT formula, eV
        return float(np.sum(self.fractions * np.array([c.E_d for c in self.constituents])))

    @property
    def label(self) -> str:
        if len(self.constituents) == 1:
            return self.constituents[0].symbol
        parts = sorted(self.constituents, key=lambda c: -c.at_pct)
        return parts[0].symbol + ''.join(f'-{c.at_pct:.3g}{c.symbol}' for c in parts[1:])

    def to_target(self) -> Target:
        layer = Layer(
            {c.symbol: {'stoich': float(x), 'E_d': c.E_d,
                        'lattice': _srim_nonzero(c.lattice), 'surface': _srim_nonzero(c.surface)}
             for c, x in zip(self.constituents, self.fractions)},
            density=self.density,
            width=self.width_A,
        )
        return Target([layer])

# --------------------------------------------------------------------------------
# Running TRIM
# --------------------------------------------------------------------------------

@dataclass
class DamageProfile:
    # Depth profiles for one TRIM run, per incident ion (fluence-independent)
    material: Material
    ion_symbol: str
    energy_eV: float
    number_ions: int
    depth_nm: np.ndarray
    Tdam_density: np.ndarray       # eV / A / ion
    disp_density: np.ndarray       # NRT displacements / A / ion
    implanted: np.ndarray          # (atoms/cm3) / (atoms/cm2), i.e. 1/cm
    Tdam_eV: float                 # total damage energy per ion
    N_NRT: float                   # total NRT displacements per ion
    Tdam_balance_eV: float         # damage energy from the energy-balance check
    meta: dict = field(default_factory=dict)

    @property
    def bin_nm(self) -> float:
        return float(np.mean(np.diff(self.depth_nm)))

    @property
    def damaged_bins(self) -> int:
        # Number of depth bins with non-negligible damage - a resolution check
        return int(np.sum(self.disp_density > 0.01 * self.disp_density.max()))

    @property
    def mean_range_nm(self) -> float:
        w = self.implanted
        return float(np.sum(self.depth_nm * w) / np.sum(w)) if w.sum() > 0 else float('nan')

    @property
    def implant_peak_nm(self) -> float:
        # depth of the implanted-ion peak (smoothed, so one noisy bin cant set it)
        return float(self.depth_nm[np.argmax(smooth(self.implanted))])

    def dpa(self, fluence: float) -> np.ndarray:
        # NRT dpa vs depth for a fluence in ions/cm2.
        return self.disp_density * fluence / (self.material.atomic_density * 1.0e-8)

    def implanted_appm(self, fluence: float) -> np.ndarray:
        # Implanted-ion concentration vs depth, atomic parts per million.
        return self.implanted * fluence / self.material.atomic_density * 1e6

    def fluence_for_peak_dpa(self, target_dpa: float) -> float:
        # Fluence (ions/cm2) at which the peak dpa equals `target_dpa`
        return float(target_dpa * self.material.atomic_density * 1.0e-8 / self.disp_density.max())

    def summary(self, fluence: float) -> dict:
        return {
            'material': self.material.label,
            'ion': self.ion_symbol,
            'energy_MeV': self.energy_eV / 1e6,
            'number_ions': self.number_ions,
            'Tdam_keV_per_ion': self.Tdam_eV / 1e3,
            'Tdam_balance_keV_per_ion': self.Tdam_balance_eV / 1e3,
            'N_NRT_per_ion': self.N_NRT,
            'mean_range_nm': self.mean_range_nm,
            'implant_peak_nm': self.implant_peak_nm,
            'peak_dpa': float(self.dpa(fluence).max()),
            'peak_depth_nm': float(self.depth_nm[np.argmax(self.disp_density)]),
            'fluence_for_1dpa_peak': self.fluence_for_peak_dpa(1.0),
        }

    def to_dataframe(self, fluence: float) -> pd.DataFrame:
        return pd.DataFrame({
            'depth_nm': self.depth_nm,
            'damage_energy_eV_per_A_per_ion': self.Tdam_density,
            'NRT_disp_per_A_per_ion': self.disp_density,
            f'dpa_at_{fluence:.3g}': self.dpa(fluence),
            'implanted_per_cm': self.implanted,
            f'implanted_appm_at_{fluence:.3g}': self.implanted_appm(fluence),
        })


def _cache_key(material: Material, ion_symbol: str, energy_eV: float, number_ions: int,
               calculation: int) -> str:
    m = asdict(material)
    m['width_A'] = float(m['width_A']) 
    m['constituents'] = [{k: (v if k == 'symbol' else float(v)) for k, v in c.items()} for c in m['constituents']]
    if m['density_override'] is not None:
        m['density_override'] = float(m['density_override'])
    payload = json.dumps({
        'material': m, 'ion': ion_symbol, 'energy_eV': float(energy_eV),
        'number_ions': int(number_ions), 'calculation': int(calculation), 'v': 2,  # bump to invalidate runs made before a physics/input fix
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def cached_profiles(min_ions: int = 31) -> List[DamageProfile]:
    # Every valid run saved in the cache, oldest first
    # Skips files whose key no longer matches their inputs (made before the last cache
    # version bump) and runs below `min_ions` (sized_material's 30-ion pilot runs).
    if not os.path.isdir(CACHE_DIR):
        return []
    paths = sorted((os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')),
                   key=os.path.getmtime)
    profiles = []
    for path in paths:
        try:
            with open(path, 'rb') as f:
                p = pickle.load(f)
            key = _cache_key(p.material, p.ion_symbol, p.energy_eV, p.number_ions, 1)
        except Exception:
            continue   # unreadable or from an older code version
        if key != os.path.basename(path)[:-4] or p.number_ions < min_ions:
            continue
        p.meta['from_cache'] = True
        profiles.append(p)
    return profiles


def check_srim_directory(srim_dir: str) -> Optional[str]:
    # Return an error message if `srim_dir` does not look like a SRIM install
    if not os.path.isdir(srim_dir):
        return f'Folder not found: {srim_dir}'
    if not os.path.isfile(os.path.join(srim_dir, 'TRIM.exe')):
        return f'TRIM.exe not found in {srim_dir}'
    return None


def run_trim(material: Material, ion_symbol: str, energy_eV: float, number_ions: int = 1000,
             srim_dir: str = DEFAULT_SRIM_DIR, calculation: int = 1,
             use_cache: bool = True) -> DamageProfile:
    # Run (or load from cache) one TRIM calculation and reduce it to damage profiles
    problems = material.validate()
    if problems:
        raise ValueError(' '.join(problems))

    key = _cache_key(material, ion_symbol, energy_eV, number_ions, calculation)
    cache_path = os.path.join(CACHE_DIR, key + '.pkl')
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            profile = pickle.load(f)
        profile.meta['from_cache'] = True
        return profile

    err = check_srim_directory(srim_dir)
    if err:
        raise FileNotFoundError(err)

    ion = Ion(ion_symbol, energy=energy_eV)
    trim = TRIM(material.to_target(), ion, number_ions=int(number_ions), calculation=calculation)
    with _trim_lock:
        results = _run_trim_exe(trim, srim_dir, timeout_s=max(600.0, 1.0 * number_ions))

    profile = reduce_results(results, material, ion_symbol, energy_eV, number_ions)
    profile.meta['cache_key'] = key

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(profile, f)
    profile.meta['from_cache'] = False
    return profile


def _run_trim_exe(trim: TRIM, srim_dir: str, timeout_s: float) -> Results:
    # Like pysrim's TRIM.run, but with a timeout and a check that outputs are fresh
    # pysrim waits forever if TRIM hangs, and if TRIM dies early it silently parses the
    # previous run's output files. Neither is acceptable when the user can't see the code
    phonon_path = os.path.join(srim_dir, 'PHONON.txt')
    started = time.time()
    current_directory = os.getcwd()
    try:
        os.chdir(srim_dir)
        trim._write_input_files()
        proc = subprocess.Popen([os.path.join(srim_dir, 'TRIM.exe')])
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise TimeoutError(f'TRIM did not finish within {timeout_s / 60:.0f} minutes and was stopped. '
                               'Try fewer ions, or check whether a TRIM error dialog was left open.')
    finally:
        os.chdir(current_directory)

    if returncode != 0:
        raise RuntimeError(f'TRIM.exe exited with code {returncode}.')
    if not os.path.exists(phonon_path) or os.path.getmtime(phonon_path) < started - 1:
        raise RuntimeError('TRIM finished but did not write new output files (PHONON.txt is missing or old). '
                           'Check the inputs, e.g. that the target is not too thin for the beam energy.')
    return Results(srim_dir)


def reduce_results(results, material: Material, ion_symbol: str, energy_eV: float,
                   number_ions: int) -> DamageProfile:
    # Turn pysrim Results into a DamageProfile (Stoller 2019 corrected Quick K-P)
    phon, ioniz, rng = results.phonons, results.ioniz, results.range
    dx = float(np.mean(np.diff(phon.depth)))  # Angstrom

    Tdam_density = phon.recoils - phon.ions
    Tdam = float(np.sum(Tdam_density) * dx)
    Ed = material.nrt_Ed
    disp_density = 0.8 * Tdam_density / (2.0 * Ed)

    dx_i = float(np.mean(np.diff(ioniz.depth)))
    Tdam_balance = float(energy_eV - np.sum(ioniz.ions) * dx_i - np.sum(ioniz.recoils) * dx_i
                         - np.sum(phon.ions) * dx)

    return DamageProfile(
        material=material, ion_symbol=ion_symbol, energy_eV=float(energy_eV),
        number_ions=int(number_ions),
        depth_nm=np.asarray(phon.depth) * 0.1,
        Tdam_density=np.asarray(Tdam_density),
        disp_density=np.asarray(disp_density),
        implanted=np.asarray(rng.ions),
        Tdam_eV=Tdam, N_NRT=0.8 * Tdam / (2.0 * Ed),
        Tdam_balance_eV=Tdam_balance,
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def find_roi(depth_nm: np.ndarray, dpa: np.ndarray, target_mean_dpa: float, tol: float,
             min_depth_nm: float = 200.0, max_width_nm: float = 500.0,
             max_end_nm: Optional[float] = None) -> Optional[dict]:
    # Widest depth window whose mean dpa is within `tol` of `target_mean_dpa`.

    # The window must start below `min_depth_nm` (avoids surface effects), be no wider
    # than `max_width_nm`, lie entirely before the dpa peak, and end no deeper than
    # `max_end_nm` (e.g. a set distance before the implanted-ion peak)
    # Returns {'start_nm', 'end_nm', 'width_nm', 'mean_dpa'} or None
    peak_idx = int(np.argmax(dpa))
    trapz_areas = (dpa[:-1] + dpa[1:]) / 2 * np.diff(depth_nm)
    cum = np.concatenate(([0.0], np.cumsum(trapz_areas)))
    n = len(depth_nm)

    best = None
    for i in range(n - 1):
        if depth_nm[i] < min_depth_nm:
            continue
        j = np.arange(i + 1, n)
        widths = depth_nm[j] - depth_nm[i]
        means = (cum[j] - cum[i]) / widths
        side_ok = j <= peak_idx
        if max_end_nm is not None:
            side_ok &= depth_nm[j] <= max_end_nm
        valid = (np.abs(means - target_mean_dpa) <= tol) & side_ok & (widths <= max_width_nm)
        if valid.any():
            k = int(np.argmax(np.where(valid, widths, -1)))
            if best is None or widths[k] > best['width_nm']:
                best = {'start_nm': float(depth_nm[i]), 'end_nm': float(depth_nm[j[k]]),
                        'width_nm': float(widths[k]), 'mean_dpa': float(means[k])}
    return best


def smooth(y: np.ndarray, window: int = 15, poly: int = 3) -> np.ndarray:
    # Savitzky-Golay smoothing of Monte-Carlo bin noise, this is for display purposes
    from scipy.signal import savgol_filter
    window = min(window, len(y) - (1 - len(y) % 2))
    return savgol_filter(y, window, poly) if window > poly else y


def convergence_table(profiles: List[DamageProfile], fluence: float) -> pd.DataFrame:
    # Per-run totals plus % difference from the run with the most ions
    df = pd.DataFrame([p.summary(fluence) for p in profiles]).sort_values('number_ions')
    ref = df.iloc[-1]
    df['N_NRT_pct_diff'] = (df['N_NRT_per_ion'] - ref['N_NRT_per_ion']) / ref['N_NRT_per_ion'] * 100
    df['peak_dpa_pct_diff'] = (df['peak_dpa'] - ref['peak_dpa']) / ref['peak_dpa'] * 100
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Beam optimiser (experimental)
# ---------------------------------------------------------------------------
# dpa scales linearly with fluence, so for a given energy every allowed depth window
# has exactly one fluence that makes its mean dpa equal the target. Only the energy
# needs TRIM runs; fluence is solved analytically.

OBJECTIVES = {
    'fluence': 'Lowest fluence (shortest beam time)',
    'implant': 'Lowest implanted ions in ROI',
    'width': 'Widest ROI',
}


def sized_material(material: Material, ion_symbol: str, energy_eV: float, srim_dir: str = DEFAULT_SRIM_DIR,
                   pilot_ions: int = 30, use_cache: bool = True, max_tries: int = 4) -> Material:
    """Copy of `material` whose thickness is about 1.3x the depth reached by damage and ions.

    A short pilot run finds the depth; if ions reach the back of the target the pilot
    is repeated with a thicker target. TRIM always uses 100 depth bins, so this keeps
    the bins small enough to resolve the ROI at every energy.
    """
    m = material
    for _ in range(max_tries):
        p = run_trim(m, ion_symbol, energy_eV, pilot_ions, srim_dir, use_cache=use_cache)
        signal = (p.disp_density > 0.005 * p.disp_density.max()) | (p.implanted > 0)
        deepest = p.depth_nm[signal].max()
        if deepest < 0.9 * p.depth_nm[-1]:
            width_nm = max(100.0, np.ceil(1.3 * deepest / 50.0) * 50.0)
            return replace(m, width_A=width_nm * 10)
        m = replace(m, width_A=m.width_A * 3)
    return m


def roi_options(p: DamageProfile, target_dpa: float, tol: float, min_width_nm: float,
                min_depth_nm: float, implant_margin_nm: float, objective: str,
                max_spread: Optional[float] = None) -> Optional[dict]:
    """Best ROI and fluence for one TRIM profile, or None if no window fits.

    Allowed windows: start at or below `min_depth_nm`, at least `min_width_nm` wide,
    wholly before the dpa peak, and ending at least `implant_margin_nm` before the
    implanted-ion peak (same rules as the Fluence & ROI tab). Each window's fluence is
    the one that makes its mean dpa equal `target_dpa`; `tol` gives the acceptable
    fluence range around it.

    `max_spread` (optional, a fraction such as 0.1) additionally requires every point of
    the smoothed dpa profile inside the window to be within target * (1 +/- max_spread).
    """
    d = p.depth_nm
    k = p.dpa(1.0)                    # dpa per (ion/cm2)
    k_smooth = smooth(k)
    imp = p.implanted_appm(1.0)       # appm per (ion/cm2)
    peak_idx = int(np.argmax(k))
    max_end = p.implant_peak_nm - implant_margin_nm
    cum = np.concatenate(([0.0], np.cumsum((k[:-1] + k[1:]) / 2 * np.diff(d))))

    best, best_score = None, None
    for i in range(len(d) - 1):
        if d[i] < min_depth_nm:
            continue
        j = np.arange(i + 1, peak_idx + 1)
        if j.size == 0:
            break
        widths = d[j] - d[i]
        ok = (widths >= min_width_nm) & (d[j] <= max_end)
        if not ok.any():
            continue
        mean_k = (cum[j] - cum[i]) / widths
        fluence = target_dpa / mean_k
        # dpa range in the window uses the smoothed profile: it should show the real gradient, not bin noise.
        seg_k, seg_imp = k_smooth[i:peak_idx + 1], imp[i:peak_idx + 1]
        max_imp = fluence * np.maximum.accumulate(seg_imp)[1:]
        dpa_min = fluence * np.minimum.accumulate(seg_k)[1:]
        dpa_max = fluence * np.maximum.accumulate(seg_k)[1:]
        if max_spread is not None:
            ok &= (dpa_min >= target_dpa * (1 - max_spread)) & (dpa_max <= target_dpa * (1 + max_spread))
            if not ok.any():
                continue
        primary = {'fluence': fluence, 'implant': max_imp, 'width': -widths}[objective]
        score = np.where(ok, primary, np.inf)
        # Tie-break on fluence so equally good windows prefer the shorter beam time.
        m = int(np.lexsort((np.where(ok, fluence, np.inf), score))[0])
        if best is None or (score[m], fluence[m]) < best_score:
            best_score = (score[m], fluence[m])
            best = {
                'energy_MeV': p.energy_eV / 1e6,
                'fluence': float(fluence[m]),
                'fluence_min': float(fluence[m] * (1 - tol / target_dpa)),
                'fluence_max': float(fluence[m] * (1 + tol / target_dpa)),
                'roi_start_nm': float(d[i]),
                'roi_end_nm': float(d[j[m]]),
                'roi_width_nm': float(widths[m]),
                'dpa_min_in_roi': float(dpa_min[m]),
                'dpa_max_in_roi': float(dpa_max[m]),
                'max_implanted_appm_in_roi': float(max_imp[m]),
                'peak_dpa': float(fluence[m] * k.max()),
                'implant_peak_nm': p.implant_peak_nm,
                'bin_nm': p.bin_nm,
            }
    return best


def rank_energies(profiles: List[DamageProfile], target_dpa: float, tol: float, min_width_nm: float,
                  min_depth_nm: float, implant_margin_nm: float, objective: str,
                  max_spread: Optional[float] = None) -> pd.DataFrame:
    """One row per energy (best window at that energy), best first; infeasible energies last."""
    rows = []
    for p in profiles:
        r = roi_options(p, target_dpa, tol, min_width_nm, min_depth_nm, implant_margin_nm, objective, max_spread)
        rows.append(r if r else {'energy_MeV': p.energy_eV / 1e6, 'bin_nm': p.bin_nm})
    df = pd.DataFrame(rows)
    if 'fluence' not in df:
        return df
    key = {'fluence': df['fluence'], 'implant': df['max_implanted_appm_in_roi'],
           'width': -df['roi_width_nm']}[objective]
    return (df.assign(_k=key.fillna(np.inf), _f=df['fluence'].fillna(np.inf))
              .sort_values(['_k', '_f']).drop(columns=['_k', '_f']).reset_index(drop=True))
