# Core SRIM/TRIM radiation-damage calculations for SRIM app by Rizqullah Rasyiddin 01/10/2026

# Follows the Quick Kinchin-Pease approach of Stoller et al. (2013, erratum 2019) 

#    damage energy density   T_dam(x) = PHONONS by recoils - PHONONS by ions   [eV / A / ion]
#    NRT displacements       N_d(x)   = 0.8 * T_dam(x) / (2 * E_d)            [disp / A / ion]
#    dpa                     dpa(x)   = N_d(x) * fluence / (atomic density * 1e-8 cm/A)

# Initialise packages
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
import types
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# pysrim imports distutils.spawn, which Python 3.12 removed. It only uses find_executable,
# so give it shutil.which under that name before pysrim is imported.
try:
    import distutils.spawn  # noqa: F401
except ImportError:
    _spawn = types.ModuleType('distutils.spawn')
    _spawn.find_executable = shutil.which
    _distutils = types.ModuleType('distutils')
    _distutils.spawn = _spawn
    sys.modules.update({'distutils': _distutils, 'distutils.spawn': _spawn})

# pysrim 0.5.10 also calls yaml.load without a Loader when it builds its element table, which
# PyYAML 6 rejects. Supply the safe loader while pysrim is imported, then restore yaml.load.
import yaml  # noqa: E402

_yaml_load = yaml.load
yaml.load = lambda stream, Loader=yaml.SafeLoader: _yaml_load(stream, Loader=Loader)
try:
    from srim import SR, TRIM, Ion, Layer, Target  # noqa: E402
    from srim.core.elementdb import ElementDB  # noqa: E402
    from srim.input import SRInput  # noqa: E402
    from srim.output import Results, SRResults  # noqa: E402
finally:
    yaml.load = _yaml_load

NA = 6.02214076e23  # atoms/mol
DEFAULT_SRIM_DIR = os.environ.get('SRIM_DIR', os.path.join(os.path.expanduser('~'), 'Documents', 'SRIM'))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'srim_outputs')
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')                  # runs made from the app and the optimiser
DATASET_CACHE_DIR = os.path.join(OUTPUT_DIR, 'dataset_cache')  # surrogate training runs (srim_dataset.py)
TEST_CACHE_DIR = os.path.join(OUTPUT_DIR, 'test_cache')        # high-statistics test runs, never trained on
SR_CACHE_DIR = os.path.join(OUTPUT_DIR, 'sr_cache')            # stopping/range tables

# TRIM and SR read and write fixed files inside their SRIM folder, so only one run may use a folder at a
# time. Parallel runs need one folder copy each, in separate processes (pysrim writes to the cwd).
_dir_locks: Dict[str, threading.Lock] = {}
_dir_locks_guard = threading.Lock()


def _dir_lock(srim_dir: str) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(srim_dir))
    with _dir_locks_guard:
        return _dir_locks.setdefault(key, threading.Lock())


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

    def to_layer(self) -> Layer:
        return Layer(
            {c.symbol: {'stoich': float(x), 'E_d': c.E_d,
                        'lattice': _srim_nonzero(c.lattice), 'surface': _srim_nonzero(c.surface)}
             for c, x in zip(self.constituents, self.fractions)},
            density=self.density,
            width=self.width_A,
        )

    def to_target(self) -> Target:
        return Target([self.to_layer()])


# ASTM E521 recommended displacement energies (eV). ASTM does not cover Si and C, so those are
# commonly used values for SiC.
E_D_ASTM = {'Al': 27.0, 'Ti': 30.0, 'Cu': 30.0, 'V': 40.0, 'Cr': 40.0, 'Mn': 40.0, 'Fe': 40.0, 'Co': 40.0,
            'Ni': 40.0, 'Zr': 40.0, 'Nb': 40.0, 'Mo': 60.0, 'Ta': 90.0, 'W': 90.0, 'Si': 35.0, 'C': 20.0}
# Elements of the named alloys that ASTM E521 does not list: assumed equal to the ASTM value of the lattice
# they sit in (Re in W; Sn and O in Zr; N in Fe; Y and O for ODS steels). plan() and screen() flag them.
E_D_ASSUMED = {'Re': 90.0, 'Sn': 40.0, 'Y': 40.0, 'O': 40.0, 'N': 40.0}
_E_D_NOT_ASTM = {'Si', 'C'} | set(E_D_ASSUMED)


def named_material(composition: Dict[str, float], unit: str = 'at',
                   density: Optional[float] = None) -> Material:
    # Material from {symbol: percent} in atomic ('at') or weight ('wt') %, with ASTM displacement energies
    symbols = list(composition)
    values = np.array([composition[s] for s in symbols], dtype=float)
    at = values if unit == 'at' else convert_composition(symbols, values, to='at')
    missing = [s for s in symbols if s not in E_D_ASTM and s not in E_D_ASSUMED]
    if missing:
        raise ValueError(f'No displacement energy known for {", ".join(missing)}: build the Material with '
                         'Constituent(symbol, at_pct, E_d) instead.')
    return Material(tuple(Constituent(s, float(x), E_D_ASTM.get(s, E_D_ASSUMED.get(s))) for s, x in zip(symbols, at)),
                    density_override=density)


def assumed_Ed_note(material: Material) -> Optional[str]:
    # A sentence naming the constituents whose displacement energy is not an ASTM E521 value, or None
    parts = [f'{c.symbol} {c.E_d:g} eV ({x:.2g} at%)' for c, x in zip(material.constituents, material.fractions * 100)
             if c.symbol in _E_D_NOT_ASTM and E_D_ASTM.get(c.symbol, E_D_ASSUMED.get(c.symbol)) == c.E_d]
    if not parts:
        return None
    return ('Displacement energies not listed in ASTM E521 were assumed: ' + ', '.join(parts)
            + '. Their effect on the mean E_d, and so on dpa, is small unless the element is a major constituent.')


def _balance(base: str, rest: Dict[str, float]) -> Dict[str, float]:
    # Composition with `base` making up 100 % minus the others
    return {base: 100.0 - sum(rest.values()), **rest}


# Targets of the surrogate training set; also the materials the LLM interface can name
MATERIALS: Dict[str, Material] = {
    **{s: named_material({s: 100.0}) for s in ['Fe', 'Cr', 'Ni', 'Cu', 'W', 'Mo', 'Ta', 'V', 'Ti', 'Zr']},
    'CuCrZr': named_material({'Cu': 99.11, 'Cr': 0.82, 'Zr': 0.07}),   # same composition as the app preset
    'Fe-9Cr': named_material({'Fe': 91.0, 'Cr': 9.0}, unit='wt'),      # binary model alloy for EUROFER97
    'SiC': named_material({'Si': 50.0, 'C': 50.0}, density=3.21),
}

# Fusion, fission and high-entropy alloys added for surrogate v2: nominal (mid-specification) compositions in
# wt % (at % for the high-entropy alloys) and room-temperature densities from the literature. Every one has its
# density set: pysrim lists gas densities for N and O, so the ideal-mixing estimate would be wrong for them.
MATERIALS.update({
    # Fusion
    'EUROFER97': named_material(_balance('Fe', {'Cr': 9.0, 'W': 1.1, 'Mn': 0.4, 'V': 0.2, 'Ta': 0.12, 'C': 0.11,
                                                'N': 0.03}), unit='wt', density=7.75),
    'V-4Cr-4Ti': named_material(_balance('V', {'Cr': 4.0, 'Ti': 4.0}), unit='wt', density=6.05),
    'W-5Re': named_material(_balance('W', {'Re': 5.0}), unit='wt', density=19.4),
    'W-5Ta': named_material(_balance('W', {'Ta': 5.0}), unit='wt', density=19.1),
    # Fission
    '316L': named_material(_balance('Fe', {'Cr': 17.0, 'Ni': 12.0, 'Mo': 2.5, 'Mn': 1.8, 'Si': 0.5, 'C': 0.02}),
                           unit='wt', density=8.00),
    'T91': named_material(_balance('Fe', {'Cr': 9.0, 'Mo': 0.95, 'Mn': 0.45, 'Si': 0.35, 'V': 0.22, 'Ni': 0.2,
                                          'Nb': 0.08, 'C': 0.10, 'N': 0.05}), unit='wt', density=7.77),
    'HT-9': named_material(_balance('Fe', {'Cr': 12.0, 'Mo': 1.0, 'Mn': 0.6, 'Ni': 0.5, 'W': 0.5, 'Si': 0.4,
                                           'V': 0.3, 'C': 0.2}), unit='wt', density=7.78),
    'Zircaloy-4': named_material(_balance('Zr', {'Sn': 1.5, 'Fe': 0.2, 'Cr': 0.1, 'O': 0.12}), unit='wt',
                                 density=6.56),
    'Inconel-718': named_material(_balance('Fe', {'Ni': 52.5, 'Cr': 19.0, 'Nb': 5.1, 'Mo': 3.05, 'Ti': 0.9,
                                                  'Al': 0.5}), unit='wt', density=8.19),
    # High-entropy alloys (equiatomic)
    'CoCrFeNi': named_material({'Co': 25.0, 'Cr': 25.0, 'Fe': 25.0, 'Ni': 25.0}, density=8.2),
    'CoCrFeMnNi': named_material({'Co': 20.0, 'Cr': 20.0, 'Fe': 20.0, 'Mn': 20.0, 'Ni': 20.0}, density=8.0),
    'NiCoCr': named_material({'Ni': 100 / 3, 'Co': 100 / 3, 'Cr': 100 / 3}, density=8.3),
})
V1_MATERIALS = ('Fe', 'Cr', 'Ni', 'Cu', 'W', 'Mo', 'Ta', 'V', 'Ti', 'Zr', 'CuCrZr', 'Fe-9Cr', 'SiC')


def material_name(material: Material) -> Optional[str]:
    # MATERIALS name of a material (at any thickness), or None if it is not one of them
    for name, m in MATERIALS.items():
        if m.constituents == material.constituents and m.density_override == material.density_override:
            return name
    return None

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


def _save_pickle(obj, path: str) -> None:
    # Write, then rename, so an interrupted run never leaves a truncated cache file behind
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:   # Windows: another process is reading the file at this moment
            time.sleep(0.2 * (attempt + 1))
    os.remove(tmp)


def _load_pickle(path: str):
    # The unpickled object, or None if the file is missing, truncated or from an incompatible code version
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def cached_profiles(min_ions: int = 31, cache_dir: str = CACHE_DIR) -> List[DamageProfile]:
    # Every valid run saved in the cache, oldest first
    # Skips files whose key no longer matches their inputs (made before the last cache
    # version bump) and runs below `min_ions` (sized_material's 30-ion pilot runs).
    if not os.path.isdir(cache_dir):
        return []
    paths = sorted((os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.pkl')),
                   key=os.path.getmtime)
    profiles = []
    for path in paths:
        try:
            p = _load_pickle(path)
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
             use_cache: bool = True, cache_dir: str = CACHE_DIR, timeout_s: Optional[float] = None,
             minimised: bool = False) -> DamageProfile:
    # Run (or load from cache) one TRIM calculation and reduce it to damage profiles
    # minimised=True starts the TRIM window minimised so background runs don't take the keyboard focus
    problems = material.validate()
    if problems:
        raise ValueError(' '.join(problems))

    key = _cache_key(material, ion_symbol, energy_eV, number_ions, calculation)
    cache_path = os.path.join(cache_dir, key + '.pkl')
    if use_cache and os.path.exists(cache_path):
        profile = _load_pickle(cache_path)
        if profile is not None:   # an unreadable file counts as a miss and is run again
            profile.meta['from_cache'] = True
            return profile

    err = check_srim_directory(srim_dir)
    if err:
        raise FileNotFoundError(err)

    ion = Ion(ion_symbol, energy=energy_eV)
    trim = TRIM(material.to_target(), ion, number_ions=int(number_ions), calculation=calculation)
    started = time.time()
    with _dir_lock(srim_dir):
        # Heavy ions at tens of MeV need ~0.5 s per ion when several TRIM runs share the CPU
        results = _run_trim_exe(trim, srim_dir, timeout_s=timeout_s or max(900.0, 2.5 * number_ions),
                                minimised=minimised)

    profile = reduce_results(results, material, ion_symbol, energy_eV, number_ions)
    profile.meta.update(cache_key=key, wall_s=time.time() - started, srim_dir=srim_dir)
    _save_pickle(profile, cache_path)
    profile.meta['from_cache'] = False
    return profile


def _popen_options(minimised: bool) -> dict:
    # Windows options for background runs: window minimised without taking the focus, below-normal priority
    if not minimised or os.name != 'nt':
        return {}
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = 7   # SW_SHOWMINNOACTIVE
    return {'startupinfo': info, 'creationflags': subprocess.BELOW_NORMAL_PRIORITY_CLASS}


class _HiddenProcess:
    # A process started on the hidden desktop, with the parts of Popen that _run_trim_exe uses
    def __init__(self, handle):
        self.handle = handle

    def wait(self, timeout: float) -> int:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        if kernel32.WaitForSingleObject(wintypes.HANDLE(self.handle), int(timeout * 1000)) == 0x102:   # WAIT_TIMEOUT
            raise subprocess.TimeoutExpired('TRIM.exe', timeout)
        code = wintypes.DWORD()
        kernel32.GetExitCodeProcess(wintypes.HANDLE(self.handle), ctypes.byref(code))
        kernel32.CloseHandle(wintypes.HANDLE(self.handle))
        return int(code.value)

    def kill(self) -> None:
        import ctypes
        from ctypes import wintypes
        ctypes.windll.kernel32.TerminateProcess(wintypes.HANDLE(self.handle), 1)
        ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(self.handle))


_hidden_desktop = None   # handle of this process's hidden desktop, kept open while it runs TRIM


def _start_hidden(exe: str, cwd: str) -> Optional[_HiddenProcess]:
    """Start exe minimised at below-normal priority on a Windows desktop that is never shown; None if Windows refuses.

    TRIM is a Visual Basic program whose message loop goes through the Windows GUI for every ion. On the person's own
    desktop the taskbar and any program that hooks windows (chat apps, overlays) slow it, by 1.6x for a lone run;
    a desktop of its own has neither. Parallel runs still share one GUI lock per Windows session, which caps them at
    about five runs' worth of speed however many cores there are.
    """
    global _hidden_desktop
    if os.name != 'nt':
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class STARTUPINFOW(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('lpReserved', wintypes.LPWSTR), ('lpDesktop', wintypes.LPWSTR),
                        ('lpTitle', wintypes.LPWSTR)] + [(f, wintypes.DWORD) for f in (
                            'dwX', 'dwY', 'dwXSize', 'dwYSize', 'dwXCountChars', 'dwYCountChars', 'dwFillAttribute',
                            'dwFlags')] + [('wShowWindow', wintypes.WORD), ('cbReserved2', wintypes.WORD),
                                           ('lpReserved2', ctypes.c_void_p), ('hStdInput', wintypes.HANDLE),
                                           ('hStdOutput', wintypes.HANDLE), ('hStdError', wintypes.HANDLE)]

        class PROCESS_INFORMATION(ctypes.Structure):
            _fields_ = [('hProcess', wintypes.HANDLE), ('hThread', wintypes.HANDLE), ('dwProcessId', wintypes.DWORD),
                        ('dwThreadId', wintypes.DWORD)]

        user32, kernel32 = ctypes.WinDLL('user32'), ctypes.WinDLL('kernel32')
        name = 'srimopt_trim'
        if _hidden_desktop is None:
            user32.CreateDesktopW.restype = wintypes.HANDLE
            user32.CreateDesktopW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p, wintypes.DWORD,
                                              wintypes.DWORD, ctypes.c_void_p]
            _hidden_desktop = user32.CreateDesktopW(name, None, None, 0, 0x10000000, None)   # GENERIC_ALL
            if not _hidden_desktop:
                _hidden_desktop = False
        if not _hidden_desktop:
            return None
        si = STARTUPINFOW(cb=ctypes.sizeof(STARTUPINFOW), lpDesktop=name, dwFlags=0x1, wShowWindow=7)
        pi = PROCESS_INFORMATION()
        kernel32.CreateProcessW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
                                            wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
                                            ctypes.POINTER(STARTUPINFOW), ctypes.POINTER(PROCESS_INFORMATION)]
        if not kernel32.CreateProcessW(exe, ctypes.create_unicode_buffer(f'"{exe}"'), None, None, False,
                                       subprocess.BELOW_NORMAL_PRIORITY_CLASS, None, cwd, ctypes.byref(si),
                                       ctypes.byref(pi)):
            return None
        kernel32.CloseHandle(pi.hThread)
        return _HiddenProcess(pi.hProcess)
    except Exception:   # anything unexpected: start TRIM the ordinary way
        return None


def _run_trim_exe(trim: TRIM, srim_dir: str, timeout_s: float, minimised: bool = False) -> Results:
    # Like pysrim's TRIM.run, but with a timeout and a check that outputs are fresh
    # pysrim waits forever if TRIM hangs, and if TRIM dies early it silently parses the
    # previous run's output files. Neither is acceptable when the user can't see the code
    phonon_path = os.path.join(srim_dir, 'PHONON.txt')
    started = time.time()
    current_directory = os.getcwd()
    try:
        os.chdir(srim_dir)
        trim._write_input_files()
        exe = os.path.join(srim_dir, 'TRIM.exe')
        proc = (_start_hidden(exe, srim_dir) if minimised else None) or subprocess.Popen([exe], **_popen_options(minimised))
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
# Stopping and range tables (SR Module): ranges in about a second, no Monte Carlo
# ---------------------------------------------------------------------------

SR_E_MAX_EV = 60e6   # one table per ion and material covers every energy the tools use


@dataclass
class RangeTable:
    ion_symbol: str
    energy_eV: np.ndarray
    Se_eV_per_A: np.ndarray    # electronic stopping
    Sn_eV_per_A: np.ndarray    # nuclear stopping
    Rp_nm: np.ndarray          # projected range
    dRp_nm: np.ndarray         # longitudinal straggling

    def at(self, energy_eV: float) -> Tuple[float, float]:
        # (projected range, straggling) in nm, interpolated in log-log
        e, x = np.log(energy_eV), np.log(self.energy_eV)
        return (float(np.exp(np.interp(e, x, np.log(self.Rp_nm)))),
                float(np.exp(np.interp(e, x, np.log(self.dRp_nm)))))

    def energy_for_range(self, Rp_nm: float) -> float:
        # Lowest energy whose projected range reaches Rp_nm; inf if no energy in the table does
        if Rp_nm > self.Rp_nm[-1]:
            return float('inf')
        return float(np.exp(np.interp(np.log(Rp_nm), np.log(self.Rp_nm), np.log(self.energy_eV))))

    def to_json(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, d: dict) -> 'RangeTable':
        return cls(d['ion_symbol'], *(np.asarray(d[k], dtype=float) for k in
                                      ('energy_eV', 'Se_eV_per_A', 'Sn_eV_per_A', 'Rp_nm', 'dRp_nm')))


def _material_key(material: Material) -> dict:
    # Composition and density only: a stopping table does not depend on the target thickness
    m = asdict(material)
    m.pop('width_A')
    m['constituents'] = [{k: (v if k == 'symbol' else float(v)) for k, v in c.items()} for c in m['constituents']]
    m['density'] = float(material.density)
    return m


def sr_table(material: Material, ion_symbol: str, e_max_eV: float = SR_E_MAX_EV,
             srim_dir: str = DEFAULT_SRIM_DIR, use_cache: bool = True, timeout_s: float = 120.0) -> RangeTable:
    # Stopping powers, projected range and straggling from 10 keV to e_max_eV for one ion in one material
    payload = json.dumps({'material': _material_key(material), 'ion': ion_symbol,
                          'e_max_eV': float(e_max_eV), 'v': 1}, sort_keys=True)
    path = os.path.join(SR_CACHE_DIR, hashlib.sha1(payload.encode()).hexdigest()[:16] + '.json')
    if use_cache and os.path.exists(path):
        try:
            with open(path) as f:
                return RangeTable.from_json(json.load(f))
        except Exception:
            pass   # unreadable: compute it again

    err = check_srim_directory(srim_dir)
    if err:
        raise FileNotFoundError(err)
    sr_dir = os.path.join(srim_dir, 'SR Module')
    exe = os.path.join(sr_dir, 'SRModule.exe')
    if not os.path.isfile(exe):
        raise FileNotFoundError(f'SRModule.exe not found in {sr_dir}')

    sr = SR(material.to_layer(), Ion(ion_symbol, energy=e_max_eV), energy_min=10e3, output_type=1)
    out_path = os.path.join(sr_dir, sr.settings.output_filename)
    with _dir_lock(srim_dir):
        started = time.time()
        current_directory = os.getcwd()
        try:
            os.chdir(sr_dir)
            SRInput(sr).write()
            proc = subprocess.Popen([exe], **_popen_options(True))
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise TimeoutError(f'SRModule did not finish within {timeout_s:.0f} s and was stopped.')
        finally:
            os.chdir(current_directory)
        if not os.path.exists(out_path) or os.path.getmtime(out_path) < started - 1:
            raise RuntimeError('SRModule finished but did not write a new SR_OUTPUT.txt.')
        data = SRResults(sr_dir).data   # keV, Se and Sn (eV/A), Rp and straggling (um), lateral straggling (um)

    table = RangeTable(ion_symbol, data[0] * 1e3, data[1], data[2], data[3] * 1e3, data[4] * 1e3)
    os.makedirs(SR_CACHE_DIR, exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(table.to_json(), f)
    os.replace(tmp, path)
    return table


def sized_material_sr(material: Material, ion_symbol: str, energy_eV: float,
                      table: Optional[RangeTable] = None, srim_dir: str = DEFAULT_SRIM_DIR) -> Material:
    """Copy of `material` about 1.4x as thick as the ions reach (Rp + 3 straggling), from SR: no pilot run.

    TRIM always uses 100 depth bins, so this keeps the bins small enough to resolve the ROI at every energy.
    """
    table = table or sr_table(material, ion_symbol, srim_dir=srim_dir)
    Rp, dRp = table.at(energy_eV)
    width_nm = max(100.0, np.ceil(1.4 * (Rp + 3 * dRp) / 50.0) * 50.0)
    return replace(material, width_A=width_nm * 10)


def energy_balance_off(p: DamageProfile) -> bool:
    # True if the damage energy from PHONON.txt and from the energy balance (beam energy less ionisation) differ by
    # more than 5 %. For H and He the damage energy is about 0.1 % of the beam energy, so the balance is a small
    # difference of large numbers: the gap must also exceed 0.5 % of the beam energy to count.
    gap = abs(p.Tdam_eV - p.Tdam_balance_eV)
    return p.Tdam_balance_eV > 0 and gap > 0.05 * p.Tdam_balance_eV and gap > 0.005 * p.energy_eV


def mean_dpa_noise(p: DamageProfile, lo_nm: float, hi_nm: float) -> float:
    # Monte Carlo noise (1 sigma, relative) of a TRIM run's mean dpa over [lo_nm, hi_nm], from the bin-to-bin scatter
    # about its smoothed profile; nan for too few bins. H and He are the noisy cases: rare high-energy recoils carry
    # most of their damage energy, so even 10,000 ions leave a few per cent over a 25-bin window.
    k = p.dpa(1.0)
    sm = smooth(k)
    inside = (p.depth_nm >= lo_nm) & (p.depth_nm <= hi_nm) & (sm > 0)
    if inside.sum() < 3:
        return float('nan')
    return float(np.std((k[inside] - sm[inside]) / sm[inside]) / np.sqrt(inside.sum()))


def target_too_thin(p: DamageProfile) -> bool:
    # True if ions stop in the last few depth bins, so some probably went straight through the target
    return bool(p.implanted.max() > 0 and p.implanted[-5:].max() > 0.02 * p.implanted.max())


def run_sized(material: Material, ion_symbol: str, energy_eV: float, number_ions: int,
              srim_dir: str = DEFAULT_SRIM_DIR, cache_dir: str = CACHE_DIR, table: Optional[RangeTable] = None,
              minimised: bool = False, use_cache: bool = True, max_tries: int = 3) -> DamageProfile:
    # TRIM run on an SR-sized target, made 1.5x thicker and run again if ions reach the back
    m = sized_material_sr(material, ion_symbol, energy_eV, table, srim_dir)
    for _ in range(max_tries):
        p = run_trim(m, ion_symbol, energy_eV, number_ions, srim_dir, use_cache=use_cache, cache_dir=cache_dir,
                     minimised=minimised)
        if not target_too_thin(p):
            return p
        m = replace(m, width_A=m.width_A * 1.5)
    p.meta['too_thin'] = True
    return p


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
                max_spread: Optional[float] = None, *, implant_pct: Optional[float] = None,
                max_fluence: Optional[float] = None, width_slack: float = 0.0) -> Optional[dict]:
    """Best ROI and fluence for one TRIM profile, or None if no window fits.

    Allowed windows: start at or below `min_depth_nm`, at least `min_width_nm` wide,
    wholly before the dpa peak, and ending at least `implant_margin_nm` before the
    implanted-ion peak (same rules as the Fluence & ROI tab). Each window's fluence is
    the one that makes its mean dpa equal `target_dpa`; `tol` gives the acceptable
    fluence range around it.

    `max_spread` (optional, a fraction such as 0.1) additionally requires every point of
    the smoothed dpa profile inside the window to be within target * (1 +/- max_spread).

    `implant_pct` (optional) replaces the distance rule: the smoothed implanted-ion
    concentration must stay at or below `implant_pct` % of its peak everywhere in the window.
    `max_fluence` (optional) drops windows that need more fluence than this (a beam-time cap).
    `width_slack` (objective 'width' only): windows within this fraction of the widest count as equally
    wide and the one needing the lowest fluence wins. A flat near-surface plateau and the rising flank
    can give almost the same width at fluences 5x apart, so e.g. 0.05 avoids paying for 2 % more width.
    """
    if objective == 'width' and width_slack > 0:
        rules = dict(implant_pct=implant_pct, max_fluence=max_fluence)
        widest = roi_options(p, target_dpa, tol, min_width_nm, min_depth_nm, implant_margin_nm, 'width',
                             max_spread, **rules)
        if widest is None:
            return None
        floor = max(min_width_nm, (1 - width_slack) * widest['roi_width_nm'])
        return roi_options(p, target_dpa, tol, floor, min_depth_nm, implant_margin_nm, 'fluence', max_spread,
                           **rules)

    d = p.depth_nm
    k = p.dpa(1.0)                    # dpa per (ion/cm2)
    k_smooth = smooth(k)
    imp = p.implanted_appm(1.0)       # appm per (ion/cm2)
    peak_idx = int(np.argmax(k))
    if implant_pct is None:
        max_end = p.implant_peak_nm - implant_margin_nm
    else:
        # The concentration rises from the surface towards the implant peak, so the window must end
        # before the first depth where the smoothed concentration goes over the limit.
        imp_smooth = smooth(imp)
        over = np.flatnonzero(imp_smooth > implant_pct / 100 * imp_smooth.max())
        first_over = int(over[0]) if over.size else len(d)
        max_end = float(d[first_over - 1]) if first_over > 0 else -np.inf
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
        ok &= mean_k > 0                  # a window with no damage at all (possible in a predicted profile)
        fluence = target_dpa / np.where(mean_k > 0, mean_k, np.nan)
        if max_fluence is not None:
            ok &= fluence <= max_fluence
            if not ok.any():
                continue
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
                'roi_limit_nm': float(max_end),
                'bin_nm': p.bin_nm,
            }
    return best


def rank_energies(profiles: List[DamageProfile], target_dpa: float, tol: float, min_width_nm: float,
                  min_depth_nm: float, implant_margin_nm: float, objective: str,
                  max_spread: Optional[float] = None, *, implant_pct: Optional[float] = None,
                  max_fluence: Optional[float] = None, width_slack: float = 0.0) -> pd.DataFrame:
    """One row per energy (best window at that energy), best first; infeasible energies last."""
    rows = []
    for p in profiles:
        r = roi_options(p, target_dpa, tol, min_width_nm, min_depth_nm, implant_margin_nm, objective, max_spread,
                        implant_pct=implant_pct, max_fluence=max_fluence, width_slack=width_slack)
        rows.append(r if r else {'energy_MeV': p.energy_eV / 1e6, 'bin_nm': p.bin_nm})
    df = pd.DataFrame(rows)
    if 'fluence' not in df:
        return df
    key = {'fluence': df['fluence'], 'implant': df['max_implanted_appm_in_roi'],
           'width': -df['roi_width_nm']}[objective]
    return (df.assign(_k=key.fillna(np.inf), _f=df['fluence'].fillna(np.inf))
              .sort_values(['_k', '_f']).drop(columns=['_k', '_f']).reset_index(drop=True))
