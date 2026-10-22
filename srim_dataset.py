# Surrogate training set: TRIM runs across ions, targets and energies, run in parallel.
#
#   python srim_dataset.py prepare-workers --n 6    copy the SRIM folder once per worker (once only)
#   python srim_dataset.py run --workers 6          run every planned job; safe to stop and start again
#   python srim_dataset.py run --kinds test,alloy   only some kinds of job
#   python srim_dataset.py status                   progress so far
#   python srim_dataset.py export                   every run as one Parquet file (for Hugging Face)
#   python srim_dataset.py throughput --workers 8 12 16    the same jobs on each number of workers
#
# Kinds of job, run in this order:
#   test    high-statistics runs (3,000-5,000 heavy ions, more for H and He) that only score models; they
#           go to their own cache so that no model trains on them
#   alloy   the v2 alloys (fusion, fission, high-entropy): 9 common ions + self-ion x 3 energies, 300 ions
#   light   H and He into all 26 targets, 8 energies each
#   train, bench   the v1 heavy-ion set (13 targets) and the optimiser benchmark sweeps
#
# TRIM and SR write fixed files inside their own folder, and pysrim writes its inputs to the current
# directory, so each worker is a separate process with its own copy of the SRIM folder.

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

import srim_core as core
from srim_optimiser import LIGHT_COST

# Worker copies of SRIM sit next to the SRIM folder (SRIM_WORKERS overrides)
WORKER_ROOT = os.environ.get('SRIM_WORKERS') or (
    os.path.join(os.path.dirname(core.DEFAULT_SRIM_DIR.rstrip('\\/')), 'SRIM_workers') if core.DEFAULT_SRIM_DIR else '')
DATASET_DIR = os.path.join(core.OUTPUT_DIR, 'dataset')
PROGRESS_CSV = os.path.join(DATASET_DIR, 'progress.csv')

TARGETS = list(core.V1_MATERIALS)
ALLOYS = ['EUROFER97', 'V-4Cr-4Ti', 'W-5Re', 'W-5Ta', '316L', 'T91', 'HT-9', 'Zircaloy-4', 'Inconel-718',
          'CoCrFeNi', 'CoCrFeMnNi', 'NiCoCr']
COMMON_IONS = ['C', 'Si', 'Fe', 'Ni', 'Cu', 'Kr', 'Xe', 'W', 'Au']
SELF_ION = {'CuCrZr': 'Cu', 'Fe-9Cr': 'Fe', 'SiC': 'Si',   # for pure elements the self-ion is the element
            'EUROFER97': 'Fe', 'V-4Cr-4Ti': 'V', 'W-5Re': 'W', 'W-5Ta': 'W', '316L': 'Fe',
            'T91': 'Fe', 'HT-9': 'Fe', 'Zircaloy-4': 'Zr', 'Inconel-718': 'Ni',
            'CoCrFeNi': 'Ni', 'CoCrFeMnNi': 'Ni', 'NiCoCr': 'Ni'}   # Ni: the usual ion for HEA irradiations

# Energies per ion-target pair: (low MeV, high MeV, points), log-uniform within each band.
# Most published heavy-ion work sits at 1-20 MeV; 20-50 MeV covers large tandems and the
# 10-50 MeV range Zinkle & Snead (2018) recommend.
STRATA_MEV = [(0.5, 1.0, 1), (1.0, 20.0, 7), (20.0, 50.0, 2)]
ALLOY_STRATA_MEV = [(0.5, 2.0, 1), (2.0, 10.0, 1), (10.0, 50.0, 1)]   # the GP already knows the energy trend
TRAIN_IONS = 300

# Light ions: their own energy ranges and ion counts (damage per ion is sparse, so many more ions per run)
LIGHT_RANGES_MEV = {'H': (0.3, 6.0), 'He': (0.1, 12.0)}
LIGHT_ENERGIES = 8
LIGHT_IONS = {'H': 10000, 'He': 3000}

# Dense reference sweeps for the optimiser benchmark (ion, target), at the optimiser's two ion counts: its
# search runs see 300-ion noise, its confirmation runs and the benchmark's ground truth use 1000 ions
BENCHMARK_PAIRS = [('Cu', 'CuCrZr'), ('Fe', 'Fe-9Cr'), ('W', 'W'), ('Ni', 'Ni'), ('Si', 'SiC')]
BENCHMARK_ENERGIES_MEV = np.geomspace(1.0, 50.0, 12)
BENCHMARK_IONS = (TRAIN_IONS, 1000)

# High-statistics test set: dose noise about 0.5 %, so shape and width errors become measurable.
#   heavy ions, v1 targets: the benchmark pairs at 3 energies inside their benchmark windows, and 13 more pairs
#     (self-ions, and the corners of the domain: C into W, Au into SiC) at one energy in 1-6 MeV and one in 6-35
#   each v2 alloy: its self-ion at 1-6 MeV and another ion at 6-35 MeV (scored before any alloy run is trained on)
#   H and He: 10 ion-target pairs at the low and high end of their range
TEST_IONS = {'heavy': 5000, 'alloy': 3000, 'H': 30000, 'He': 10000}
TEST_BENCH_WINDOWS_MEV = {('Cu', 'CuCrZr'): (1.0, 10.0), ('Fe', 'Fe-9Cr'): (1.2, 18.0), ('W', 'W'): (2.0, 35.0),
                          ('Ni', 'Ni'): (1.0, 20.0), ('Si', 'SiC'): (1.0, 30.0)}
TEST_PAIRS = [('Fe', 'Fe'), ('Au', 'Fe'), ('Kr', 'Cr'), ('Xe', 'Cu'), ('Mo', 'Mo'), ('Ta', 'Ta'), ('V', 'V'),
              ('Ti', 'Ti'), ('Zr', 'Zr'), ('Au', 'SiC'), ('C', 'W'), ('Ni', 'Fe-9Cr'), ('Xe', 'W')]
TEST_ALLOY_IONS = {'EUROFER97': 'W', 'V-4Cr-4Ti': 'Kr', 'W-5Re': 'Cu', 'W-5Ta': 'C', '316L': 'Xe',
                   'T91': 'Si', 'HT-9': 'Ni', 'Zircaloy-4': 'Kr', 'Inconel-718': 'Au', 'CoCrFeNi': 'Xe',
                   'CoCrFeMnNi': 'Cu', 'NiCoCr': 'Au'}   # the second ion for each alloy
TEST_LIGHT_PAIRS = [('H', 'Fe-9Cr'), ('H', 'W'), ('H', '316L'), ('H', 'Zircaloy-4'), ('H', 'SiC'),
                    ('He', 'W'), ('He', 'EUROFER97'), ('He', 'Cu'), ('He', 'V-4Cr-4Ti'), ('He', 'CoCrFeNi')]

# Parts of the SRIM install the worker copies leave out: the installer and the manuals. TRIM silently
# writes no output if its other folders are missing (e.g. 'SRIM Restore'), so everything else is copied.
_SKIP_COPY = {'SRIM-2013-Std.exe', 'SRIM-Setup'}

FIELDS = ['time', 'job_id', 'kind', 'ion', 'target', 'energy_MeV', 'ions', 'ok', 'from_cache', 'wall_s',
          'worker', 'cache_key', 'too_thin', 'error']

Job = Tuple[str, str, str, float, int]   # kind, ion, target, energy_eV, number of ions


def job_id(job: Job) -> str:
    kind, ion, target, energy_eV, n = job
    return f'{kind}|{ion}|{target}|{energy_eV:.6g}|{n}'


def target_name(material: core.Material) -> str:
    # Registry name of a (sized) material, or its label if it is not one of MATERIALS
    return core.material_name(material) or material.label


def ions_for(target: str) -> List[str]:
    self_ion = SELF_ION.get(target, target)
    return COMMON_IONS + ([self_ion] if self_ion not in COMMON_IONS else [])


def _energies(tag: str, strata) -> List[float]:
    # Scrambled log-uniform energies (MeV), one LHS per (low, high, points) band, the same every time for a tag
    seed = zlib.crc32(tag.encode())
    energies = []
    for k, (lo, hi, n) in enumerate(strata):
        u = qmc.LatinHypercube(d=1, seed=seed + k).random(n)[:, 0]
        energies += [float(f'{np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))):.3g}') for x in u]
    return energies


def pair_energies_MeV(ion: str, target: str, strata=STRATA_MEV) -> List[float]:
    return _energies(f'{ion}->{target}', strata)


def _log_bands(lo: float, hi: float, n: int) -> list:
    # n equal bands in log energy between lo and hi, one point each
    edges = np.geomspace(lo, hi, n + 1)
    return [(float(a), float(b), 1) for a, b in zip(edges[:-1], edges[1:])]


def v1_jobs() -> List[Job]:
    # The v1 heavy-ion set and the benchmark sweeps, in a fixed shuffled order so that a partly finished set still
    # covers the whole space (all done in Part 1; their job ids must not change)
    jobs = [('train', ion, t, E * 1e6, TRAIN_IONS) for t in TARGETS for ion in ions_for(t)
            for E in pair_energies_MeV(ion, t)]
    jobs += [('bench', ion, t, float(f'{E:.3g}') * 1e6, n)
             for n in BENCHMARK_IONS for ion, t in BENCHMARK_PAIRS for E in BENCHMARK_ENERGIES_MEV]
    order = np.random.default_rng(0).permutation(len(jobs))
    return [jobs[i] for i in order]


def test_jobs() -> List[Job]:
    jobs = []
    for (ion, t), (lo, hi) in TEST_BENCH_WINDOWS_MEV.items():
        jobs += [('test', ion, t, E * 1e6, TEST_IONS['heavy']) for E in _energies(f'test:{ion}->{t}', _log_bands(lo, hi, 3))]
    for ion, t in TEST_PAIRS:
        jobs += [('test', ion, t, E * 1e6, TEST_IONS['heavy'])
                 for E in _energies(f'test:{ion}->{t}', [(1.0, 6.0, 1), (6.0, 35.0, 1)])]
    for t, other in TEST_ALLOY_IONS.items():
        for ion, band in ((SELF_ION[t], (1.0, 6.0, 1)), (other, (6.0, 35.0, 1))):
            jobs += [('test', ion, t, E * 1e6, TEST_IONS['alloy']) for E in _energies(f'test:{ion}->{t}', [band])]
    for ion, t in TEST_LIGHT_PAIRS:
        jobs += [('test', ion, t, E * 1e6, TEST_IONS[ion])
                 for E in _energies(f'test:{ion}->{t}', _log_bands(*LIGHT_RANGES_MEV[ion], 2))]
    return jobs


def alloy_jobs() -> List[Job]:
    jobs = [('alloy', ion, t, E * 1e6, TRAIN_IONS) for t in ALLOYS for ion in ions_for(t)
            for E in pair_energies_MeV(ion, t, ALLOY_STRATA_MEV)]
    order = np.random.default_rng(1).permutation(len(jobs))
    return [jobs[i] for i in order]


def light_jobs() -> List[Job]:
    jobs = [('light', ion, t, E * 1e6, n) for t in TARGETS + ALLOYS for ion, n in LIGHT_IONS.items()
            for E in pair_energies_MeV(ion, t, [(*LIGHT_RANGES_MEV[ion], LIGHT_ENERGIES)])]
    order = np.random.default_rng(2).permutation(len(jobs))
    return [jobs[i] for i in order]


def job_cost(job: Job) -> float:
    # TRIM time relative to one heavy ion at 3 MeV, for ordering jobs and for ETAs (heavy: Part 1's timings;
    # H and He: LIGHT_COST, measured with `throughput --light`)
    kind, ion, target, energy_eV, n = job
    per_ion = LIGHT_COST.get(ion, 1.0) * (energy_eV / 3e6) ** 0.37
    return n * per_ion


def plan_jobs() -> List[Job]:
    # Every planned run in priority order (see the top of the file). Test runs go longest first, so that the last
    # few do not hold up the pool; each training kind keeps a fixed shuffled order.
    return sorted(test_jobs(), key=job_cost, reverse=True) + alloy_jobs() + light_jobs() + v1_jobs()


def cache_dir_for(kind: str) -> str:
    return core.TEST_CACHE_DIR if kind == 'test' else core.DATASET_CACHE_DIR


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def prepare_workers(n: int) -> List[str]:
    err = core.check_srim_directory(core.DEFAULT_SRIM_DIR)
    if err or not WORKER_ROOT:
        raise SystemExit(err or 'Set SRIM_WORKERS to a folder for the worker copies of SRIM.')
    dirs = [os.path.join(WORKER_ROOT, f'w{k}') for k in range(n)]
    for d in dirs:
        if os.path.isfile(os.path.join(d, 'TRIM.exe')):
            continue
        shutil.copytree(core.DEFAULT_SRIM_DIR, d, dirs_exist_ok=True,
                        ignore=lambda _, names: [x for x in names if x in _SKIP_COPY or x.lower().endswith('.pdf')])
        print(f'Copied SRIM to {d}')
    return dirs


_worker_dir = None


def _init_worker(slots) -> None:
    global _worker_dir
    _worker_dir = slots.get()


def _run_job(job: Job, cache_dir: Optional[str] = None) -> dict:
    kind, ion, target, energy_eV, n = job
    row = {'job_id': job_id(job), 'kind': kind, 'ion': ion, 'target': target, 'energy_MeV': energy_eV / 1e6,
           'ions': n, 'worker': os.path.basename(_worker_dir), 'ok': False, 'from_cache': False, 'wall_s': 0.0,
           'cache_key': '', 'too_thin': False, 'error': ''}
    try:
        material = core.MATERIALS[target]
        table = core.sr_table(material, ion, srim_dir=_worker_dir)
        p = core.run_sized(material, ion, energy_eV, n, srim_dir=_worker_dir,
                           cache_dir=cache_dir or cache_dir_for(kind), table=table, minimised=True)
        row.update(ok=True, from_cache=bool(p.meta.get('from_cache')), wall_s=float(p.meta.get('wall_s', 0.0)),
                   cache_key=p.meta.get('cache_key', ''), too_thin=bool(p.meta.get('too_thin', False)))
    except Exception as e:   # one bad run must not stop the others
        row['error'] = f'{type(e).__name__}: {e}'.replace('\n', ' ')
    row['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    return row


def _pool(jobs: List[Job], dirs: List[str], cache_dir: Optional[str] = None) -> Iterator[dict]:
    # Runs the jobs on one process per SRIM folder in `dirs`, yielding each job's progress row as it finishes.
    # If the caller stops early (Ctrl+C), jobs not yet started are cancelled; running ones finish and are saved.
    ctx = mp.get_context('spawn')
    with ctx.Manager() as manager:
        slots = manager.Queue()
        for d in dirs:
            slots.put(d)
        with ProcessPoolExecutor(max_workers=len(dirs), mp_context=ctx, initializer=_init_worker,
                                 initargs=(slots,)) as pool:
            futures = [pool.submit(_run_job, j, cache_dir) for j in jobs]
            try:
                for fut in as_completed(futures):
                    yield fut.result()
            finally:
                for fut in futures:
                    fut.cancel()


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def read_progress() -> pd.DataFrame:
    if not os.path.exists(PROGRESS_CSV):
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(PROGRESS_CSV)


def _append_progress(row: dict) -> None:
    new = not os.path.exists(PROGRESS_CSV)
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(PROGRESS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def _fmt_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return '--'
    h, m = divmod(int(seconds) // 60, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m'


KINDS = ('test', 'alloy', 'light', 'train', 'bench')
LIVE_STATUS = os.path.join(DATASET_DIR, 'live_status.json')   # read by the Claude Code status line (trim-progress)


def _write_live(**fields) -> None:
    # Best effort: a reader holding the file for a moment must not stop the run
    tmp = f'{LIVE_STATUS}.{os.getpid()}.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump({**fields, 'updated_unix': time.time(), 'updated': time.strftime('%Y-%m-%d %H:%M:%S')}, f)
        os.replace(tmp, LIVE_STATUS)
    except OSError:
        pass


def coverage_line(done_ids: set, jobs: List[Job]) -> str:
    # Finished / planned runs of each kind
    parts = []
    for kind in KINDS:
        planned = [j for j in jobs if j[0] == kind]
        if planned:
            parts.append(f'{kind} {sum(job_id(j) in done_ids for j in planned)}/{len(planned)}')
    return '  '.join(parts)


def run(workers: int, limit: int = 0, kinds: Sequence[str] = KINDS) -> None:
    jobs = plan_jobs()
    progress = read_progress()
    done_ids = set(progress.loc[progress['ok'] == True, 'job_id'])   # noqa: E712 (pandas bool column)
    todo = [j for j in jobs if job_id(j) not in done_ids and j[0] in kinds]
    if limit:
        todo = todo[:limit]
    n_done = sum(job_id(j) in done_ids for j in jobs)
    print(f'{n_done} of {len(jobs)} runs already done; running {len(todo)} ({", ".join(kinds)}) on {workers} workers.')
    print('TRIM runs on a hidden desktop at below-normal priority. Stop with Ctrl+C; run again to resume.')
    if not todo:
        return

    # Progress and ETA in estimated TRIM time rather than runs: a 5000-ion test run costs as much as 17 training runs.
    # The pool starts jobs in order, so the first `workers` unfinished ones are running: count them as half done,
    # or the first long runs would make the ETA many times too long.
    cost = {job_id(j): job_cost(j) for j in todo}
    unfinished = [job_id(j) for j in todo]
    left = sum(cost.values())
    started, finished, failed, histories = time.time(), 0, 0, 0
    planned = {k: [job_id(j) for j in jobs if j[0] == k] for k in KINDS if k in kinds}

    def live(state: str, eta: Optional[float], last: str = '') -> None:
        _write_live(state=state, workers=workers, failed=failed, last=last,
                    ions_per_hour=histories / max(time.time() - started, 1.0) * 3600,
                    eta_s=eta if eta is not None and np.isfinite(eta) else None,
                    kinds=[[k, sum(i in done_ids for i in ids), len(ids)] for k, ids in planned.items() if ids])

    live('running', None)
    state = 'stopped'
    try:
        for row in _pool(todo, prepare_workers(workers)):
            _append_progress(row)
            finished += 1
            left -= cost[row['job_id']]
            unfinished.remove(row['job_id'])
            if row['ok']:
                done_ids.add(row['job_id'])
                histories += row['ions']
            else:
                failed += 1
            elapsed = max(time.time() - started, 1.0)
            running = 0.5 * sum(cost[i] for i in unfinished[:workers])
            rate_cost = (sum(cost.values()) - left + running) / elapsed
            eta = (left - running) / rate_cost if rate_cost > 0 else float('inf')
            n_done += row['ok']
            what = (f"{row['worker']}: {row['kind']} {row['ion']} -> {row['target']} {row['energy_MeV']:.3g} MeV, "
                    f"{row['ions']} ions, " + (f"{row['wall_s']:.0f} s" if row['ok'] else f"FAILED {row['error'][:80]}"))
            print(f'[{n_done:5d}/{len(jobs)} {n_done / len(jobs):4.0%}]  {histories / elapsed * 3600 / 1e3:4.0f}k ions/h  '
                  f'ETA {_fmt_duration(eta):>6}  failed {failed} | {what}', flush=True)
            live('running', eta, what)
            if finished % 50 == 0:
                print('  done: ' + coverage_line(done_ids, jobs), flush=True)
        state = 'finished'
    finally:
        live(state, None)
    print('  done: ' + coverage_line(done_ids, jobs))
    print(f'Finished: {n_done} of {len(jobs)} runs done, {failed} failed this session.')


def status() -> None:
    jobs = plan_jobs()
    progress = read_progress()
    ok = progress[progress['ok'] == True]   # noqa: E712
    done_ids = set(ok['job_id'])
    planned_ids = {job_id(j) for j in jobs}
    failed_ids = set(progress.loc[progress['ok'] == False, 'job_id']) - done_ids   # noqa: E712
    n_done = len(done_ids & planned_ids)
    print(f'{n_done} of {len(jobs)} runs done ({n_done / len(jobs):.0%}), '
          f'{len(failed_ids)} failed and not yet redone.')
    if len(ok):
        recent = pd.to_datetime(ok['time']) > pd.Timestamp.now() - pd.Timedelta(hours=1)
        print(f'{int(recent.sum())} runs finished in the last hour; median TRIM time '
              f"{ok.loc[~ok['from_cache'].astype(bool), 'wall_s'].median():.0f} s.")
    print('done: ' + coverage_line(done_ids, jobs))
    for err in progress.loc[progress['job_id'].isin(failed_ids), 'error'].value_counts().head(5).index:
        print(f'  error: {err[:150]}')


THROUGHPUT_CSV = os.path.join(DATASET_DIR, 'throughput.csv')


def throughput(worker_counts: Sequence[int], n_jobs: int = 48, light: bool = False) -> None:
    # The same jobs on each number of workers, each time into an empty cache so that every run uses TRIM.
    # Heavy: n_jobs v1 training jobs spread evenly over its energies. Light: H and He at 500 ions across their
    # energy ranges and a light, a medium and a heavy target, for the cost per ion.
    if light:
        jobs = [('light', ion, t, E * 1e6, 500) for ion, (lo, hi) in LIGHT_RANGES_MEV.items()
                for t in ('Ti', 'Fe-9Cr', 'W') for E in np.geomspace(lo, hi, 4).round(3)]   # 500 ions: enough to time
    else:
        train = sorted((j for j in v1_jobs() if j[0] == 'train'), key=lambda j: j[3])
        jobs = [train[int(k)] for k in np.linspace(0, len(train) - 1, n_jobs)]
    jobs = sorted(jobs, key=job_cost, reverse=True)
    dirs = prepare_workers(max(worker_counts))
    rows = []
    for w in worker_counts:
        cache = os.path.join(core.OUTPUT_DIR, 'throughput', f'{"light" if light else "heavy"}_{w}')
        shutil.rmtree(cache, ignore_errors=True)
        started = time.time()
        results = list(_pool(jobs, dirs[:w], cache_dir=cache))
        wall = time.time() - started
        failed = [r for r in results if not r['ok']]
        for r in results:
            rows.append({**r, 'workers': w, 'trial_wall_s': wall, 's_per_ion': r['wall_s'] / r['ions']})
        print(f'{w:2d} workers: {wall / 60:4.1f} min for {len(jobs)} runs ({len(failed)} failed), '
              f'{sum(j[4] for j in jobs) / wall * 3600:9,.0f} ion histories per hour, median '
              f'{np.median([r["wall_s"] / r["ions"] for r in results if r["ok"]]):.3f} s per ion per worker', flush=True)
        for r in failed[:3]:
            print(f'   failed: {r["job_id"]}: {r["error"][:120]}')
    df = pd.DataFrame(rows)
    path = THROUGHPUT_CSV.replace('.csv', '_light.csv') if light else THROUGHPUT_CSV
    df.to_csv(path, index=False)
    if light:
        ok = df[df.ok == True]   # noqa: E712
        print(ok.groupby(['ion', 'target', 'energy_MeV']).s_per_ion.median().unstack('target').round(4).to_string())
    print(f'Saved {path}')


def _profile_rows(cache_dir: str, kinds: Dict[str, str]) -> List[dict]:
    # One row per run: inputs, totals and the depth profiles as lists (per incident ion)
    rows = []
    for p in core.cached_profiles(min_ions=TRAIN_IONS, cache_dir=cache_dir):
        if core.target_too_thin(p):
            continue   # a first attempt that ions went through; run_sized redid it with a thicker target
        name = target_name(p.material)
        rows.append({
            'kind': kinds.get(p.meta.get('cache_key', ''), 'other'),
            'target': name, 'target_label': p.material.label, 'ion': p.ion_symbol,
            'energy_MeV': p.energy_eV / 1e6, 'number_ions': p.number_ions,
            'density_g_cm3': p.material.density, 'atomic_density_cm3': p.material.atomic_density,
            'nrt_Ed_eV': p.material.nrt_Ed, 'target_thickness_nm': p.material.width_A / 10,
            'damage_energy_keV_per_ion': p.Tdam_eV / 1e3, 'NRT_displacements_per_ion': p.N_NRT,
            'mean_range_nm': p.mean_range_nm, 'implant_peak_nm': p.implant_peak_nm,
            'depth_nm': p.depth_nm.tolist(),
            'damage_energy_eV_per_A_per_ion': p.Tdam_density.tolist(),
            'NRT_disp_per_A_per_ion': p.disp_density.tolist(),
            'implanted_per_cm_per_ion': p.implanted.tolist(),
        })
    return rows


def export(path: str = os.path.join(DATASET_DIR, 'srim_profiles.parquet')) -> None:
    # The training runs (every kind but 'test') to `path`, the high-statistics test runs beside it (_test.parquet)
    progress = read_progress()
    ok = progress[progress['ok'] == True]   # noqa: E712
    kinds = dict(zip(ok['cache_key'].astype(str), ok['kind']))
    order = ['target', 'ion', 'energy_MeV', 'number_ions']
    df = pd.DataFrame(_profile_rows(core.DATASET_CACHE_DIR, kinds)).sort_values(order)
    test = pd.DataFrame(_profile_rows(core.TEST_CACHE_DIR, kinds))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    if len(test):
        test.sort_values(order).to_parquet(path.replace('.parquet', '_test.parquet'), index=False)
    write_datasheet(df, test, os.path.join(os.path.dirname(path), 'DATASET.md'))
    print(f'Wrote {len(df)} training runs to {path}' + (f' and {len(test)} test runs beside it' if len(test) else ''))


def write_datasheet(df: pd.DataFrame, test: pd.DataFrame, path: str) -> None:
    light = df['ion'].isin(list(LIGHT_IONS))
    heavy, lt = df[~light], df[light]
    targets = ', '.join(f'{t} ({n})' for t, n in df['target'].value_counts().sort_index().items())
    count = lambda d, k: int((d['kind'] == k).sum())   # noqa: E731
    test_line = ''
    if len(test):
        tl = test['ion'].isin(list(LIGHT_IONS))
        counts = lambda d: ' and '.join(f'{n:,}' for n in sorted(d['number_ions'].unique()))   # noqa: E731
        test_line = (f"\n- **Test split** ({len(test)} runs, never trained on), for scoring models: {int((~tl).sum())} "
                     f"heavy-ion runs at {counts(test[~tl])} ions and {int(tl.sum())} H and He runs at "
                     f"{counts(test[tl])}. Their mean dpa over 0.2-0.6 of the range has about 1 % Monte Carlo noise "
                     '(1 sigma) for heavy ions and 2-7 % for H and He.')
    text = f"""# Ion damage profiles in fusion and fission materials (SRIM-2013, Quick K-P)

{len(df)} TRIM runs into {df['target'].nunique()} targets (pure metals, fusion and fission alloys, high-entropy alloys and
SiC). Made with srim_opt (`srim_dataset.py`) to train fast surrogates for planning ion irradiations.
- **Heavy ions** ({len(heavy)} runs): {heavy['ion'].nunique()} ions from C to Au, {heavy['energy_MeV'].min():.2g}-{heavy['energy_MeV'].max():.3g} MeV, {TRAIN_IONS} ions
  per run ({int((heavy['number_ions'] > TRAIN_IONS).sum())} benchmark runs with 1000): {count(df, 'train')} on the 13 Part 1 targets,
  {count(df, 'bench')} benchmark sweeps and {count(df, 'alloy')} on the 12 alloys added in v2.
- **H and He** ({len(lt)} runs): H at {LIGHT_RANGES_MEV['H'][0]:g}-{LIGHT_RANGES_MEV['H'][1]:g} MeV with {LIGHT_IONS['H']:,} ions per run, He at
  {LIGHT_RANGES_MEV['He'][0]:g}-{LIGHT_RANGES_MEV['He'][1]:g} MeV with {LIGHT_IONS['He']:,}: light-ion damage is sparse, so many more ions per run.{test_line}

## How each run was made
- SRIM-2013 TRIM (driven by pysrim), "Ion distribution and quick calculation of damage" (Quick Kinchin-Pease).
- Displacement energies from ASTM E521 (Si 35 eV and C 20 eV for SiC; for Re, Sn, O and N, which ASTM does not list,
  the value of the lattice they sit in: Re 90 eV in W, Sn and O 40 eV in Zr, N 40 eV in steel); lattice and surface
  binding energies 0, as Stoller et al. (2013) recommend for Quick K-P damage calculations.
- Alloys: nominal (mid-specification) compositions and literature room-temperature densities, as in
  `srim_core.MATERIALS`.
- Damage energy per depth from PHONON.txt following Stoller et al. (2013, erratum 2019); NRT displacements are
  0.8 T_dam / (2 E_d) with the atom-fraction mean E_d.
- Target thickness 1.4 x (projected range + 3 straggling) from SRIM's stopping and range tables, so the 100
  depth bins resolve the profile at every energy; thickened and re-run if ions reached the back.
- Energies per ion-target pair, scrambled log-uniform: Part 1 targets 1 in 0.5-1 MeV, 7 in 1-20 MeV, 2 in 20-50 MeV,
  plus 12-energy sweeps (1-50 MeV, at 300 and at 1000 ions) for five benchmark pairs; v2 alloys 1 each in 0.5-2,
  2-10 and 10-50 MeV; H and He {LIGHT_ENERGIES} across their range.

## Targets (runs)
{targets}

Ions: {', '.join(sorted(df['ion'].unique()))}.

## Columns
- `kind` (train, bench, alloy, light; test in the test split), `target`, `target_label`, `ion`, `energy_MeV`,
  `number_ions`, `density_g_cm3`, `atomic_density_cm3`, `nrt_Ed_eV`, `target_thickness_nm`
- `damage_energy_keV_per_ion`, `NRT_displacements_per_ion`, `mean_range_nm`, `implant_peak_nm`
- Depth profiles per incident ion, 100 bins (depth = upper bin edge): `depth_nm`,
  `damage_energy_eV_per_A_per_ion`, `NRT_disp_per_A_per_ion`, `implanted_per_cm_per_ion`
  (implanted-ion concentration per unit fluence: (atoms/cm3) / (ions/cm2)).
- dpa at a fluence F (ions/cm2): `NRT_disp_per_A_per_ion * F / (atomic_density_cm3 * 1e-8)`.

## Caveats
- SRIM treats the target as amorphous: no channelling or crystal effects.
- Quick K-P damage, not full-cascade; NRT dpa, not arc-dpa.
- {TRAIN_IONS} ions per heavy-ion training run: bin-to-bin Monte Carlo noise of a few per cent, enough to make or break a
  narrow-tolerance flat window; check such windows with more ions.
- H and He: rare high-energy recoils carry most of their damage energy, so even 10,000 ions leave 10-35 % noise per
  depth bin and a few per cent over a 25-bin window.
- SRIM's electronic stopping is less accurate for slow heavy ions in light targets.
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare-workers')
    p.add_argument('--n', type=int, default=6)
    p = sub.add_parser('run')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--limit', type=int, default=0, help='run at most this many jobs (0 = all)')
    p.add_argument('--kinds', default=','.join(KINDS), help=f'comma-separated kinds of job ({", ".join(KINDS)})')
    sub.add_parser('status')
    sub.add_parser('export')
    sub.add_parser('plan')
    p = sub.add_parser('throughput')
    p.add_argument('--workers', type=int, nargs='+', default=[8, 12, 16])
    p.add_argument('--jobs', type=int, default=48)
    p.add_argument('--light', action='store_true', help='time H and He runs instead')
    args = parser.parse_args(argv)
    if args.cmd == 'prepare-workers':
        prepare_workers(args.n)
    elif args.cmd == 'run':
        kinds = [k.strip() for k in args.kinds.split(',') if k.strip()]
        unknown = set(kinds) - set(KINDS)
        if unknown:
            parser.error(f'unknown kinds: {", ".join(sorted(unknown))}')
        run(args.workers, args.limit, kinds)
    elif args.cmd == 'status':
        status()
    elif args.cmd == 'export':
        export()
    elif args.cmd == 'plan':
        jobs = plan_jobs()
        print(f'{len(jobs)} jobs, {len({(j[1], j[2]) for j in jobs})} ion-target pairs: '
              + ', '.join(f'{sum(j[0] == k for j in jobs)} {k}' for k in KINDS))
        for kind in KINDS:
            some = [j for j in jobs if j[0] == kind]
            print(f'  {kind:5}: {sum(j[4] for j in some):9,d} ion histories, e.g. ' + '; '.join(job_id(j) for j in some[:2]))
    elif args.cmd == 'throughput':
        throughput(args.workers, args.jobs, args.light)


if __name__ == '__main__':
    main(sys.argv[1:])
