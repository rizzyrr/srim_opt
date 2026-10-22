# Surrogate training set: TRIM runs across ions, targets and energies, run in parallel.
#
#   python srim_dataset.py prepare-workers --n 6    copy the SRIM folder once per worker (once only)
#   python srim_dataset.py run --workers 6          run every planned job; safe to stop and start again
#   python srim_dataset.py status                   progress so far
#   python srim_dataset.py export                   every run as one Parquet file (for Hugging Face)
#
# TRIM and SR write fixed files inside their own folder, and pysrim writes its inputs to the current
# directory, so each worker is a separate process with its own copy of the SRIM folder.

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

import srim_core as core

WORKER_ROOT = os.environ.get('SRIM_WORKERS', os.path.join(os.path.dirname(core.DEFAULT_SRIM_DIR), 'SRIM_workers'))
DATASET_DIR = os.path.join(core.OUTPUT_DIR, 'dataset')
PROGRESS_CSV = os.path.join(DATASET_DIR, 'progress.csv')

TARGETS = ['Fe', 'Cr', 'Ni', 'Cu', 'W', 'Mo', 'Ta', 'V', 'Ti', 'Zr', 'CuCrZr', 'Fe-9Cr', 'SiC']
COMMON_IONS = ['C', 'Si', 'Fe', 'Ni', 'Cu', 'Kr', 'Xe', 'W', 'Au']
SELF_ION = {'CuCrZr': 'Cu', 'Fe-9Cr': 'Fe', 'SiC': 'Si'}   # for pure elements the self-ion is the element

# Energies per ion-target pair: (low MeV, high MeV, points), log-uniform within each band.
# Most published heavy-ion work sits at 1-20 MeV; 20-50 MeV covers large tandems and the
# 10-50 MeV range Zinkle & Snead (2018) recommend.
STRATA_MEV = [(0.5, 1.0, 1), (1.0, 20.0, 7), (20.0, 50.0, 2)]
TRAIN_IONS = 300

# Dense reference sweeps for the optimiser benchmark (ion, target)
BENCHMARK_PAIRS = [('Cu', 'CuCrZr'), ('Fe', 'Fe-9Cr'), ('W', 'W'), ('Ni', 'Ni'), ('Si', 'SiC')]
BENCHMARK_ENERGIES_MEV = np.geomspace(1.0, 50.0, 12)

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


def pair_energies_MeV(ion: str, target: str) -> List[float]:
    # Scrambled log-uniform energies, the same every time for a given pair
    seed = zlib.crc32(f'{ion}->{target}'.encode())
    energies = []
    for k, (lo, hi, n) in enumerate(STRATA_MEV):
        u = qmc.LatinHypercube(d=1, seed=seed + k).random(n)[:, 0]
        energies += [float(f'{np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))):.3g}') for x in u]
    return energies


def plan_jobs() -> List[Job]:
    # Every planned run, in a fixed shuffled order so that a partly finished set still covers the whole space
    jobs = [('train', ion, t, E * 1e6, TRAIN_IONS) for t in TARGETS for ion in ions_for(t)
            for E in pair_energies_MeV(ion, t)]
    jobs += [('bench', ion, t, float(f'{E:.3g}') * 1e6, TRAIN_IONS)
             for ion, t in BENCHMARK_PAIRS for E in BENCHMARK_ENERGIES_MEV]
    order = np.random.default_rng(0).permutation(len(jobs))
    return [jobs[i] for i in order]


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def prepare_workers(n: int) -> List[str]:
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


def _run_job(job: Job) -> dict:
    kind, ion, target, energy_eV, n = job
    row = {'job_id': job_id(job), 'kind': kind, 'ion': ion, 'target': target, 'energy_MeV': energy_eV / 1e6,
           'ions': n, 'worker': os.path.basename(_worker_dir), 'ok': False, 'from_cache': False, 'wall_s': 0.0,
           'cache_key': '', 'too_thin': False, 'error': ''}
    try:
        material = core.MATERIALS[target]
        table = core.sr_table(material, ion, srim_dir=_worker_dir)
        p = core.run_sized(material, ion, energy_eV, n, srim_dir=_worker_dir, cache_dir=core.DATASET_CACHE_DIR,
                           table=table, minimised=True)
        row.update(ok=True, from_cache=bool(p.meta.get('from_cache')), wall_s=float(p.meta.get('wall_s', 0.0)),
                   cache_key=p.meta.get('cache_key', ''), too_thin=bool(p.meta.get('too_thin', False)))
    except Exception as e:   # one bad run must not stop the others
        row['error'] = f'{type(e).__name__}: {e}'.replace('\n', ' ')
    row['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    return row


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


def coverage_line(done_ids: set, jobs: List[Job]) -> str:
    # Finished / planned training runs per target
    parts = []
    for t in TARGETS:
        planned = [j for j in jobs if j[2] == t and j[0] == 'train']
        parts.append(f'{t} {sum(job_id(j) in done_ids for j in planned)}/{len(planned)}')
    return '  '.join(parts)


def run(workers: int, limit: int = 0) -> None:
    jobs = plan_jobs()
    progress = read_progress()
    done_ids = set(progress.loc[progress['ok'] == True, 'job_id'])   # noqa: E712 (pandas bool column)
    todo = [j for j in jobs if job_id(j) not in done_ids]
    if limit:
        todo = todo[:limit]
    print(f'{len(done_ids)} of {len(jobs)} runs already done; running {len(todo)} on {workers} workers.')
    print('TRIM windows open minimised at below-normal priority. Stop with Ctrl+C; run again to resume.')
    if not todo:
        return

    dirs = prepare_workers(workers)
    ctx = mp.get_context('spawn')
    with ctx.Manager() as manager:
        slots = manager.Queue()
        for d in dirs:
            slots.put(d)
        started, finished, failed = time.time(), 0, 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(slots,)) as pool:
            futures = [pool.submit(_run_job, j) for j in todo]
            for fut in as_completed(futures):
                row = fut.result()
                _append_progress(row)
                finished += 1
                if row['ok']:
                    done_ids.add(row['job_id'])
                else:
                    failed += 1
                rate = finished / max(time.time() - started, 1.0) * 3600
                eta = (len(todo) - finished) / rate * 3600 if rate > 0 else float('inf')
                total_done = len(done_ids)
                what = (f"{row['worker']}: {row['ion']} -> {row['target']} {row['energy_MeV']:.3g} MeV, "
                        + (f"{row['wall_s']:.0f} s" if row['ok'] else f"FAILED {row['error'][:80]}"))
                print(f'[{total_done:5d}/{len(jobs)} {total_done / len(jobs):4.0%}]  {rate:4.0f} runs/h  '
                      f'ETA {_fmt_duration(eta):>6}  failed {failed} | {what}', flush=True)
                if finished % 50 == 0:
                    print('  coverage: ' + coverage_line(done_ids, jobs), flush=True)
    print('  coverage: ' + coverage_line(done_ids, jobs))
    print(f'Finished: {len(done_ids)} of {len(jobs)} runs done, {failed} failed this session.')


def status() -> None:
    jobs = plan_jobs()
    progress = read_progress()
    ok = progress[progress['ok'] == True]   # noqa: E712
    done_ids = set(ok['job_id'])
    failed_ids = set(progress.loc[progress['ok'] == False, 'job_id']) - done_ids   # noqa: E712
    print(f'{len(done_ids)} of {len(jobs)} runs done ({len(done_ids) / len(jobs):.0%}), '
          f'{len(failed_ids)} failed and not yet redone.')
    if len(ok):
        recent = pd.to_datetime(ok['time']) > pd.Timestamp.now() - pd.Timedelta(hours=1)
        print(f'{int(recent.sum())} runs finished in the last hour; median TRIM time '
              f"{ok.loc[~ok['from_cache'].astype(bool), 'wall_s'].median():.0f} s.")
    print('coverage: ' + coverage_line(done_ids, jobs))
    for err in progress.loc[progress['job_id'].isin(failed_ids), 'error'].value_counts().head(5).index:
        print(f'  error: {err[:150]}')


def export(path: str = os.path.join(DATASET_DIR, 'srim_profiles.parquet')) -> None:
    # One row per run: inputs, totals and the depth profiles as lists (per incident ion)
    rows = []
    for p in core.cached_profiles(min_ions=TRAIN_IONS, cache_dir=core.DATASET_CACHE_DIR):
        name = target_name(p.material)
        rows.append({
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
    df = pd.DataFrame(rows).sort_values(['target', 'ion', 'energy_MeV'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    write_datasheet(df, os.path.join(os.path.dirname(path), 'DATASET.md'))
    print(f'Wrote {len(df)} runs to {path}')


def write_datasheet(df: pd.DataFrame, path: str) -> None:
    targets = ', '.join(f'{t} ({n})' for t, n in df['target'].value_counts().sort_index().items())
    text = f"""# Heavy-ion damage profiles in fusion and fission materials (SRIM-2013, Quick K-P)

{len(df)} TRIM runs: {df['ion'].nunique()} ions into {df['target'].nunique()} targets,
{df['energy_MeV'].min():.2g}-{df['energy_MeV'].max():.3g} MeV, {TRAIN_IONS} ions per run. Made with srim_opt
(`srim_dataset.py`) to train a fast surrogate for planning ion irradiations.

## How each run was made
- SRIM-2013 TRIM (driven by pysrim), "Ion distribution and quick calculation of damage" (Quick Kinchin-Pease).
- Displacement energies from ASTM E521 (Si 35 eV and C 20 eV for SiC); lattice and surface binding energies 0,
  as Stoller et al. (2013) recommend for Quick K-P damage calculations.
- Damage energy per depth from PHONON.txt following Stoller et al. (2013, erratum 2019); NRT displacements are
  0.8 T_dam / (2 E_d) with the atom-fraction mean E_d.
- Target thickness 1.4 x (projected range + 3 straggling) from SRIM's stopping and range tables, so the 100
  depth bins resolve the profile at every energy; thickened and re-run if ions reached the back.
- Energies per ion-target pair: 1 in 0.5-1 MeV, 7 in 1-20 MeV, 2 in 20-50 MeV (scrambled log-uniform), plus
  12-energy sweeps for five benchmark pairs.

## Targets (runs)
{targets}

Ions: {', '.join(sorted(df['ion'].unique()))}.

## Columns
- `target`, `target_label`, `ion`, `energy_MeV`, `number_ions`, `density_g_cm3`, `atomic_density_cm3`,
  `nrt_Ed_eV`, `target_thickness_nm`
- `damage_energy_keV_per_ion`, `NRT_displacements_per_ion`, `mean_range_nm`, `implant_peak_nm`
- Depth profiles per incident ion, 100 bins (depth = upper bin edge): `depth_nm`,
  `damage_energy_eV_per_A_per_ion`, `NRT_disp_per_A_per_ion`, `implanted_per_cm_per_ion`
  (implanted-ion concentration per unit fluence: (atoms/cm3) / (ions/cm2)).
- dpa at a fluence F (ions/cm2): `NRT_disp_per_A_per_ion * F / (atomic_density_cm3 * 1e-8)`.

## Caveats
- SRIM treats the target as amorphous: no channelling or crystal effects.
- Quick K-P damage, not full-cascade; NRT dpa, not arc-dpa.
- {TRAIN_IONS} ions per run: bin-to-bin Monte Carlo noise of a few per cent.
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
    sub.add_parser('status')
    sub.add_parser('export')
    sub.add_parser('plan')
    args = parser.parse_args(argv)
    if args.cmd == 'prepare-workers':
        prepare_workers(args.n)
    elif args.cmd == 'run':
        run(args.workers, args.limit)
    elif args.cmd == 'status':
        status()
    elif args.cmd == 'export':
        export()
    elif args.cmd == 'plan':
        jobs = plan_jobs()
        print(f'{len(jobs)} jobs: {sum(j[0] == "train" for j in jobs)} training, '
              f'{sum(j[0] == "bench" for j in jobs)} benchmark sweeps, '
              f'{len({(j[1], j[2]) for j in jobs})} ion-target pairs')
        for j in jobs[:8]:
            print('  ', job_id(j))


if __name__ == '__main__':
    main(sys.argv[1:])
