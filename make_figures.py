# Figures for the write-up, saved as PNGs in srim_outputs/figures/. Each one is skipped, with a note, until
# the command that makes its data has been run:
#   learning_curve.png   python srim_surrogate.py learning-curve
#   parity.png           python srim_surrogate.py validate
#   overlay.png          (fits its own model with every Cu and CuCrZr run left out)
#   benchmark.png        python bench_optimiser.py
#
#   python make_figures.py [learning_curve parity overlay benchmark]

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import Patch                 # noqa: E402
from matplotlib.ticker import ScalarFormatter        # noqa: E402

import srim_core as core                             # noqa: E402
import srim_surrogate as sur                         # noqa: E402

FIG_DIR = os.path.join(core.OUTPUT_DIR, 'figures')
BENCH_CSV = os.path.join(core.OUTPUT_DIR, 'benchmark', 'benchmark.csv')
# The app's colours in fixed order (blue, orange, green, purple: every pair stays apart under colour-blindness);
# markers and line styles carry the same identity
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']
MARKERS = ['o', 's', '^', 'D']
INK, MUTED, GRID = '#1f2328', '#59636e', '#d8dee4'
HOLDOUT_LABELS = {'unseen pairs': 'ion-target pairs not in training', 'Cu + CuCrZr': 'Cu and CuCrZr targets not in training',
                  'W': 'W target not in training', 'SiC': 'SiC target not in training'}

plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.axisbelow': True, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'savefig.dpi': 200,
                     'savefig.bbox': 'tight'})


def _save(fig, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


def learning_curve() -> None:
    path = os.path.join(sur.MODEL_DIR, 'learning_curve.csv')
    if not os.path.exists(path):
        print('learning_curve.png: run "python srim_surrogate.py learning-curve" first')
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(df.training_runs, 100 * df.fluence_error_median, color=COLORS[0], marker='o', lw=2, ms=5,
            label='fluence for the target dose, median')
    if 'fluence_error_p90' in df:
        ax.plot(df.training_runs, 100 * df.fluence_error_p90, color=COLORS[0], marker='o', lw=1.2, ms=4, ls='--',
                mfc='white', label='fluence, 90th percentile')
    ax.plot(df.training_runs, 100 * df.shape_error_median, color=COLORS[1], marker='s', lw=2, ms=5,
            label='damage profile shape, median')
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.set_xticks(df.training_runs)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(bottom=0)
    ax.set_xlabel('TRIM runs in the training set')
    ax.set_ylabel('Error on unseen ion-target pairs (%)')
    ax.legend(fontsize=8.5)
    _save(fig, 'learning_curve.png')


def parity() -> None:
    path = os.path.join(sur.MODEL_DIR, 'validation.csv')
    if not os.path.exists(path):
        print('parity.png: run "python srim_surrogate.py validate" first')
        return
    df = pd.read_csv(path).dropna(subset=['fluence_1dpa_trim', 'fluence_1dpa_pred'])
    lo = 0.6 * min(df.fluence_1dpa_trim.min(), df.fluence_1dpa_pred.min())
    hi = 1.6 * max(df.fluence_1dpa_trim.max(), df.fluence_1dpa_pred.max())
    xs = np.array([lo, hi])
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    ax.fill_between(xs, xs / 1.2, xs * 1.2, color=GRID, alpha=0.7, lw=0, label='within 20 %')
    ax.plot(xs, xs, color=MUTED, lw=0.8)
    for k, (group, g) in enumerate(df.groupby('holdout', sort=False)):
        err = 100 * np.median(np.abs(g.fluence_error))
        ax.scatter(g.fluence_1dpa_trim, g.fluence_1dpa_pred, s=24, marker=MARKERS[k % 4], color=COLORS[k % 4],
                   edgecolor='white', linewidth=0.6, zorder=3,
                   label=f'{HOLDOUT_LABELS.get(group, group)}: median error {err:.0f} % (n = {len(g)})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('Fluence for 1 dpa over the ROI, TRIM (ions/cm²)')
    ax.set_ylabel('Predicted by the surrogate (ions/cm²)')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.13), fontsize=8)
    _save(fig, 'parity.png')


def _pick(runs, energies_MeV):
    # The run nearest each energy, preferring the one with the most ions among those within 15 %
    picked = []
    for E in energies_MeV:
        near = [p for p in runs if abs(np.log(p.energy_eV / 1e6 / E)) < 0.15]
        p = (max(near, key=lambda q: q.number_ions) if near
             else min(runs, key=lambda q: abs(np.log(q.energy_eV / 1e6 / E))))
        if p not in picked:
            picked.append(p)
    return sorted(picked, key=lambda q: q.energy_eV)


def overlay(group: str = 'Cu + CuCrZr', ion: str = 'Cu', target: str = 'CuCrZr',
            energies_MeV=(2.0, 5.0, 12.0)) -> None:
    runs = sur.load_runs()
    names = sur.HOLDOUT_GROUPS[group]
    cands = [p for n, p, _ in runs if n == target and p.ion_symbol == ion]
    if not cands:
        print(f'overlay.png: no {ion} -> {target} runs yet')
        return
    print(f'overlay.png: training a surrogate without any {group} runs...', flush=True)
    model = sur.Surrogate.fit(sur.build_training_set([r for r in runs if r[0] not in names]))
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    x_max = 0.0
    for k, p in enumerate(_pick(cands, energies_MeV)):
        mean = model.predict_profiles(p.material, ion, [p.energy_eV])[0]
        draws = model.predict_profiles(p.material, ion, [p.energy_eV], n_samples=50)
        band = np.percentile([d.dpa(1e15) for d in draws], [10, 90], axis=0)
        ax.fill_between(mean.depth_nm / 1e3, band[0], band[1], color=COLORS[k], alpha=0.18, lw=0)
        ax.plot(mean.depth_nm / 1e3, mean.dpa(1e15), color=COLORS[k], lw=2, ls='--')
        ax.plot(p.depth_nm / 1e3, p.dpa(1e15), color=COLORS[k], lw=1.1)
        peak = int(np.argmax(mean.dpa(1.0)))     # label the smooth predicted peak, not a noise spike in TRIM
        ax.annotate(f'{p.energy_eV / 1e6:.3g} MeV', xy=(mean.depth_nm[peak] / 1e3, mean.dpa(1e15)[peak]),
                    xytext=(6, 10), textcoords='offset points', fontsize=8.5, color=INK)
        k_true = p.dpa(1.0)
        x_max = max(x_max, p.depth_nm[k_true > 0.01 * k_true.max()].max() / 1e3)
    ax.set_xlim(0, 1.08 * x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Depth (µm)')
    ax.set_ylabel('dpa at 10¹⁵ ions/cm²')
    ax.set_title(f'{ion} → {target}: surrogate trained without any Cu or CuCrZr target', fontsize=9.5, loc='left')
    ax.legend(handles=[Line2D([], [], color=MUTED, lw=1.1, label='TRIM'),
                       Line2D([], [], color=MUTED, lw=2, ls='--', label='surrogate (mean)'),
                       Patch(color=MUTED, alpha=0.25, lw=0, label="surrogate's 80 % band")],
              loc='upper right', fontsize=8.5)
    _save(fig, 'overlay.png')


def benchmark() -> None:
    if not os.path.exists(BENCH_CSV):
        print('benchmark.png: run "python bench_optimiser.py" first')
        return
    df = pd.read_csv(BENCH_CSV)
    cases = list(dict.fromkeys(df.case))
    strategies = list(dict.fromkeys(df.strategy))
    h = 0.8 / len(strategies)
    fig, ax = plt.subplots(figsize=(7.0, 0.8 + 0.85 * len(cases)))
    for k, s in enumerate(strategies):
        sub = df[df.strategy == s].set_index('case').reindex(cases)
        y = np.arange(len(cases)) + (k - (len(strategies) - 1) / 2) * h
        ax.barh(y, sub.trim_minutes, height=0.9 * h, color=COLORS[k], label=s)
        for yi, minutes, n, short in zip(y, sub.trim_minutes, sub.trim_runs, sub.shortfall_vs_best):
            if pd.isna(minutes):
                continue
            verdict = 'no answer' if pd.isna(short) else ('best energy' if short < 0.005 else f'{short:.0%} short of best')
            ax.text(minutes, yi, f'  {n:.0f} runs, {verdict}', va='center', fontsize=7.5, color=MUTED)
    ax.set_yticks(np.arange(len(cases)), cases)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, 1.45 * df.trim_minutes.max())
    ax.set_xlabel('TRIM time spent choosing the energy (minutes)')
    ax.legend(loc='lower right', fontsize=8)
    _save(fig, 'benchmark.png')


FIGURES = {'learning_curve': learning_curve, 'parity': parity, 'overlay': overlay, 'benchmark': benchmark}

if __name__ == '__main__':
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
