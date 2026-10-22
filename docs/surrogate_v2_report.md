# Surrogate v2: results of the desktop session (22-23 Oct 2026)

What was done, what it showed, and what Riz needs to decide. The plan is `docs/surrogate_v2_plan.md`.

## The decision

**v2's heavy-ion model is better than v1 on the high-statistics test set, and is the default on `riz-desktop`.**
(`srim_outputs/surrogate/surrogate.npz`; v1 is kept in `srim_outputs/surrogate/v1/`.) Merging into `riz` for the
submission is Riz's call.

| Heavy-ion test runs | v1 (as shipped) | v2 | Paired 90 % interval of the change |
|---|---|---|---|
| Part 1 targets: fluence error, median / p90 | 1.41 % / 4.6 % | **0.91 % / 2.7 %** | -0.97 to +0.05 points |
| Part 1 targets: 0.2-0.6 Rp dose error, median | 1.58 % | **1.00 %** | -1.18 to -0.11 (significant) |
| Part 1 targets: ROI width error, median / p90 | 10.7 % / 34 % | 10.3 % / **27 %** | -3.5 to +5.6 |
| Part 1 targets: damage energy error, median | 0.46 % | **0.27 %** | -0.42 to -0.09 (significant) |
| Part 1 targets: shape error, median | 2.02 % | 1.87 % | -0.34 to +0.02 |
| Alloys: every metric | | | no significant change (below) |
| Fluence band (meant to hold 80 %): factor; coverage on Part 1 targets / alloys | x1.063; 96 % / 94 % | x1.032 (x1.026 for pairs in training); 80 % / 75 % | calibrated on these test runs, so coverage is leave-one-out |

What v2 is: v1's model (same inputs, PCA sizes and kernels), with the kernels tuned on every training run instead of
a random 500, trained on 1,680 runs (v1's 1,350 plus 330 on the alloys), with its fluence band recalibrated on the
high-statistics test runs. Every modelling change the plan proposed was tried and none beat v1 (below). Beside the
heavy-ion model, v2 adds a light-ion model for H and He (below), which v1 did not have at all.

## What was built

| Step | Done | Where |
|---|---|---|
| 0. Setup and throughput | Python 3.13 venv, SRIM-2013 set up, `SRIM_DIR` now defaults to `~/Documents/SRIM`; 8/12/16-worker test (below) | `srim_core.py`, `srim_dataset.py throughput` |
| 1. High-statistics test set | 85 TRIM runs that no model trains on: 41 heavy-ion runs at 5,000 ions on the v1 targets, 24 at 3,000 on the new alloys, 20 H and He runs at 30,000 and 10,000 | `srim_dataset.py` (`kind='test'`), `srim_outputs/test_cache/`, `srim_surrogate.py evaluate-test` |
| 2. More materials | 12 alloys (14YWT dropped at Riz's request) with nominal compositions and literature densities; 330 training runs | `core.MATERIALS`, `srim_app.PRESETS` |
| 3. H and He | A separate light-ion model on its own grid; 400 training runs | `srim_outputs/surrogate/surrogate_light.npz` |
| 4. Modelling changes | Ablations: 10 changes against v1 for heavy ions, 6 for the light model, scored on the test set | `srim_surrogate.py ablation`, `srim_outputs/surrogate/ablation.csv` |
| 5. Wrap-up | Figures, end-to-end `plan()` runs, the app, this report | `make_figures.py`, `srim_outputs/figures/` |

## Throughput: TRIM is limited by Windows, not by the CPU

More workers did not make TRIM faster: total speed is flat from 4 workers to 16 (`srim_dataset.py throughput`).

| Workers | Heavy ions (48 training jobs, 300 ions, 0.5-50 MeV) | H and He (24 jobs, 500 ions) |
|---|---|---|
| 4 | (not run) | 903k ion histories per hour |
| 8 | 156k ion histories per hour | 911k |
| 12 | 144k | (not run) |
| 16 | 149k | 874k |

- Each TRIM process used only about 54 % of a core, and the whole CPU about 25 %. TRIM is a Visual Basic program
  whose message loop goes through the Windows GUI for every ion, and the GUI has one lock per Windows session. One
  job run alone, then 4 and 8 copies at once: 14.4, 46 and 51 ions/s in all.
- With its window visible, or hidden with SW_HIDE, TRIM is about 2.6x slower than minimised.
- On a desktop of its own (never shown; `core._start_hidden`), a lone run is 1.6x faster (23.5 ions/s): there the
  taskbar and the programs that hook windows on the user's desktop (Discord used 0.6 of a core while TRIM ran) no
  longer see it. Eight together reach 56 ions/s, so the session-wide ceiling stays. Background runs
  (`minimised=True`, which the dataset and `plan(..., minimised=True)` use) now start there.
- Against the laptop (6 workers, 107k heavy-ion histories per hour) the desktop is 1.5x faster in all.
- Getting past the ceiling needs a second Windows session or another OS (TRIM under Wine, one wineserver per
  worker); both need admin rights or WSL, which this PC does not have set up.
- H costs TRIM about a sixth of a heavy ion's time per ion and He about a third, so light-ion runs with
  10,000 and 3,000 ions take about as long as 300- to 1,000-ion heavy runs.

## The test set and its own noise

85 runs, none failed (`srim_dataset.py`, `kind='test'`; `srim_outputs/test_cache/`):
- 41 heavy-ion runs at 5,000 ions on the v1 targets: the 5 benchmark pairs at 3 energies inside their benchmark
  windows, and 13 more pairs (self-ions, Au into Fe, Kr into Cr, Xe into Cu, and the corners C into W, Au into
  SiC) at one energy in 1-6 MeV and one in 6-35 MeV;
- 24 runs at 3,000 ions on the new alloys, 2 each (the self-ion at 1-6 MeV, another ion at 6-35 MeV);
- 20 H and He runs, 30,000 and 10,000 ions, on 10 ion-target pairs.

How noisy is this "truth"? From the bin-to-bin scatter about each run's smoothed profile, the mean dpa over 0.2-0.6 Rp
has a 1 sigma noise of **1.0 % (median) in the 5,000-ion runs** and **2-4 % in the 30,000-ion H runs**. H is the
noisy case: rare high-energy recoils carry most of its damage energy (with a 1/T² recoil spectrum up to 10⁴ E_d, one
recoil's damage energy scatters by about 11 times its mean), so even 30,000 H ions leave 14-20 % noise per depth
bin. Errors below these levels cannot be told from TRIM's own noise.

Two metrics were added because the plan's fluence error is defined only where TRIM's own profile has an ROI flat
to ±10 %, which 25 of the 41 v1-target runs and 16 of the 24 alloy runs have (high-energy profiles rise too steeply):
- **0.2-0.6 Rp dose error**: the fluence error for a fixed window on the rising side, defined for every run;
- **truth noise**, as above, beside every error (`truth_noise_fixed` in the per-run CSVs).

## v1 against the test set

The saved v1 model, exactly as shipped (`evaluate-test --model srim_outputs/surrogate/v1/surrogate.npz`):

| | v1 targets (41 runs) | v2 alloys, which v1 never saw (24 runs) |
|---|---|---|
| Fluence error, median / p90 | **1.4 % / 4.6 %** | 2.0 % / 3.1 % |
| 0.2-0.6 Rp dose error, median / p90 | 1.6 % / 3.3 % | 1.5 % / 3.6 % |
| ROI width error, median / p90 | **10.7 %** / 34 % | 10.4 % / 64 % |
| Damage energy error, median | 0.5 % | 0.5 % |
| Shape error, median | 2.0 % | 2.2 % |
| Fluence band (x1.063) covers | 96 % (meant to be 80 %) | 94 % |

Against clean truth v1 is about twice as good as Part 1 could measure: 1.4 % against the 2.8 % measured with 300-ion
truth, and 10.7 % width error against about 25 %. The band is too wide for the same reason: it was calibrated on
noisy truth. The GP's own posterior band (10-90 %) covers only 16-25 %, as Part 1 found; the calibrated band is
what the optimiser reports.

## The ablation

Every variant trained on the same 1,350 runs on the v1 targets (`srim_surrogate.py ablation`); scored on the 41
v1-target test runs with their pairs in training ("seen", at other energies), with those 18 pairs' training runs
left out ("unseen"), and on the 24 alloy runs (zero-shot for every variant). Medians in %; in brackets the bootstrap
90 % interval of the change against v1's configuration refitted on the same runs, in percentage points (an interval
below zero is a significant improvement).

| Variant | Fluence, seen / unseen / alloys | 0.2-0.6 Rp dose, seen / unseen / alloys | Shape, seen / unseen / alloys |
|---|---|---|---|
| v1 (refit) | 1.22 / 1.92 / 2.54 | 1.34 / 1.39 / 1.71 | 2.11 / 2.21 / 2.23 |
| SR inputs | 1.01 / 3.06 / 2.38 | 1.13 / **3.67** (+1.4 to +2.9) / 1.82 | 1.68 / 2.79 (+0.2 to +1.1) / 2.06 |
| ion-count weighting | 1.18 / 1.92 / 2.44 | 1.32 / 1.39 / 1.98 | 1.99 / 2.21 / 2.23 |
| SR + weighting | 0.93 / 3.06 / 2.13 | 1.06 / **3.67** / 1.89 | 1.69 / 2.79 / 2.03 |
| 12 + 10 PCA components | 1.04 / 1.92 / 2.60 | 1.58 / 1.55 / 1.79 | 1.91 / 2.08 / 2.05 |
| kernel restarts | 1.22 / 1.65 / 2.54 | 1.34 / 1.62 / 1.71 | 2.11 / 2.10 / 2.23 |
| kernels on 1000 runs | 1.15 / 1.68 / 2.27 | 1.33 / 1.84 / 1.83 | 2.00 / 2.11 / 2.20 |
| **kernels on every run** | 0.99 / 1.69 / 2.26 | 1.38 / 1.80 / 1.70 | 1.95 / 2.15 / 2.14 |
| sqrt shapes | 0.92 / **0.84** (-1.7 to -0.4) / 1.89 | 1.41 / 2.51 (+0.1 to +1.4) / 1.90 | **1.65 / 1.86 / 1.78** (all < 0) |
| sqrt + kernels on every run | 0.76 / **0.81** (-1.8 to -0.4) / 1.88 | 1.48 / 2.16 / 1.74 | **1.65 / 1.82 / 1.79** (all < 0) |
| sqrt + SR inputs | 0.85 / 2.06 / 1.41 | 1.34 / 4.88 (+1.8 to +5.4) / 1.88 | 1.52 / 2.95 / 1.71 |

ROI width errors (10-19 % medians) moved by less than their own uncertainty (intervals about ±8 points) for every
variant, and are left out of the table; they are in `srim_outputs/surrogate/ablation.csv`.

What it showed:
- **The plan's three changes do not beat v1.** SR inputs help on pairs in training but are significantly worse on
  unseen pairs: the model leans on two inputs that are specific to each pair. Ion-count weighting does nothing, as
  only 60 training runs have more than 300 ions. More PCA components fit noise.
- **The kernel fit is a lottery.** v1's hyperparameters are tuned on a random 500 of the runs. The saved v1 file
  (1,357 runs) and a refit here without the laptop's 7 confirmation runs (1,350) tune on different subsets, and their
  predictions differ by up to 4 % (Si into SiC at 21.5 MeV: +3.8 % against -0.2 % in the 0.2-0.6 Rp dose), as large
  as any variant's effect. Tuning on every run removes it.
- **Square-root shapes** (PCA on sqrt of the shape, which gives a profile's low parts more weight) are the only
  significant gain: better shapes everywhere and a better fluence over the ROI for unseen pairs. But they move error
  to the shallow side of the profile (the 0.2-0.6 Rp dose gets worse for unseen pairs), and that is where the
  optimiser checks flatness: in the optimiser benchmark the sqrt model saw no valid ROI for Fe into Fe-9Cr and needed a
  TRIM search for Ni into Ni, 13 TRIM runs for the five cases against v1's 5. **Rejected.**
- **Adopted for v2: v1's configuration with the kernels tuned on every run.** It is not significantly better, but
  it is not worse anywhere, its point estimates are better in 7 of the 9 columns above, the optimiser benchmark is
  identical to v1's (5 cases, 5 TRIM runs, the same energies), and the fit no longer depends on a random draw.

## The new alloys: zero-shot, then trained

v1 never saw the alloys; v2 trained on 330 runs of them (9 ions plus the self-ion, 3 energies each). On the 24 alloy
test runs:

| | v1 (zero-shot) | v2 (trained) |
|---|---|---|
| Fluence error, median / p90 (16 runs with a window) | 2.0 % / 3.1 % | 2.0 % / 3.3 % |
| 0.2-0.6 Rp dose error, median / p90 | 1.5 % / 3.6 % | 2.0 % / 3.3 % |
| ROI width error, median | 10.4 % | 10.9 % |
| Damage energy error, median | 0.5 % | 0.4 % |
| Shape error, median | 2.2 % | 2.2 % |

No difference is significant (paired bootstrap). **v1 already predicted these alloys as well as a model trained on
them**: their mean Z and mass sit among the 13 v1 targets, and the inputs are physics (reduced energy, mass ratio,
Z1, Z2) rather than material labels. The one large error is C into W-5Ta (+31 % in fluence for v1, +28 % for v2),
the same corner as C into W in the v1 targets: a light ion into a heavy target, whose shallow damage is small and
whose own 3,000-ion TRIM truth carries 6.5 % noise there. What the alloy runs do buy: they are now in the training set,
so their predictions get the narrower in-training band, and they are in the Hugging Face dataset.

## H and He

**TRIM's own noise sets what can be measured.** In the test runs, the mean dpa over 0.2-0.6 Rp has a 1 sigma noise
of 2-4 % with 30,000 H ions and 1-7 % with 10,000 He ions (the most at a few MeV), from 9-35 % noise per depth bin.
Scaled to 10,000 ions, H is at 3.4-6.8 % and He up to 6.9 %. This changed two things in `plan()` (`LIGHT_MIN_IONS`):
H and He are confirmed with at least 30,000 ions (about 3 % for H and 5 % for He at a few MeV, and 2-7 minutes of
TRIM; a 10,000-ion He confirmation at 9 MeV into W left 8 %), and every confirmation now carries a flag when its own
noise leaves the ROI's dose uncertain by more than 2 %.

The training runs (10,000 H ions, 3,000 He) are noisier than that one by one; the model pools about 200 runs per ion.

**The light-ion model** (`srim_outputs/surrogate/surrogate_light.npz`) is the same `Surrogate` class on its own grid
(0-2 Rp in bins of 0.0125 Rp, twice v1's resolution, about TRIM's own bin size for H and He), with 10 + 8 PCA
components of the square root of the shapes: the plateau where an ROI sits is only 1-10 % of the end-of-range peak, so
on the plain scale the PCA would spend itself on the peak. Its ablation (`ablation --family light`), on the 20 test
runs with their pairs in training ("seen") and left out ("unseen"):

| Variant | Fluence, seen / unseen | 0.2-0.6 Rp dose, seen / unseen | Shape, seen / unseen |
|---|---|---|---|
| first try: SR inputs, sqrt shapes | 3.98 / 5.81 | 2.03 / 3.44 | 2.81 / 2.92 |
| plain shapes | 3.25 / 3.73 | 2.90 / 2.86 | 3.18 / 3.28 (worse, significant) |
| **v1 inputs, sqrt shapes (adopted)** | 4.51 / **5.21** (better, significant) | 2.08 / 3.06 | 2.83 / 2.92 |
| v1 inputs, plain shapes | 3.19 / 4.56 (p90 14 %) | 2.50 / 3.08 (p90 14 %) | 3.17 / 3.22 |
| 8 + 6 or 14 + 10 PCA components | 4.5-4.7 / 5.4-5.5 | 2.1 / 3.1-3.2 | 2.8 / 2.9 |

As for heavy ions, the SR inputs were worse on pairs left out of training; without them the model generalises better.
The adopted model against the test runs (seen pairs):

| | H (10 runs, 30,000 ions) | He (10 runs, 10,000 ions) |
|---|---|---|
| 0.2-0.6 Rp dose error, median / p90 | **1.8 % / 4.3 %** | **2.7 % / 7.6 %** |
| Fluence error over TRIM's ROI, median | 4.1 % (10 runs with a window) | 5.8 % (4 runs) |
| Damage energy error, median | 0.7 % | 0.9 % |
| Shape error, median | 3.0 % | 2.5 % |
| Fluence band x1.067 covers | 80 % | 2 of 4 |

The dose errors are at the level of the truth's own noise (2-4 % for H, 1-7 % for He), so the model is about as good
as these runs can show; the fluence over TRIM's flat window is less steady because that window is picked on a noisy
profile. On ion-target pairs it never saw, the dose error is 3.1 % and the fluence error 5.2 % (median).
`srim_outputs/figures/light_overlay.png` shows four test runs against a model trained without their pairs.

## End-to-end

**The optimiser benchmark** (`bench_optimiser.py`; the surrogate retrained without the 5 benchmark pairs) is unchanged
by v2: 5 TRIM runs for the five cases against 25 for the TRIM search and 33 for the app's grid, the best energy in 4
of 5 and 4.8 % short of it in the fifth (Ni into Ni, where both v1 and v2 choose 4.15 MeV over 8.45).

**`plan()` for real, each confirmed with one 1,000-ion TRIM run** (`bench_optimiser.py --confirm`, with the v2 model
as saved):

| Case | Energy | Fluence | ROI | Surrogate's fluence for that ROI against the TRIM run: v1 / v2 | The run's own noise there |
|---|---|---|---|---|---|
| Cu into CuCrZr, 3 dpa, 1-10 MeV | 10 MeV | 7.24e15 | 552-920 nm | +2.0 % / +3.0 % | 2.8 % |
| Fe into Fe-9Cr, 10 dpa, HZDR 3 MV | 18 MeV | 1.07e17 | 208-624 nm | +15.2 % / +10.5 % | 4.2 % |
| W into W, 1 dpa, Helsinki 5 MV | 35 MeV | 4.69e14 | 1472-1853 nm | +1.8 % / +1.6 % | 2.3 % |
| Ni into Ni, 5 dpa, lowest fluence | 3.15 MeV | 5.00e15 | 765-969 nm | +9.5 % / +6.4 % | 1.9 % |
| Si into SiC, 1 dpa, 1-30 MeV | 30 MeV | 1.92e17 | 221-1658 nm | +7.6 % / +7.3 % | 4.1 % |
| **Fe into EUROFER97**, 10 dpa, HZDR 3 MV | 18 MeV | 9.46e16 | 368-683 nm | - / +4.3 % | 8 % (flagged) |

On the same runs and ROIs v2 is the closer in 4 of 5. Both models predict more dose than these runs give, by up to
15 %: the ROI is chosen on the confirmation run itself, a shallow window spans only 8-14 bins of a 1,000-ion run, and
the run's own noise there is 2-4 %. That is why a single confirmation run is a poor yardstick for the model and the
high-statistics test runs are the fair one. It also shows why the new noise flag matters: for the EUROFER97 case,
`plan()` now says the confirmed dose is uncertain by about 8 %.

**H and He, the plan's other two end-to-end cases**, each `screen()` in 1-2 s and then one TRIM confirmation:

| Case | Energy | Fluence | ROI | Surrogate against the TRIM run | The run's own noise there |
|---|---|---|---|---|---|
| H into Fe-9Cr, 1 dpa, 0.5-2 MeV, 30,000 ions (132 s) | 2 MeV | 2.36e19 | 7.8-12.8 µm | -6.8 % | 3 % (flagged) |
| He into W, 1 dpa, HZDR 3 MV, 30,000 ions (386 s) | 9 MeV (He2+, 3 MV) | 9.12e18 | 2.7-5.9 µm | +10.5 % | 5 % (flagged) |
| (the same with 10,000 He ions, 131 s) | 9 MeV | 8.13e18 | 3.2-6.8 µm | +1.8 % | 8 % (flagged) |

The two He confirmations differ from each other by 12 %, which is the point: one TRIM run of a light ion is itself
uncertain by several per cent, and `plan()` now says by how much.

**The app** (Streamlit's test harness, real TRIM runs): the single-run tab and the one-click tab both ran without an
exception (Cu into CuCrZr: 1,069.5 keV of damage energy per ion at 5 MeV; one-click at 1.5 MeV with the then-default
*Lowest fluence* objective, the surrogate within 1 % of TRIM on the confirmed ROI). The alloy presets set their
density; switching back to a pure metal returns to the estimate.

## Left for later sessions

The queue is resumable (`srim_dataset.py run`); each item below adds jobs or a variant without touching what is done.
- **Cleaner heavy-ion training data.** The model is now close to the 300-ion training runs' noise; pooling repeat runs
  (a replicate number in the cache key) to 1,000+ ions per point is the next real gain, about 6 hours of TRIM here.
- **More He ions per training run.** He at a few MeV is as noisy per ion as H (up to 7 % over 0.2-0.6 Rp at 10,000
  ions); the 3,000-ion training runs could go to 10,000 (about 1.5 hours).
- **The light-ion-into-heavy-target corner** (C into W, W-5Ta) has the largest errors, 13-31 % in fluence: more
  training runs there, or the C ion's own model.
- **The ±10 % ROI width** is noise-limited even in 5,000-ion runs (median error about 10 %, wide tails); a flatness
  test on a fitted profile instead of a smoothed one would make it steadier.
- **Throughput** is capped by Windows' GUI lock at about 5 TRIM runs' worth; a second Windows session (a scheduled
  task as another user) or TRIM under Wine would raise it.
- 14YWT was left out; Hugging Face still has v1 (updating needs a token for the `srimopt` organisation, and Riz's
  go-ahead).
- Active learning, and a GPU model once the dataset is past about 10,000 runs, as the plan said.

Also fixed on the way: the app's one-click tab defaulted to *Lowest fluence*, against `BeamSpec`'s and
`OPTIMISER_PROBLEM.md`'s *Widest ROI*; the screen message printed the narrower v2 band as 1.3e+16-1.3e+16 (now three
significant figures); the energy-balance check flagged every H run, whose damage energy is only about 0.1 % of the
beam energy (the gap must now also exceed 0.5 % of the beam energy, as it always does when a heavy-ion run is wrong).
