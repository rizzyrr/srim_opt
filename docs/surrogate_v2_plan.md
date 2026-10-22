# Surrogate v2 on the desktop PC: plan for the first 3–4 h session

> **Status (23 Oct 2026):** carried out in the session of 22-23 Oct. Results, the v1-v2 comparison and what is left
> are in `docs/surrogate_v2_report.md`. Differences from this plan: 14YWT was dropped (12 alloys, not 13); TRIM
> throughput is capped by Windows' GUI lock, so 8 workers were used, not 16.

For the Claude Code session on Riz's desktop: start here. It's on the branch `riz-desktop`, made from `riz` after Part 1 was finished.

Hardware: Ryzen 7 5800X3D (8 cores / 16 threads), RTX 2070 Super. Part 1 ran on a laptop with a Ryzen 7 PRO 4750U.

## Context

Today's surrogate looks accurate, but checking it against TRIM's own run-to-run noise changes what "better" means:

- **Noise in one TRIM run:** the five benchmark pairs have 61 points run at both 300 and 1,000 ions. pysrim draws a fresh random seed per run (`srim/srim.py:104`), so the pairs are independent. The dose over the ROI differs by 3.0% median between the two.
  - So a single 300-ion run has about 2.6% median error, a 1,000-ion run about 1.4%, and a 10,000-ion run about 0.5%.
  - The surrogate's measured 2.8% (against 300-ion truth) is therefore mostly the truth's noise. Its own error is probably about 1%.
- **ROI width (±10% flatness) is dominated by noise:**
  - For the same case, the 300- and 1,000-ion runs disagree on whether a window exists in 24 of 61 cases.
  - Where both find one, the widths differ by 20% median.
  - So the surrogate's "25% width error" mostly reflects TRIM's noise, and no improvement can be seen without high-statistics truth.
- **Coverage is the other limit:** heavy ions (C–Au) into 13 targets only. Riz chose:
  - sharper profiles and widths
  - more materials: fusion alloys, fission steels and Zr, HEAs
  - H and He beams

**Throughput**
- TRIM runs single-threaded on the CPU. The full Part 1 dataset took 23.6 worker-hours (447k ion histories, 0.19 s per ion) on the laptop.
- Expect roughly 250–350k heavy-ion histories per hour on the desktop; Step 0 measures it.
- **The GPU is not needed in this session.** sklearn GPs, with hyperparameters tuned on a 500-run subset, scale to about 10k runs. A GPU model (GPyTorch or neural) only pays off beyond that.

**How the session runs**
- Work on `riz-desktop`.
- The hackathon submission (due 08:00 Fri 23 Oct, UTC+8) keeps v1 unless v2 is shown to be better on the high-statistics test set. Merging into `riz` is Riz's decision.
- The API (`BeamSpec`, `screen`, `plan`) must stay the same, because Riz's teammate is building Part 2 (the LLM interface) on it.
- Every queue is resumable, so a later session can carry on.

## Step 0: setup and throughput (~45 min)

**Riz:**
1. Install Claude Code, git and Python 3.13.
2. Clone `rizzyrr/srim_opt` and `git checkout riz-desktop`.
3. Copy `C:\Users\Riz\Documents\SRIM` from the laptop (36 MB), or install SRIM-2013 from srim.org.
4. Run `SRIM-Setup\_SRIM-Setup (Right-Click).bat` **as administrator**. It registers SRIM's VB runtime and needs a UAC click.

**Claude:**
1. Run `py -3.13 -m venv .venv313`, then `.venv313\Scripts\pip install -r requirements.txt`, then `pytest` (40 tests).
2. Make one TRIM test run.
3. Change the default `SRIM_DIR` from the hard-coded `C:\Users\Riz\...` to `~/Documents/SRIM` (`srim_core.py:52`). `SRIM_WORKERS` already follows it.
4. Measure throughput:
   - 8, 12 and 16 workers on the same 48 jobs, then pick the fastest.
   - Time a few H and He runs. Their cost per ion is unknown, and it sets their ion counts and how many fit.

## Step 1: high-statistics test set, run first (~45 min of compute)

New `kind='test'` jobs in `srim_dataset.plan_jobs`, run before anything else:
- about 40 heavy-ion cases at 5,000 ions, spread over ions, targets and energies (1–35 MeV), including the 5 benchmark pairs
- 2 cases at 3,000 ions for each new material (Step 2)
- about 20 H and He cases at high ion counts (Step 3)

The dose noise in this truth is about 0.5%, so shape and width errors become measurable.

A new `srim_surrogate.py evaluate-test` command scores a model on this set. It reports fluence, shape and width errors, and whether the 80% band covers 80%. Run it on v1 first: that's the baseline every change must beat. It reuses `compare()`, `evaluate_on()` and `summarise()`.

## Step 2: more materials (~40 min of compute)

**13 new `core.MATERIALS` entries**, built with `named_material()` from wt% compositions, with literature densities checked during implementation:

| Group | Materials |
|---|---|
| Fusion | EUROFER97 (full composition), V-4Cr-4Ti, W-5Re, W-5Ta, ODS Fe-14Cr-2W-0.3Ti-0.3Y₂O₃ (14YWT) |
| Fission | 316L, T91, HT-9, Zircaloy-4, Inconel 718 |
| High-entropy alloys | CoCrFeNi, CoCrFeMnNi (Cantor), NiCoCr |

- **Displacement energies:** ASTM E521 where it lists one (`E_D_ASTM`). Elements it doesn't list (Re, Sn, Y, O, N) get flagged, assumed values. The surrogate learns damage energy, so E_d only changes the reported dpa.
- **Zero-shot check first:** score v1 on the Step 1 cases for these materials before training on them. That's an honest number for unseen alloys.
- **Training runs:** for each material, the 9 common ions plus the self-ion (V, Zr) × 3 energies at 300 ions, about 360 runs.
- **App:** add the new alloys to the sidebar presets (`srim_app.PRESETS`).

## Step 3: H and He beams (~50 min of compute; exact counts from Step 0 timing)

**A separate "light" model for Z₁ ≤ 2.** Light-ion profiles look different, so one shared PCA basis would serve neither family well:
- a long flat plateau, then a sharp end-of-range peak
- a damage-energy fraction of only about 0.1–1%

**Data:** H at 0.3–6 MeV and He at 0.1–12 MeV, log-uniform. That's 26 targets × 8 energies = 416 runs, aiming for at least 10k ions for H and 3k for He, because light-ion damage is sparse.

**Model:** the same `Surrogate` class.
- The depth grid is stored in the model file, with finer bins near the end of range. `U_EDGES` becomes a per-model array; old files load unchanged.
- The SR nuclear-fraction baseline works as it is.

**Integration:**
- `srim_surrogate.load_latest(ion)` picks the model family.
- `srim_optimiser._surrogate_proposal`, `plan` and `screen` pass the ion through.
- The facility windows already handle H (q = 1, E = 2V) and He.

## Step 4: sharper profiles and widths (modelling, alongside the compute)

Each change is kept only if it beats v1 on the Step 1 test set. The result is an ablation table.
- **Physics inputs from the SR tables:** log(ΔRp/Rp) and the SR nuclear fraction at E, using `sr_nuclear_fraction()` and `RangeTable.at()`. Both encode SRIM's own stopping for the exact ion and target, alloys included.
- **Weighting by ion count:** a per-run noise term proportional to 1/N, via sklearn's `alpha` in `_PerOutputGP.fit`. Then 1,000- to 10,000-ion runs count for more than 300-ion ones.
- **Finer grid and more PCA components** (`_n_components` cap), if the cleaner data supports them.

## Step 5: wrap-up (~30 min)

1. Final fits of the heavy and light models; then `validate`, the zero-shot results for new materials, and recalibration of the band against high-statistics truth.
2. Figures: parity against 5,000-ion truth, width error before and after, and a light-ion overlay.
3. Commit on `riz-desktop` (only when Riz says) and push it.
4. Report v1 against v2. Riz decides whether to merge it into `riz` for the submission.
5. Updating Hugging Face (private repos in the `srimopt` organization) needs a token with access to that organization (optional).

## If time runs short

- **Must:** Step 0, Step 1 with v1's baseline, Step 2 materials registered with the zero-shot check.
- **Should:** Step 3 (light ions), Step 4's SR inputs and ion-count weighting.
- **Stretch, for later sessions:**
  - the finer grid
  - pooling repeat runs to bring 300-ion training runs up to 1,000+ ions (needs a replicate number in the cache key)
  - active learning
  - a GPU model once the dataset is past about 10k runs

## Files

- `srim_core.py`: new materials and E_d values, `SRIM_DIR` default
- `srim_dataset.py`: test, new-material and light-ion jobs with their own ion counts, in priority order
- `srim_surrogate.py`: model families, per-model grid, SR inputs, ion-count weighting, `evaluate-test`
- `srim_optimiser.py`: picks the model family by ion
- `srim_app.py` (presets), `make_figures.py`, `tests/`

## Verification

- `pytest`: the existing 40 tests, plus new ones for:
  - choosing the light model for H and He
  - compositions summing to 100% and sensible densities
  - per-model grid round trip
  - the ion-count weighting
- `srim_dataset.py status` and the terminal progress line while the runs go.
- v1 against v2 on the 5,000-ion test set: fluence median and p90, shape, width, and band coverage.
- New materials: zero-shot error before their runs were added.
- Light ions: error on held-out pairs against high-count truth.
- End to end:
  - `plan()` for 2 MeV H into Fe-9Cr, He into W, and Fe into EUROFER97, each confirmed by TRIM
  - the app's test run

## Working with Riz, and lessons from Part 1

**Conventions**
- British English.
- Commit only when Riz says so, and push only when asked.
- Ask before publishing anything, including Hugging Face. Keep repos private until Riz decides.
- Riz is a radiation-damage researcher (SRIM, NRT dpa, fusion alloys). Explain the ML; don't explain the physics.
- `PART2_HANDOFF.md` is the teammate's guide to the API. Keep it accurate.

**Environment**
- Use `.venv313` (Python 3.13). numpy is pinned to 2.3.5 and pandas to 2.3.3, because newer versions warn together.
- `srim_core` patches pysrim for Python 3.13 (the `distutils` and `yaml.load` shims).
- matplotlib is only for `make_figures.py`.

**SRIM**
- TRIM writes `TRIM.IN` and its outputs in its own folder, so parallel runs need one SRIM copy per worker process.
- A worker copy must include everything except the installer and PDFs. Without folders such as `SRIM Restore`, TRIM silently writes no output.
- Dataset runs start TRIM minimised at below-normal priority.

**Shell**
- PowerShell mangles quoted arguments to native programs, so write commit messages to a file and use `git commit -F <file>`.
- `git push` prints its progress on stderr, which PowerShell reports as an error (exit code 255) even when the push worked.

**Running long jobs**
- The Grep tool (ripgrep) obeys `.gitignore`, so it won't search `.venv313`; use `grep` in Bash for installed packages.
- Run commands longer than 10 minutes in the background.
- GP fits compete with TRIM for the CPU. Run them with `OPENBLAS_NUM_THREADS=4` while TRIM workers are busy, or the TRIM timings used in the benchmark get skewed.

**Analysis**
- Always judge against data the model never saw.
- Report noise, failures and out-of-domain cases honestly.
- Look at every figure before calling it done.
