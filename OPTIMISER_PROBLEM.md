# One-click beam optimiser: problem description

## Goal

Given a target **mean dpa** in a **region of interest (ROI)** below the surface of a sample, find the
**incident ion energy E** and **fluence Φ** that achieve it, with one button press and no manual
energy sweeps.

The user supplies only what they care about experimentally (target dpa, how deep the ROI must
start, how wide it must be, etc.). The tool returns E, Φ, the ROI position and enough diagnostics to
trust the answer.

## Context

- Damage is computed with SRIM/TRIM (via pysrim) using the Quick Kinchin-Pease method of Stoller et
  al. (2013, erratum 2019). See `srim_core.py`.
- Ion species and target material (composition, density, displacement energies) are fixed inputs
  from the app sidebar. They are **not** optimised.
- An experimental optimiser already exists (the *Optimise* tab in `srim_app.py`, backed by
  `roi_options`, `rank_energies` and `sized_material` in `srim_core.py`). It works but needs the user to
  choose an energy range, the number of grid points and whether to refine. This document describes
  what the one-click version should do.

## Physics that shapes the problem

For one TRIM run at energy E, the NRT displacement profile per ion is `k(x; E)`, in dpa per
(ion/cm²). Then

```
dpa(x; E, Φ) = Φ · k(x; E)
```

Consequences:

1. **Fluence is not a search variable.** For a fixed E and a fixed window [x₀, x₁], the fluence that
   gives the target mean dpa is exact:

   ```
   Φ = D_target / mean_{x₀..x₁} k(x; E)
   ```

2. **Only energy is expensive.** Each new E needs a TRIM run, which takes seconds to minutes
   depending on energy and ion count. Changing dpa, width, depth or objective only re-ranks
   existing runs.
3. **TRIM output is noisy.** It is a Monte Carlo result, so `k(x; E)` has bin-to-bin noise that
   shrinks with more ions.
4. **Resolution is fixed.** TRIM always uses 100 depth bins across the target thickness, so the
   target must be sized for each E (`sized_material`) or the ROI is poorly resolved.

So the problem is a **1-D optimisation over E of an expensive, noisy function**, with an inner,
cheap, exact search over ROI windows and fluence.

## Inputs (what the user enters)

| Input | Symbol | Example | Notes |
|---|---|---|---|
| Target mean dpa in ROI | D | 3.0 dpa | Required |
| Tolerance on mean dpa | ±δ | 0.15 dpa | Gives an acceptable fluence range |
| Minimum ROI start depth | d_min | 100 nm | ROI may start anywhere at or deeper than this |
| Minimum ROI width | w | 100 nm | Wider is fine |
| Implant limit, distance mode | m | 100 nm | ROI ends at least m before the implanted-ion peak |
| Implant limit, percentage mode | p | 10 % | Implanted concentration in the ROI ≤ p % of its peak value |
| Max dpa spread in ROI | ±s | 10 % | On by default (as flat as possible) |
| Accelerator energy limits | E_min, E_max | 1–10 MeV | Hardware limits, not search settings |
| Objective | | widest ROI | Selectable; see below |

Ion, material and number of ions come from the sidebar as they do now.

## Constraints on a valid ROI [x₀, x₁]

- `x₀ ≥ d_min`
- `x₁ − x₀ ≥ w`
- `x₁ ≤` depth of the dpa peak (the ROI sits on the rising side of the profile)
- Implant limit, one of:
  - distance mode: `x₁ ≤ implant_peak(E) − m`
  - percentage mode: `c_imp(x) ≤ (p/100) · max(c_imp)` for every x in the ROI (smoothed profile)
- Flatness (on by default): `D(1 − s) ≤ Φ·k_smooth(x) ≤ D(1 + s)` for every x in the ROI
- `E_min ≤ E ≤ E_max`

## Objectives (pick one)

1. **Widest ROI** (default): most material for post-irradiation characterisation.
2. **Lowest fluence**: shortest beam time.
3. **Fewest implanted ions in the ROI**: least chemical contamination from the beam.

Ties are broken by lower fluence.

## Outputs

- Best E (MeV) and Φ (ions/cm²), plus the acceptable fluence range from ±δ
- ROI start, end and width
- dpa minimum and maximum inside the ROI, peak dpa, implanted appm in the ROI
- Depth plot of dpa and implanted ions with the ROI, implant peak and ROI limit marked
- Objective vs energy curve, showing how flat the optimum is
- Quality flags: bin size relative to ROI width, TRIM noise or convergence, and whether the optimum
  sits at E_min or E_max (meaning the hardware limit is the binding constraint)

## What "one click" should mean

- **No energy grid settings.** The tool brackets the feasible energy range itself, starting from a
  physics-based guess (for example, the energy whose range is about `d_min + w + m` beyond the ROI),
  then narrows down using as few TRIM runs as possible.
- **Automatic target sizing** for every energy, as `sized_material` does now.
- **Clear stopping rule.** Stop when another TRIM run would not change E by more than a set amount,
  or not improve the objective beyond TRIM noise.
- **Two-stage statistics.** Search with 300 ions per run, then confirm the chosen E with at least 1000
  ions and report the final numbers from that run.
- **Time budget of about 5 minutes.** Estimate the time per run from earlier runs and stop searching
  early if the budget would be exceeded, keeping enough time for the confirmation run.
- **Reuse the cache.** Never repeat a run that is already cached.
- **Helpful failure.** If nothing is feasible, say which constraint blocks it, for example "E_max is
  too low to reach d_min + w + m".

## Search strategy ideas (to evaluate)

- Golden-section or Brent search on E, assuming the objective is unimodal in E (needs checking)
- Coarse log grid, then a 1-D surrogate (spline or Gaussian process) to pick the next E
- Interpolating `k(x; E)` between runs to estimate the optimum without another TRIM run, then
  confirming it with one final run
- Few ions while exploring, then a full-statistics run at the chosen E

## Decisions

| Question | Decision |
|---|---|
| ROI width | **Minimum.** Wider is fine. |
| ROI start | **Anywhere at or deeper than d_min.** Deeper is fine; it keeps clear of surface effects. |
| Past the dpa peak? | **No.** The ROI stays on the rising side of the profile. |
| Flatness | **As flat as possible.** The flatness limit is on by default (±s, default 10 %). |
| Default objective | **Widest ROI**, and the user can still pick another objective. |
| Ions per run | **300 while searching**, then **at least 1000** for the final confirmation run at the chosen E. |
| Implant limit | **The user chooses the type of limit:** a distance gap before the implant peak (m, as now), *or* a cap of p % of the peak implanted concentration anywhere in the ROI. |
| Runtime budget | **About 5 minutes** per click, including the confirmation run. |

### Note: flatness vs widest ROI

These two goals pull in opposite directions. A tighter flatness limit forces a narrower ROI. So
"widest ROI" is solved **at the chosen flatness limit**, with flatness as a constraint and width as the
objective. The existing "fluence vs allowed spread" trade-off plot then shows the cost of choosing a
tighter limit.

### Still to check

- [ ] **Is the objective unimodal in E?** "Unimodal" means the objective, plotted against E, has a
      single best point and gets steadily worse on either side, with no second smaller hump to
      mislead the search. If it holds, a simple bracketing search (golden-section or Brent) finds the
      optimum in a few runs. If it doesn't, the search needs a coarse grid first. Check with a dense
      sweep, for example Cu → CuCrZr, 1–10 MeV, about 15 energies.
- [ ] **Does "widest ROI" just push to E_max?** Higher energies give a longer, flatter rising side, so
      the widest ROI may simply land at the highest energy allowed. If that holds, the one-click search
      is mostly about confirming feasibility at E_max and reporting the fluence cost. In that case,
      consider a beam-time cap (maximum Φ) as an extra optional constraint.
- [ ] Is 300 ions enough for the flatness check? The spread uses the smoothed profile, but noise at
      300 ions may still reject or accept windows wrongly. Compare 300 vs 1000 at a few energies.

## Out of scope (for now)

- Optimising ion species or multi-energy (stacked) irradiations
- Full-cascade TRIM mode (calculation = 2)
- Dose-rate or temperature effects
