---
license: cc-by-4.0
library_name: scikit-learn
tags:
- materials-science
- radiation-damage
- ion-irradiation
- srim
- gaussian-process
- surrogate-model
datasets:
- srimopt/srim-heavy-ion-damage-profiles
---

# SRIM/TRIM surrogates for ion damage profiles (heavy ions, H and He)

Gaussian-process surrogates that predict SRIM-2013 TRIM (Quick Kinchin-Pease) damage-energy and implanted-ion depth
profiles for an ion, target and energy in milliseconds. They were built for srim_opt, a tool that plans ion
irradiations: it chooses the energy and fluence for a target dpa over a region of interest (ROI). A surrogate
proposes them, and one TRIM run confirms the result.

Two models, trained on
[srimopt/srim-heavy-ion-damage-profiles](https://huggingface.co/datasets/srimopt/srim-heavy-ion-damage-profiles):
- **Heavy ions** (`surrogate.npz`): 1,680 TRIM runs, 15 ions from C to Au into 25 targets at 0.5-50 MeV.
- **H and He** (`surrogate_light.npz`): 400 TRIM runs, H at 0.3-6 MeV and He at 0.1-12 MeV into the same 25 targets.

Targets: Fe, Cr, Ni, Cu, W, Mo, Ta, V, Ti, Zr, CuCrZr, Fe-9Cr and SiC; the fusion alloys EUROFER97, V-4Cr-4Ti,
W-5Re and W-5Ta; the fission alloys 316L, T91, HT-9, Zircaloy-4 and Inconel 718; and the high-entropy alloys CoCrFeNi,
CoCrFeMnNi and NiCoCr.

## How it works
The physics that is already known is taken out before anything is learnt:
- **Fluence:** dpa is linear in fluence, so fluence is applied afterwards.
- **Displacement energy:** NRT dpa = 0.8 T_dam / 2E_d, so the model predicts damage energy and E_d is applied afterwards.
- **Depth:** profiles are expressed on u = depth / R_p, with R_p from SRIM's SR Module.
- **Damage energy:** it is learnt relative to the nuclear share of stopping, integrated from SR's S_n and S_e.

Inputs: log ZBL reduced energy, log(M1/M2), Z1 and Z2, with atom-fraction means for alloys and compounds. Outputs: log
damage-energy fraction against the baseline, the stopped fraction, and the PCA weights of the damage and implant
shapes. One scikit-learn `GaussianProcessRegressor` per output, with a Matérn 5/2 kernel (a length scale per input)
plus a white-noise term, tuned on every training run. The predictive spread excludes the fitted noise, which is TRIM's
run-to-run scatter. The light-ion model has its own depth grid (bins of 0.0125 R_p, twice as fine) and compresses the
square root of the shapes, because an H or He profile's plateau is only 1-10 % of its end-of-range peak.

## Accuracy against high-statistics TRIM runs no model trained on
85 test runs (in the dataset's test split): 5,000 ions per heavy-ion run (3,000 on the alloys), 30,000 per H run and
10,000 per He run. Their own Monte Carlo noise in the mean dpa over 0.2-0.6 R_p is about 1 % for heavy ions and 2-7 %
for H and He.

| Test runs | Fluence for the target dose over the ROI, median / p90 | Dose over 0.2-0.6 R_p, median | Damage energy, median |
|---|---|---|---|
| Heavy ions, ion-target pairs in training (41) | 0.9 % / 2.7 % | 1.0 % | 0.3 % |
| Heavy ions into the alloys (24) | 2.0 % / 3.3 % | 2.0 % | 0.4 % |
| Heavy ions, pairs left out of training (41) | 1.7 % / 4.1 % | 1.8 % | 0.3 % |
| H (10) | 4.1 % / 6.8 % | 1.8 % | 0.7 % |
| He (10) | 5.8 % / 7.9 % | 2.7 % | 0.9 % |
| H and He, pairs left out of training (20) | 5.2 % / 7.3 % | 3.1 % | 0.8 % |

The fluence error is measured over the widest window of TRIM's own profile that is flat to ±10 %; it exists for about
two thirds of the heavy-ion runs. "Pairs left out" are scored with a model fitted without those pairs' training runs. The 80 % fluence band is x1.032 (x1.026 for an ion-target pair in training) for heavy ions and
x1.067 for H and He, calibrated on these runs; leave-one-out, it covers 75-80 %. Against v1 (the first release,
trained on the 13 Part 1 targets), the fluence error on the heavy-ion pairs in training fell from 1.4 % to 0.9 %
median and from 4.6 % to 2.7 % at p90. Per-run numbers and the full comparison: `docs/surrogate_v2_report.md` in
the code repository.

**Limits**
- The width of an ROI flat to ±10 % is predicted to about 10 % (median), with wide tails: confirm the window with TRIM.
- Light ions into heavy targets (C into W or W alloys) are the hard corner: 13-31 % fluence error.
- H and He damage is noisy in TRIM itself (rare high-energy recoils carry most of it); a 30,000-ion H confirmation
  still leaves a few per cent of noise over the ROI.
- Outside the training range (lighter than SiC or heavier than W targets, Li to B ions) do not trust a prediction;
  `in_domain` in the prediction says so.

## Files
- `surrogate.npz`, `surrogate.json`: the heavy-ion model (PCA basis, training inputs and outputs, kernel
  hyperparameters; metadata, domain box, calibrated band); no pickled objects.
- `surrogate_light.npz`, `surrogate_light.json`: the H and He model, the same way.

## About the project
srim_opt (code: [github.com/rizzyrr/srim_opt](https://github.com/rizzyrr/srim_opt)) plans ion irradiations. You give a
target dpa and an ROI (start depth, minimum width, distance from the implanted-ion peak, flatness), and it returns the
beam energy and fluence, verified with SRIM. It was built for the OSI Hackathon 2026.
- **The dataset:** 2,080 training runs and 85 test runs, with no failures.
- **The surrogates** (this repository).
- **A one-click optimiser:** a range check with SRIM's SR Module, a surrogate proposal, then one TRIM confirmation
  run (1,000 heavy ions, 30,000 H or He). Over five benchmark cases it needs 5 TRIM runs, against 33 for a
  grid search, and finds the best energy in 4 of 5 (within 5 % in the fifth).
- **Accelerator presets** (11 machines) set the energy window and report the charge state and terminal voltage.
- Part 2: a natural-language interface in which an LLM calls these tools.

## Use
1. Get the code: `git clone https://github.com/rizzyrr/srim_opt.git`, then `pip install -r requirements.txt`
   (Python 3.13). The repository already contains these models in `srim_outputs/surrogate/`.
2. Predict without TRIM. This works for the dataset's ion-target pairs without SRIM installed; other pairs need
   SRIM for the range tables.

```python
import srim_core as core, srim_surrogate as sur
model = sur.load_latest('W')        # the heavy-ion model; load_latest('H') gives the H and He one
p = model.predict_profiles(core.MATERIALS['W'], 'W', [20e6])[0]   # 20 MeV W into W
dpa = p.dpa(1e15)                                                  # dpa vs depth at 1e15 ions/cm2
```

3. Plan an irradiation (needs SRIM-2013 installed; runs one TRIM confirmation):
```python
import srim_optimiser as opt
spec = opt.BeamSpec(target_dpa=3, min_depth_nm=100, min_width_nm=100, implant_margin_nm=100,
                    facility='HZDR 3 MV Tandetron')
print(opt.screen('EUROFER97', 'Fe', spec).message)   # instant prediction
print(opt.plan('EUROFER97', 'Fe', spec).message)     # confirmed with TRIM
```
Or use the app's **Optimise (one-click)** tab (`run_srim_gui.bat`).

To retrain on new runs: `python srim_surrogate.py fit` (and `fit --family light`); `evaluate-test` scores a model on
the test split.

Predictions are of SRIM-2013 Quick K-P, NRT dpa with ASTM E521 displacement energies, not of experiment; SRIM's own
limits apply (amorphous target, no channelling).
