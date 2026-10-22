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

# SRIM/TRIM surrogate for heavy-ion damage profiles

A Gaussian-process surrogate that predicts SRIM-2013 TRIM (Quick Kinchin-Pease) damage-energy and
implanted-ion depth profiles for a heavy ion, target and energy in milliseconds. It was built for srim_opt,
a tool that plans ion irradiations: it chooses the energy and fluence for a target dpa over a region of
interest. The surrogate proposes them, and one TRIM run confirms the result.

Trained on 1,357 TRIM runs: 13 targets (Fe, Cr, Ni, Cu, W, Mo, Ta, V, Ti, Zr, CuCrZr, Fe-9Cr, SiC),
15 heavy ions (C to Au) and 0.5-50 MeV, from
[srimopt/srim-heavy-ion-damage-profiles](https://huggingface.co/datasets/srimopt/srim-heavy-ion-damage-profiles).

## How it works
The physics that is already known is taken out before anything is learnt:
- **Fluence:** dpa is linear in fluence, so fluence is applied afterwards.
- **Displacement energy:** NRT dpa = 0.8 T_dam / 2E_d, so the model predicts damage energy and E_d is applied afterwards.
- **Depth:** profiles are expressed on u = depth / R_p, with R_p from SRIM's SR Module.
- **Damage energy:** it is learnt relative to the nuclear share of stopping, integrated from SR's S_n and S_e.

Inputs: log ZBL reduced energy, log(M1/M2), Z1 and Z2, with atom-fraction means for alloys and compounds.
Outputs: log damage-energy fraction against the baseline, the stopped fraction, and the PCA weights of the
damage shape (8) and implant shape (6). There is one scikit-learn `GaussianProcessRegressor` per output,
with a Matérn 5/2 kernel with a length scale per input plus a white-noise term. The predictive spread
excludes the fitted noise, which is TRIM's run-to-run scatter. The 80 % fluence band is calibrated on
held-out ion-target pairs.

## Accuracy (data never seen in training)
Fluence for a target dose over the ROI:

| Left out of training | Median error | 90th percentile |
|---|---|---|
| Ion-target pairs | 2.8 % | 7.8 % |
| Cu and CuCrZr targets | 3.3 % | 12 % |
| W target | 4.9 % | 10 % |
| SiC target (outside the training range, flagged out of domain) | 8.1 % | 13 % |

The damage energy is within 0.6 % (median). Against 1000-ion TRIM runs, the error on never-seen Cu,
CuCrZr and W targets is about 2.4 %; much of the rest is noise in the 300-ion runs it is scored against.
Feeding two runs of a new material back in (`Surrogate.update`) cut the SiC error from 12 % to 4.7 %.
Per-run numbers are in `results/`.

**Limits**
- ROI width under a strict flatness limit is predicted poorly (about 25 % median error), so confirm the
  final window with TRIM.
- Do not trust predictions outside the training range (H or He beams, targets lighter than SiC or heavier
  than W); `in_domain` in the prediction says so.

## Files
- `surrogate.npz`: arrays (PCA basis, training inputs and outputs, kernel hyperparameters); no pickled objects.
- `surrogate.json`: metadata (training coverage, domain box, calibrated band).

## About the project
srim_opt (code: [github.com/rizzyrr/srim_opt](https://github.com/rizzyrr/srim_opt), branch `riz`) plans heavy-ion
irradiations. You give a target dpa and a region of interest (ROI: start depth, minimum width, distance from the
implanted-ion peak, flatness), and it returns the beam energy and fluence, verified with SRIM. It was built for
the OSI Hackathon 2026.

What has been done (Part 1, the physics core):
- **This dataset:** 1,350 TRIM runs on 6 parallel SRIM copies, about 4 hours, with no failures.
- **The surrogate** ([srimopt/heavysrim](https://huggingface.co/srimopt/heavysrim)): a
  Gaussian process that predicts a damage profile in milliseconds. On ion-target pairs it never saw, the
  fluence error is 2.8 % median.
- **A one-click optimiser:** a range check with SRIM's SR Module, a surrogate proposal, then one 1000-ion TRIM
  confirmation run. Over five benchmark cases it needed 5 TRIM runs, against 33 for a grid search. It found
  the best energy in 4 of 5 cases and was within 5 % in the fifth.
- **Accelerator presets** (11 machines) set the energy window and report the charge state and terminal voltage.
- Part 2 (in progress): a natural-language interface in which an LLM calls these tools.

## Use
1. Get the code: `git clone https://github.com/rizzyrr/srim_opt.git`, then `git checkout riz`, then
   `pip install -r requirements.txt` (Python 3.13). The repo already contains this model in
   `srim_outputs/surrogate/`, or download it with `hf download srimopt/heavysrim --local-dir srim_outputs/surrogate`.
2. Predict without TRIM. This works for the dataset's ion-target pairs without SRIM installed; other pairs
   need SRIM for the range tables.

```python
import srim_core as core, srim_surrogate as sur
model = sur.load_latest()          # srim_outputs/surrogate/surrogate.npz
p = model.predict_profiles(core.MATERIALS['W'], 'W', [20e6])[0]   # 20 MeV W into W
dpa = p.dpa(1e15)                                                  # dpa vs depth at 1e15 ions/cm2
```

3. Plan an irradiation (needs SRIM-2013 installed; runs one TRIM confirmation):
```python
import srim_optimiser as opt
spec = opt.BeamSpec(target_dpa=3, min_depth_nm=100, min_width_nm=100, implant_margin_nm=100,
                    facility='HZDR 3 MV Tandetron')
print(opt.screen('CuCrZr', 'Cu', spec).message)   # instant prediction
print(opt.plan('CuCrZr', 'Cu', spec).message)     # confirmed with TRIM
```
Or use the app's **Optimise (one-click)** tab (`run_srim_gui.bat`).

To retrain on new runs: `python srim_surrogate.py fit`; `validate`, `learning-curve` and `few-shot`
reproduce the accuracy numbers.

Predictions are of SRIM-2013 Quick K-P, NRT dpa with ASTM E521 displacement energies, not of
experiment; SRIM's own limits apply (amorphous target, no channelling).
