---
license: cc-by-4.0
pretty_name: SRIM heavy-ion damage profiles in fusion and fission materials
tags:
- materials-science
- radiation-damage
- ion-irradiation
- srim
- nuclear-materials
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files: data/srim_profiles.parquet
---

# Heavy-ion damage profiles in fusion and fission materials (SRIM-2013, Quick K-P)

1350 TRIM runs: 15 ions into 13 targets,
0.5-50 MeV, 300 ions per run (60 benchmark
runs with 1000). Made with srim_opt (`srim_dataset.py`) to train a fast surrogate for planning ion irradiations.

## How each run was made
- SRIM-2013 TRIM (driven by pysrim), "Ion distribution and quick calculation of damage" (Quick Kinchin-Pease).
- Displacement energies from ASTM E521 (Si 35 eV and C 20 eV for SiC); lattice and surface binding energies 0,
  as Stoller et al. (2013) recommend for Quick K-P damage calculations.
- Damage energy per depth from PHONON.txt following Stoller et al. (2013, erratum 2019); NRT displacements are
  0.8 T_dam / (2 E_d) with the atom-fraction mean E_d.
- Target thickness 1.4 x (projected range + 3 straggling) from SRIM's stopping and range tables, so the 100
  depth bins resolve the profile at every energy; thickened and re-run if ions reached the back.
- Energies per ion-target pair: 1 in 0.5-1 MeV, 7 in 1-20 MeV, 2 in 20-50 MeV (scrambled log-uniform), plus
  12-energy sweeps (1-50 MeV, at 300 and at 1000 ions) for five benchmark pairs.

## Targets (runs)
Cr (100), Cu (90), CuCrZr (114), Fe (90), Fe-9Cr (114), Mo (100), Ni (114), SiC (114), Ta (100), Ti (100), V (100), W (114), Zr (100)

Ions: Au, C, Cr, Cu, Fe, Kr, Mo, Ni, Si, Ta, Ti, V, W, Xe, Zr.

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
- 300 ions per run: bin-to-bin Monte Carlo noise of a few per cent, enough to make or break a
  narrow-tolerance flat window; check such windows with more ions.
- SRIM's electronic stopping is less accurate for slow heavy ions in light targets.

## Licence
CC-BY-4.0: free to reuse with credit. The profiles were computed with SRIM-2013 (J. F. Ziegler, www.srim.org);
cite SRIM as well as this dataset.

## Citation
srim_opt heavy-ion damage profiles (2026), Hugging Face dataset srimopt/srim-heavy-ion-damage-profiles.
Made for the OSI Hackathon 2026.

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

## How to use this dataset
```python
from datasets import load_dataset
ds = load_dataset("srimopt/srim-heavy-ion-damage-profiles", split="train")
row = ds[0]          # one TRIM run: inputs, totals and depth profiles (per incident ion)
```
- **dpa at fluence F (ions/cm2):** `NRT_disp_per_A_per_ion * F / (atomic_density_cm3 * 1e-8)`
- **Implanted ions in appm:** `implanted_per_cm_per_ion * F / atomic_density_cm3 * 1e6`
- **A different displacement energy:** scale the NRT columns by `nrt_Ed_eV / E_d_new`.
- **Retraining the surrogate:** use the srim_opt code. The repo holds the same runs as raw files in
  `srim_outputs/dataset_cache/`; run `python srim_surrogate.py fit`.
