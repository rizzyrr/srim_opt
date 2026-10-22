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
