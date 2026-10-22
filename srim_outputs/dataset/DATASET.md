# Ion damage profiles in fusion and fission materials (SRIM-2013, Quick K-P)

2080 TRIM runs into 25 targets (pure metals, fusion and fission alloys, high-entropy alloys and
SiC). Made with srim_opt (`srim_dataset.py`) to train fast surrogates for planning ion irradiations.
- **Heavy ions** (1680 runs): 15 ions from C to Au, 0.5-50 MeV, 300 ions
  per run (60 benchmark runs with 1000): 1230 on the 13 Part 1 targets,
  120 benchmark sweeps and 330 on the 12 alloys added in v2.
- **H and He** (400 runs): H at 0.3-6 MeV with 10,000 ions per run, He at
  0.1-12 MeV with 3,000: light-ion damage is sparse, so many more ions per run.
- **Test split** (85 runs, never trained on), for scoring models: 65 heavy-ion runs at 3,000 and 5,000 ions and 20 H and He runs at 10,000 and 30,000. Their mean dpa over 0.2-0.6 of the range has about 1 % Monte Carlo noise (1 sigma) for heavy ions and 2-7 % for H and He.

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
  2-10 and 10-50 MeV; H and He 8 across their range.

## Targets (runs)
316L (43), CoCrFeMnNi (43), CoCrFeNi (43), Cr (116), Cu (106), CuCrZr (130), EUROFER97 (43), Fe (106), Fe-9Cr (130), HT-9 (43), Inconel-718 (43), Mo (116), Ni (130), NiCoCr (43), SiC (130), T91 (43), Ta (116), Ti (116), V (116), V-4Cr-4Ti (46), W (130), W-5Re (43), W-5Ta (43), Zircaloy-4 (46), Zr (116)

Ions: Au, C, Cr, Cu, Fe, H, He, Kr, Mo, Ni, Si, Ta, Ti, V, W, Xe, Zr.

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
- 300 ions per heavy-ion training run: bin-to-bin Monte Carlo noise of a few per cent, enough to make or break a
  narrow-tolerance flat window; check such windows with more ions.
- H and He: rare high-energy recoils carry most of their damage energy, so even 10,000 ions leave 10-35 % noise per
  depth bin and a few per cent over a 25-bin window.
- SRIM's electronic stopping is less accurate for slow heavy ions in light targets.
