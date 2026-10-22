# Part 2 handoff: the physics core's API

Part 1 left the physics core as a Python API in `srim_optimiser.py`. The app's **Optimise (one-click)** tab
already uses it. The LLM interface should call the same functions as tools and quote only the numbers they
return.

## What Part 1 delivered (branch `riz`)

| Piece | File | Result |
|---|---|---|
| Dataset: 1,350 TRIM runs, 15 heavy ions into 13 nuclear materials, 0.5-50 MeV | `srim_dataset.py`, `srim_outputs/dataset/` | 0 failures; Parquet file and datasheet |
| Surrogate: Gaussian process on range-scaled profiles | `srim_surrogate.py`, `srim_outputs/surrogate/` | fluence error 2.8 % median on unseen ion-target pairs; 0.3 s for 60 energies |
| One-click optimiser: surrogate proposes, one TRIM run confirms | `srim_optimiser.py` | 5 TRIM runs instead of 33 over five benchmark cases; all five real plans in 1-3 min each |
| Accelerator presets: energy window, charge state, terminal voltage | `srim_optimiser.FACILITIES` | 11 machines |
| App | `srim_app.py` | **Optimise (one-click)** tab; periodic-table element picker |
| Benchmark and figures | `bench_optimiser.py`, `make_figures.py`, `srim_outputs/figures/` | five PNGs for the write-up |

The repo includes the trained model, the dataset (raw runs in `srim_outputs/dataset_cache/`, enough to
retrain), the SR range tables and the figures, so nothing needs re-running. The same dataset and model are
on Hugging Face as private repos in the `srimopt` organization (`srimopt/srim-heavy-ion-damage-profiles`, `srimopt/heavysrim`); ask Riz to add you to it.

## Setup

1. `git clone https://github.com/rizzyrr/srim_opt.git`, then `git checkout riz`.
2. Python 3.13: `py -3.13 -m venv .venv313`, then `.venv313\Scripts\pip install -r requirements.txt anthropic mcp`.
   The Anthropic and MCP SDKs need Python 3.10 or newer.
3. `.venv313\Scripts\python -m pytest tests`: 40 tests, no SRIM needed. They should pass before and after your changes.
4. Try it without SRIM: `screen()` works for the ion-target pairs in the dataset (the 13 named materials
   with C, Si, Fe, Ni, Cu, Kr, Xe, W, Au or the self-ion), because their SR range tables are in the repo.
   Any other pair needs SRIM installed.
5. `plan()` and the app need SRIM-2013 (Windows, from srim.org). By default the code looks in
   `C:\Users\Riz\Documents\SRIM`; set the `SRIM_DIR` environment variable to point elsewhere.
6. The app: `run_srim_gui.bat`.

To retrain the surrogate after new TRIM runs: `python srim_surrogate.py fit` (about 2 min).
`validate`, `learning-curve` and `few-shot` reproduce the accuracy numbers.

## The calls

```python
import srim_core as core
import srim_optimiser as opt

spec = opt.BeamSpec(target_dpa=3.0, min_depth_nm=100, min_width_nm=100, implant_margin_nm=100,
                    facility='HZDR 3 MV Tandetron')          # or e_min_MeV=1, e_max_MeV=10 instead of a facility

s = opt.screen('CuCrZr', 'Cu', spec)                         # about 1 s, no TRIM: a prediction
r = opt.plan('CuCrZr', 'Cu', spec, minimised=True,           # minutes: runs TRIM to confirm
             progress=lambda stage, step, total, msg: print(stage, msg))
r.ok, r.message, r.to_dict()                                 # to_dict() is JSON-safe

opt.beam_time(r.fluence, current_nA=50, charge_state=r.beam_setting['charge_state'], area_cm2=1.0,
              roi_mean_dpa=spec.target_dpa, peak_dpa=r.peak_dpa)   # hours and dpa/s
opt.energy_window('Helsinki 5 MV tandem', 'W')               # (e_min, e_max, notes) in MeV
opt.beam_setting('Helsinki 5 MV tandem', 20.0, 'W')          # {'charge_state': 3, 'terminal_MV': 5.0}
```

| Call | Time | Runs TRIM | Returns |
|---|---|---|---|
| `screen(material, ion, spec)` | 0.2–2 s (about 5 s the first time, while it loads the surrogate) | no | `ScreenResult`: `ok`, `message`, `feasible_from_MeV`, `best` (predicted energy, fluence, ROI, 80 % fluence band, uncertainty), `source` (`'surrogate'` or `'sr'`) |
| `plan(material, ion, spec)` | about 2–15 min | yes, usually once | `PlanResult`: `ok`, `message`, `energy_MeV`, `fluence` (+ `fluence_min/max`), `roi_start/end/width_nm`, `dpa_min/max_in_roi`, `peak_dpa`, `max_implanted_appm_in_roi`, `beam_setting`, `prediction` (what the surrogate said first), `flags`, `runs`, `provenance` |
| `beam_time(...)` | instant | no | `seconds`, `hours`, `flux_ions_per_cm2_s`, dpa rates |
| `energy_window(facility, ion)` | instant | no | energy range the machine can deliver for that ion, plus caveats |
| `beam_setting(facility, E, ion)` | instant | no | lowest charge state that reaches E, and its terminal voltage; `None` if it can't |

### Inputs the LLM has to fill in

- **material**: a key of `core.MATERIALS`: Fe, Cr, Ni, Cu, W, Mo, Ta, V, Ti, Zr, CuCrZr, Fe-9Cr, SiC.
  - For anything else, build one with `core.named_material({'Fe': 86, 'Cr': 14}, unit='wt')`. It uses ASTM E521 displacement energies.
  - The surrogate was trained on the 13 named materials, but its inputs are Z, mass and energy, so it predicts for any composition in their range (Fe-14Cr works). The range is a mean target Z from 10 (SiC) to 74 (W), and ions from C to Au. Outside it, for example with H or He beams or a U target, `screen()` gives only the SR energy window, and `plan()` searches with TRIM. Inside the range it still knows nothing about chemistry, so treat a prediction for a new compound such as an oxide with care.
- **ion**: an element symbol. Heavy ions are what the surrogate was trained on.
- **facility**: a key of `opt.FACILITIES` (11 machines). Without one, set `e_min_MeV` and `e_max_MeV`.
- **BeamSpec** (all lengths in nm, energies in MeV, fluence in ions/cm²):
  - `target_dpa` is required.
  - `objective`: `'width'` (default), `'fluence'` or `'implant'`.
  - Implant rule: `implant_mode='distance'` with `implant_margin_nm`, or `'percent'` with `implant_pct`.
  - `max_spread=0.10` keeps the dpa within ±10 % across the ROI.
  - `max_fluence` caps beam time.
  - `BeamSpec.from_dict(d)` ignores unknown keys; `spec.validate()` returns problems as readable sentences.

## Things to get right

1. **Only quote tool output.** Every number the user sees should come from a tool result: energy, fluence, ROI, dpa and hours. Mark `screen()` numbers as predicted. Mark `plan()` numbers as confirmed by TRIM.
2. **Show `flags` and failure messages.** When `ok` is False, `message` says why and what to change. For example: *"E_max = 10 MeV is too low … Raise the maximum energy to about 14 MeV"*. Let the LLM pass that on, or act on it by calling again with the new limit.
3. **`plan()` blocks for minutes.**
   - Run it in a worker thread, or as a background task in MCP.
   - Stream `progress(stage, step, total, message)` to the user.
   - Use `screen()` first, so the user sees an answer straight away.
4. **One TRIM at a time per SRIM folder.**
   - TRIM writes its input and output files in its own folder. A lock serialises runs within one Python process, but not across processes.
   - If the Streamlit app and an MCP server both run plans, give them different SRIM folders (the `srim_dir` argument). `C:\Users\Riz\Documents\SRIM_workers\w0` … `w5` are full copies, free once the dataset run has finished.
5. **Results are cached.** A repeated `plan()` with the same spec reuses its TRIM runs and returns in seconds.
6. **Ask, don't guess.** Ask the user for missing essentials: target dpa, material, ion, and facility or energy range. The defaults (ROI at least 100 nm wide, starting at 100 nm, ±10 % flatness) are reasonable, but say that they were used.

## Suggested tools

| Tool | Wraps |
|---|---|
| `list_options` | `core.MATERIALS`, `opt.FACILITIES` (with `energy_window` per ion), `core.OBJECTIVES` |
| `screen_beam` | `screen()`: arguments are material, ion and the `BeamSpec` fields |
| `plan_beam` | `plan()` |
| `beam_time` | `beam_time()` |

These suit the Anthropic SDK's tool runner (`client.beta.messages.tool_runner` with `@beta_tool` functions) for the Streamlit agent tab, and an MCP server for other clients. Use model `claude-opus-5-5`.

## Where the numbers come from

- NRT dpa from SRIM Quick Kinchin-Pease damage energy (Stoller et al. 2013), with ASTM E521 displacement energies.
- `plan()` reports numbers from a TRIM run of at least 1,000 ions.
- The surrogate (`srim_surrogate.py`) only proposes the energy. It is a Gaussian process trained on about 1,300 TRIM runs (13 targets × 10 ions × 0.5–50 MeV). Validation numbers come from `python srim_surrogate.py validate`.
- `PlanResult.provenance` records all of this for each answer.
