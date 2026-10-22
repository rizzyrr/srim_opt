# SRIM Damage Calculator - a point-and-click front end for srim_core.
# Launch with:  streamlit run srim_app.py   (or double-click run_srim_gui.bat)
# Built by Rizqullah Rasyiddin 01/10/2026


import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import srim_core as core
import srim_optimiser as opt

st.set_page_config(page_title='SRIM Damage Calculator', layout='wide')

# Default st.metric values are large enough to truncate long numbers and push deltas out of view
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem; }
[data-testid="stMetricValue"] > div, [data-testid="stMetricDelta"] > div { white-space: normal; overflow: visible; }
</style>
""", unsafe_allow_html=True)

# Fixed categorical order: a series keeps its colour no matter how many are shown
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
MAX_SERIES = len(COLORS)

PRESETS = {
    'CuCrZr (Cu-0.82Cr-0.07Zr)': [('Cu', 99.11, 25.0), ('Cr', 0.82, 25.0), ('Zr', 0.07, 25.0)],
    'CuCrTa (Cu-0.8Cr-0.4Ta)': [('Cu', 98.8, 25.0), ('Cr', 0.8, 25.0), ('Ta', 0.4, 32.0)],
    'Pure Cu': [('Cu', 100.0, 25.0)],
    'Pure Fe': [('Fe', 100.0, 40.0)],
    'Pure W': [('W', 100.0, 90.0)],
    'Custom (start empty)': [('Cu', 100.0, 25.0)],
}

SYMBOLS = core.all_element_symbols()

# Composition table: entered in atomic % or weight %, always converted to atomic % for TRIM.
UNIT_COLS = {'at': 'Atomic %', 'wt': 'Weight %'}


def to_atomic_pct(df, unit):
    symbols, values = df['Element'].astype(str).tolist(), df[UNIT_COLS[unit]].to_numpy(dtype=float)
    return values if unit == 'at' else core.convert_composition(symbols, values, to='at')


def _seed_table(symbols, values, e_d, unit):
    """Give the composition editor new starting data (a new key makes it forget old edits)."""
    st.session_state.comp_seed = pd.DataFrame({'Element': symbols, UNIT_COLS[unit]: values, 'E_d (eV)': e_d})
    st.session_state.comp_ver = st.session_state.get('comp_ver', 0) + 1


def _on_preset_change():
    symbols, at, e_d = zip(*PRESETS[st.session_state.preset])
    values = at if st.session_state.comp_unit == 'at' else core.convert_composition(symbols, at, to='wt')
    _seed_table(list(symbols), np.round(values, 4), list(e_d), st.session_state.comp_unit)
    st.session_state.comp_balance = symbols[0]   # presets list the base metal first


def _on_unit_change():
    """Convert what is currently in the table into the newly selected unit."""
    old_unit, df = st.session_state.comp_current
    new_unit = st.session_state.comp_unit
    values = df[UNIT_COLS[old_unit]].to_numpy(dtype=float)
    try:
        values = core.convert_composition(df['Element'].astype(str).tolist(), values, to=new_unit)
    except Exception:
        pass  # unknown element or empty table: keep the numbers, validation will flag the problem
    _seed_table(df['Element'].tolist(), np.round(values, 4), df['E_d (eV)'].tolist(), new_unit)


if 'comp_seed' not in st.session_state:
    st.session_state.preset, st.session_state.comp_unit = list(PRESETS)[0], 'at'
    _on_preset_change()

# Helper functions for parsing input, running simulations, and managing session state

def parse_list(text, scale=1.0):
    values = [float(v) * scale for v in text.replace(';', ',').split(',') if v.strip()]
    if not values:
        raise ValueError('Enter at least one value.')
    return values


def run_label(p):
    return f'{p.material.label} | {p.ion_symbol} {p.energy_eV / 1e6:g} MeV | {p.number_ions:,} ions'


def remember(p):
    st.session_state.history[run_label(p)] = p


if 'history' not in st.session_state:
    # label -> DamageProfile, in run order; starts with every run saved in the cache by earlier sessions
    st.session_state.history = {}
    for _p in core.cached_profiles():
        remember(_p)


def run_one(material, ion, energy_eV, n_ions, srim_dir, use_cache):
    p = core.run_trim(material, ion, energy_eV, n_ions, srim_dir, use_cache=use_cache)
    remember(p)
    return p


def run_many(material_ion_energy_n, srim_dir, use_cache):
    # Run a list of (material, ion, energy_eV, n_ions) jobs with a progress bar 
    profiles = []
    bar = st.progress(0.0, text='Starting TRIM...')
    for k, (m, ion, E, n) in enumerate(material_ion_energy_n):
        bar.progress(k / len(material_ion_energy_n),
                     text=f'Run {k + 1} of {len(material_ion_energy_n)}: {ion} {E / 1e6:g} MeV, {n:,} ions')
        profiles.append(run_one(m, ion, E, n, srim_dir, use_cache))
    bar.progress(1.0, text='Done')
    return profiles


def maybe_smooth(y, on):
    return core.smooth(y) if on else y


def depth_figure(curves, show_implanted=True, hline=None, segments=(), xmax=None, vlines=(), overlay=False):
    # dpa (top) and implanted-ion concentration (bottom) vs depth on a shared x-axis.
    # curves: list of dicts with keys name, depth, dpa, appm.
    # segments: list of (name, x0, x1, y, color) drawn on the dpa panel (ROIs).
    # vlines: list of (x, label) reference depths drawn across the whole figure.
    # overlay: draw dpa (solid, left axis) and implanted ions (dashed, right axis) on one chart.
    if overlay and show_implanted:
        return _overlay_figure(curves, hline, segments, xmax, vlines)
    rows = 2 if show_implanted else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.62, 0.38] if rows == 2 else [1.0])
    for i, c in enumerate(curves):
        color = COLORS[i % MAX_SERIES]
        line = dict(color=color, width=2, dash=c.get('dash'))   # optional 'dash' key, e.g. for predictions
        fig.add_trace(go.Scatter(x=c['depth'], y=c['dpa'], name=c['name'], legendgroup=c['name'],
                                 line=line, hovertemplate='%{y:.3g} dpa'), row=1, col=1)
        if show_implanted:
            fig.add_trace(go.Scatter(x=c['depth'], y=c['appm'], name=c['name'], legendgroup=c['name'],
                                     showlegend=False, line=line,
                                     hovertemplate='%{y:.3g} appm'), row=2, col=1)
    for name, x0, x1, y, color in segments:
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode='lines+markers', name=name,
                                 line=dict(color=color, width=6), marker=dict(size=9, symbol='line-ns-open'),
                                 hovertemplate=f'{name}<br>{x0:.0f}-{x1:.0f} nm, mean %{{y:.3g}} dpa'
                                               '<extra></extra>'), row=1, col=1)
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color='gray', dash='dash', width=1),
                      annotation_text=f'{hline:g} dpa', annotation_position='top left', row=1, col=1)
    fig.update_yaxes(title_text='NRT dpa', rangemode='tozero', row=1, col=1)
    if show_implanted:
        fig.update_yaxes(title_text='Implanted ions (appm)', rangemode='tozero', row=2, col=1)
    fig.update_xaxes(title_text='Depth (nm)', row=rows, col=1)
    _add_vlines(fig, vlines)
    if xmax:
        fig.update_xaxes(range=[0, xmax])
    fig.update_layout(height=620 if show_implanted else 420, hovermode='x unified',
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    return fig


def _add_vlines(fig, vlines):
    for x, label in vlines:
        fig.add_vline(x=x, line=dict(color='gray', dash='dot', width=1),
                      annotation_text=label, annotation_position='top right')


def _overlay_figure(curves, hline, segments, xmax, vlines):
    """dpa (solid, left y-axis) and implanted ions (dashed, right y-axis) on one chart."""
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    for i, c in enumerate(curves):
        color = COLORS[i % MAX_SERIES]
        fig.add_trace(go.Scatter(x=c['depth'], y=c['dpa'], name=c['name'], legendgroup=c['name'],
                                 line=dict(color=color, width=2),
                                 hovertemplate='%{y:.3g} dpa'), secondary_y=False)
        fig.add_trace(go.Scatter(x=c['depth'], y=c['appm'], name=c['name'], legendgroup=c['name'],
                                 showlegend=False, line=dict(color=color, width=2, dash='dash'),
                                 hovertemplate='%{y:.3g} appm'), secondary_y=True)
    for name, x0, x1, y, color in segments:
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode='lines+markers', name=name,
                                 line=dict(color=color, width=6), marker=dict(size=9, symbol='line-ns-open'),
                                 hovertemplate=f'{name}<br>{x0:.0f}-{x1:.0f} nm, mean %{{y:.3g}} dpa'
                                               '<extra></extra>'), secondary_y=False)
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color='gray', dash='dash', width=1),
                      annotation_text=f'{hline:g} dpa', annotation_position='top left', secondary_y=False)
    _add_vlines(fig, vlines)
    fig.update_yaxes(title_text='NRT dpa (solid)', rangemode='tozero', secondary_y=False)
    fig.update_yaxes(title_text='Implanted ions, appm (dashed)', rangemode='tozero', showgrid=False,
                     secondary_y=True)
    fig.update_xaxes(title_text='Depth (nm)')
    if xmax:
        fig.update_xaxes(range=[0, xmax])
    fig.update_layout(height=480, hovermode='x unified', margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    return fig


def xy_figure(x, y, xlabel, ylabel, logx=True):
    fig = go.Figure(go.Scatter(x=x, y=y, mode='lines+markers', line=dict(color=COLORS[0], width=2),
                               marker=dict(size=9), hovertemplate=f'{xlabel}: %{{x:.3g}}<br>{ylabel}: %{{y:.4g}}'
                                                                  '<extra></extra>'))
    fig.update_layout(title=ylabel, height=340, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(title_text=xlabel, type='log' if logx else 'linear')
    fig.update_yaxes(title_text=ylabel, rangemode='tozero')
    return fig


def auto_xmax(profiles):
    """Depth range that shows the damage and implantation without empty target."""
    edge = 0.0
    for p in profiles:
        mask = (p.disp_density > 0.005 * p.disp_density.max()) | (p.implanted > 0.005 * p.implanted.max())
        if mask.any():
            edge = max(edge, p.depth_nm[mask].max())
    return edge * 1.1 if edge else None


def quality_checks(p):
    if p.damaged_bins < 20:
        st.warning(
            f'Only {p.damaged_bins} of 100 depth bins contain damage (bin size {p.bin_nm:.0f} nm), so the '
            f'profile is coarse. Reduce **Target thickness** in the sidebar to about '
            f'{max(100, round(2 * auto_xmax([p]) / 100) * 100):,.0f} nm and run again.')
    diff = abs(p.Tdam_eV - p.Tdam_balance_eV) / p.Tdam_balance_eV * 100
    if diff > 5:
        st.warning(f'Damage energy from PHONON.txt and from the energy balance differ by {diff:.1f}%. '
                   'Check that the target is thick enough to stop the ions and that the run used enough ions.')
    if p.number_ions < 1000:
        st.info('Fewer than 1,000 ions: fine for a quick look, but use the **Convergence** tab before '
                'trusting the numbers.')


# ---------------------------------------------------------------------------
# Sidebar: inputs shared by every tab
# --------------------------------------------------------------------------

with st.sidebar:
    st.title('SRIM Damage Calculator')
    st.caption('Quick Kinchin-Pease, NRT dpa (Stoller et al. 2013/2019)')

    st.subheader('1. SRIM installation')
    srim_dir = st.text_input('SRIM folder (contains TRIM.exe)', core.DEFAULT_SRIM_DIR)
    srim_error = core.check_srim_directory(srim_dir)
    if srim_error:
        st.error(srim_error)
    else:
        st.success('TRIM.exe found', icon='✅')

    st.subheader('2. Ion beam')
    ion = st.selectbox('Incident ion', SYMBOLS, index=SYMBOLS.index('Cu'),
                       help='Self-ion irradiation uses the same element as the main target element.')
    n_ions = st.number_input('Number of ions', min_value=10, max_value=100000, value=1000, step=100,
                             help='More ions = smoother profiles but longer runs. 1,000 is a good start; '
                                  'check with the Convergence tab.')

    st.subheader('3. Target material')
    st.selectbox('Start from', list(PRESETS), key='preset', on_change=_on_preset_change)
    unit = st.radio('Composition entered as', list(UNIT_COLS), key='comp_unit', horizontal=True,
                    format_func=lambda u: UNIT_COLS[u], on_change=_on_unit_change,
                    help='Switching converts the values already in the table.')
    value_col = UNIT_COLS[unit]
    comp_df = st.data_editor(
        st.session_state.comp_seed, key=f'composition_{st.session_state.comp_ver}', num_rows='dynamic',
        hide_index=True,
        column_config={
            'Element': st.column_config.SelectboxColumn(options=SYMBOLS, required=True),
            value_col: st.column_config.NumberColumn(min_value=0.0, format='%.4g', required=True,
                                                     help=f'{value_col.split()[0]} percent. Normalised to 100%.'),
            'E_d (eV)': st.column_config.NumberColumn(min_value=0.1, format='%.4g', required=True,
                                                      help='Displacement threshold energy of this element.'),
        })
    comp_df = comp_df.dropna()

    # Balance element: its amount is whatever the other rows leave out of 100%
    balance_options = ['None'] + list(dict.fromkeys(comp_df['Element'].astype(str)))
    if st.session_state.get('comp_balance') not in balance_options:
        st.session_state.comp_balance = 'None'
    balance = st.selectbox('Balance element', balance_options, key='comp_balance',
                           help='This element makes up the rest of the alloy (100% minus the other rows), '
                                'so the value typed in its row is ignored. E.g. Cu-0.82Cr-0.07Zr: pick Cu.')
    balance_problem = None
    if balance != 'None':
        row = (comp_df['Element'] == balance).idxmax()
        rest = comp_df[value_col].drop(index=row).sum()
        comp_df.loc[row, value_col] = 100 - rest
        if rest > 100:
            balance_problem = f'The other elements add up to {rest:g} {value_col}, leaving nothing for {balance}.'
        else:
            st.caption(f'{balance} (balance) = {100 - rest:.4g} {value_col}')

    st.session_state.comp_current = (unit, comp_df)
    total = comp_df[value_col].sum()
    if comp_df.shape[0] and abs(total - 100) > 0.01:
        st.caption(f'{value_col} adds up to {total:g}; it will be normalised to 100%.')

    with st.expander('Typical displacement energies'):
        st.markdown('ASTM E521 recommended values (eV): Al 27, Ti 30, Cu 30, '
                    'V/Cr/Mn/Fe/Co/Ni/Zr/Nb 40, Mo 60, Ta/W 90. ')

    st.subheader('4. Target geometry')
    density_mode = st.radio('Density', ['Estimate from composition', 'Enter measured value'], horizontal=True)
    density_override = None
    if density_mode == 'Enter measured value':
        density_override = st.number_input('Density (g/cm³)', min_value=0.01, value=8.94, format='%.3f')
    thickness_nm = st.number_input('Target thickness (nm)', min_value=10.0, value=5000.0, step=500.0,
                                   help='TRIM splits the target into 100 depth bins. Make it roughly 2x the '
                                        'ion range so the profile is well resolved.')
    st.caption(f'Depth bin size: {thickness_nm / 100:.0f} nm')

    st.subheader('5. Display')
    overlay = st.checkbox('Overlay dpa and implanted ions on one chart (two y-axes)', value=False,
                          help='dpa is drawn solid on the left axis, implanted ions dashed on the right axis. '
                               'Off: two panels sharing the depth axis.')

    with st.expander('Advanced'):
        lattice = st.number_input('Lattice binding energy (eV)', min_value=0.0, value=0.0,
                                  help='Stoller et al. recommend 0 for Quick K-P damage calculations.')
        surface = st.number_input('Surface binding energy (eV)', min_value=0.0, value=0.0)
        use_cache = st.checkbox('Re-use previous identical runs', value=True,
                                help='Runs are saved in srim_outputs/cache and reloaded instantly if every '
                                     'input is identical. Untick to force a fresh TRIM run.')

try:
    at_pct = to_atomic_pct(comp_df, unit)
except Exception:
    at_pct = comp_df[value_col].to_numpy(dtype=float)   # invalid rows: validate() reports them below
material = core.Material(
    constituents=tuple(core.Constituent(str(r['Element']), float(x), float(r['E_d (eV)']), lattice, surface)
                       for (_, r), x in zip(comp_df.iterrows(), at_pct)),
    density_override=density_override,
    width_A=thickness_nm * 10,
)
problems = material.validate() + ([balance_problem] if balance_problem else [])

with st.sidebar:
    if problems:
        for msg in problems:
            st.error(msg)
    else:
        st.caption(f'**{material.label}** (atomic %): {material.density:.3f} g/cm³, '
                   f'{material.atomic_density:.4e} atoms/cm³, NRT E_d = {material.nrt_Ed:.1f} eV')
        st.dataframe(pd.DataFrame({'Element': [c.symbol for c in material.constituents],
                                   'Atomic %': material.fractions * 100, 'Weight %': material.weight_pct}),
                     hide_index=True, column_config={'Atomic %': st.column_config.NumberColumn(format='%.4g'),
                                                     'Weight %': st.column_config.NumberColumn(format='%.4g')})

can_run = not problems and not srim_error


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_guide, tab_single, tab_energy, tab_roi, tab_one, tab_opt, tab_compare, tab_conv = st.tabs(
    ['Start here', 'Single run', 'Energy sweep', 'Fluence & ROI', 'Optimise (one-click)', 'Optimise (manual grid)',
     'Compare runs', 'Convergence'])

with tab_guide:
    st.markdown(r"""
### What's up!
This tool was made to make it less daunting to run heavy ion SRIM experiments for dpa calculations.
It runs **SRIM** for an ion beam hitting your material and turns the output into a
**radiation-damage profile (dpa vs depth)** and an **implanted-ion profile**, using the
corrected Quick Kinchin-Pease method of Stoller et al. (2013, erratum 2019).

### Typical workflow
1. **Sidebar:** choose the ion, the material (start from a preset and edit the table) and the target thickness.
2. **Single run:** pick a beam energy and fluence, press *Run*. A TRIM window opens and closes by itself.
3. **Convergence:** check that your number of ions is large enough (results should change by less than a few %).
4. **Energy sweep:** see how damage and ion range change with beam energy.
5. **Fluence & ROI:** find the fluence that gives your target dpa, and the depth window (region of interest)
   where the dpa is close to that value, away from the surface and the implanted ions.
6. **Optimise (one-click):** give the damage you need (dpa, ROI depth and width, implant limit) and your
   accelerator; it finds the beam energy and fluence and confirms them with one TRIM run of 1,000+ ions.
7. **Optimise (manual grid):** the same question answered by running TRIM on an energy grid you choose.
8. **Compare runs:** overlay any runs from this session (e.g. different alloys) at the same fluence.

Every run is remembered for this session and saved to disk, so repeating an identical run is instant.

### Glossary
| Term | Meaning |
|---|---|
| **Fluence** | Ions delivered per cm² of sample surface (ions/cm²). |
| **$E_d$** | Displacement threshold energy: the energy needed to knock an atom off its lattice site. |
| **Damage energy** | The part of the ion's energy that goes into moving atoms rather than electrons. |
| **NRT dpa** | Displacements per atom from the Norgett-Robinson-Torrens model, $N_d$: $0.8 \frac{T_{dam}(E_{PKA})}{2E_d}$. |
| **Implanted ions (appm)** | Concentration of the beam ions that stopped in the sample, in atomic parts per million. |
| **ROI** | Region of interest: a depth band with roughly uniform dpa, used for TEM or nano-indentation. |
""")

# --- Single run --------------------------------------------------------------
with tab_single:
    c1, c2, c3 = st.columns([1, 1, 1])
    energy_MeV = c1.number_input('Beam energy (MeV)', min_value=0.001, max_value=2000.0, value=5.0,
                                 key='single_E')
    fluence = c2.number_input('Fluence (ions/cm²)', min_value=1e8, value=2.5e15, format='%.3e',
                              key='single_F')
    c3.write('')
    if c3.button('Run TRIM', type='primary', disabled=not can_run, key='single_run'):
        with st.status('Running TRIM... a TRIM window will open and close by itself.', expanded=False) as s:
            try:
                st.session_state.single = run_one(material, ion, energy_MeV * 1e6, n_ions, srim_dir, use_cache)
                s.update(label='TRIM finished' + (' (loaded from cache)'
                                                  if st.session_state.single.meta.get('from_cache') else ''),
                         state='complete')
            except Exception as e:
                s.update(label=f'TRIM failed: {e}', state='error')

    p = st.session_state.get('single')
    if p is not None:
        st.markdown(f'#### {run_label(p)}')
        summ = p.summary(fluence)
        m = st.columns(5)
        m[0].metric('Damage energy / ion', f'{summ["Tdam_keV_per_ion"]:.1f} keV')
        m[1].metric('NRT displacements / ion', f'{summ["N_NRT_per_ion"]:,.0f}')
        m[2].metric('Peak dpa', f'{summ["peak_dpa"]:.3g}', help=f'at {fluence:.2e} ions/cm²')
        m[3].metric('Mean ion range', f'{summ["mean_range_nm"]:,.0f} nm')
        m[4].metric('Fluence for 1 dpa at peak', f'{summ["fluence_for_1dpa_peak"]:.2e}')
        st.caption(f'Energy-balance check: {summ["Tdam_balance_keV_per_ion"]:.1f} keV/ion '
                   f'(should match the damage energy above). Peak damage at {summ["peak_depth_nm"]:.0f} nm.')
        quality_checks(p)

        sm = st.checkbox('Smooth Monte-Carlo noise (display only)', key='single_smooth')
        fig = depth_figure([{'name': f'{fluence:.2e} ions/cm²', 'depth': p.depth_nm,
                             'dpa': maybe_smooth(p.dpa(fluence), sm),
                             'appm': maybe_smooth(p.implanted_appm(fluence), sm)}],
                           xmax=auto_xmax([p]), overlay=overlay)
        st.plotly_chart(fig, key='single_fig')
        st.download_button('Download depth profile (CSV)', p.to_dataframe(fluence).to_csv(index=False),
                           file_name=f'{run_label(p).replace(" | ", "_").replace(" ", "")}.csv',
                           mime='text/csv')
    else:
        st.info('Set up the beam and target in the sidebar, then press **Run TRIM**.')

# -----------------------------------------------------
# Energy sweep
# -----------------------------------------------------

with tab_energy:
    c1, c2 = st.columns([2, 1])
    energies_txt = c1.text_input('Beam energies (MeV, comma-separated)', '0.1, 0.25, 0.5, 1, 2, 5')
    fluence_e = c2.number_input('Fluence (ions/cm²)', min_value=1e8, value=1e15, format='%.3e', key='sweep_F')
    try:
        energies = parse_list(energies_txt, 1e6)
    except ValueError:
        energies = []
        st.error('Could not read the energies. Use numbers separated by commas, e.g. 0.5, 1, 2')
    if len(energies) > MAX_SERIES:
        st.warning(f'More than {MAX_SERIES} energies: only the first {MAX_SERIES} are drawn in the profile plot.')

    if st.button('Run energy sweep', type='primary', disabled=not (can_run and energies), key='sweep_run'):
        try:
            st.session_state.sweep = run_many([(material, ion, E, n_ions) for E in energies], srim_dir, use_cache)
        except Exception as e:
            st.error(f'TRIM failed: {e}')

    sweep = st.session_state.get('sweep')
    if sweep:
        df = pd.DataFrame([p.summary(fluence_e) for p in sweep])
        a, b, c = st.columns(3)
        a.plotly_chart(xy_figure(df['energy_MeV'], df['N_NRT_per_ion'], 'Beam energy (MeV)',
                                 'NRT displacements / ion'), key='sweep_nrt')
        b.plotly_chart(xy_figure(df['energy_MeV'], df['peak_dpa'], 'Beam energy (MeV)',
                                 f'Peak dpa at {fluence_e:.2e}'), key='sweep_dpa')
        c.plotly_chart(xy_figure(df['energy_MeV'], df['mean_range_nm'], 'Beam energy (MeV)',
                                 'Mean ion range (nm)'), key='sweep_range')
        shown = sweep[:MAX_SERIES]
        sm = st.checkbox('Smooth Monte-Carlo noise (display only)', key='sweep_smooth')
        st.plotly_chart(depth_figure([{'name': f'{p.energy_eV / 1e6:g} MeV', 'depth': p.depth_nm,
                                       'dpa': maybe_smooth(p.dpa(fluence_e), sm),
                                       'appm': maybe_smooth(p.implanted_appm(fluence_e), sm)} for p in shown],
                                     xmax=auto_xmax(shown), overlay=overlay), key='sweep_fig')
        st.dataframe(df, hide_index=True)
        st.download_button('Download sweep summary (CSV)', df.to_csv(index=False),
                           file_name='energy_sweep.csv', mime='text/csv')
        for p in sweep:
            if p.damaged_bins < 20:
                st.warning(f'{p.energy_eV / 1e6:g} MeV: only {p.damaged_bins} depth bins contain damage. '
                           'Low-energy profiles are coarse with a thick target; run them separately with a '
                           'thinner target for detail.')

# -----------------------------------------------------
# Fluence and ROI
# -----------------------------------------------------

with tab_roi:
    history = st.session_state.history
    if not history:
        st.info('Run something first (Single run or Energy sweep). Every run of this session can be analysed here.')
    else:
        labels = list(history)
        choice = st.selectbox('Run to analyse', labels, index=len(labels) - 1)
        p = history[choice]

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        target_dpa = c1.number_input('Target mean dpa', min_value=0.001, value=3.0)
        tol = c2.number_input('Tolerance (± dpa)', min_value=0.0001, value=0.05, format='%.3g')
        min_depth = c3.number_input('ROI starts below (nm)', min_value=0.0, value=200.0,
                                    help='Skip the near-surface region, which is affected by surface effects.')
        max_width = c4.number_input('Max ROI width (nm)', min_value=1.0, value=500.0)
        implant_margin = c5.number_input('Min. distance before implant peak (nm)', min_value=0.0, value=100.0,
                                         help='The ROI must end at least this far above the implanted-ion peak, '
                                              'so the injected ions do not affect the measured region.')
        sm = c6.checkbox('Smooth (display only)', key='roi_smooth')

        implant_peak = p.implant_peak_nm
        max_end = implant_peak - implant_margin
        dpa_peak = float(p.depth_nm[np.argmax(p.disp_density)])
        st.caption(f'Implant peak at **{implant_peak:.0f} nm**, so the ROI must end before **{max_end:.0f} nm**. '
                   f'It must also lie before the dpa peak at **{dpa_peak:.0f} nm**.')

        needed = p.fluence_for_peak_dpa(target_dpa)
        st.markdown(f'Fluence for **peak** dpa = {target_dpa:g}: **{needed:.3e} ions/cm²**')
        fluences_txt = st.text_input('Fluences to compare (ions/cm², comma-separated)',
                                     f'{needed * 0.5:.2e}, {needed:.2e}, {needed * 1.5:.2e}, {needed * 2:.2e}')
        try:
            fluences = parse_list(fluences_txt)[:MAX_SERIES]
        except ValueError:
            fluences = []
            st.error('Could not read the fluences. Example: 1e15, 2.5e15, 5e15')

        rows, curves, segments = [], [], []
        for i, F in enumerate(fluences):
            dpa = p.dpa(F)
            roi = core.find_roi(p.depth_nm, dpa, target_dpa, tol, min_depth, max_width, max_end_nm=max_end)
            curves.append({'name': f'{F:.2e} ions/cm²', 'depth': p.depth_nm, 'dpa': maybe_smooth(dpa, sm),
                           'appm': maybe_smooth(p.implanted_appm(F), sm)})
            row = {'fluence_ions_per_cm2': F, 'peak_dpa': dpa.max()}
            if roi:
                appm = p.implanted_appm(F)
                in_roi = (p.depth_nm >= roi['start_nm']) & (p.depth_nm <= roi['end_nm'])
                row.update(roi_start_nm=roi['start_nm'], roi_end_nm=roi['end_nm'], roi_width_nm=roi['width_nm'],
                           roi_mean_dpa=roi['mean_dpa'], max_implanted_appm_in_roi=appm[in_roi].max())
                segments.append((f'ROI @ {F:.2e}', roi['start_nm'], roi['end_nm'], roi['mean_dpa'],
                                 COLORS[i % MAX_SERIES]))
            rows.append(row)

        if curves:
            st.plotly_chart(depth_figure(curves, hline=target_dpa, segments=segments, xmax=auto_xmax([p]),
                                         vlines=[(implant_peak, 'implant peak'), (max_end, 'ROI limit')],
                                         overlay=overlay), key='roi_fig')
            st.caption('Thick horizontal bars mark the widest depth window whose mean dpa is within the '
                       'tolerance of the target, lying wholly before the damage peak and ending before the '
                       'ROI limit.')
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            if not segments:
                st.info('No ROI found. Try a larger tolerance, a different fluence, a smaller "starts below" '
                        'depth or a smaller distance before the implant peak.')

# -----------------------------------------------------
# One-click optimiser (srim_optimiser.plan)
# -----------------------------------------------------

OBJECTIVE_LABELS = {'width': 'Widest ROI (nm)', 'fluence': 'Fluence for target (ions/cm²)',
                    'implant': 'Max implanted ions in ROI (appm)'}


def objective_figure(curve, objective):
    # Objective vs energy: the surrogate's scan as a line, every TRIM run as a point
    fig = go.Figure()
    scan = [c for c in curve if c['source'] == 'surrogate' and c['objective'] is not None]
    if scan:
        fig.add_trace(go.Scatter(x=[c['energy_MeV'] for c in scan], y=[c['objective'] for c in scan], mode='lines',
                                 name='Surrogate scan', line=dict(color=COLORS[1], width=2),
                                 hovertemplate='%{x:.3g} MeV: %{y:.4g}<extra>surrogate</extra>'))
    for c in curve:
        if c['source'].startswith('trim') and c['objective'] is not None:
            fig.add_trace(go.Scatter(x=[c['energy_MeV']], y=[c['objective']], mode='markers', showlegend=False,
                                     marker=dict(color=COLORS[0], size=10,
                                                 symbol='circle' if c['source'] != 'trim300' else 'circle-open'),
                                     hovertemplate=f"%{{x:.3g}} MeV: %{{y:.4g}}<extra>TRIM, {c['source'][4:]} ions"
                                                   "</extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='TRIM runs (open: 300 ions)',
                             marker=dict(color=COLORS[0], size=10)))
    fig.update_layout(title=OBJECTIVE_LABELS[objective], height=340, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    fig.update_xaxes(title_text='Beam energy (MeV)', type='log')
    fig.update_yaxes(title_text=OBJECTIVE_LABELS[objective], rangemode='tozero')
    return fig


with tab_one:
    st.markdown('Give the damage you need and your accelerator; this finds the **beam energy and fluence** with '
                'as few TRIM runs as possible and confirms the answer with one TRIM run of at least 1,000 ions. '
                'It uses the ion and material from the sidebar.')
    st.caption('With a trained surrogate model (`python srim_surrogate.py fit`) the energies are scanned in under a '
               'second and only the confirmation needs TRIM. Without one, a short search with 300-ion runs comes '
               'first. The number of ions in the sidebar is not used here.')
    c1, c2, c3, c4 = st.columns(4)
    oc_dpa = c1.number_input('Target mean dpa in ROI', min_value=0.001, value=3.0, key='oc_dpa')
    oc_tol = c2.number_input('Tolerance (± dpa)', min_value=0.0, value=0.15, format='%.3g', key='oc_tol',
                             help='Sets the acceptable fluence range around the answer.')
    oc_depth = c3.number_input('ROI starts below (nm)', min_value=0.0, value=100.0, key='oc_depth')
    oc_width = c4.number_input('Minimum ROI width (nm)', min_value=1.0, value=100.0, key='oc_width')
    c1, c2, c3, c4 = st.columns(4)
    oc_obj = c1.selectbox('Optimise for', list(core.OBJECTIVES), format_func=core.OBJECTIVES.get, key='oc_obj',
                          help='Widest ROI: windows within 5 % of the widest count as equal, and the one needing '
                               'the lowest fluence wins.')
    oc_imode = c2.radio('Implant limit', ['distance', 'percent'], horizontal=True, key='oc_imode',
                        format_func={'distance': 'Gap before peak', 'percent': '% of peak'}.get)
    if oc_imode == 'distance':
        oc_margin = c3.number_input('ROI ends this far before the implant peak (nm)', min_value=0.0, value=100.0,
                                    key='oc_margin')
        oc_pct = 10.0
    else:
        oc_pct = c3.number_input('Implanted ions in ROI at most (% of their peak)', min_value=0.1, max_value=100.0,
                                 value=10.0, key='oc_pct')
        oc_margin = 0.0
    oc_flat = c4.checkbox('Limit dpa spread in ROI', value=True, key='oc_flat')
    oc_spread = c4.number_input('Max spread (± % of target)', min_value=1.0, max_value=100.0, value=10.0,
                                key='oc_spread', disabled=not oc_flat)
    c1, c2, c3, c4 = st.columns(4)
    oc_fac = c1.selectbox('Accelerator', ['Custom energy limits'] + list(opt.FACILITIES), key='oc_fac',
                          help='A facility sets the energy window from its terminal voltage and charge states.')
    if oc_fac == 'Custom energy limits':
        oc_emin = c2.number_input('Lowest energy (MeV)', min_value=0.01, value=1.0, key='oc_emin')
        oc_emax = c3.number_input('Highest energy (MeV)', min_value=0.01, value=10.0, key='oc_emax')
        oc_qmax = None
    else:
        oc_qmax = int(c2.number_input('Highest charge state', min_value=1, max_value=30,
                                      value=opt.FACILITIES[oc_fac].q_max, key=f'oc_qmax_{oc_fac}'))
        oc_emin, oc_emax, fac_notes = opt.energy_window(oc_fac, ion, oc_qmax)
        c3.metric(f'Energy window for {ion}', f'{oc_emin:.3g}–{oc_emax:.3g} MeV')
        for note in fac_notes:
            c3.caption(note)
    oc_cap = c4.number_input('Max fluence (ions/cm², 0 = no limit)', min_value=0.0, value=0.0, format='%.3e',
                             key='oc_cap', help='A beam-time cap: windows that need more fluence are rejected.')
    with st.expander('Beam current, for the irradiation time'):
        b1, b2, b3 = st.columns(3)
        oc_current = b1.number_input('Beam current (nA, electrical)', min_value=0.0, value=100.0, key='oc_current')
        oc_q = int(b2.number_input('Charge state (when no accelerator is chosen)', min_value=1, value=2, key='oc_q'))
        oc_area = b3.number_input('Irradiated area (cm²)', min_value=0.001, value=1.0, key='oc_area')

    spec = opt.BeamSpec(target_dpa=oc_dpa, tol_dpa=oc_tol, min_depth_nm=oc_depth, min_width_nm=oc_width,
                        implant_mode=oc_imode, implant_margin_nm=oc_margin, implant_pct=oc_pct,
                        max_spread=oc_spread / 100 if oc_flat else None, e_min_MeV=oc_emin, e_max_MeV=oc_emax,
                        objective=oc_obj, max_fluence=oc_cap or None,
                        facility=None if oc_fac == 'Custom energy limits' else oc_fac, q_max=oc_qmax)
    spec_problems = spec.validate()
    for msg in spec_problems:
        st.error(msg)
    if st.button('Find best energy and fluence', type='primary', disabled=not can_run or bool(spec_problems),
                 key='oc_run'):
        with st.status('Optimising... TRIM windows will open and close by themselves.', expanded=False) as s:
            def show_progress(stage, k, n, msg):
                s.update(label=f'{stage.capitalize()} (run {k}): {msg}')
            try:
                result = opt.plan(material, ion, spec, srim_dir, progress=show_progress)
                st.session_state.oneclick = (result, material, ion, spec)
                for r in result.runs:   # every run joins this session's history
                    run_p = core._load_pickle(os.path.join(core.CACHE_DIR, f"{r['cache_key']}.pkl"))
                    if run_p is not None:
                        remember(run_p)
                s.update(label=f'Done in {result.elapsed_s:.0f} s with {len(result.runs)} TRIM runs',
                         state='complete' if result.ok else 'error')
            except Exception as e:
                s.update(label=f'Optimiser failed: {e}', state='error')

    oneclick = st.session_state.get('oneclick')
    if oneclick:
        res, res_material, res_ion, res_spec = oneclick
        (st.success if res.ok else st.warning)(res.message)
        if res.ok:
            m = st.columns(5)
            setting = res.beam_setting
            m[0].metric('Beam energy', f'{res.energy_MeV:g} MeV',
                        f"{res_ion}{setting['charge_state']}+ at {setting['terminal_MV']:g} MV" if setting else None,
                        delta_color='off')
            m[1].metric('Fluence', f'{res.fluence:.3e}',
                        help=f'{res.fluence_min:.3e} to {res.fluence_max:.3e} ions/cm² keeps the ROI mean within '
                             f'±{res_spec.tol_dpa:g} dpa.')
            m[2].metric('ROI', f'{res.roi_start_nm:.0f}–{res.roi_end_nm:.0f} nm', f'{res.roi_width_nm:.0f} nm wide',
                        delta_color='off')
            m[3].metric('dpa across the ROI', f'{res.dpa_min_in_roi:.2f}–{res.dpa_max_in_roi:.2f}')
            m[4].metric('Max implanted ions in ROI', f'{res.max_implanted_appm_in_roi:.3g} appm')
            q = setting['charge_state'] if setting else oc_q
            if oc_current > 0:
                t = opt.beam_time(res.fluence, oc_current, q, oc_area, roi_mean_dpa=res_spec.target_dpa,
                                  peak_dpa=res.peak_dpa)
                st.markdown(f'**Beam time:** {t["hours"]:.2f} h at {oc_current:g} nA of {res_ion}{q}+ over '
                            f'{oc_area:g} cm² ({t["flux_ions_per_cm2_s"]:.2e} ions/cm²/s; dose rate '
                            f'{t["dpa_rate_roi_per_s"]:.2e} dpa/s in the ROI, {t["dpa_rate_peak_per_s"]:.2e} at the '
                            f'peak).')
            pred = res.prediction
            if pred and pred.get('fluence_on_confirmed_roi') is not None:
                moved = (f' It expected the best ROI at {pred["roi_start_nm"]:.0f}-{pred["roi_end_nm"]:.0f} nm; '
                         f'the TRIM run found a better one.' if pred.get('roi_moved') else '')
                st.caption(f'Surrogate check: for this energy and ROI it predicted '
                           f'{pred["fluence_on_confirmed_roi"]:.3e} ions/cm², and TRIM gave {res.fluence:.3e} '
                           f'({pred["fluence_error"]:+.0%}).' + moved)
            for flag in res.flags:
                st.warning(flag)

            p_conf = core._load_pickle(os.path.join(core.CACHE_DIR, f'{res.confirm_cache_key}.pkl'))
            if p_conf is not None:
                curves = [{'name': f'TRIM, {p_conf.number_ions:,} ions', 'depth': p_conf.depth_nm,
                           'dpa': p_conf.dpa(res.fluence), 'appm': p_conf.implanted_appm(res.fluence)}]
                if pred:
                    try:
                        import srim_surrogate
                        model = srim_surrogate.load_latest()
                        p_pred = model.predict_profiles(res_material, res_ion, [res.energy_MeV * 1e6])[0]
                        curves.append({'name': 'Surrogate prediction', 'depth': p_pred.depth_nm,
                                       'dpa': p_pred.dpa(res.fluence), 'appm': p_pred.implanted_appm(res.fluence),
                                       'dash': 'dash'})
                    except Exception:
                        pass   # the model changed or is gone: show the TRIM profile alone
                vlines = [(res.implant_peak_nm, 'implant peak')]
                if res.roi_limit_nm is not None and res.roi_limit_nm > 0:
                    vlines.append((res.roi_limit_nm, 'ROI limit'))
                st.plotly_chart(depth_figure(curves, hline=res_spec.target_dpa, xmax=auto_xmax([p_conf]),
                                             segments=[('ROI', res.roi_start_nm, res.roi_end_nm, res_spec.target_dpa,
                                                        COLORS[2])],
                                             vlines=vlines, overlay=overlay), key='oc_fig')
        if res.curve:
            st.plotly_chart(objective_figure(res.curve, res_spec.objective), key='oc_curve')
        st.download_button('Download the plan (JSON)', json.dumps(res.to_dict(), indent=1),
                           file_name='irradiation_plan.json', mime='application/json')
        with st.expander(f'TRIM runs used ({len(res.runs)})'):
            st.dataframe(pd.DataFrame(res.runs), hide_index=True)
            if res.provenance:
                st.json(res.provenance)

# -----------------------------------------------------
# Optimise beam energy and fluence (manual grid)
# -----------------------------------------------------

def run_optimiser_energies(energies_eV, auto_size, label):
    # TRIM runs for the optimiser: optional pilot to size the target, then the real run
    profiles = []
    bar = st.progress(0.0, text=label)
    for k, E in enumerate(energies_eV):
        bar.progress(k / len(energies_eV), text=f'{label}: {E / 1e6:.3g} MeV ({k + 1} of {len(energies_eV)})')
        m = core.sized_material(material, ion, E, srim_dir, use_cache=use_cache) if auto_size else material
        profiles.append(run_one(m, ion, E, n_ions, srim_dir, use_cache))
    bar.progress(1.0, text=f'{label}: done')
    return profiles


with tab_opt:
    st.markdown('Finds the beam **energy** and **fluence** that give an ROI of at least a minimum width with '
                'a chosen **mean** dpa. It uses the same ROI rules as the *Fluence & ROI* tab and the ion, '
                'material and number of ions from the sidebar.')
    st.caption('Only energy needs TRIM runs: dpa scales linearly with fluence, so the fluence is calculated '
               'exactly for every candidate window. Changing the targets below re-ranks the results instantly '
               'without re-running TRIM.')

    c1, c2, c3, c4 = st.columns(4)
    opt_dpa = c1.number_input('Target mean dpa', min_value=0.001, value=3.0, key='opt_dpa')
    opt_tol = c2.number_input('Tolerance (± dpa)', min_value=0.0, value=0.15, format='%.3g', key='opt_tol',
                              help='Sets the acceptable fluence range around the optimum.')
    opt_width = c3.number_input('Minimum ROI width (nm)', min_value=1.0, value=100.0, key='opt_width')
    objective = c4.selectbox('Optimise for', list(core.OBJECTIVES), format_func=core.OBJECTIVES.get,
                             key='opt_obj')
    c1, c2, c3, c4, c5 = st.columns(5)
    opt_min_depth = c1.number_input('ROI starts below (nm)', min_value=0.0, value=100.0, key='opt_min_depth')
    opt_margin = c2.number_input('Min. distance before implant peak (nm)', min_value=0.0, value=100.0,
                                 key='opt_margin')
    e_min = c3.number_input('Lowest energy (MeV)', min_value=0.01, value=1.0, key='opt_emin')
    e_max = c4.number_input('Highest energy (MeV)', min_value=0.01, value=10.0, key='opt_emax',
                            help='Set to the highest energy your accelerator can deliver for this ion.')
    n_e = int(c5.number_input('Energies to try', min_value=2, max_value=12, value=6, key='opt_ne'))
    c1, c2 = st.columns(2)
    auto_size = c1.checkbox('Size the target for each energy (recommended)', value=True, key='opt_auto',
                            help='A quick 30-ion pilot run sets the target thickness to about 1.3x the damaged '
                                 'depth, so the 100 depth bins resolve the ROI at every energy. Overrides the '
                                 'sidebar thickness.')
    refine = c2.checkbox('Refine around the best energy', value=True, key='opt_refine',
                         help='After the grid, runs two extra energies either side of the best one.')
    c1, c2 = st.columns(2)
    limit_spread = c1.checkbox('Limit dpa spread inside the ROI', value=False, key='opt_limit_spread',
                               help='Off: only the mean dpa of the ROI must hit the target, so the ROI can sit on '
                                    'a steep part of the profile. On: every point of the (smoothed) dpa profile in '
                                    'the ROI must also be within the spread below. Flatter ROIs usually need a '
                                    'higher fluence or a narrower ROI.')
    spread_pct = c2.number_input('Max spread (± % of target dpa)', min_value=1.0, max_value=100.0, value=10.0,
                                 step=1.0, key='opt_spread', disabled=not limit_spread)
    max_spread = spread_pct / 100 if limit_spread else None
    st.caption(f'About {n_e + (2 if refine else 0)} TRIM runs of {n_ions:,} ions'
               + (' plus short pilot runs' if auto_size else '') + '. High energies with many ions can take '
               'several minutes each.')

    if st.button('Run optimiser', type='primary', disabled=not can_run or e_max <= e_min, key='opt_run'):
        try:
            grid = [float(f'{E:.3g}') for E in np.geomspace(e_min, e_max, n_e) * 1e6]
            profiles = run_optimiser_energies(grid, auto_size, 'Energy grid')
            if refine:
                ranked = core.rank_energies(profiles, opt_dpa, opt_tol, opt_width, opt_min_depth, opt_margin,
                                            objective, max_spread)
                if 'fluence' in ranked and pd.notna(ranked.loc[0, 'fluence']):
                    k = grid.index(min(grid, key=lambda E: abs(E / 1e6 - ranked.loc[0, 'energy_MeV'])))
                    extra = [float(f'{np.sqrt(grid[k] * grid[n]):.3g}') for n in (k - 1, k + 1)
                             if 0 <= n < len(grid)]
                    profiles += run_optimiser_energies(extra, auto_size, 'Refining')
            st.session_state.opt = sorted(profiles, key=lambda p: p.energy_eV)
        except Exception as e:
            st.error(f'TRIM failed: {e}')

    opt_profiles = st.session_state.get('opt')
    if opt_profiles:
        ranked = core.rank_energies(opt_profiles, opt_dpa, opt_tol, opt_width, opt_min_depth, opt_margin,
                                    objective, max_spread)
        feasible = ranked.dropna(subset=['fluence']) if 'fluence' in ranked else ranked.iloc[0:0]
        if feasible.empty:
            st.warning('No energy gives an ROI with these settings. Try a smaller minimum width, a smaller '
                       'distance before the implant peak, a shallower start depth or a higher maximum energy'
                       + (', or allow a larger dpa spread.' if limit_spread else '.'))
        else:
            best = feasible.iloc[0]
            st.markdown(f'#### Best: {best.energy_MeV:.3g} MeV, {best.fluence:.3e} ions/cm²')
            m = st.columns(4)
            m[0].metric('Beam energy', f'{best.energy_MeV:.3g} MeV')
            m[1].metric('Fluence', f'{best.fluence:.3e}',
                        help=f'Any fluence from {best.fluence_min:.3e} to {best.fluence_max:.3e} keeps the ROI '
                             f'mean within ±{opt_tol:g} dpa.')
            m[2].metric('ROI', f'{best.roi_start_nm:.0f}–{best.roi_end_nm:.0f} nm', f'{best.roi_width_nm:.0f} nm wide',
                        delta_color='off')
            m[3].metric('Max implanted ions in ROI', f'{best.max_implanted_appm_in_roi:.3g} appm')
            lo, hi = (best.dpa_min_in_roi / opt_dpa - 1) * 100, (best.dpa_max_in_roi / opt_dpa - 1) * 100
            msg = (f'Mean dpa in the ROI is {opt_dpa:g}; the dpa **varies from {best.dpa_min_in_roi:.2f} to '
                   f'{best.dpa_max_in_roi:.2f}** across it ({lo:+.0f}% / {hi:+.0f}% of the target'
                   + (f', within the ±{spread_pct:g}% limit).' if limit_spread else ').'))
            (st.warning if not limit_spread and max(-lo, hi) > 10 else st.info)(msg)
            st.caption(f'Acceptable fluence range: {best.fluence_min:.3e} to {best.fluence_max:.3e} ions/cm². '
                       f'Peak dpa at this fluence: {best.peak_dpa:.2f}. Depth bins: {best.bin_nm:.0f} nm.')

            p_best = min(opt_profiles, key=lambda p: abs(p.energy_eV / 1e6 - best.energy_MeV))
            max_end = p_best.implant_peak_nm - opt_margin
            st.plotly_chart(depth_figure(
                [{'name': f'{best.energy_MeV:.3g} MeV, {best.fluence:.2e} ions/cm²', 'depth': p_best.depth_nm,
                  'dpa': p_best.dpa(best.fluence), 'appm': p_best.implanted_appm(best.fluence)}],
                hline=opt_dpa, xmax=auto_xmax([p_best]), overlay=overlay,
                segments=[('ROI', best.roi_start_nm, best.roi_end_nm, opt_dpa, COLORS[1])],
                vlines=[(p_best.implant_peak_nm, 'implant peak'), (max_end, 'ROI limit')]), key='opt_fig')

            metric_col, metric_label = {
                'fluence': ('fluence', 'Fluence for target (ions/cm²)'),
                'implant': ('max_implanted_appm_in_roi', 'Max implanted ions in ROI (appm)'),
                'width': ('roi_width_nm', 'Widest ROI (nm)'),
            }[objective]
            curve = feasible.sort_values('energy_MeV')
            st.plotly_chart(xy_figure(curve['energy_MeV'], curve[metric_col], 'Beam energy (MeV)', metric_label),
                            key='opt_curve')

        # Cost of uniformity: lowest fluence (over all energies tried) for each allowed spread.
        st.markdown('##### How the fluence rises as the ROI must be flatter')
        tradeoff = []
        for s in [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]:
            r = core.rank_energies(opt_profiles, opt_dpa, opt_tol, opt_width, opt_min_depth, opt_margin,
                                   'fluence', s)
            if 'fluence' in r and pd.notna(r.loc[0, 'fluence']):
                tradeoff.append({'spread_pct': s * 100, 'fluence': r.loc[0, 'fluence'],
                                 'energy_MeV': r.loc[0, 'energy_MeV']})
        mean_only = core.rank_energies(opt_profiles, opt_dpa, opt_tol, opt_width, opt_min_depth, opt_margin, 'fluence')
        if tradeoff:
            t = pd.DataFrame(tradeoff)
            st.plotly_chart(xy_figure(t['spread_pct'], t['fluence'], 'Allowed dpa spread in ROI (± %)',
                                      'Lowest fluence for target (ions/cm²)', logx=False), key='opt_tradeoff')
            note = f'Tightest spread achievable: ±{t["spread_pct"].min():g}%.'
            if 'fluence' in mean_only and pd.notna(mean_only.loc[0, 'fluence']):
                note += f' Mean-only (no spread limit) needs {mean_only.loc[0, "fluence"]:.3e} ions/cm².'
            st.caption(note + ' Uses the energies tried, the minimum width and the ROI depth limits above.')
        else:
            st.caption('No energy gives an ROI within ±50% spread with these settings.')

        st.markdown('##### All energies tried (best first)')
        st.dataframe(ranked, hide_index=True)
        infeasible = ranked[ranked['fluence'].isna()]['energy_MeV'] if 'fluence' in ranked else ranked['energy_MeV']
        if len(infeasible):
            st.caption('No valid ROI at: ' + ', '.join(f'{e:.3g} MeV' for e in infeasible))
        st.download_button('Download optimiser table (CSV)', ranked.to_csv(index=False),
                           file_name='beam_optimiser.csv', mime='text/csv')

# -----------------------------------------------------
# Compare runs
# -----------------------------------------------------

with tab_compare:
    history = st.session_state.history
    if len(history) < 2:
        st.info('Make at least two runs (e.g. change the material preset and run again), then compare them here.')
    else:
        picked = st.multiselect('Runs to compare', list(history), default=list(history)[-2:],
                                max_selections=MAX_SERIES)
        c1, c2 = st.columns(2)
        fluence_c = c1.number_input('Common fluence (ions/cm²)', min_value=1e8, value=2.5e15, format='%.3e',
                                    key='cmp_F')
        sm = c2.checkbox('Smooth (display only)', value=True, key='cmp_smooth')
        if picked:
            ps = [history[k] for k in picked]
            st.plotly_chart(depth_figure([{'name': k, 'depth': p.depth_nm,
                                           'dpa': maybe_smooth(p.dpa(fluence_c), sm),
                                           'appm': maybe_smooth(p.implanted_appm(fluence_c), sm)}
                                          for k, p in zip(picked, ps)], xmax=auto_xmax(ps), overlay=overlay), key='cmp_fig')
            df = pd.DataFrame([p.summary(fluence_c) for p in ps])
            df.insert(0, 'run', picked)
            st.dataframe(df, hide_index=True)
            st.download_button('Download comparison (CSV)', df.to_csv(index=False),
                               file_name='comparison.csv', mime='text/csv')

# -----------------------------------------------------
# Convergence
# -----------------------------------------------------

with tab_conv:
    st.markdown(r'Re-runs TRIM for the sidebar material and ion with increasing numbers of ions. '
                r'When the % difference from the largest run is small (a few %), you have enough ions.')
    c1, c2, c3 = st.columns(3)
    conv_E = c1.number_input('Beam energy (MeV)', min_value=0.001, value=5.0, key='conv_E')
    conv_F = c2.number_input('Fluence (ions/cm²)', min_value=1e8, value=1e15, format='%.3e', key='conv_F')
    counts_txt = c3.text_input('Ion counts', '100, 500, 1000, 2500, 5000')
    try:
        counts = sorted({int(v) for v in parse_list(counts_txt)})
    except ValueError:
        counts = []
        st.error('Could not read the ion counts. Example: 100, 500, 1000')
    st.caption('Large counts can take many minutes: each run blocks the app until TRIM finishes.')

    if st.button('Run convergence test', type='primary', disabled=not (can_run and counts), key='conv_run'):
        try:
            st.session_state.conv = run_many([(material, ion, conv_E * 1e6, n) for n in counts], srim_dir, use_cache)
        except Exception as e:
            st.error(f'TRIM failed: {e}')

    conv = st.session_state.get('conv')
    if conv:
        df = core.convergence_table(conv, conv_F)
        a, b = st.columns(2)
        a.plotly_chart(xy_figure(df['number_ions'], df['N_NRT_per_ion'], 'Number of ions',
                                 'NRT displacements / ion'), key='conv_nrt')
        b.plotly_chart(xy_figure(df['number_ions'], df['peak_dpa'], 'Number of ions',
                                 f'Peak dpa at {conv_F:.2e}'), key='conv_dpa')
        shown = sorted(conv, key=lambda p: p.number_ions)[-MAX_SERIES:]
        st.plotly_chart(depth_figure([{'name': f'{p.number_ions:,} ions', 'depth': p.depth_nm,
                                       'dpa': p.dpa(conv_F), 'appm': p.implanted_appm(conv_F)} for p in shown],
                                     show_implanted=False, xmax=auto_xmax(shown)), key='conv_fig')
        st.dataframe(df[['number_ions', 'N_NRT_per_ion', 'N_NRT_pct_diff', 'peak_dpa', 'peak_dpa_pct_diff']],
                     hide_index=True)
