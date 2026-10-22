import numpy as np

import srim_core as core
import srim_surrogate as sur


def test_shape_on_u_preserves_area():
    depth = np.linspace(20, 2000, 100)            # TRIM-style upper bin edges
    density = np.exp(-0.5 * ((depth - 900) / 200) ** 2)
    shape, total = sur.shape_on_u(depth, density, Rp_nm=1000.0)
    assert abs(np.sum(shape) * sur.DU - 1.0) < 1e-9
    assert abs(total - np.sum(density) * 19.98) / total < 1e-3


def test_refine_keeps_area_and_shape():
    u = (sur.U_EDGES[:-1] + sur.U_EDGES[1:]) / 2
    s = np.exp(-0.5 * ((u - 0.8) / 0.2) ** 2)
    fine = sur._refine(s)
    assert abs(np.sum(fine) * sur.DU_OUT - np.sum(s) * sur.DU) < 1e-12
    uf = (sur.OUT_EDGES[:-1] + sur.OUT_EDGES[1:]) / 2
    assert np.max(np.abs(fine - np.exp(-0.5 * ((uf - 0.8) / 0.2) ** 2))) < 0.01


def test_lindhard_eta_matches_nrt_for_self_ions():
    # Cu -> Cu at 10 MeV: k = 0.1337 Z^(1/6) (Z/A)^(1/2) and eps = E / E_L as in the NRT standard
    Z, A, E = 29.0, 63.546, 1e7
    k = 0.1337 * Z ** (1 / 6) * (Z / A) ** 0.5
    eps = E / (30.724 * Z * Z * np.sqrt(2 * Z ** (2 / 3)) * 2)
    nrt = 1 / (1 + k * (3.4008 * eps ** (1 / 6) + 0.40244 * eps ** 0.75 + eps))
    assert abs(sur.lindhard_eta(Z, A, Z, A, E) / nrt - 1) < 0.005


def test_reduced_energy_scales_with_energy():
    eps = sur.reduced_energy(29, 63.5, 29, 63.5, [1e6, 2e6])
    assert abs(eps[1] / eps[0] - 2.0) < 1e-12
    X = sur.features(core.MATERIALS['W'], 'W', [5e6, 20e6])
    assert X.shape == (2, 4) and X[0, 1] == 0.0      # self-ion: log mass ratio is zero


def _synthetic_set(n=40, seed=0):
    # Shapes whose peak position moves smoothly with the first input: something a GP should learn
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-0.5, 0.5, n), rng.choice([26, 29, 74], n),
                         rng.choice([26, 29, 74], n)])
    u = (sur.U_EDGES[:-1] + sur.U_EDGES[1:]) / 2
    peaks = 0.8 + 0.1 * X[:, 0]
    S = np.exp(-0.5 * ((u[None] - peaks[:, None]) / 0.2) ** 2)
    S /= S.sum(1, keepdims=True) * sur.DU
    G = np.exp(-0.5 * ((u[None] - 1.0) / 0.25) ** 2)
    G = np.repeat(G / (G.sum() * sur.DU), n, axis=0)
    return sur.TrainingSet(targets=np.array(['Cu'] * n), ions=np.array(['Cu'] * n), energy_eV=np.full(n, 5e6),
                           X=X, log_eta=-1.0 + 0.2 * X[:, 0], stopped=np.ones(n), S=S, G=G, order=np.arange(n))


def test_sr_nuclear_fraction():
    E = np.geomspace(1e4, 6e7, 200)
    half = core.RangeTable('X', E, np.full_like(E, 2.0), np.full_like(E, 2.0), E, E)
    assert np.allclose(sur.sr_nuclear_fraction(half, [1e5, 1e7]), 0.5)
    # Electronic stopping growing with energy: the nuclear share falls as the energy rises
    falling = core.RangeTable('X', E, 1e-3 * E ** 0.5, np.full_like(E, 1.0), E, E)
    f = sur.sr_nuclear_fraction(falling, [1e5, 1e6, 1e7])
    assert np.all(np.diff(f) < 0) and np.all((0 < f) & (f < 1))


def test_fit_save_load_predict_round_trip(tmp_path):
    ts = _synthetic_set()
    model = sur.Surrogate.fit(ts, eta_baseline=None)
    path = str(tmp_path / 'model.npz')
    model.save(path)
    loaded = sur.Surrogate.load(path)
    m1, _ = model.predict_outputs(ts.X[:5])
    m2, _ = loaded.predict_outputs(ts.X[:5])
    for g in sur.GROUPS:
        assert np.allclose(m1[g], m2[g], atol=1e-6)
    # A training point's damage shape comes back closely
    s_pred = np.clip(loaded.a['damage_mean'] + m2['damage'][0] @ loaded.a['damage_comp'], 0, None)
    assert 0.5 * np.sum(np.abs(s_pred - ts.S[0])) * sur.DU < 0.05


def test_predicted_profile_works_with_roi_options(tmp_path):
    model = sur.Surrogate.fit(_synthetic_set(), eta_baseline=None)
    m = core.MATERIALS['Cu']
    E = np.geomspace(1e4, 6e7, 60)
    table = core.RangeTable('Cu', E, E * 0, E * 0, 1e-3 * E ** 0.8, 3e-4 * E ** 0.8)
    p = model.predict_profiles(m, 'Cu', [5e6], table=table)[0]
    assert p.meta['surrogate'] and len(p.depth_nm) == len(sur.OUT_EDGES) - 1
    assert abs(np.sum(p.implanted) * p.bin_nm * 1e-7 - 1.0) < 0.05   # ~all ions stop
    assert abs(p.Tdam_eV - np.sum(p.Tdam_density) * p.bin_nm * 10) / p.Tdam_eV < 1e-9
    r = core.roi_options(p, 1.0, 0.05, 0.1 * p.meta['Rp_nm'], 0.1 * p.meta['Rp_nm'], 0.0, 'width', None)
    assert r is not None and r['fluence'] > 0
    samples = model.predict_profiles(m, 'Cu', [5e6], table=table, n_samples=5)
    assert len(samples) == 5
