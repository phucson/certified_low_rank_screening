"""Rank ladder: certified fraction, band fraction and cost by rank (main-text table).

TVaR(0.99); threshold at the population median (the adverse case, since
candidates then cluster on it).  Asserts the bracket and verifies that the
certified band contains the true tail set on every candidate, and asserts the
Wasserstein-robust bracket at two radii.

The speedup columns are operation counts from the fallback-cost proposition,
evaluated by `certproj.cost` on the measured f and beta; they are not timings.
The wall-clock experiment reports which of them survive conversion.
"""
import argparse, json, sys
import numpy as np
sys.path.insert(0, "..")
from certproj import (Projection, FAMILIES, weights, rho, certified_band,
                      screen, robust_bracket, speedups, iota,
                      assert_version, dump)
from certproj.data import yield_changes, block_bootstrap
from certproj.populations import bond_books

ALPHA = 0.99
RADII = (0.01, 0.1)          # radii at which the robust bracket is asserted


def run(F, cands, r, g, iota_val, alpha=ALPHA):
    proj = Projection(F, r)
    w = weights(len(F), g)
    m = int(np.ceil((1 - alpha) * len(F)))
    Ce = proj.constant(g)
    true, surr, half, band, ok = [], [], [], [], True
    rob_ok = True
    for c in cands:
        P, Pt = c.evaluate(F), proj.surrogate(c)
        gam = proj.gamma(c)
        true.append(rho(P, w)); surr.append(rho(Pt, w)); half.append(gam * Ce)
        B = certified_band(Pt, gam * proj.enorm, m)
        band.append(B.mean())
        top = np.argpartition(P, len(F) - m)[len(F) - m:]
        ok &= bool(B[top].all())
        # Robust bracket.  Under the closed form each robust value is its
        # nominal value inflated by radius * lip * iota, so the robust deviation
        # is checked directly against the robust half-width.
        for rad in RADII:
            hrob = robust_bracket(proj, c, g, rad, iota_val)
            dev = abs((true[-1] + rad * c.lipschitz() * iota_val)
                      - (surr[-1] + rad * proj.surrogate_lipschitz(c) * iota_val))
            rob_ok &= bool(dev <= hrob + 1e-9)
    true, surr, half = map(np.array, (true, surr, half))
    assert np.all(np.abs(true - surr) <= half + 1e-9), "BRACKET VIOLATED"
    assert rob_ok, "ROBUST BRACKET VIOLATED"
    Q = np.median(true)
    f = float(np.mean([screen(s, h, Q) != "undecided"
                       for s, h in zip(surr, half)]))
    Sig = np.cov(F, rowvar=False)
    ev = np.linalg.eigvalsh(Sig)[::-1]
    beta_med = float(np.median(band))
    K, d = cands[0].K, F.shape[1]
    sp_dir, sp_env = speedups(K, d, r, f, beta_med)
    return dict(r=r, certified=f, band=beta_med,
                band_mean=float(np.mean(band)),
                band_correct=ok, robust_bracket_ok=rob_ok,
                var_captured=float(ev[:r].sum() / ev.sum()),
                tightness=float(np.median(np.abs(true - surr)
                                          / np.maximum(half, 1e-12))),
                speedup_direct=sp_dir, speedup_envelope=sp_env, K=K, d=d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--casdatasets", required=True)
    ap.add_argument("-N", type=int, default=40_000)
    ap.add_argument("-B", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    assert_version()
    F_src = yield_changes(a.casdatasets)     # 371 monthly changes, 8 key rates
    F = block_bootstrap(F_src, a.N, a.seed, block=6)
    g = FAMILIES["TVaR(0.99)"]
    iota_val = iota(g)                       # finite for TVaR: (1-alpha)^-1
    rows = []
    print(f"{'books':>7}{'r':>3}{'var':>9}{'f':>8}{'beta':>9}{'tight':>8}"
          f"{'dir':>9}{'env':>9}{'band ok':>9}{'rob ok':>8}")
    for m_hold in (4, 8):
        cands = bond_books(a.B, m_hold, K=5, seed=100 + m_hold)
        for r in (1, 2, 3, 4):
            o = run(F, cands, r, g, iota_val)
            o["books"] = f"m={m_hold}"; rows.append(o)
            print(f"{o['books']:>7}{r:>3}{o['var_captured']:>9.4f}"
                  f"{o['certified']:>8.3f}{o['band']:>9.4f}"
                  f"{o['tightness']:>8.3f}{o['speedup_direct']:>8.1f}x"
                  f"{o['speedup_envelope']:>8.1f}x"
                  f"{str(o['band_correct']):>9}{str(o['robust_bracket_ok']):>8}")
    dump("../results/table2_treasury.json", rows,
         dict(dataset="US Treasury key rates (FedYieldCurve)",
              sigma="TVaR(0.99)", threshold="median of true values",
              N=a.N, B=a.B, seed=a.seed, block_length=6,
              n_changes=len(F_src), d=F.shape[1], K=5,
              holdings=[4, 8], ranks=[1, 2, 3, 4],
              robust_radii=list(RADII),
              beta_reported="median across candidates",
              speedup_note="operation counts, not timings; see wall-clock section"))
