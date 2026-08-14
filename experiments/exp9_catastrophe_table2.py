"""Catastrophe block of the main certified-fraction table.

Same protocol as the Treasury block (TVaR(0.99), threshold at the population
median, ranks 1-4, nominal and robust brackets asserted, band containment
checked on every candidate), on the five-peril catalogue at the design of the
main numerical study: N = 40,000 simulated events against excess-of-loss
treaties over peril subsets.

This is the adverse dataset for projection: log-scale peril correlations are
weak, so principal components capture far less variance at low rank than the
Treasury key rates do.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "..")
from certproj import (Projection, FAMILIES, weights, rho, certified_band,
                      screen, robust_bracket, speedups, iota,
                      assert_version, dump)
from certproj.data import catastrophe_catalogue, PERILS
from certproj.populations import treaties

ALPHA, K = 0.99, 4
RADII = (0.01, 0.1)


def run(F, cands, r, g, iota_val, alpha=ALPHA):
    proj = Projection(F, r)
    w = weights(len(F), g)
    m = int(np.ceil((1 - alpha) * len(F)))
    Ce = proj.constant(g)
    true, surr, half, band, ok, rob_ok = [], [], [], [], True, True
    for c in cands:
        P, Pt = c.evaluate(F), proj.surrogate(c)
        gam = proj.gamma(c)
        true.append(rho(P, w)); surr.append(rho(Pt, w)); half.append(gam * Ce)
        B = certified_band(Pt, gam * proj.enorm, m)
        band.append(B.mean())
        top = np.argpartition(P, len(F) - m)[len(F) - m:]
        ok &= bool(B[top].all())
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
    ev = np.linalg.eigvalsh(np.cov(F, rowvar=False))[::-1]
    beta_med = float(np.median(band))
    d = F.shape[1]
    sp_dir, sp_env = speedups(K, d, r, f, beta_med)
    return dict(r=r, certified=f, band=beta_med,
                band_mean=float(np.mean(band)), band_correct=ok,
                robust_bracket_ok=rob_ok,
                var_captured=float(ev[:r].sum() / ev.sum()),
                tightness=float(np.median(np.abs(true - surr)
                                          / np.maximum(half, 1e-12))),
                speedup_direct=sp_dir, speedup_envelope=sp_env, K=K, d=d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--casdatasets", required=True)
    ap.add_argument("-N", type=int, default=40_000)
    ap.add_argument("-B", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    assert_version()

    F, C = catastrophe_catalogue(a.casdatasets, a.N, a.seed)
    iu = np.triu_indices(len(PERILS), 1)
    print(f"catastrophe catalogue: N={a.N}, d={F.shape[1]} perils, B={a.B}")
    print(f"log-correlation range: [{C[iu].min():+.3f}, {C[iu].max():+.3f}]")
    cands = treaties(a.B, K=K, seed=0)
    g = FAMILIES["TVaR(0.99)"]
    iota_val = iota(g)

    rows = [run(F, cands, r, g, iota_val) for r in (1, 2, 3, 4)]
    print(f"\n{'r':>3}{'var':>9}{'f':>8}{'beta':>9}{'tight':>8}"
          f"{'dir':>9}{'env':>9}{'band ok':>9}{'rob ok':>8}")
    for o in rows:
        print(f"{o['r']:>3}{o['var_captured']:>9.4f}{o['certified']:>8.3f}"
              f"{o['band']:>9.4f}{o['tightness']:>8.3f}"
              f"{o['speedup_direct']:>8.1f}x{o['speedup_envelope']:>8.1f}x"
              f"{str(o['band_correct']):>9}{str(o['robust_bracket_ok']):>8}")
    dump("../results/table2_catastrophe.json", rows,
         dict(dataset="Australian multi-peril catalogue (auscathist)",
              sigma="TVaR(0.99)", threshold="median of true values",
              N=a.N, B=a.B, seed=a.seed, d=F.shape[1], K=K,
              ranks=[1, 2, 3, 4], robust_radii=list(RADII),
              beta_reported="median across candidates"))
