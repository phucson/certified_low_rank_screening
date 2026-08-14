"""Choice of subspace at fixed rank, and the population features that favour it.

Three rank-r subspaces are natural, all computable before any candidate is
evaluated: U_M (leading eigenvectors of M = sum_jt b_jt b_jt'), U_F (leading
eigenvectors of Sigma_F), and U_G (leading generalised eigenvectors of the
pencil (M, Sigma_F)).  None dominates, because each trades the numerator of the
signal-to-noise ratio against its denominator differently.

Also reports what bounds any exact solution: the ratio of gamma_j(U) to the
median orthogonal slope norm, and the two population features -- sparsity and
heterogeneity of alignment -- that raise the certified fraction at fixed rank.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "..")
from certproj import (Projection, FAMILIES, weights, rho, screen,
                      assert_version, dump)
from certproj.data import synthetic_catalogue
from certproj.populations import bond_books
from certproj.certificate import Candidate

ALPHA = 0.99


def subspaces(F, cands, r):
    Sig = np.cov(F - F.mean(axis=0), rowvar=False)
    M = sum(c.b.T @ c.b for c in cands)
    UF = np.linalg.eigh(Sig)[1][:, ::-1][:, :r]
    UM = np.linalg.eigh(M)[1][:, ::-1][:, :r]
    from scipy.linalg import eigh as geigh
    UG = geigh(M, Sig)[1][:, ::-1][:, :r]
    UG, _ = np.linalg.qr(UG)                    # orthonormalise the pencil basis
    return {"U_M": UM, "U_F": UF, "U_G": UG}


def certified(F, cands, U, g):
    proj = Projection(F, U.shape[1], U=U)
    w = weights(len(F), g)
    Ce = proj.constant(g)
    true = np.array([rho(c.evaluate(F), w) for c in cands])
    surr = np.array([rho(proj.surrogate(c), w) for c in cands])
    gam = np.array([proj.gamma(c) for c in cands])
    half = gam * Ce
    assert np.all(np.abs(true - surr) <= half + 1e-9), "BRACKET VIOLATED"
    Q = float(np.median(true))
    f = float(np.mean([screen(s, h, Q) != "undecided"
                       for s, h in zip(surr, half)]))
    med = np.median([np.median(np.linalg.norm(
        c.b - (c.b @ proj.U) @ proj.U.T, axis=1)) for c in cands])
    return f, float(np.median(gam) / max(med, 1e-12))


def variant(B, d, loaded, spread, seed):
    """Bond-book-like population with controlled sparsity and heterogeneity."""
    rng = np.random.default_rng(seed)
    dur = np.geomspace(0.25, 10.0, d)
    out = []
    for _ in range(B):
        S = rng.choice(d, size=loaded, replace=False)
        w = rng.exponential(size=loaded); w /= w.sum()
        base = np.zeros(d); base[S] = -w * dur[S] * 100.0
        tilt = rng.uniform(1 - spread, 1 + spread)      # alignment dispersion
        b = np.zeros((5, d))
        for t in range(5):
            keep = rng.random(loaded) < 0.75
            v = np.zeros(d); v[S[keep]] = base[S[keep]] * tilt
            b[t] = v
        out.append(Candidate(rng.normal(scale=0.5, size=5), b))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=40_000)
    ap.add_argument("-B", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert_version()

    F = synthetic_catalogue(a.N, seed=a.seed)
    cands = bond_books(a.B, m=4, K=5, seed=a.seed)
    g = FAMILIES["TVaR(0.99)"]

    print("subspace choice at fixed rank (synthetic population)")
    print(f"{'r':>3}{'U_M':>10}{'U_F':>10}{'U_G':>10}   best")
    rows, ratios = [], []
    for r in (1, 2, 3, 4):
        U = subspaces(F, cands, r)
        fs = {}
        for name, Ur in U.items():
            f, ratio = certified(F, cands, Ur, g)
            fs[name] = f; ratios.append(ratio)
        best = max(fs, key=fs.get)
        rows.append(dict(r=r, **fs, best=best))
        print(f"{r:>3}{fs['U_M']:>10.3f}{fs['U_F']:>10.3f}{fs['U_G']:>10.3f}"
              f"   {best}")
    print(f"\ngamma_j(U) over the median orthogonal slope norm: "
          f"[{min(ratios):.2f}, {max(ratios):.2f}]")

    # Sparsity and heterogeneity, at rank 2, d = 10.
    print("\npopulation features at r=2, d=10")
    feats = {}
    for label, loaded, spread in [("dense (8 of 10 loaded)", 8, 0.0),
                                  ("sparse (2 of 10 loaded)", 2, 0.0),
                                  ("homogeneous alignment", 4, 0.0),
                                  ("heterogeneous alignment", 4, 0.6)]:
        Fv = synthetic_catalogue(a.N, seed=a.seed, d=10)
        cv = variant(a.B, 10, loaded, spread, seed=a.seed)
        UF = np.linalg.eigh(np.cov(Fv, rowvar=False))[1][:, ::-1][:, :2]
        f, _ = certified(Fv, cv, UF, g)
        feats[label] = f
        print(f"  {label:<28} f = {f:.3f}")

    dump("../results/subspace_and_population.json",
         dict(subspace_by_rank=rows,
              gamma_ratio_range=[min(ratios), max(ratios)],
              population_features=feats),
         dict(population="synthetic (S.4)", N=a.N, B=a.B, seed=a.seed,
              sigma="TVaR(0.99)", threshold="population median"))
