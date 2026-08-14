"""Wall-clock timings of the three evaluation paths.

Operation counts are not timings.  This measures per-candidate wall-clock of
exact evaluation, the direct surrogate pass, and envelope point location, so
the reported speedups can be qualified where they do not survive conversion.

At r = 1 the envelope is the exact upper envelope of K lines (O(K log K) build,
O(log K) query, verified exact against direct evaluation).  At r >= 2 no
one-dimensional ordering exists and the direct pass is used, which is why the
envelope column pays only at r = 1 or at large K.
"""
import argparse, sys, timeit
import numpy as np
sys.path.insert(0, "..")
from certproj import Projection, assert_version, dump
from certproj.data import synthetic_catalogue
from certproj.populations import bond_books


def envelope_r1(c, proj, z):
    """Upper envelope of K lines in one variable.

    Lines y = a + b x; the envelope is the upper convex hull in the dual.
    Sort by slope, keeping only the largest intercept among equal slopes (equal
    slopes never both appear on the envelope, and keeping both would put a zero
    in the breakpoint denominator), then discard any line made redundant by its
    neighbours via the standard convex-hull-trick predicate.
    """
    bt = (c.b @ proj.U).ravel()
    at = c.a + c.b @ proj.mu
    o = np.lexsort((-at, bt))                   # slope asc, intercept desc
    bt, at = bt[o], at[o]
    keep_first = np.concatenate([[True], np.diff(bt) > 0])
    bt, at = bt[keep_first], at[keep_first]

    hull = []
    for i in range(len(bt)):
        while len(hull) >= 2:
            j, k = hull[-1], hull[-2]
            # j (the middle line) is redundant when the k-i intersection occurs
            # no later than the k-j one, i.e. j never attains the maximum
            if (at[k] - at[i]) * (bt[j] - bt[k]) <= (at[k] - at[j]) * (bt[i] - bt[k]):
                hull.pop()
            else:
                break
        hull.append(i)
    bt, at = bt[hull], at[hull]
    xs = ((at[:-1] - at[1:]) / (bt[1:] - bt[:-1])) if len(bt) > 1 \
        else np.empty(0)
    idx = np.searchsorted(xs, z.ravel())
    return at[idx] + bt[idx] * z.ravel()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=40_000)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert_version()

    F = synthetic_catalogue(a.N, seed=a.seed)
    rows = []
    print(f"N={a.N}, d={F.shape[1]}, best of {a.repeat}\n")
    print(f"{'K':>5}{'r':>3}{'exact(ms)':>11}{'direct(ms)':>12}"
          f"{'envelope(ms)':>14}{'exact/direct':>14}{'direct/env':>12}")
    for K in (5, 125, 625):
        c = bond_books(1, m=4, K=K, seed=7)[0]
        t_exact = min(timeit.repeat(lambda: c.evaluate(F), number=1,
                                    repeat=a.repeat)) * 1e3
        for r in (1, 2, 3):
            proj = Projection(F, r)
            t_dir = min(timeit.repeat(lambda: proj.surrogate(c), number=1,
                                      repeat=a.repeat)) * 1e3
            if r == 1:
                exact_ref = proj.surrogate(c)
                assert np.allclose(envelope_r1(c, proj, proj.z), exact_ref), \
                    "envelope disagrees with direct evaluation"
                t_env = min(timeit.repeat(lambda: envelope_r1(c, proj, proj.z),
                                          number=1, repeat=a.repeat)) * 1e3
            else:
                t_env = t_dir          # no 1-D ordering above r = 1
            rows.append(dict(K=K, r=r, exact_ms=t_exact, direct_ms=t_dir,
                             envelope_ms=t_env,
                             exact_over_direct=t_exact / t_dir,
                             direct_over_envelope=t_dir / t_env))
            print(f"{K:>5}{r:>3}{t_exact:>11.2f}{t_dir:>12.2f}{t_env:>14.2f}"
                  f"{t_exact / t_dir:>13.2f}x{t_dir / t_env:>11.2f}x")
    dump("../results/tableS4_wallclock.json", rows,
         dict(population="synthetic (S.4)", N=a.N, d=int(F.shape[1]),
              repeat=a.repeat, seed=a.seed,
              note="envelope available at r=1 only; r>=2 reuses the direct pass"))
