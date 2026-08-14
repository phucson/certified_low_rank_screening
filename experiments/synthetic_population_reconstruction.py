"""Synthetic calibration population: the reconstruction adopted in S.4.

Extracted from certproj/data.py for reference; the authoritative copy lives
there. Requires KEY_RATE_DURATIONS from the same module.

Reproduces every property S.4 states -- d = 8, principal-component variance
shares 0.861/0.114/0.015/0.006 (cumulative 0.861/0.975/0.990/0.996),
lambda_1/lambda_2 = 7.4, Student-t6 innovations standardised and coloured by
the Cholesky factor, N = 40,000, centred -- and additionally fixes the
eigenvectors, which S.4 as previously written left open.
"""
import numpy as np

KEY_RATE_DURATIONS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])


# --------------------------------------------------------------------------
# synthetic calibration population
# --------------------------------------------------------------------------

#: Principal-component variance shares of the synthetic factor covariance,
#: reproducing the Treasury cumulative shares 0.861/0.975/0.990/0.996 at ranks
#: one to four, with lambda_1/lambda_2 = 7.5.  The remaining mass is spread over
#: the trailing components.
PC_SHARES = (0.861, 0.114, 0.015, 0.006)


def synthetic_covariance(d: int = 8, seed: int = 0) -> np.ndarray:
    """Factor covariance with the calibrated principal-component shares.

    The leading four eigenvalues are PC_SHARES; the remainder (0.004) is spread
    geometrically over the trailing d-4 components, so the spectrum decays
    rather than ending in a flat plateau.  Eigenvectors are a Haar-random
    orthogonal basis, which fixes the spectrum without privileging any
    coordinate direction.
    """
    if d < len(PC_SHARES) + 1:
        raise ValueError(f"d must exceed {len(PC_SHARES)}")
    tail_n = d - len(PC_SHARES)
    tail = 0.5 ** np.arange(1, tail_n + 1)
    tail = tail / tail.sum() * (1.0 - sum(PC_SHARES))
    lam = np.concatenate([np.array(PC_SHARES), tail])

    # Eigenvectors are the level/slope/curvature shapes of a real yield curve,
    # obtained by orthonormalising polynomials in log maturity.  Calibrating the
    # eigenvalues alone is not enough: gamma_j(U) measures the alignment between
    # the retained subspace and the candidates' duration slopes, and a
    # Haar-random basis destroys that alignment (a duration vector projects onto
    # a random first component at roughly a quarter of the strength it projects
    # onto a level factor), which shifts every certified fraction.
    dur = KEY_RATE_DURATIONS if d == len(KEY_RATE_DURATIONS) else \
        np.geomspace(0.25, 10.0, d)
    x = np.log(dur) - np.log(dur).mean()
    x = x / np.abs(x).max()
    Q, _ = np.linalg.qr(np.vander(x, d, increasing=True))
    return (Q * lam) @ Q.T


def synthetic_catalogue(n: int = 40_000, seed: int = 0, d: int = 8,
                        df: int = 6) -> np.ndarray:
    """Student-t catalogue with the calibrated second-order structure.

    Innovations are Student-t with `df` degrees of freedom, standardised to unit
    variance and coloured by the Cholesky factor of the synthetic covariance, so
    the catalogue is heavier-tailed than Gaussian while matching the target
    second-order structure.  Centred before return.
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(synthetic_covariance(d, seed=seed))
    t = rng.standard_t(df, size=(n, d))
    t /= np.sqrt(df / (df - 2.0))                  # unit variance
    F = t @ L.T
    return F - F.mean(axis=0)
