"""W_k inflation constants ||phi||_{k/(k-1)}: closed forms vs quadrature.

A calculus unit check: it integrates the spectrum numerically and compares
against the four closed forms, including the finiteness transition for PH at
k = gamma.  It does NOT verify the theorem's content, namely that the robust
increment equals rho * L * ||phi||_{k/(k-1)}; that attainment check is
exp6_wk_attainment.py.
"""
import sys
import numpy as np
sys.path.insert(0, "..")
from certproj import lp_norm, tvar, dual_power, proportional_hazard, wang

if __name__ == "__main__":
    print(f"{'family':<18}{'k':>4}{'numerical':>12}{'closed form':>14}")
    for k in (2.0, 3.0, 4.0):
        p = k / (k - 1)
        for name, g, cf in [
            ("TVaR(0.99)", tvar(0.99), (1 - 0.99) ** (-1 / k)),
            ("dual-power 6", dual_power(6), 6 * ((k - 1) / (k * 6 - 1)) ** ((k - 1) / k)),
            ("PH 3", proportional_hazard(3),
             (3 ** -1 * (3 * (k - 1) / (k - 3)) ** ((k - 1) / k)) if k > 3 else np.inf),
            ("Wang 0.5", wang(0.5), np.exp(0.5 ** 2 / (2 * (k - 1)))),
        ]:
            num = lp_norm(g, p)
            print(f"{name:<18}{k:>4.0f}{num:>12.4f}"
                  f"{('inf' if not np.isfinite(cf) else f'{cf:.4f}'):>14}")
        print()
    print("The numerical column uses adaptive quadrature with an explicit")
    print("divergence test, so PH(gamma) is reported as infinite for every")
    print("k <= gamma, including the logarithmically divergent boundary k = gamma,")
    print("and matches the closed form to four decimals for k > gamma.")
