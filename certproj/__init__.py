"""Certified dimension reduction for spectral risk measures of max-affine exposures.

Reference implementation accompanying the paper of the same title.
"""
from .spectral import (weights, rho, spectrum, iota, lp_norm,
                       effective_tail_support, tvar, dual_power,
                       proportional_hazard, wang, FAMILIES)
from .certificate import (Candidate, Projection, bracket, robust_bracket,
                          tight_envelope, certified_band, screen)
from .cost import surrogate_cost, saving_factor, speedup, speedups
from .provenance import assert_version, git_commit, dump

__version__ = "1.2.1"
