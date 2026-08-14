# Certified Low-Rank Screening of Spectral Risk Measures

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducible](https://img.shields.io/badge/results-15%2F15%20reproducible-brightgreen.svg)](#reproducing-the-paper)

Reference implementation for *Certified Low-Rank Screening of Spectral Risk Measures for Max-Affine Exposures* (Nguyen, Vu & Dang).

## The problem, in one paragraph

Selecting a reinsurance programme, screening business units against risk limits,
or generating columns in a risk-constrained optimisation all share one
computational core: evaluating a spectral risk measure of a max-affine loss

```
P_j(F) = max_t ( a_jt + b_jt' F ),        F in R^d
```

repeatedly, across thousands of candidates sharing one N-scenario catalogue.
Each evaluation costs `Θ(N K d)`, and with `d ≥ 2` **no sublinear exact query
structure exists** under near-linear storage in the semigroup model — the tail
of a convex loss is the complement of a polytope, so a tail query is a simplex
range-counting query.

This code gives up exactness, but only in a way that is *quantified*.

## What it computes

Project the factor onto a rank-`r` subspace. The surrogate is a different random
variable, but the gap between their risk measures is bracketed:

```
| ϱ_σ(P_j) − ϱ_σ(P̃_j) |  ≤  γ_j(U) · C^e_σ(U)
                             └──┬──┘   └──┬──┘
                     per candidate,    per catalogue,
                     O(K d), no        computed once,
                     scenario access   reused for every
                                       candidate and every
                                       risk measure
```

**The accept/reject label is exact even though the value is not.** Candidates the
bracket cannot resolve are evaluated exactly on a certified *band* of scenarios
rather than the whole catalogue.

## Install

```bash
git clone https://github.com/phucson/certified-projection.git
cd certified-projection/code
pip install -r requirements.txt
```

Both catalogues are public and ship with the `CASdatasets` R package. Clone it
once (~530 MB) into `code/`, and `run_all.sh` will find it automatically:

```bash
git clone --depth 1 https://github.com/dutangc/CASdatasets.git
```

## Quick start

```python
import numpy as np
from collections import Counter
from certproj import Projection, FAMILIES, weights, rho, screen
from certproj.data import synthetic_catalogue
from certproj.populations import bond_books

F     = synthetic_catalogue(40_000, seed=0)    # d = 8, concentrated factors
proj  = Projection(F, r=2)                     # shared preprocessing, once
g     = FAMILIES["TVaR(0.99)"]
Ce    = proj.constant(g)                       # shared constant
w     = weights(len(F), g)
cands = bond_books(400, m=4, K=5, seed=1)

v = np.array([rho(proj.surrogate(c), w) for c in cands])   # surrogate values
h = np.array([proj.gamma(c) * Ce for c in cands])          # half-widths, O(K d)

Q      = float(np.quantile(v, 0.9))                        # risk limit
labels = [screen(vi, hi, Q) for vi, hi in zip(v, h)]

Counter(labels)   # {'below': 323, 'undecided': 74, 'above': 3}
```

82% of the population is labelled without ever evaluating the true loss, and
those labels are **exact**. The 74 `undecided` candidates are then resolved
exactly on the certified band (`certproj.certificate.certified_band`) rather
than the whole catalogue — typically a few per cent of the scenarios.

Certification is hardest when candidates cluster on the threshold; at the
population median the same run certifies far fewer. Both regimes are swept in
`exp7_threshold_sweep.py`.

## Reproducing the paper

```bash
./run_all.sh                 # finds ./CASdatasets automatically
./run_all.sh /other/path     # or point it somewhere else
```

Runs without `CASdatasets` too: the four dataset-driven targets are skipped and
the rest still execute, so the closed-form and extremiser checks are verifiable
by anyone. Output goes to `results/`, each artefact stamped with the library
version, git commit and full run configuration.

| Result | Paper | Script | Needs data |
|---|---|---|---|
| Certified fraction, band, speedups — Treasury | Table 2 | `exp1_certified_fraction_table2.py` | ✔ |
| Certified fraction, band, speedups — catastrophe | Table 2 | `exp9_catastrophe_table2.py` | ✔ |
| Certification inside a selection loop | Table 3 | `exp10_selection_loop.py` | ✔ |
| Bootstrap sensitivity of the dependence fit | Table S.6 | `exp4_sensitivity.py` | ✔ |
| Effective tail support / localisation boundary | Prop. 5.5 | `exp3_localisation_boundary.py` | |
| Threshold sweep | Table S.3 | `exp7_threshold_sweep.py` | |
| γ-diagnostic as a pre-commitment rule | Table S.5 | `exp8_gamma_diagnostic.py` | |
| Wall-clock timings | Table S.4 | `exp12_wallclock.py` | |
| Limitations and rejected refinements | §8 | `exp11_rejected_refinements.py` | |
| Subspace choice and population features | §6.1, Rem. 6.1/6.5 | `exp13_subspace_choice.py` | |
| W₁ closed form, invariance, lift lemma | §S.5 | `exp14_w1_verification.py` | |
| Attainment of the W_k constant | §S.5 | `exp6_wk_attainment.py` | |
| Certified-band figure | Figure 1 | `exp15_figure_band.py` | |
| Sharpness constants R\*_σ | Table S.2 | `check_sharpness_extremisers.py` | |
| W_k inflation constants | §S.2.15 | `check_wk_closed_forms.py` | |

Every draw is seeded through `numpy.random.default_rng`; seeds are script
arguments defaulting to the paper's values.

## What the code asserts

Correctness claims are checked on every run rather than reported once. The
experiments **fail loudly** if:

- the nominal bracket is violated on any candidate;
- the robust bracket is violated at either radius `ρ ∈ {0.01, 0.1}`;
- the certified band fails to contain the true tail set;
- the upper envelope disagrees with direct surrogate evaluation;
- the installed `certproj` version is not the pinned one.

A failing run writes no artefact, so a truncated file can never be mistaken for
a result.

## Layout

```
certproj/
  spectral.py      risk measures, spectra, ι(σ), L^p norms, tail support
  certificate.py   Projection, bracket, robust bracket, tight envelope,
                   certified band, screening rule
  cost.py          operation-count model behind the speedup columns
  data.py          FedYieldCurve, auscathist, synthetic calibration catalogue
  populations.py   bond-book and treaty candidate generators
  provenance.py    version pinning and run manifests
experiments/       one script per reported result
results/           artefacts, with provenance
```

## Known deviation from the published numbers

The synthetic calibration population of §S.4 is a **reconstruction**. It matches
every property the paper states — `d = 8`, PC variance shares
0.861/0.114/0.015/0.006, `λ₁/λ₂ = 7.4`, Student-t₆ innovations coloured by the
Cholesky factor — and additionally fixes the eigenvectors, which the paper's
description left open but which materially affect `γ_j(U)`. Results derived from
it (Table S.3, Remarks 6.1/6.4/6.5, §8) reproduce the paper's *qualitative*
conclusions but not all of its exact figures. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Validity under misspecification

The bracket is stated for whatever empirical law is supplied, so a misspecified,
resampled or importance-weighted catalogue still yields a valid certificate:
**misspecification degrades the certified fraction, never correctness.**
`exp4_sensitivity.py` exercises this directly — 60 refitted catalogues,
4 ranks, 200 candidates, zero bracket violations.

## Licence and citation

MIT (see [`LICENSE`](LICENSE)). Citation metadata in
[`CITATION.cff`](CITATION.cff); please cite the paper rather than the code alone.
