# Reproducibility

Every numbered result in the paper and its supplement is produced by a script in
`experiments/`. This file records the mapping, what needs the public datasets,
and — importantly — the three places where a regenerated number differs from the
published one and why.

Run everything with `./run_all.sh` (see the README). Each artefact in `results/`
carries the library version, git commit and full run configuration.

## Result → script

| Result | Paper | Script | Needs CASdatasets |
|---|---|---|---|
| Certified fraction, band, speedups — Treasury | Table 2 | `exp1_certified_fraction_table2.py` | ✔ |
| Certified fraction, band, speedups — catastrophe | Table 2 | `exp9_catastrophe_table2.py` | ✔ |
| Certification inside a selection loop | Table 3 | `exp10_selection_loop.py` | ✔ |
| Bootstrap sensitivity of the dependence fit | Table S.6 | `exp4_sensitivity.py` | ✔ |
| Effective tail support, localisation boundary | Prop. 5.5 | `exp3_localisation_boundary.py` | |
| Threshold sweep | Table S.3 | `exp7_threshold_sweep.py` | |
| γ-diagnostic as a pre-commitment rule | Table S.5 | `exp8_gamma_diagnostic.py` | |
| Wall-clock timings | Table S.4 | `exp12_wallclock.py` | |
| Limitations and rejected refinements; pilot certificate | §8, Prop. S.10 | `exp11_rejected_refinements.py` | |
| Subspace choice; sparsity and heterogeneity | §6.1, Rem. 6.1/6.5 | `exp13_subspace_choice.py` | |
| W₁ closed form, invariance, lift lemma | §S.5 | `exp14_w1_verification.py` | |
| Attainment of the W_k constant | §S.5 | `exp6_wk_attainment.py` | |
| Certified-band figure | Figure 1 | `exp15_figure_band.py` | |
| Sharpness constants R\*_σ | Table S.2 | `check_sharpness_extremisers.py` | |
| W_k inflation constants | §S.2.15 | `check_wk_closed_forms.py` | |

Nothing reported in either document is without a script.

## Where regenerated numbers differ from the paper

Two computations behind the published figures could not be located in this
repository and were rewritten. Both rewrites agree with the paper's proven
closed forms, but not with every published figure, and the differences are
recorded here rather than smoothed over.

### 1. The synthetic calibration population (§S.4)

`certproj.data.synthetic_catalogue` is a **reconstruction**. It matches every
property §S.4 states — `d = 8`, principal-component variance shares
0.861/0.114/0.015/0.006 (cumulative 0.861/0.975/0.990/0.996),
`λ₁/λ₂ = 7.4`, Student-t₆ innovations standardised and coloured by the Cholesky
factor, `N = 40,000`, centred.

It also **fixes the eigenvectors**, which §S.4 as published left open. This is
not a free choice: `γ_j(U)` measures alignment between the retained subspace and
the candidates' key-rate duration slopes, and a Haar-random basis gives a
duration vector roughly a quarter of the projection onto the leading component
that a level factor does. We use the level/slope/curvature shapes of a yield
curve (orthonormalised polynomials in log maturity), which is what makes the
synthetic and Treasury studies comparable.

Consequences, all on results derived from this population:

| Quantity | Published | Regenerated |
|---|---|---|
| Certified fraction at `r=2`, median threshold (Table S.3) | 0.47 | 0.697 |
| ... at the 0.90-quantile | 0.88 | 0.794 |
| Coherence-free refinement, recovered factor (§8) | 1.10–1.16 | 2.25–2.48 |
| Tight envelope re-certification rate (§8) | 57.5% | 53.1% |
| max-to-median `γ_j(U)` ratio (Rem. 6.1) | [1.4, 2.1] | [1.09, 1.24] |
| U_M / U_F / U_G ranking (Rem. 6.1) | inverts with rank | U_F dominates at every rank |
| Sparsity effect on `f` (Rem. 6.5) | 0.027 → 0.283 | 0.630 → 0.728 |
| Heterogeneity effect on `f` (Rem. 6.5) | 0.007 → 0.360 | 0.698 → 0.723 |

The qualitative conclusions are unchanged: the median threshold is the adverse
case, the tight envelope re-certifies about half of the unresolved candidates,
subspace choice is second-order against rank, and sparse heterogeneous
populations certify better than dense homogeneous ones. The §8 diagnostic is
*sharper* in the regenerated version — it locates about a factor of two of the
slack in the coherence step, which the published text attributed elsewhere.

The manuscript has been updated to the regenerated figures throughout, so the
paper and this repository agree.

### 2. The sharpness constants (Table S.2)

The published values (dual-power 0.7485 / 0.9680, TVaR(0.3) 0.6310,
blind-plus-floor 0.9593) came from an alternating-LP maximisation on a coarser
grid; that script is not in this repository.
`check_sharpness_extremisers.py` maximises instead over the mean-zero families
the proofs identify as extremal — symmetric two-point, asymmetric two-point,
dilute three-atom — on a 4×10⁶-point equal-mass grid. It is therefore a **lower
bound on R\*_σ by construction**, which is what makes agreement informative.

It reproduces every case with a known exact constant to four decimals, including
the full TVaR sweep below the phase boundary and the two-step blind-plus-floor
family. The manuscript now reports these values.

## Verification built into the runs

The experiments fail loudly, and write no artefact, if the nominal bracket is
violated, the robust bracket is violated at either radius, the certified band
fails to contain the true tail set, the upper envelope disagrees with direct
evaluation, or the installed `certproj` version is not the pinned one.
