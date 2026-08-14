#!/usr/bin/env bash
#
# Reproduce the results in this repository.
#
#   ./run_all.sh /path/to/CASdatasets    everything
#   ./run_all.sh                         only the runs that need no data
#
# The two public catalogues are distributed with the CASdatasets R package
# (https://github.com/dutangc/CASdatasets); clone it once and pass the path.
# Without it the dataset-driven runs are skipped and the rest still execute,
# so the closed-form and extremiser checks can be verified by anyone.
#
# Every run is logged to results/ and stamped with the library version and
# commit. NOT every number in the paper is reproduced here: see
# MISSING_SCRIPTS.md for the reported results that still have no script.

set -euo pipefail

PY="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The referee archive ships CASdatasets/ inside this directory, so the path
# argument is optional; pass one to override.
CAS="${1:-}"
if [[ -z "$CAS" && -d "$ROOT/CASdatasets" ]]; then CAS="$ROOT/CASdatasets"; fi
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

VERSION="$(cd "$ROOT" && "$PY" -c 'import certproj; print(certproj.__version__)')"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "certproj $VERSION @ $COMMIT"
echo "started  $STARTED"
if [[ -z "$CAS" ]]; then
  echo "CASdatasets: not found (no ./CASdatasets, no path given)"
  echo "             dataset-driven runs will be SKIPPED"
elif [[ ! -d "$CAS" ]]; then
  echo "error: '$CAS' is not a directory" >&2; exit 2
else
  echo "CASdatasets: $CAS"
  for f in FedYieldCurve.rda auscathist.rda; do
    [[ -f "$CAS/data/$f" ]] || {
      echo "error: $CAS/data/$f not found; expected a CASdatasets clone" >&2
      exit 2; }
  done
  "$PY" - <<'PREFLIGHT' || { echo "error: pyreadr is required to read the .rda files (pip install -r requirements.txt)" >&2; exit 2; }
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("pyreadr") else 1)
PREFLIGHT
fi
echo

# Preflight: every script this driver invokes must exist, and no superseded
# copy may linger beside it.  Duplicate scripts are the dangerous case: a
# referee runs the stale one and gets numbers that differ from the paper.
REQUIRED=(exp1_certified_fraction_table2.py exp3_localisation_boundary.py
          exp4_sensitivity.py exp6_wk_attainment.py exp7_threshold_sweep.py
          exp8_gamma_diagnostic.py exp9_catastrophe_table2.py
          exp10_selection_loop.py exp11_rejected_refinements.py
          exp12_wallclock.py exp13_subspace_choice.py exp14_w1_verification.py
          exp15_figure_band.py check_sharpness_extremisers.py check_wk_closed_forms.py)
SUPERSEDED=(exp1_rank_ladder.py exp2_sharpness.py exp5_wk_constants.py)
fail=0
for f in "${REQUIRED[@]}"; do
  [[ -f "$ROOT/experiments/$f" ]] || { echo "missing: experiments/$f" >&2; fail=1; }
done
for f in "${SUPERSEDED[@]}"; do
  [[ -f "$ROOT/experiments/$f" ]] && {
    echo "superseded copy still present: experiments/$f -- delete it" >&2; fail=1; }
done
(( fail )) && { echo "preflight failed; see README" >&2; exit 2; }

ran=(); skipped=()

# run <output-stem> <paper object> <script> [args...]
# Results are written to a temporary file and moved into place only on success,
# so a failed run never leaves a truncated artefact that reads like a result.
run () {
  local stem="$1" what="$2"; shift 2
  local tmp="$RESULTS/.$stem.partial"
  printf '=== %-26s %s\n' "$stem" "$what"
  if ( cd "$ROOT/experiments" && "$PY" "$@" ) | tee "$tmp"; then
    mv "$tmp" "$RESULTS/$stem.txt"
    ran+=("$stem")
  else
    local rc=$?
    rm -f "$tmp"
    echo "FAILED: $stem (exit $rc); no artefact written" >&2
    exit "$rc"
  fi
  echo
}

skip () {
  printf '=== %-26s SKIPPED (needs CASdatasets): %s\n\n' "$1" "$2"
  skipped+=("$1")
}

# ---------------------------------------------------------------- reported
# Each of these is the source of a numbered object in the paper.

if [[ -n "$CAS" ]]; then
  run table2_treasury "main text, Table 2 (Treasury block)" \
      exp1_certified_fraction_table2.py --casdatasets "$CAS"
  run table2_catastrophe "main text, Table 2 (Catastrophe block)" \
      exp9_catastrophe_table2.py --casdatasets "$CAS"
  run table3_selection_loop "main text, Table 3" \
      exp10_selection_loop.py --casdatasets "$CAS"
  run tableS6_sensitivity "supplement, Table S.6" \
      exp4_sensitivity.py --casdatasets "$CAS"
else
  skip table2_treasury      "main text, Table 2 (Treasury block)"
  skip table2_catastrophe   "main text, Table 2 (Catastrophe block)"
  skip table3_selection_loop "main text, Table 3"
  skip tableS6_sensitivity  "supplement, Table S.6"
fi

run prop55_tail_support "main text, Proposition 5.5 tail supports" \
    exp3_localisation_boundary.py

# Synthetic calibration population (no external data needed).
run tableS3_threshold_sweep "supplement, Table S.3" \
    exp7_threshold_sweep.py

run tableS5_gamma_diagnostic "supplement, Table S.5" \
    exp8_gamma_diagnostic.py

run tableS4_wallclock "supplement, Table S.4 (wall-clock)" \
    exp12_wallclock.py

run rejected_refinements "main text, limitations and rejected refinements" \
    exp11_rejected_refinements.py

run subspace_and_population "main text, subspace choice and population features" \
    exp13_subspace_choice.py

run w1_verification "supplement S.5, W_1 closed form, invariance, lift lemma" \
    exp14_w1_verification.py

run figure_band "main text, certified-band figure" \
    exp15_figure_band.py

run wk_attainment "supplement S.5, attainment of the W_k constant" \
    exp6_wk_attainment.py

# ------------------------------------------------------------- unit checks
# These verify closed forms and extremiser families. check_sharpness_extremisers
# regenerates the supplementary sharpness table; see the provenance note in
# MISSING_SCRIPTS.md before citing it as the source of those values.

run check_sharpness "supplement, sharpness constants (see provenance note)" \
    check_sharpness_extremisers.py

run check_wk_closed_forms "supplement S.2.15, W_k closed forms" \
    check_wk_closed_forms.py

# ----------------------------------------------------------------- manifest
{
  echo "certproj_version: $VERSION"
  echo "git_commit: $COMMIT"
  echo "python: $("$PY" -c 'import sys; print(sys.version.split()[0])')"
  echo "casdatasets: ${CAS:-<not supplied>}"
  echo "started: $STARTED"
  echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "ran: ${ran[*]}"
  echo "skipped: ${skipped[*]:-<none>}"
} > "$RESULTS/RUN_MANIFEST.txt"

echo "output in results/ (manifest: results/RUN_MANIFEST.txt)"
if (( ${#skipped[@]} )); then
  echo "SKIPPED without CASdatasets: ${skipped[*]}"
fi
echo "reported results with no script yet: see MISSING_SCRIPTS.md"
