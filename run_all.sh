#!/usr/bin/env bash
#
# Reproduce every reported result.
#
#   ./run_all.sh                        run all stages
#   ./run_all.sh table2_treasury        run one stage (repeatable)
#   ./run_all.sh --list                 list stage names and exit
#   CASDATASETS=/path ./run_all.sh      point at the CASdatasets checkout
#
# Stages are named for what they produce in the paper, not for the file that
# produces them, so a failure names the table the reader would be missing.
# Every stage is independent: seeds live in the scripts, no stage consumes
# another's output, and any stage may be rerun alone.
#
# Exit status is 0 only if every requested stage succeeded.

set -uo pipefail

# ---------------------------------------------------------------- layout ----
# ROOT is the directory holding certproj/ (the experiment scripts import it as
# "..").  EXPDIR is the directory holding the exp*.py files.  Both are located
# rather than assumed, so this driver works whether it sits in ROOT or beside
# the scripts.  Override either by exporting ROOT= or EXPDIR=.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${ROOT:-}" ]]; then
  d="$HERE"
  while [[ "$d" != "/" ]]; do
    [[ -d "$d/certproj" ]] && { ROOT="$d"; break; }
    d="$(dirname "$d")"
  done
fi
[[ -n "${ROOT:-}" ]] || { echo "preflight: certproj/ not found in any parent of $HERE; set ROOT=" >&2; exit 2; }

PROBE="exp1_certified_fraction_table2.py"
if [[ -z "${EXPDIR:-}" ]]; then
  if [[ -f "$HERE/$PROBE" ]]; then
    EXPDIR="$HERE"
  else
    EXPDIR="$(dirname "$(find "$ROOT" -maxdepth 2 -name "$PROBE" -print -quit 2>/dev/null)")"
  fi
fi
[[ -n "${EXPDIR:-}" && -f "$EXPDIR/$PROBE" ]] || { echo "preflight: $PROBE not found under $ROOT; set EXPDIR=" >&2; exit 2; }

cd "$EXPDIR"
CASDATASETS="${CASDATASETS:-$ROOT/CASdatasets}"
RESULTS="$ROOT/results"
PY="${PYTHON:-python3}"
LOG="$RESULTS/run_all.log"

# ---------------------------------------------------------------- stages ----
# name | script | extra arguments
STAGES=(
  "table2_treasury|exp1_certified_fraction_table2.py|--casdatasets $CASDATASETS"
  "table2_catastrophe|exp9_catastrophe_table2.py|--casdatasets $CASDATASETS"
  "table3_selection_loop|exp10_selection_loop.py|--casdatasets $CASDATASETS"
  "tableS6_sensitivity|exp4_sensitivity.py|--casdatasets $CASDATASETS"
  "prop_tail_support|exp3_localisation_boundary.py|"
  "tableS3_threshold_sweep|exp7_threshold_sweep.py|"
  "tableS5_gamma_diagnostic|exp8_gamma_diagnostic.py|"
  "tableS4_wallclock|exp12_wallclock.py|"
  "rejected_refinements|exp11_rejected_refinements.py|"
  "second_moment|exp16_second_moment.py|--casdatasets $CASDATASETS"
  "subspace_and_population|exp13_subspace_choice.py|"
  "w1_verification|exp14_w1_verification.py|"
  "wk_attainment|exp6_wk_attainment.py|"
  "sharpness_constants|check_sharpness_extremisers.py|"
  "wk_closed_forms|check_wk_closed_forms.py|"
  "figure_band|exp15_figure_band.py|"
)

# What each stage feeds, printed as a banner so the log maps onto the paper.
declare -A CAPTION=(
  [table2_treasury]="main text, Table 2 (Treasury block)"
  [table2_catastrophe]="main text, Table 2 (catastrophe block)"
  [table3_selection_loop]="main text, Table 3"
  [tableS6_sensitivity]="supplement, Table S.6"
  [prop_tail_support]="main text, tail supports of Proposition 5.5"
  [tableS3_threshold_sweep]="supplement, Table S.3"
  [tableS5_gamma_diagnostic]="supplement, Table S.5"
  [tableS4_wallclock]="supplement, Table S.4 (wall-clock)"
  [rejected_refinements]="main text, section 8, refinements (1) and (3)"
  [second_moment]="main text, section 8, refinement (2)  [RECONSTRUCTED]"
  [subspace_and_population]="supplement, Table S.9; main text, Remarks 6.2 and 6.4"
  [w1_verification]="supplement S.5, W_1 closed form, invariance, lift lemma"
  [wk_attainment]="supplement S.5, attainment of the W_k constant"
  [sharpness_constants]="supplement, Table S.2 (sharpness constants)"
  [wk_closed_forms]="supplement S.2.15, W_k closed forms"
  [figure_band]="main text, Figure 1 (certified band)"
)

stage_names() { for s in "${STAGES[@]}"; do echo "${s%%|*}"; done; }

if [[ "${1:-}" == "--list" ]]; then
  for s in "${STAGES[@]}"; do
    n="${s%%|*}"; printf '  %-26s %s\n' "$n" "${CAPTION[$n]:-}"
  done
  exit 0
fi

# ------------------------------------------------------------- preflight ----
fail() { echo "preflight: $*" >&2; exit 2; }

command -v "$PY" >/dev/null || fail "$PY not found (set PYTHON=...)"
[[ -d "$ROOT/certproj" ]] || fail "certproj package not found at $ROOT/certproj"
mkdir -p "$RESULTS" || fail "cannot create $RESULTS"

# Version and commit are asserted by the library, not guessed here; a mismatch
# between the installed package and the pinned version must stop the run rather
# than silently produce numbers that do not correspond to any released code.
# certproj.assert_version() is the real gate: it fails when the installed
# library is not the pinned REQUIRED_VERSION.  The commit is a label from
# certproj.git_commit(), which returns "unknown" outside a checkout.
VERSION_LINE="$(ROOT="$ROOT" "$PY" -c '
import os, sys
sys.path.insert(0, os.environ["ROOT"])
import certproj
certproj.assert_version()
print(certproj.__version__)
print(certproj.git_commit())
' 2>&1)" || fail "certproj import or version assertion failed:
$VERSION_LINE"

CERT_VERSION="$(sed -n 1p <<<"$VERSION_LINE")"
CERT_COMMIT="$(sed -n 2p <<<"$VERSION_LINE")"
[[ "$CERT_COMMIT" == "unknown" ]] && CERT_COMMIT=""
CERT_COMMIT="${CERT_COMMIT:0:7}"
# A run from uncommitted code is not a reproducible one; say so in the header.
if [[ -n "$CERT_COMMIT" ]] && git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1 \
   && ! git -C "$ROOT" diff --quiet HEAD 2>/dev/null; then
  CERT_COMMIT="$CERT_COMMIT-dirty"
fi
VERSION_LINE="certproj $CERT_VERSION${CERT_COMMIT:+ @ $CERT_COMMIT}"

# CASdatasets is needed only by the four real-data stages; check it if any of
# them are in the requested set.
NEEDS_CAS=0
REQUESTED=("$@")
[[ ${#REQUESTED[@]} -eq 0 ]] && mapfile -t REQUESTED < <(stage_names)
for name in "${REQUESTED[@]}"; do
  for s in "${STAGES[@]}"; do
    [[ "${s%%|*}" == "$name" ]] || continue
    [[ "${s##*|}" == *casdatasets* ]] && NEEDS_CAS=1
  done
done
if [[ $NEEDS_CAS -eq 1 ]]; then
  [[ -d "$CASDATASETS" ]] || fail "CASdatasets not found at $CASDATASETS
set CASDATASETS=/path/to/CASdatasets, or run only the synthetic stages"
  # data.py reads "$CASDATASETS/data/<name>.rda"; check exactly that path.
  for f in FedYieldCurve auscathist; do
    [[ -f "$CASDATASETS/data/$f.rda" ]] \
      || fail "$CASDATASETS/data/$f.rda not found
CASDATASETS must point at the CASdatasets checkout itself (the directory
containing data/), not at its data/ subdirectory"
  done
fi

# Reject unknown stage names before doing any work.
for name in "${REQUESTED[@]}"; do
  stage_names | grep -qx "$name" \
    || fail "unknown stage '$name' (see ./run_all.sh --list)"
done

# ------------------------------------------------------------------ run ----
{
  echo "$VERSION_LINE"
  echo "started  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python   $("$PY" --version 2>&1)"
  [[ $NEEDS_CAS -eq 1 ]] && echo "CASdatasets: $CASDATASETS"
  echo "root     $ROOT"
  echo "scripts  $EXPDIR"
  echo "results  $RESULTS"
  echo
} | tee "$LOG"

PASSED=(); FAILED=()
START=$SECONDS
for name in "${REQUESTED[@]}"; do
  for s in "${STAGES[@]}"; do
    [[ "${s%%|*}" == "$name" ]] || continue
    rest="${s#*|}"; script="${rest%%|*}"; args="${rest#*|}"
    [[ -f "$script" ]] || { echo "=== $name: MISSING $script" | tee -a "$LOG"
                            FAILED+=("$name"); continue; }
    printf '=== %-26s %s\n' "$name" "${CAPTION[$name]:-}" | tee -a "$LOG"
    t0=$SECONDS
    # shellcheck disable=SC2086
    if "$PY" "$script" $args 2>&1 | tee -a "$LOG"; then
      PASSED+=("$name")
    else
      FAILED+=("$name")                       # pipefail propagates the status
      echo "--- $name FAILED" | tee -a "$LOG"
    fi
    echo "    ($((SECONDS - t0))s)" | tee -a "$LOG"
  done
done

{
  echo
  echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ) in $((SECONDS - START))s"
  echo "passed   ${#PASSED[@]}/${#REQUESTED[@]}"
  if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf 'FAILED   %s\n' "${FAILED[*]}"
  fi
} | tee -a "$LOG"

[[ ${#FAILED[@]} -eq 0 ]]
