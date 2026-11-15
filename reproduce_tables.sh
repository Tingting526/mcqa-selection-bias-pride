#!/usr/bin/env bash
# Regenerate the 28 thesis tables from the bundled result files and compare
# them byte for byte with Paper/generated.
#
# The four generators write directly into <root>/Paper/generated. To make sure a
# run can never overwrite the approved tables, everything runs in a temporary
# copy of this repository. The committed files are only read.
#
# Usage: ./reproduce_tables.sh [--keep]     (--keep retains the temporary copy)
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PDF_REL="analysis/outputs/rq1_restructure_20261021/sources/zheng_iclr2024.pdf"
PDF_URL="https://proceedings.iclr.cc/paper_files/paper/2024/file/54dd9e0cff6d9214e20d97eb2a3bae49-Paper-Conference.pdf"
PDF_SHA="7ee57c2648851c9e4754371d3c6d89d1a078ce47044c00ee48c6ed64191e0f14"
PY="${PYTHON:-python3}"

WORK="$(mktemp -d)"
trap '[[ "${1:-}" == "--keep" ]] || rm -rf "$WORK"' EXIT
echo "Working copy: $WORK"
cp -R "$REPO/." "$WORK/"
rm -rf "$WORK/.git"

# The Zheng et al. (2024) paper PDF is not redistributed. Only its SHA-256 is
# recorded by restructure_rq1_presentation.py, so download the public copy.
if [[ ! -f "$WORK/$PDF_REL" ]]; then
  curl -L --fail --silent --show-error -o "$WORK/$PDF_REL" "$PDF_URL"
fi
got="$(shasum -a 256 "$WORK/$PDF_REL" 2>/dev/null || sha256sum "$WORK/$PDF_REL")"
[[ "${got%% *}" == "$PDF_SHA" ]] || echo "Note: downloaded PDF hash differs from the recorded one (only the provenance file is affected)."

# build_phase7_robustness_assets.py reads its output directory from this file.
mkdir -p "$WORK/tmp"
echo "$WORK/analysis/outputs/phase7_20261018_20261018T122106Z" > "$WORK/tmp/phase7_active_output.txt"

export MPLCONFIGDIR="$WORK/tmp/mpl"
cd "$WORK/analysis"
rm -rf "$WORK/Paper/generated" && mkdir -p "$WORK/Paper/generated"
rm -rf outputs/phase4_20261018_20261018T091030Z/tables/latex   # generator refuses to overwrite

# Steps 1-4: original generators, in this order.
"$PY" scripts/build_phase4_thesis_tables.py --output "$WORK/analysis/outputs/phase4_20261018_20261018T091030Z" >/dev/null
"$PY" scripts/restructure_rq1_presentation.py >/dev/null
"$PY" scripts/build_strengthening_tables.py >/dev/null
"$PY" scripts/build_phase7_robustness_assets.py >/dev/null

# Step 5: structural revision of 3 Nov 2026, approved caption/note revisions
# and the two hand-maintained tables. Compare with the committed tables.
"$PY" scripts/finalize_thesis_tables.py \
  --input "$WORK/Paper/generated" \
  --output "$WORK/final_tables" \
  --check "$REPO/Paper/generated"
