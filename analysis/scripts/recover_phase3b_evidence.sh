#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PHASE3B_PYTHON:-python3}" "$script_dir/recover_phase3b_evidence.py" "$@"
