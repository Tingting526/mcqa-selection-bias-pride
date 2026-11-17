#!/usr/bin/env python3
"""Verify submitted evidence and recompute the 36 fixed-prior bootstrap intervals.

Optionally replay the original raw-score archive; no model inference is run.
All outputs go to a new directory. Submitted inputs are read-only.
"""
import argparse
import json
from pathlib import Path
import sys

from _bootstrap import PROJECT_ROOT
from src.offline_phase3b import write_json_new, sha256
from src.release_validation import (
    PRIMARY, STATISTICS, read_json, check_input_hashes, recompute_statistics, replay_raw,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--archive-root", type=Path,
                        help="Root of the original project archive containing analysis/")
    args = parser.parse_args()
    root = PROJECT_ROOT.parent
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must be a new directory; existing results are never overwritten")
    try:
        files = check_input_hashes(root)
        primary = read_json(root / PRIMARY)
        statistics = read_json(root / STATISTICS)
        recomputed = recompute_statistics(primary, statistics)
        raw = replay_raw(primary, args.archive_root.resolve(),
                         progress=lambda message: print(message, flush=True)) if args.archive_root else None
        # Create the output only after every scientific check has passed.
        output.mkdir(parents=True, exist_ok=False)
        write_json_new(output / "statistics_results.json", recomputed)
        report = dict(status="passed", submitted_files_verified=files,
                      primary_conditions=30, transfer_directions=12, bootstrap_intervals=36,
                      raw_scores_replayed=raw is not None, raw_replay=raw,
                      new_inference=False, python=sys.version,
                      source_sha256={PRIMARY: sha256(root / PRIMARY), STATISTICS: sha256(root / STATISTICS)},
                      scope="Saved predictions and exact-count summaries; raw observation provenance only checked when --archive-root is supplied.")
        write_json_new(output / "verification.json", report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("raw_replay", "source_sha256", "python")}))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Reproduction failed: {error}\n")


if __name__ == "__main__":
    main()
