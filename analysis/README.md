# Analysis package

This package contains the experimental implementation, offline analyses and tests
for the thesis. Installation, reproduction commands and archive requirements are
documented in the [repository README](../README.md).

Run tests from this directory with `python -m pytest -ra`. External reference
and raw-archive tests skip when their inputs are absent; `--require-reference`
and `--require-archive` make those missing inputs errors.

The portable submission check is `scripts/reproduce_results.py`. Earlier
`phase3b`, `phase4`, `phase6` and manuscript delivery scripts retain the paths
and workflow of the original project archive. Their historical audit reports
are not prerequisites for the portable check.
