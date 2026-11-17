"""Optional external fixtures; a present but invalid archive must still fail."""
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--require-reference", action="store_true",
                     help="Fail instead of skipping tests when reference data are absent")
    parser.addoption("--require-archive", action="store_true",
                     help="Fail instead of skipping tests when raw LRZ data are absent")


def require_directory(request, path, option, instruction):
    if not path.exists():
        if request.config.getoption(option):
            pytest.fail(instruction)
        pytest.skip(instruction)
    return path


@pytest.fixture(scope="session")
def reference_data(request):
    return require_directory(
        request, ANALYSIS / "data/reference/zheng2024", "--require-reference",
        "Reference data absent: run python analysis/scripts/download_zheng2024.py from the repository root.",
    )


@pytest.fixture(scope="session")
def archive_data(request, reference_data):
    return require_directory(
        request, ANALYSIS / "outputs/lrz-results", "--require-archive",
        "Raw LRZ archive absent: restore the original outputs/lrz-results and outputs/preprocessed directories.",
    )
