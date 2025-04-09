import sys
from importlib.metadata import version
from importlib.util import find_spec


def pytest_report_header(config, start_path):
    is_gil_enabled = sys.version_info < (3, 13) or sys._is_gil_enabled()

    return [
        f"{is_gil_enabled = }",
        f"NumPy: {version('numpy')}",
        f"rlic._core loads from {find_spec('rlic._core').origin}",
    ]
