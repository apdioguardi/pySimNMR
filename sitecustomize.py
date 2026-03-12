"""
Test-time helpers for sandbox-friendly defaults.

This module is imported automatically when PYTHONPATH includes the repo root.
The only customization we keep is nudging simulations to prefer thread-based
parallelism when running under pytest, which avoids issues with process-based
backends in restricted environments.
"""

from __future__ import annotations
import os


def _under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def _hint_thread_parallel() -> None:
    if _under_pytest() and not os.environ.get("PY_SIMNMR_THREADS"):
        os.environ["PY_SIMNMR_THREADS"] = "1"


try:
    _hint_thread_parallel()
except Exception:
    pass
