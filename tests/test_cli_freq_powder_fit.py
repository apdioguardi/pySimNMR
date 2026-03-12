import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_freq_powder_fit(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PY_SIMNMR_THREADS"] = "1"
    cmd = [sys.executable, "-m", "pysimnmr.powder_freq_fit", *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)


@pytest.mark.freq_powder_fit
@pytest.mark.cfg("freq_powder_fit.modern")
def test_freq_powder_fit_cli(tmp_path: Path) -> None:
    out_base = tmp_path / "freq_powder_fit"
    result = _run_freq_powder_fit(
        [
            "--config",
            "tests/configs/freq_powder_fit_exact.py",
            "--out",
            str(out_base),
            "--no-show",
            "--no-progress",
            "--max-nfev",
            "20",
        ],
        tmp_path,
    )
    assert result.returncode == 0, f"nmr-freq-powder-fit failed: {result.stderr}\n{result.stdout}"
    assert out_base.with_suffix(".txt").exists()
    assert out_base.with_suffix(".png").exists()


@pytest.mark.freq_powder_fit
@pytest.mark.cfg("freq_powder_fit.perturbative")
def test_freq_powder_fit_cli_perturbative(tmp_path: Path) -> None:
    out_base = tmp_path / "freq_powder_fit_2nd_order"
    result = _run_freq_powder_fit(
        [
            "--config",
            "tests/configs/freq_powder_fit_perturbative.py",
            "--out",
            str(out_base),
            "--no-show",
            "--no-progress",
            "--max-nfev",
            "20",
        ],
        tmp_path,
    )
    assert result.returncode == 0, f"nmr-freq-powder-fit (2nd order) failed: {result.stderr}\n{result.stdout}"
    assert out_base.with_suffix(".txt").exists()
    assert out_base.with_suffix(".png").exists()
