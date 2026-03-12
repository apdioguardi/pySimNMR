import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_freq_fit(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, "-m", "pysimnmr.freq_fit", *args]
    return subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)


@pytest.mark.freq_fit
@pytest.mark.cfg("freq_fit.multisite")
def test_freq_fit_cli(tmp_path: Path) -> None:
    out_base = tmp_path / "freq_fit"
    result = _run_freq_fit(
        [
            "--config",
            "tests/configs/freq_fit_multisite.py",
            "--out",
            str(out_base),
            "--no-show",
            "--no-progress",
            "--max-nfev",
            "20",
        ],
        tmp_path,
    )
    assert result.returncode == 0, f"nmr-freq-fit failed: {result.stderr}\n{result.stdout}"
    assert out_base.with_suffix(".txt").exists()
    assert out_base.with_suffix(".png").exists()
