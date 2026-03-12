import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_field_fit(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, "-m", "pysimnmr.field_fit", *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)


@pytest.mark.field_fit
@pytest.mark.cfg("field_fit.multisite")
def test_field_fit_cli(tmp_path: Path) -> None:
    out_base = tmp_path / "field_fit"
    result = _run_field_fit(
        [
            "--config",
            "tests/configs/field_fit_multisite.py",
            "--out",
            str(out_base),
            "--no-show",
            "--no-progress",
            "--max-nfev",
            "20",
        ],
        tmp_path,
    )
    assert result.returncode == 0, f"nmr-field-fit failed: {result.stderr}\n{result.stdout}"
    assert out_base.with_suffix(".txt").exists()
    assert out_base.with_suffix(".png").exists()
