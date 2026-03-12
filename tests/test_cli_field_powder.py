import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_field_powder(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PY_SIMNMR_THREADS"] = "1"
    cmd = [sys.executable, "-m", "pysimnmr.powder_field_cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)


@pytest.mark.field_powder
@pytest.mark.cfg("field_powder.modern")
def test_field_powder_cli(tmp_path: Path) -> None:
    out_base = tmp_path / "field_powder"
    result = _run_field_powder(
        [
            "--config",
            "tests/configs/field_powder_perturbative.py",
            "--out",
            str(out_base),
            "--no-show",
            "--no-progress",
        ],
        tmp_path,
    )
    assert result.returncode == 0, f"nmr-field-powder failed: {result.stderr}\n{result.stdout}"
    assert out_base.with_suffix(".txt").exists()
    assert out_base.with_suffix(".png").exists()


@pytest.mark.field_powder
@pytest.mark.cfg("field_powder.ed")
def test_field_powder_cli_ed(tmp_path: Path) -> None:
    out_base = tmp_path / "field_powder_ed"
    result = _run_field_powder(
        [
            "--config",
            "tests/configs/field_powder_exact.py",
            "--out",
            str(out_base),
            "--no-show",
            "--no-progress",
        ],
        tmp_path,
    )
    assert result.returncode == 0, f"nmr-field-powder (ED) failed: {result.stderr}\n{result.stdout}"
    assert out_base.with_suffix(".txt").exists()
    assert out_base.with_suffix(".png").exists()
