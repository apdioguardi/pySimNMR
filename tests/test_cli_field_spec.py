import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_field_cli(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, "-m", "pysimnmr.field_cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)


@pytest.mark.field_spec
@pytest.mark.cfg("field_spec.multisite")
def test_field_cli(tmp_path: Path) -> None:
    out_base = tmp_path / "field_spec"
    result = _run_field_cli(
        [
            "--config",
            "tests/configs/field_spec_multisite_basic.py",
            "--out",
            str(out_base),
            "--no-show",
        ],
        tmp_path,
    )
    assert result.returncode == 0, f"nmr-field-spec failed: {result.stderr}\n{result.stdout}"
    assert out_base.with_suffix(".txt").exists()
    assert out_base.with_suffix(".png").exists()
