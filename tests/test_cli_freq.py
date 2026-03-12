import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_freq_cli(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, "-m", "pysimnmr.freq_cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)


@pytest.mark.freq
@pytest.mark.cfg("freq.multisite_ed")
def test_freq_cli_ed(tmp_path: Path) -> None:
    out_base = tmp_path / "freq_ed"
    result = _run_freq_cli(
        [
            "--config",
            "tests/configs/freq_multisite_basic.py",
            "--out",
            str(out_base),
            "--no-show",
        ],
        tmp_path,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}\n{result.stdout}"
    assert out_base.with_suffix(".txt").exists()
    assert out_base.with_suffix(".png").exists()


def test_freq_cli_invalid(tmp_path: Path) -> None:
    cfg = {
        "isotope_list": [],
        "Ka_list": [],
        "Kb_list": [],
        "Kc_list": [],
        "vc_list": [],
        "eta_list": [],
        "H0": 5.0,
        "min_freq": 10.0,
        "max_freq": 20.0,
        "n_freq_points": 128,
        "convolution_function_list": [],
        "conv_FWHM_list": [],
        "conv_vQ_FWHM_list": [],
    }
    cfg_path = tmp_path / "invalid_freq.json"
    cfg_path.write_text(json.dumps(cfg))
    result = _run_freq_cli(
        [
            "--config",
            str(cfg_path),
            "--out",
            str(tmp_path / "invalid"),
            "--no-show",
        ],
        tmp_path,
    )
    assert result.returncode != 0
