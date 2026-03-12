import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_BASE = Path(os.environ.get("PY_SIMNMR_TEST_ARTIFACTS", "")).resolve() if os.environ.get("PY_SIMNMR_TEST_ARTIFACTS") else None

import pytest
import yaml


def _run_vary(args: list[str], tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, "-m", "pysimnmr.vary_cli", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert result.returncode == 0, f"vary CLI failed: {result.stderr}\n{result.stdout}"


@pytest.mark.vary
@pytest.mark.cfg("vary.freq_vs_field")
def test_vary_freq_vs_field(tmp_path: Path) -> None:
    outdir = tmp_path / "out_freq_vs_field"
    args = [
        "freq-vs-field",
        "--config",
        "tests/configs/vary_freq_vs_field.py",
        "--out",
        str(outdir),
        "--no-progress",
    ]
    _run_vary(args, tmp_path)
    png = outdir.with_suffix(".png")
    assert png.exists() and png.stat().st_size > 0


@pytest.mark.vary
@pytest.mark.cfg("vary.elevels_vs_field")
def test_vary_elevels_vs_field(tmp_path: Path) -> None:
    outdir = tmp_path / "out_elevels_vs_field"
    args = [
        "elevels-vs-field",
        "--config",
        "tests/configs/vary_elevels_vs_field.py",
        "--out",
        str(outdir),
    ]
    _run_vary(args, tmp_path)
    png = outdir.with_suffix(".png")
    assert png.exists() and png.stat().st_size > 0


@pytest.mark.vary
@pytest.mark.cfg("vary.freq_vs_eta")
def test_vary_freq_vs_eta(tmp_path: Path) -> None:
    outbase = tmp_path / "out_freq_vs_eta"
    args = [
        "freq-vs-eta",
        "--config",
        "tests/configs/vary_freq_vs_eta.py",
        "--out",
        str(outbase),
    ]
    _run_vary(args, tmp_path)
    png = outbase.with_suffix(".png")
    assert png.exists() and png.stat().st_size > 0


@pytest.mark.vary
@pytest.mark.cfg("vary.freq_vs_angle")
def test_vary_freq_vs_angle(tmp_path: Path) -> None:
    outbase = tmp_path / "out_freq_vs_angle"
    args = [
        "freq-vs-angle",
        "--config",
        "tests/configs/vary_freq_vs_angle.py",
        "--out",
        str(outbase),
    ]
    _run_vary(args, tmp_path)
    png = outbase.with_suffix(".png")
    assert png.exists() and png.stat().st_size > 0


def test_vary_invalid_config(tmp_path: Path) -> None:
    cfg = {
        "isotope": "75As",
        "B_min_T": 5.0,
        "B_max_T": 4.0,
    }
    cfg_path = tmp_path / "bad.yml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [
        sys.executable,
        "-m",
        "pysimnmr.vary_cli",
        "freq-vs-field",
        "--config",
        str(cfg_path),
        "--out",
        str(tmp_path / "out"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert result.returncode != 0
    assert "Invalid configuration" in result.stderr


def test_vary_freq_vs_field_aliases(tmp_path: Path) -> None:
    cfg = {
        "isotope": "75As",
        "min_field": 0.0,
        "max_field": 0.5,
        "n_fields": 8,
        "Ka": 0.0,
        "Kb": 0.0,
        "Kc": 0.0,
        "vc": 11.1,
        "eta": 0.1,
        "phi_z_deg": 0.0,
        "theta_x_prime_deg": 0.0,
        "psi_z_prime_deg": 0.0,
        "field_axis": "y",
    }
    cfg_path = tmp_path / "freq_alias.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    outdir = tmp_path / "freq_alias_out"
    args = ["freq-vs-field", "--config", str(cfg_path), "--out", str(outdir)]
    _run_vary(args, tmp_path)
    assert outdir.with_suffix(".txt").exists()


def test_vary_freq_vs_angle_aliases(tmp_path: Path) -> None:
    cfg = {
        "isotope": "75As",
        "Ka": 0.0,
        "Kb": 0.0,
        "Kc": 0.0,
        "vc": 5.0,
        "eta": 0.2,
        "H0": 1.0,
        "angle_to_vary": "theta_x_prime_deg",
        "angle_start": 0.0,
        "angle_stop": 5.0,
        "n_angles": 5,
        "phi_z_deg_init_list": [0.0],
        "theta_x_prime_deg_init_list": [0.0],
        "psi_z_prime_deg_init_list": [0.0],
    }
    cfg_path = tmp_path / "angle_alias.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    outdir = tmp_path / "angle_alias_out"
    args = ["freq-vs-angle", "--config", str(cfg_path), "--out", str(outdir)]
    _run_vary(args, tmp_path)
    assert outdir.with_suffix(".txt").exists()
