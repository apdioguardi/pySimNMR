"""CLI utilities for hyperfine-distribution powder spectra."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import importlib
import numpy as np
import yaml

from .core import SimNMR
from .hyperfine_distribution import (
    freq_spectrum_from_internal_field_distribution,
    validate_internal_field_samples,
)
from .config_aliases import normalize_common_aliases
from .plotly_utils import save_internal_field_distribution_html
from .config_loader import load_python_config

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

__all__ = ["load_hyperfine_config", "run_hyperfine_simulation", "main"]

def load_hyperfine_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".py":
        data = load_python_config(path)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("configuration must be a mapping at the top level")
    return data


def _plane_axes(plane: str) -> Tuple[int, int, int]:
    plane = plane.lower()
    if plane == "xy":
        return 0, 1, 2
    if plane == "xz":
        return 0, 2, 1
    if plane == "yz":
        return 1, 2, 0
    raise ValueError(f"unsupported plane '{plane}', expected one of xy/xz/yz")




def _weights_from_config(weights_cfg: Any, n_samples: int) -> np.ndarray | None:
    if weights_cfg is None:
        return None
    if isinstance(weights_cfg, dict):
        if str(weights_cfg.get('type', '')).lower() == 'equal' or bool(weights_cfg.get('equal', False)):
            return None
        if 'values' in weights_cfg:
            values = np.asarray(weights_cfg['values'], dtype=float)
            if values.shape != (n_samples,):
                raise ValueError('weights.values must have the same length as the vectors')
            return values
    values = np.asarray(weights_cfg, dtype=float)
    if values.shape != (n_samples,):
        raise ValueError('weights must have the same length as the vectors')
    return values


def _coerce_generator_output(result: Any) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(result, tuple) and len(result) == 2:
        vectors, weights = result
    elif isinstance(result, dict):
        vectors = result.get('vectors')
        weights = result.get('weights')
    else:
        vectors = result
        weights = None
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError('custom generator must return an array of shape (n,3)')
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
    return vectors, weights


def _run_custom_generator(custom_cfg: Any, weights_cfg: Any, allow_custom: bool) -> tuple[np.ndarray, np.ndarray]:
    if not allow_custom:
        raise ValueError('Custom generators are disabled; set allow_custom_generators: true to enable them.')
    if not isinstance(custom_cfg, dict):
        raise ValueError('custom_generator must be a mapping with module/function keys')
    module_path = custom_cfg.get('module')
    function_name = custom_cfg.get('function')
    if not module_path or not function_name:
        raise ValueError("custom_generator requires 'module' and 'function' keys")
    params = custom_cfg.get('params') or {}
    module = importlib.import_module(module_path)
    func = getattr(module, function_name)
    result = func(**params)
    vectors, weights_raw = _coerce_generator_output(result)
    if weights_raw is None:
        weights_raw = _weights_from_config(weights_cfg, vectors.shape[0])
    else:
        if weights_raw.shape != (vectors.shape[0],):
            raise ValueError('custom generator weights must have same length as vectors')
    return validate_internal_field_samples(vectors, weights_raw)


def _sample_internal_fields(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    dist = dict(cfg.get("hyperfine_distribution", {}))
    if not dist:
        raise ValueError("'hyperfine_distribution' section is required")

    allow_custom = bool(dist.get('allow_custom_generators') or cfg.get('allow_custom_generators'))
    custom_cfg = dist.get('custom_generator')
    if custom_cfg is not None:
        return _run_custom_generator(custom_cfg, dist.get('weights'), allow_custom)

    kind = dist.get("kind")
    if kind is not None:
        kind_lower = str(kind).lower()
        if kind_lower == "xy-plane-gaussian":
            dist = {
                'samples': dist.get('samples', 1000),
                'seed': dist.get('seed'),
                'plane': dist.get('plane', 'xy'),
                'angle_distribution': {'type': 'uniform'},
                'magnitude_distribution': {
                    'type': 'gaussian',
                    'mean_T': dist.get('mean_T', 0.0),
                    'sigma_T': dist.get('sigma_T', 0.01),
                },
                'weights': dist.get('weights'),
            }
        elif kind_lower == "explicit":
            vectors = np.asarray(dist.get('vectors'), dtype=float)
            weights_raw = _weights_from_config(dist.get('weights'), vectors.shape[0])
            return validate_internal_field_samples(vectors, weights_raw)
        else:
            raise ValueError(f"unsupported hyperfine distribution kind '{kind_lower}'")

    if "vectors" in dist and dist["vectors"] is not None:
        vectors = np.asarray(dist["vectors"], dtype=float)
        weights_raw = _weights_from_config(dist.get("weights"), vectors.shape[0])
        return validate_internal_field_samples(vectors, weights_raw)

    samples = int(dist.get("samples", 0))
    if samples <= 0:
        raise ValueError("hyperfine_distribution.samples must be > 0 when explicit vectors are not provided")

    rng = np.random.default_rng(dist.get("seed"))
    plane = dist.get("plane", "xy")
    ax_a, ax_b, _ = _plane_axes(plane)

    angle_cfg = dist.get("angle_distribution", {})
    angle_type = str(angle_cfg.get("type", "uniform")).lower()
    if angle_type == "uniform":
        angles = rng.uniform(0.0, 2.0 * np.pi, size=samples)
    else:
        raise ValueError(f"unsupported angle_distribution.type '{angle_type}'")

    mag_cfg = dist.get("magnitude_distribution", {"type": "delta", "value_T": 0.0})
    mag_type = str(mag_cfg.get("type", "delta")).lower()
    if mag_type == "gaussian":
        mean = float(mag_cfg.get("mean_T", 0.0))
        sigma = float(mag_cfg.get("sigma_T", 0.01))
        magnitudes = rng.normal(loc=mean, scale=sigma, size=samples)
        magnitudes = np.clip(magnitudes, 0.0, None)
    elif mag_type == "delta":
        value = float(mag_cfg.get("value_T", 0.0))
        magnitudes = np.full(samples, value, dtype=float)
    elif mag_type == "uniform":
        low = float(mag_cfg.get("low_T", 0.0))
        high = float(mag_cfg.get("high_T", low))
        if high <= low:
            raise ValueError("magnitude_distribution.uniform requires high_T > low_T")
        magnitudes = rng.uniform(low, high, size=samples)
    else:
        raise ValueError(f"unsupported magnitude_distribution.type '{mag_type}'")

    vectors = np.zeros((samples, 3), dtype=float)
    vectors[:, ax_a] = magnitudes * np.cos(angles)
    vectors[:, ax_b] = magnitudes * np.sin(angles)

    weights_raw = _weights_from_config(dist.get("weights"), samples)

    vectors, weights = validate_internal_field_samples(vectors, weights_raw)
    return vectors, weights



def run_hyperfine_simulation(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    enable_plotly: bool = True,
) -> Dict[str, Path | None]:
    """Run a hyperfine-distribution spectrum simulation using config dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = normalize_common_aliases(cfg)
    isotope = str(cfg.get("isotope", ""))
    if not isotope:
        raise ValueError("'isotope' must be provided in the configuration")
    sim = SimNMR(isotope)
    vectors, weights = _sample_internal_fields(cfg)

    freq_cfg = cfg.get("freq_axis", {})
    freq_min = float(freq_cfg.get("min_MHz", 0.0))
    freq_max = float(freq_cfg.get("max_MHz", 40.0))
    freq_points = int(freq_cfg.get("points", 4096))
    freq_axis = np.linspace(freq_min, freq_max, freq_points)

    Ka = float(cfg.get("Ka", 0.0))
    Kb = float(cfg.get("Kb", 0.0))
    Kc = float(cfg.get("Kc", 0.0))
    vQ_MHz = float(cfg.get("vQ_MHz", 0.0))
    eta = float(cfg.get("eta", 0.0))
    H0_T = float(cfg.get("H0_T", 0.0))
    mtx_cutoff = float(cfg.get("matrix_element_cutoff", 1e-6))
    FWHM_MHz = float(cfg.get("FWHM_MHz", 0.05))
    FWHM_vQ_MHz = float(cfg.get("FWHM_vQ_MHz", 0.0))
    line_shape = str(cfg.get("line_shape", "gauss"))

    spectrum = freq_spectrum_from_internal_field_distribution(
        sim,
        freq_axis,
        hint_pas_samples_T=vectors,
        weights=weights,
        Ka=Ka,
        Kb=Kb,
        Kc=Kc,
        vQ_MHz=vQ_MHz,
        eta=eta,
        H0_T=H0_T,
        mtx_elem_min=mtx_cutoff,
        FWHM_MHz=FWHM_MHz,
        FWHM_vQ_MHz=FWHM_vQ_MHz,
        line_shape_func=line_shape,
    )

    spectrum_path = out_dir / cfg.get("spectrum_basename", "hyperfine_spectrum.txt")
    np.savetxt(
        spectrum_path,
        np.column_stack((freq_axis, spectrum)),
        header="freq_MHz intensity",
    )

    html_path: Path | None = None
    plotly_cfg = cfg.get("plotly", {})
    if enable_plotly and plotly_cfg.get("html", True):
        html_path = out_dir / cfg.get(
            "plotly_basename",
            "hyperfine_internal_fields.html",
        )
        try:
            save_internal_field_distribution_html(
                vectors,
                weights=weights,
                title=plotly_cfg.get("title", "Hyperfine field distribution"),
                out_html=html_path,
            )
        except RuntimeError:
            html_path = None

    return {"spectrum_path": spectrum_path, "html_path": html_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nmr-hyperfine",
        description="Hyperfine-distribution powder spectrum generator",
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML config file")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/hyperfine_distribution"),
        help="Output directory for spectrum/plots",
    )
    parser.add_argument(
        "--no-plotly",
        action="store_true",
        help="Skip Plotly HTML export even if enabled in the config",
    )
    args = parser.parse_args(argv)
    cfg = load_hyperfine_config(args.config)
    run_hyperfine_simulation(cfg, args.out, enable_plotly=not args.no_plotly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
