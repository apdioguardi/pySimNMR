from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import warnings
import re

AXIS_KEY_ALIASES = {
    'x_low_limit': 'x_axis_min',
    'x_high_limit': 'x_axis_max',
    'y_low_limit': 'y_axis_min',
    'y_high_limit': 'y_axis_max',
}


def normalize_axis_keys(cfg: dict[str, Any] | None, *, warn: bool = True) -> dict[str, Any] | None:
    """Map deprecated plotting keys (x_low_limit, etc.) to the canonical names."""
    if not cfg:
        return cfg
    for old_key, new_key in AXIS_KEY_ALIASES.items():
        if old_key in cfg:
            if new_key in cfg:
                if warn:
                    warnings.warn(
                        f"Ignoring deprecated config key '{old_key}' because '{new_key}' "
                        "is also provided.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                cfg.pop(old_key)
            else:
                if warn:
                    warnings.warn(
                        f"Config key '{old_key}' is deprecated; use '{new_key}' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                cfg[new_key] = cfg.pop(old_key)
    return cfg


def _finite_arrays(arrays: Iterable[Any]) -> list[np.ndarray]:
    finite_chunks: list[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        data = np.asarray(arr, dtype=float).ravel()
        if data.size == 0:
            continue
        mask = np.isfinite(data)
        if not np.any(mask):
            continue
        finite_chunks.append(data[mask])
    return finite_chunks


def _data_extent(arrays: Iterable[Any]) -> tuple[float, float]:
    finite_chunks = _finite_arrays(arrays)
    if not finite_chunks:
        return 0.0, 1.0
    concat = np.concatenate(finite_chunks)
    return float(concat.min()), float(concat.max())


def resolve_axis_limits(
    data_arrays: Iterable[Any],
    axis_min: float | None,
    axis_max: float | None,
    *,
    pad_fraction: float = 0.02,
) -> tuple[float, float]:
    """Return axis limits that span the data unless explicitly overridden."""
    data_min, data_max = _data_extent(data_arrays)
    if axis_min is None:
        axis_min = data_min
    if axis_max is None:
        axis_max = data_max
    if axis_min == axis_max:
        pad = abs(axis_min) if axis_min else 1.0
        pad *= pad_fraction if pad_fraction > 0 else 0.05
        axis_min -= pad
        axis_max += pad
    return axis_min, axis_max


def load_experimental_data(
    path: Path,
    *,
    delimiter: str = ' ',
    skip_header: int = 0,
    missing_values: str | None = None,
) -> np.ndarray:
    """Load experimental data ensuring it exists and has at least two columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experimental data file not found: {path}")
    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=skip_header,
        missing_values=missing_values,
    )
    if data.size == 0:
        raise ValueError(f"Experimental data file is empty: {path}")
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(
            f"Experimental data file must have at least two columns (field/frequency and intensity): {path}" 
        ) # ahoy, plotting should eventually incude plotting for vary
    return data


def format_isotope_label(isotope: str) -> str:
    """Return an HTML-friendly isotope label, e.g., 75As -> <sup>75</sup>As."""
    match = re.match(r"(\d+)(.*)", isotope.strip())
    if match:
        digits, rest = match.groups()
        rest = rest.strip()
        if rest:
            return f"<sup>{digits}</sup>{rest}"
        return f"<sup>{digits}</sup>"
    return isotope


def format_parameter_value(value: Any) -> str:
    """Stringify parameter values for Plotly tables."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "<br>".join(format_parameter_value(v) for v in value)
    return str(value)


def _extract_value(entry: Any) -> Any:
    if hasattr(entry, "value"):
        val = getattr(entry, "value")
        return _extract_value(val)
    if isinstance(entry, (list, tuple)) and len(entry) == 1:
        return _extract_value(entry[0])
    return entry


def _coerce_sequence(values: Any, length: int) -> list[Any]:
    if values is None:
        seq: list[Any] = []
    elif isinstance(values, np.ndarray):
        seq = values.tolist()
    elif isinstance(values, (list, tuple)):
        seq = list(values)
    else:
        seq = [values]
    seq = [_extract_value(v) for v in seq]
    if len(seq) < length:
        seq.extend([""] * (length - len(seq)))
    elif len(seq) > length:
        seq = seq[:length]
    return seq


def build_parameter_table(
    site_labels: Sequence[str],
    per_site_params: dict[str, Sequence[Any]],
    global_params: dict[str, Any] | None = None,
) -> tuple[Optional[List[str]], Optional[List[List[str]]], Optional[List[str]], Optional[List[List[str]]]]:
    """Return headers/rows for per-site and global parameter tables."""
    site_labels = list(site_labels)
    site_header: Optional[List[str]] = None
    site_rows: Optional[List[List[str]]] = None
    n_sites = len(site_labels)

    if per_site_params:
        site_header = ["Parameter"] + site_labels
        site_rows = []
        for name, values in per_site_params.items():
            seq = _coerce_sequence(values, n_sites)
            row = [name] + [format_parameter_value(v) for v in seq]
            site_rows.append(row)

    global_header: Optional[List[str]] = None
    global_rows: Optional[List[List[str]]] = None
    if global_params:
        global_header = ["Global parameter", "Value"]
        global_rows = []
        for name, value in global_params.items():
            global_rows.append([name, format_parameter_value(value)])

    return site_header, site_rows, global_header, global_rows

