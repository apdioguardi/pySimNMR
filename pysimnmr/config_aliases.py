"""Helpers to normalize legacy YAML keys onto canonical CLI names."""
from __future__ import annotations

from typing import Any, Dict
import math


def _pop_first(cfg: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in cfg:
            return cfg.pop(name)
    return None


def _first_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _as_float(value: Any) -> float | None:
    val = _first_value(value)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    fval = _as_float(value)
    if fval is None:
        return None
    return int(round(fval))


def _set_angle_rad(cfg: Dict[str, Any], key: str, *deg_aliases: str) -> None:
    if key in cfg:
        return
    val = _pop_first(cfg, *deg_aliases)
    deg = _as_float(val)
    if deg is not None:
        cfg[key] = math.radians(deg)


def normalize_common_aliases(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``raw_cfg`` with canonical keys populated.

    Applies aliases shared across CLIs (shift tensor names, quadrupole values,
    Euler angles, hint vectors, etc.). Original keys are popped so Pydantic
    models do not see duplicates.
    """
    cfg = dict(raw_cfg)

    shift_aliases = {
        'Ka': ('K_a', 'Ka_percent'),
        'Kb': ('K_b', 'Kb_percent'),
        'Kc': ('K_c', 'Kc_percent'),
    }
    for canon, aliases in shift_aliases.items():
        if canon not in cfg:
            val = _pop_first(cfg, *aliases)
            fval = _as_float(val)
            if fval is not None:
                cfg[canon] = fval

    if 'vQ_MHz' not in cfg:
        val = _pop_first(cfg, 'vQ', 'v_Q', 'vc', 'vc_MHz', 'nu_c', 'nu_c_MHz', 'nu_cMHz')
        fval = _as_float(val)
        if fval is not None:
            cfg['vQ_MHz'] = fval

    _set_angle_rad(cfg, 'phi', 'phi_deg', 'phi_z_deg', 'phi0_deg')
    _set_angle_rad(cfg, 'theta', 'theta_deg', 'theta_x_prime_deg')
    _set_angle_rad(cfg, 'psi', 'psi_deg', 'psi_z_prime_deg')

    if 'B_axis' not in cfg:
        val = _pop_first(cfg, 'field_axis', 'axis')
        if val is not None:
            cfg['B_axis'] = str(val)

    if 'Hint_pas' not in cfg:
        hx = _as_float(_pop_first(cfg, 'Hintx', 'Hinta', 'Hint_a'))
        hy = _as_float(_pop_first(cfg, 'Hinty', 'Hintb', 'Hint_b'))
        hz = _as_float(_pop_first(cfg, 'Hintz', 'Hintc', 'Hint_c'))
        if hx is not None or hy is not None or hz is not None:
            cfg['Hint_pas'] = {
                'x': hx if hx is not None else 0.0,
                'y': hy if hy is not None else 0.0,
                'z': hz if hz is not None else 0.0,
            }

    # Allow legacy field naming for fixed-field configs (used directly in hyperfine CLI)
    if 'H0_T' not in cfg:
        val = _pop_first(cfg, 'H0', 'H0_Tesla', 'B0', 'B0_T', 'field_T')
        fval = _as_float(val)
        if fval is not None:
            cfg['H0_T'] = fval

    return cfg


def normalize_vary_config(raw_cfg: Dict[str, Any], cmd: str) -> Dict[str, Any]:
    """Apply canonical key mapping tailored to single-crystal variation CLIs."""
    cfg = normalize_common_aliases(raw_cfg)

    def pop(*names: str) -> Any:
        return _pop_first(cfg, *names)

    def set_float(key: str, *aliases: str) -> None:
        if key in cfg:
            return
        val = pop(*aliases)
        fval = _as_float(val)
        if fval is not None:
            cfg[key] = fval

    def set_int(key: str, *aliases: str) -> None:
        if key in cfg:
            return
        val = pop(*aliases)
        ival = _as_int(val)
        if ival is not None:
            cfg[key] = ival

    cmd = (cmd or '').lower()
    if cmd in {'freq-vs-field', 'elevels-vs-field'}:
        set_float('B_min_T', 'min_field', 'field_min_T', 'H_min_T', 'B_min')
        set_float('B_max_T', 'max_field', 'field_max_T', 'H_max_T', 'B_max')
        set_int('B_points', 'n_fields', 'n_field_points', 'field_points')
    if cmd == 'freq-vs-eta':
        set_float('eta_min', 'eta_start')
        set_float('eta_max', 'eta_stop')
        set_int('eta_points', 'n_etas', 'eta_steps')
        if 'B0_T' not in cfg:
            val = pop('B0', 'H0', 'H0_T', 'field_T')
            fval = _as_float(val)
            if fval is not None:
                cfg['B0_T'] = fval
    if cmd == 'freq-vs-angle':
        if 'angle_name' not in cfg:
            val = pop('angle_to_vary')
            if isinstance(val, str):
                val_lower = val.strip().lower()
                mapping = {
                    'phi_z_deg': 'phi',
                    'phi': 'phi',
                    'theta_x_prime_deg': 'theta',
                    'theta': 'theta',
                    'psi_z_prime_deg': 'psi',
                    'psi': 'psi',
                }
                mapped = mapping.get(val_lower)
                if mapped:
                    cfg['angle_name'] = mapped
        set_float('angle_min_deg', 'angle_start')
        set_float('angle_max_deg', 'angle_stop')
        set_int('angle_points', 'n_angles', 'angle_steps')
        def set_fixed_deg(key: str, *aliases: str) -> None:
            if key in cfg:
                return
            val = pop(*aliases)
            fval = _as_float(val)
            if fval is not None:
                cfg[key] = fval
        set_fixed_deg('fixed_phi_deg', 'phi_z_deg_init', 'phi_z_deg_init_list', 'phi_z_deg')
        set_fixed_deg('fixed_theta_deg', 'theta_x_prime_deg_init', 'theta_x_prime_deg_init_list', 'theta_x_prime_deg')
        set_fixed_deg('fixed_psi_deg', 'psi_z_prime_deg_init', 'psi_z_prime_deg_init_list', 'psi_z_prime_deg')
        if 'H0_T' not in cfg and 'B0_T' in cfg:
            cfg['H0_T'] = cfg['B0_T']
        if 'H0_T' not in cfg:
            val = pop('H0', 'H0_Tesla')
            fval = _as_float(val)
            if fval is not None:
                cfg['H0_T'] = fval

    return cfg


__all__ = [
    'normalize_common_aliases',
    'normalize_vary_config',
    'normalize_freq_config',
    'normalize_field_config',
    'normalize_powder_freq_config',
    'normalize_powder_field_config',
]


def normalize_freq_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize configs for single-crystal frequency spectra."""
    cfg = dict(raw_cfg)
    isotopes = cfg.get('isotope_list')
    if isotopes is None:
        raise ValueError("Config must provide 'isotope_list'")
    isotopes = list(isotopes)
    if not isotopes:
        raise ValueError("'isotope_list' cannot be empty")
    cfg['isotope_list'] = isotopes
    n_sites = len(isotopes)

    def _ensure_list(key: str, default: Any | None = None) -> None:
        if key not in cfg or cfg[key] is None:
            if default is None:
                return
            cfg[key] = default() if callable(default) else default
        value = cfg[key]
        if isinstance(value, (list, tuple)):
            seq = list(value)
        else:
            seq = [value]
        if len(seq) == 1 and n_sites > 1:
            seq = seq * n_sites
        cfg[key] = seq

    _ensure_list('site_multiplicity_list', lambda: [1.0] * n_sites)
    _ensure_list('Ka_list')
    _ensure_list('Kb_list')
    _ensure_list('Kc_list')
    _ensure_list('va_list', lambda: [None] * n_sites)
    _ensure_list('vb_list', lambda: [None] * n_sites)
    _ensure_list('vc_list')
    _ensure_list('eta_list')
    _ensure_list('convolution_function_list')
    _ensure_list('conv_FWHM_list')
    _ensure_list('conv_vQ_FWHM_list')
    _ensure_list('Hint_list', lambda: [[0.0, 0.0, 0.0] for _ in range(n_sites)])
    _ensure_list('phi_deg_list')
    _ensure_list('theta_deg_list')
    _ensure_list('phi_z_deg_list')
    _ensure_list('theta_x_prime_deg_list')
    _ensure_list('psi_z_prime_deg_list')

    sim_type = str(cfg.get('sim_type', 'ed')).strip().lower()
    if sim_type in {'exact diag', 'exact_diag', 'exact-diag', 'ed'}:
        cfg['sim_type'] = 'ed'
    elif sim_type in {'2nd order', '2nd-order', '2nd', 'perturbative'}:
        cfg['sim_type'] = 'perturbative'
    else:
        cfg['sim_type'] = 'ed'

    return cfg


def normalize_field_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw_cfg)
    isotopes = list(cfg.get('isotope_list', []))
    if not isotopes:
        raise ValueError("Config must provide 'isotope_list'")
    n_sites = len(isotopes)

    def ensure_list(key: str, default=None):
        if key not in cfg or cfg[key] is None:
            if default is None:
                raise ValueError(f"{key} is required")
            cfg[key] = default() if callable(default) else default
        value = cfg[key]
        if isinstance(value, (list, tuple)):
            seq = list(value)
        else:
            seq = [value]
        if len(seq) == 1 and n_sites > 1:
            seq = seq * n_sites
        cfg[key] = seq

    ensure_list('site_multiplicity_list', lambda: [1.0] * n_sites)
    for key in ['Ka_list', 'Kb_list', 'Kc_list', 'vc_list', 'eta_list',
                'Hinta_list', 'Hintb_list', 'Hintc_list',
                'conv_FWHM_list', 'conv_vQ_FWHM_list',
                'convolution_function_list']:
        ensure_list(key)
    ensure_list('phi_z_deg_list', lambda: [0.0] * n_sites)
    ensure_list('theta_x_prime_deg_list', lambda: [0.0] * n_sites)
    ensure_list('psi_z_prime_deg_list', lambda: [0.0] * n_sites)
    ensure_list('phi_deg_list', lambda: [0.0] * n_sites)
    ensure_list('theta_deg_list', lambda: [0.0] * n_sites)

    if 'va_list' not in cfg or cfg['va_list'] is None:
        cfg['va_list'] = [None] * n_sites
    else:
        ensure_list('va_list')
    if 'vb_list' not in cfg or cfg['vb_list'] is None:
        cfg['vb_list'] = [None] * n_sites
    else:
        ensure_list('vb_list')

    sim_type = str(cfg.get('sim_type', 'ed')).strip().lower()
    if sim_type in {'exact diag', 'exact_diag', 'exact-diag', 'ed'}:
        cfg['sim_type'] = 'ed'
    elif sim_type in {'2nd order', '2nd-order', '2nd', 'perturbative'}:
        cfg['sim_type'] = 'perturbative'
    else:
        cfg['sim_type'] = 'ed'

    return cfg


def normalize_powder_freq_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw_cfg)
    isotopes = list(cfg.get('isotope_list', []))
    if not isotopes:
        raise ValueError("Config must provide 'isotope_list'")
    n_sites = len(isotopes)

    def ensure_list(key: str, default=None):
        if key not in cfg or cfg[key] is None:
            if default is None:
                raise ValueError(f"{key} is required")
            cfg[key] = default() if callable(default) else default
        value = cfg[key]
        if isinstance(value, (list, tuple)):
            seq = list(value)
        else:
            seq = [value]
        if len(seq) == 1 and n_sites > 1:
            seq = seq * n_sites
        cfg[key] = seq

    ensure_list('site_multiplicity_list', lambda: [1.0] * n_sites)
    for key in ['Ka_list', 'Kb_list', 'Kc_list', 'eta_list',
                'conv_FWHM_list', 'conv_vQ_FWHM_list',
                'convolution_function_list']:
        ensure_list(key)
    if 'vc_list' not in cfg:
        if 'vQ_list' not in cfg:
            raise ValueError("Either vc_list or vQ_list must be provided")
        ensure_list('vQ_list')
    else:
        ensure_list('vc_list')

    sim_type = str(cfg.get('sim_type', 'ed')).strip().lower()
    if sim_type in {'exact diag', 'exact_diag', 'exact-diag', 'ed'}:
        cfg['sim_type'] = 'ed'
    elif sim_type in {'2nd order', '2nd-order', '2nd', 'perturbative'}:
        cfg['sim_type'] = 'perturbative'
    else:
        cfg['sim_type'] = 'ed'

    return cfg


def normalize_powder_field_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw_cfg)
    isotopes = list(cfg.get('isotope_list', []))
    if not isotopes:
        raise ValueError("Config must provide 'isotope_list'")
    n_sites = len(isotopes)

    def ensure_list(key: str, default=None):
        if key not in cfg or cfg[key] is None:
            if default is None:
                raise ValueError(f"{key} is required")
            cfg[key] = default() if callable(default) else default
        value = cfg[key]
        if isinstance(value, (list, tuple)):
            seq = list(value)
        else:
            seq = [value]
        if len(seq) == 1 and n_sites > 1:
            seq = seq * n_sites
        cfg[key] = seq

    hint_vectors = cfg.get('Hint_list')
    if hint_vectors is not None:
        if isinstance(hint_vectors, (list, tuple)):
            vectors = list(hint_vectors)
        else:
            vectors = [hint_vectors]
        if len(vectors) == 1 and n_sites > 1:
            vectors = vectors * n_sites
        if len(vectors) != n_sites:
            raise ValueError(f"Hint_list must have {n_sites} entries")
        parsed = []
        for idx, vec in enumerate(vectors):
            if vec is None:
                parsed.append((0.0, 0.0, 0.0))
                continue
            if not isinstance(vec, (list, tuple)) or len(vec) != 3:
                raise ValueError(f"Hint_list[{idx}] must be a list/tuple of three values")
            parsed.append(tuple(float(v) for v in vec))
        cfg['Hinta_list'] = [vec[0] for vec in parsed]
        cfg['Hintb_list'] = [vec[1] for vec in parsed]
        cfg['Hintc_list'] = [vec[2] for vec in parsed]

    ensure_list('site_multiplicity_list', lambda: [1.0] * n_sites)
    for key in ['Ka_list', 'Kb_list', 'Kc_list',
                'conv_FWHM_list', 'conv_vQ_FWHM_list',
                'convolution_function_list']:
        ensure_list(key)

    if 'eta_list' in cfg and cfg['eta_list'] is not None:
        ensure_list('eta_list')
    else:
        cfg['eta_list'] = [None] * n_sites

    ensure_list('Hinta_list', lambda: [0.0] * n_sites)
    ensure_list('Hintb_list', lambda: [0.0] * n_sites)
    ensure_list('Hintc_list', lambda: [0.0] * n_sites)

    if 'va_list' not in cfg or cfg['va_list'] is None:
        cfg['va_list'] = [None] * n_sites
    else:
        ensure_list('va_list')
    if 'vb_list' not in cfg or cfg['vb_list'] is None:
        cfg['vb_list'] = [None] * n_sites
    else:
        ensure_list('vb_list')

    if 'vc_list' not in cfg:
        if 'vQ_list' not in cfg:
            raise ValueError("Either vc_list or vQ_list must be provided")
        ensure_list('vQ_list')
    else:
        ensure_list('vc_list')

    sim_type = str(cfg.get('sim_type', 'ed')).strip().lower()
    if sim_type in {'exact diag', 'exact_diag', 'exact-diag', 'ed'}:
        cfg['sim_type'] = 'ed'
    elif sim_type in {'2nd order', '2nd-order', '2nd', 'perturbative'}:
        cfg['sim_type'] = 'perturbative'
    else:
        cfg['sim_type'] = 'ed'

    return cfg
