# -- coding: utf-8 --
from __future__ import annotations
import argparse, json
import warnings
from pathlib import Path
from typing import List
import numpy as np
import yaml
from .core import SimNMR
from .vary_helpers import (freq_vs_field_sweep, elevels_vs_field_sweep,
                           freq_vs_eta_sweep, freq_vs_angle_sweep)
from pydantic import ValidationError
from .config_schema import CMD_TO_MODEL
from .config_aliases import normalize_vary_config
from .config_loader import load_python_config
from .progress import ProgressManager
from .plotting import save_line_plot, finalize_rendered_figure

try:
    from .isotopeDict import isotope_dict as _ISO
except Exception:
    try:
        from .isotopeDict import isotope_data_dict as _ISO
    except Exception as e:
        raise ImportError("Could not import isotope dictionary from pysimnmr.isotopeDict") from e

def _load_config(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == '.py':
        return load_python_config(path)
    with open(path, 'r', encoding='utf-8') as f:
        if suffix in {'.yml', '.yaml'}:
            return yaml.safe_load(f)
        return json.load(f)

def _save_all(x, Y, outbase: Path, xlab: str, ylab: str, title: str) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        outbase.with_suffix('.txt'),
        np.column_stack([x, Y if Y.ndim == 1 else Y]),
        header=f"{xlab} {ylab}"
    )
    series = []
    labels = []
    if Y.ndim == 1:
        series.append(np.asarray(Y))
        labels.append(title or 'Series')
    else:
        for j in range(Y.shape[1]):
            series.append(np.asarray(Y[:, j]))
            labels.append(f"Series {j + 1}")
    rendered = save_line_plot(
        np.asarray(x),
        series,
        labels=labels,
        title=title,
        x_label=xlab,
        y_label=ylab,
        save_path=outbase.with_suffix('.png'),
        html_path=outbase.with_suffix('.html'),
        dpi=150,
    )
    finalize_rendered_figure(rendered, show_plot=False)


def _run_command(cmd: str, config_path: Path, out_path: Path, *, progress_enabled: bool) -> int:
    raw_cfg = _load_config(config_path)
    normalized_cfg = normalize_vary_config(raw_cfg, cmd)
    model_cls = CMD_TO_MODEL.get(cmd)
    if model_cls is None:
        raise SystemExit(f"Unknown command {cmd}")
    try:
        cfg_model = model_cls(**normalized_cfg)
    except ValidationError as exc:
        raise SystemExit(f"Invalid configuration: {exc}") from exc
    cfg = cfg_model
    progress = ProgressManager(progress_enabled)
    sim = SimNMR(cfg.isotope)
    if cmd == 'freq-vs-field':
        B, Y = freq_vs_field_sweep(
            sim,
            B_min_T=cfg.B_min_T,
            B_max_T=cfg.B_max_T,
            B_points=cfg.B_points,
            Ka=cfg.Ka,
            Kb=cfg.Kb,
            Kc=cfg.Kc,
            vQ_MHz=cfg.vQ_MHz,
            eta=cfg.eta,
            phi=cfg.phi,
            theta=cfg.theta,
            psi=cfg.psi,
            Hint_pas=cfg.Hint_pas.as_tuple() if cfg.Hint_pas else None,
            axis=cfg.B_axis.lower(),
            progress=progress if progress.enabled else None,
        )
        _save_all(
            B,
            Y,
            out_path,
            xlab='Field (T)',
            ylab='Frequencies (MHz) - columns = transitions',
            title='Field-swept spectrum',
        )
        return 0
    if cmd == 'elevels-vs-field':
        B, Y = elevels_vs_field_sweep(
            sim,
            B_min_T=cfg.B_min_T,
            B_max_T=cfg.B_max_T,
            B_points=cfg.B_points,
            Ka=cfg.Ka,
            Kb=cfg.Kb,
            Kc=cfg.Kc,
            vQ_MHz=cfg.vQ_MHz,
            eta=cfg.eta,
            phi=cfg.phi,
            theta=cfg.theta,
            psi=cfg.psi,
            axis=cfg.B_axis.lower(),
            Hint_pas=cfg.Hint_pas.as_tuple() if cfg.Hint_pas else None,
            progress=progress if progress.enabled else None,
        )
        _save_all(
            B,
            Y,
            out_path,
            xlab='Field (T)',
            ylab='Energy levels (MHz) - columns = levels',
            title='Energy levels vs field',
        )
        return 0
    if cmd == 'freq-vs-eta':
        X, Y = freq_vs_eta_sweep(
            sim,
            eta_min=cfg.eta_min,
            eta_max=cfg.eta_max,
            eta_points=cfg.eta_points,
            Ka=cfg.Ka,
            Kb=cfg.Kb,
            Kc=cfg.Kc,
            vQ_MHz=cfg.vQ_MHz,
            phi=cfg.phi,
            theta=cfg.theta,
            psi=cfg.psi,
            H0_T=cfg.B0_T,
            axis=cfg.B_axis.lower(),
            Hint_pas=cfg.Hint_pas.as_tuple() if cfg.Hint_pas else None,
            progress=progress if progress.enabled else None,
        )
        _save_all(
            X,
            Y,
            out_path,
            xlab='eta (unitless)',
            ylab='Frequencies (MHz) - columns = transitions',
            title='Frequency vs eta',
        )
        return 0
    if cmd == 'freq-vs-angle':
        X, Y = freq_vs_angle_sweep(
            sim,
            angle_name=cfg.angle_name,
            angle_min_deg=cfg.angle_min_deg,
            angle_max_deg=cfg.angle_max_deg,
            angle_points=cfg.angle_points,
            Ka=cfg.Ka,
            Kb=cfg.Kb,
            Kc=cfg.Kc,
            vQ_MHz=cfg.vQ_MHz,
            eta=cfg.eta,
            fixed_phi_deg=cfg.fixed_phi_deg,
            fixed_theta_deg=cfg.fixed_theta_deg,
            fixed_psi_deg=cfg.fixed_psi_deg,
            H0_T=cfg.H0_T,
            axis=cfg.B_axis.lower(),
            Hint_pas=cfg.Hint_pas.as_tuple() if cfg.Hint_pas else None,
            progress=progress if progress.enabled else None,
        )
        axis_label = f"{cfg.angle_name} (deg)"
        _save_all(
            X,
            Y,
            out_path,
            xlab=axis_label,
            ylab='Frequencies (MHz) - columns = transitions',
            title='Frequency vs angle',
        )
        return 0
    raise SystemExit(f"Unknown command {cmd}")


def _single_command_main(cmd: str, prog: str, description: str, default_out: Path,
                         argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        '--config',
        required=True,
        type=Path,
        help='Configuration file (Python module, YAML, or JSON).',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=default_out,
        help='Output stem for generated data/plots (default: %(default)s).',
    )
    parser.set_defaults(progress=True)
    parser.add_argument(
        '--progress',
        dest='progress',
        action='store_true',
        help='Show progress bars (default; auto-disables when not on a TTY).',
    )
    parser.add_argument(
        '--no-progress',
        dest='progress',
        action='store_false',
        help='Disable progress bars.',
    )
    args = parser.parse_args(argv)
    return _run_command(cmd, args.config, args.out,
                        progress_enabled=args.progress)


def field_main(argv: List[str] | None = None) -> int:
    """CLI entry point for `nmr-field`."""
    return _single_command_main(
        'freq-vs-field',
        prog='nmr-field',
        description='Single-crystal field-swept spectrum (frequency vs field).',
        default_out=Path('output/freq_vs_field'),
        argv=argv,
    )


def elevels_main(argv: List[str] | None = None) -> int:
    """CLI entry point for `nmr-elevels`."""
    return _single_command_main(
        'elevels-vs-field',
        prog='nmr-elevels',
        description='Single-crystal energy levels vs field.',
        default_out=Path('output/elevels_vs_field'),
        argv=argv,
    )


def vary_eta_main(argv: List[str] | None = None) -> int:
    """CLI entry point for `nmr-vary-eta`."""
    return _single_command_main(
        'freq-vs-eta',
        prog='nmr-vary-eta',
        description='Frequency response vs asymmetry parameter eta.',
        default_out=Path('output/freq_vs_eta'),
        argv=argv,
    )


def vary_angle_main(argv: List[str] | None = None) -> int:
    """CLI entry point for `nmr-vary-angle`."""
    return _single_command_main(
        'freq-vs-angle',
        prog='nmr-vary-angle',
        description='Frequency response vs Euler angles.',
        default_out=Path('output/freq_vs_angle'),
        argv=argv,
    )


def legacy_main(argv: List[str] | None = None) -> int:
    """Backward-compatible multi-command parser retained for `nmr-sweep`."""
    warnings.warn(
        "nmr-sweep is deprecated; use nmr-field, nmr-elevels, nmr-vary-eta, or nmr-vary-angle instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(
        prog='nmr-sweep',
        description='Single-crystal variation utilities (legacy multi-command entry).',
    )
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name, default_out in [
        ('freq-vs-field', Path('output/freq_vs_field')),
        ('elevels-vs-field', Path('output/elevels_vs_field')),
        ('freq-vs-eta', Path('output/freq_vs_eta')),
        ('freq-vs-angle', Path('output/freq_vs_angle')),
    ]:
        sp = sub.add_parser(name)
        sp.add_argument('--config', required=True, type=Path, help='Config file path')
        sp.add_argument(
            '--out',
            type=Path,
            default=default_out,
            help='Output stem for generated data/plots.',
        )
        sp.set_defaults(progress=True)
        sp.add_argument(
            '--progress',
            dest='progress',
            action='store_true',
            help='Show progress bars (default; auto-disables on non-TTY).',
        )
        sp.add_argument(
            '--no-progress',
            dest='progress',
            action='store_false',
            help='Disable progress bars.',
        )
    args = parser.parse_args(argv)
    return _run_command(args.cmd, args.config, args.out,
                        progress_enabled=args.progress)


main = legacy_main


if __name__ == '__main__':
    raise SystemExit(legacy_main())
