from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import yaml

from .config_loader import load_python_config
from .powder_field_simulation import PowderFieldConfig, simulate_powder_field
from .plotting import SpectrumPlotData, save_spectrum_plot, finalize_rendered_figure
from .plotting_utils import (
    normalize_axis_keys,
    resolve_axis_limits,
    build_parameter_table,
    format_isotope_label,
)
from .config_aliases import normalize_powder_field_config
from .progress import ProgressManager


def _load_config(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == '.py':
        return load_python_config(path)
    with open(path, 'r', encoding='utf-8') as fh:
        if suffix in {'.yml', '.yaml'}:
            return yaml.safe_load(fh)
        return json.load(fh)


def _apply_background(field: np.ndarray, spec: np.ndarray, bgd: List[float]) -> np.ndarray:
    out = np.array(spec, dtype=float)
    if not bgd:
        return out
    if len(bgd) >= 1:
        denom = np.nanmax(out) + bgd[0] if np.nanmax(out) else 1.0
        out = (out + bgd[0]) / denom
    if len(bgd) >= 2:
        offset, slope = bgd[0], bgd[1]
        out = offset + slope * field + out
        norm = np.nanmax(out)
        out = out / norm if norm else out
    if len(bgd) >= 3:
        center, width, intensity = bgd[0], bgd[1], bgd[2]
        corr = (1.0 / (np.sqrt(2 * np.pi) * width)) * np.exp(-(field - center) ** 2 / (2 * width ** 2))
        if np.nanmax(corr):
            corr = corr / np.nanmax(corr) * intensity
        out = corr + out
        norm = np.nanmax(out)
        out = out / norm if norm else out
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Powder field-swept spectra.')
    parser.add_argument('--config', type=Path, required=True, help='Configuration file (.py/.yml/.json).')
    parser.add_argument('--out', type=Path, default=Path('output/field_powder_spectrum'),
                        help='Output stem for exported data/plots (default: %(default)s).')
    parser.add_argument('--save-fig', type=Path, help='Optional figure path.')
    parser.add_argument('--fig-format', default='png', choices=['png','svg','pdf'],
                        help='Figure format when --save-fig is omitted.')
    parser.add_argument('--dpi', type=int, default=150, help='Figure DPI.')
    parser.add_argument('--no-show', action='store_true', help='Disable GUI windows.')
    parser.set_defaults(progress=True)
    parser.add_argument('--progress', dest='progress', action='store_true',
                        help='Show progress bars (default; auto-disables when --no-show is set).')
    parser.add_argument('--no-progress', dest='progress', action='store_false',
                        help='Disable progress bars.')
    args = parser.parse_args(argv)
    config_path = args.config.resolve()

    progress_flag = args.progress and not args.no_show
    progress = ProgressManager(progress_flag)

    raw_cfg = _load_config(args.config)
    axis_cfg = normalize_axis_keys(raw_cfg, warn=False) or raw_cfg
    normalized = normalize_powder_field_config(axis_cfg)
    exp_path = normalized.get('exp_data_file')
    exp_path_str = None
    if exp_path:
        exp_path = Path(exp_path)
        if not exp_path.is_absolute():
            exp_path = args.config.parent / exp_path
        exp_path = exp_path.resolve()
        normalized['exp_data_file'] = str(exp_path)
        exp_path_str = str(exp_path)

    cfg = PowderFieldConfig(**normalized)
    try:
        result = simulate_powder_field(cfg, progress=progress if progress.enabled else None)
    except RuntimeError as exc:  # pragma: no cover - passthrough for clarity
        raise SystemExit(str(exc)) from exc
    show_plot = not args.no_show
    per_site = result.per_site
    total = np.sum(np.vstack(per_site), axis=0)
    max_val = np.nanmax(total) or 1.0
    per_site_norm = [spec / max_val for spec in per_site]
    total_norm = total / max_val
    total_with_bg = _apply_background(result.field_axis, total_norm, normalized.get('bgd', [0.0]))

    out_base = args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)
    output_txt = out_base.with_suffix('.txt').resolve()
    np.savetxt(output_txt, np.column_stack((result.field_axis, total_with_bg)))

    plot_individual = normalized.get('plot_individual_bool', True)
    plot_sum = normalized.get('plot_sum_bool', True)
    x_min, x_max = resolve_axis_limits([result.field_axis, result.exp_x], normalized.get('x_axis_min'),
                                       normalized.get('x_axis_max'))
    y_min, y_max = resolve_axis_limits([total_with_bg, result.exp_y], normalized.get('y_axis_min'),
                                       normalized.get('y_axis_max'))

    site_headers = [format_isotope_label(iso) for iso in cfg.isotope_list]
    per_site_params = {
        r'$m_i$': cfg.site_multiplicity_list,
        r'$K_a$ (%)': cfg.Ka_list,
        r'$K_b$ (%)': cfg.Kb_list,
        r'$K_c$ (%)': cfg.Kc_list,
        r'$\nu_c$ (MHz)': cfg.vc_list,
        r'$\eta$': cfg.eta_list,
        'Conv. func.': cfg.convolution_function_list,
        'FWHM (T)': cfg.conv_FWHM_list,
        r'$\mathrm{FWHM}_{d\nu_Q}$ (T)': cfg.conv_vQ_FWHM_list,
    }
    global_params = {
        r'$f_0$ (MHz)': cfg.f0_MHz,
        'Simulation': cfg.sim_mode,
        r'$\Delta f_0$ (MHz)': cfg.delta_f0,
        'Matrix elem. min': cfg.mtx_elem_min,
        'n samples': cfg.n_samples,
    }
    site_table_header, site_table_rows, global_table_header, global_table_rows = build_parameter_table(
        site_headers, per_site_params, global_params
    )
    save_fig = args.save_fig or args.out.with_suffix(f".{args.fig_format}")
    metadata_entries: list[tuple[str, str]] = [("Config file", str(config_path))]
    if exp_path_str:
        metadata_entries.append(("Experimental data", exp_path_str))
    metadata_entries.append(("Output spectrum", str(output_txt)))
    metadata_entries.append(("Figure", str(save_fig.resolve())))
    plot_data = SpectrumPlotData(
        x=result.field_axis,
        total_with_background=total_with_bg,
        per_site_normalized=per_site_norm,
        site_labels=result.site_labels,
        plot_individual=plot_individual,
        plot_sum=plot_sum,
        title='Powder field spectrum',
        x_label='Field (T)',
        y_label='Intensity (arb. units)',
        legend_lines=[],
        legend_width_ratio=normalized.get('plot_legend_width_ratio', [3.25, 1.0]),
        x_limits=(x_min, x_max),
        y_limits=(y_min, y_max),
        exp_x=result.exp_x,
        exp_y=result.exp_y,
        exp_label='experiment',
        site_table_header=site_table_header,
        site_table_rows=site_table_rows,
        global_table_header=global_table_header,
        global_table_rows=global_table_rows,
        metadata_entries=metadata_entries,
    )

    rendered = save_spectrum_plot(
        plot_data,
        save_path=save_fig,
        html_path=out_base.with_suffix('.html'),
        dpi=args.dpi,
    )

    finalize_rendered_figure(rendered, show_plot)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
