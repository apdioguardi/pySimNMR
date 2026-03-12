from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

import numpy as np
import yaml

from .config_aliases import normalize_freq_config
from .config_loader import load_python_config
from .freq_simulation import (
    SingleCrystalFreqConfig,
    SingleCrystalFreqResult,
    simulate_single_crystal_freq,
)
from .plotting import (
    SpectrumPlotData,
    RenderedPlot,
    save_spectrum_plot,
    finalize_rendered_figure,
)
from .plotting_utils import (
    normalize_axis_keys,
    resolve_axis_limits,
    load_experimental_data,
    build_parameter_table,
    format_isotope_label,
)


def _load_config(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == '.py':
        return load_python_config(path)
    with open(path, 'r', encoding='utf-8') as fh:
        if suffix in {'.yml', '.yaml'}:
            return yaml.safe_load(fh)
        return json.load(fh)


def _apply_background(freq: np.ndarray, spec: np.ndarray, bgd: List[float]) -> np.ndarray:
    out = np.array(spec, dtype=float)
    if not bgd:
        return out
    max_val = np.nanmax(out)
    if len(bgd) == 1:
        denom = max_val + bgd[0] if max_val else 1.0
        return (out + bgd[0]) / denom
    if len(bgd) == 2:
        offset, slope = bgd
        out = offset + slope * freq + out
        norm = np.nanmax(out)
        return out / norm if norm else out
    if len(bgd) >= 3:
        center, width, intensity = bgd[:3]
        corr = (1.0 / (np.sqrt(2 * np.pi) * width)) * np.exp(-(freq - center) ** 2 / (2 * width ** 2))
        corr = corr / np.nanmax(corr) * intensity if np.nanmax(corr) else corr
        out = corr + out
        norm = np.nanmax(out)
        return out / norm if norm else out
    return out


def _plot_and_save(result: SingleCrystalFreqResult,
                   cfg: SingleCrystalFreqConfig,
                   out_base: Path,
                   save_fig_path: Path,
                   dpi: int,
                   config_dir: Path,
                   config_path: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    freq_axis = result.freq_axis
    per_site = result.per_site
    total = np.sum(np.vstack(per_site), axis=0)
    max_val = np.nanmax(total) if np.size(total) else 1.0
    if not np.isfinite(max_val) or max_val == 0:
        max_val = 1.0
    total_norm = total / max_val
    total_with_bg = _apply_background(freq_axis, total_norm, cfg.bgd)
    per_site_norm = [spec / max_val for spec in per_site]

    output_txt = out_base.with_suffix('.txt').resolve()
    np.savetxt(output_txt, np.column_stack((freq_axis, total_with_bg)))

    exp_x = exp_y = None
    exp_path_str = None
    if cfg.exp_data_file:
        exp_path = Path(cfg.exp_data_file)
        if not exp_path.is_absolute():
            exp_path = config_dir / exp_path
        exp_path = exp_path.resolve()
        try:
            exp_data = load_experimental_data(
                exp_path,
                delimiter=cfg.exp_data_delimiter,
                skip_header=cfg.number_of_header_lines,
                missing_values=cfg.missing_values_string,
            )
        except Exception as exc:  # pragma: no cover - simple error propagation
            raise SystemExit(f"Failed to load experimental data from '{exp_path}': {exc}") from exc
        exp_x = exp_data[:, 0] * cfg.exp_x_scaling + cfg.exp_x_offset
        exp_y = exp_data[:, 1] * cfg.exp_y_scaling + cfg.exp_y_offset
        exp_path_str = str(exp_path)

    all_x = [freq_axis] + ([exp_x] if exp_x is not None else [])
    all_y = [total_with_bg] + ([exp_y] if exp_y is not None else [])
    x_axis_min, x_axis_max = resolve_axis_limits(all_x, cfg.x_axis_min, cfg.x_axis_max)
    y_axis_min, y_axis_max = resolve_axis_limits(all_y, cfg.y_axis_min, cfg.y_axis_max)
    x_limits = (x_axis_min, x_axis_max)
    y_limits = (y_axis_min, y_axis_max)

    site_headers = [format_isotope_label(iso) for iso in cfg.isotope_list]
    hint_values = [f"({hint.x:.3g}, {hint.y:.3g}, {hint.z:.3g})" for hint in cfg.Hint_list]
    per_site_params = {
        r'$m_i$': cfg.site_multiplicity_list,
        r'$K_a$ (%)': cfg.Ka_list,
        r'$K_b$ (%)': cfg.Kb_list,
        r'$K_c$ (%)': cfg.Kc_list,
        r'$\nu_c$ (MHz)': cfg.vc_list,
        r'$\eta$': cfg.eta_list,
        'Hint (T)': hint_values,
        r'$\phi$ (deg)': cfg.phi_z_deg_list,
        r'$\theta$ (deg)': cfg.theta_x_prime_deg_list,
        r'$\psi$ (deg)': cfg.psi_z_prime_deg_list,
        'Conv. func.': cfg.convolution_function_list,
        'FWHM (MHz)': cfg.conv_FWHM_list,
        r'$\mathrm{FWHM}_{d\nu_Q}$ (MHz)': cfg.conv_vQ_FWHM_list,
    }
    global_params = {
        r'$H_0$ (T)': cfg.H0_T,
        'Simulation': cfg.sim_mode,
        'Matrix elem. min': cfg.matrix_element_cutoff,
    }
    site_table_header, site_table_rows, global_table_header, global_table_rows = build_parameter_table(
        site_headers, per_site_params, global_params
    )
    metadata_entries: list[tuple[str, str]] = [("Config file", str(config_path))]
    if exp_path_str:
        metadata_entries.append(("Experimental data", exp_path_str))
    metadata_entries.append(("Output spectrum", str(output_txt)))
    if cfg.sim_export_file:
        metadata_entries.append(("Sim export", str(Path(cfg.sim_export_file).resolve())))
    metadata_entries.append(("Figure", str(save_fig_path.resolve())))

    plot_data = SpectrumPlotData(
        x=freq_axis,
        total_with_background=total_with_bg,
        per_site_normalized=per_site_norm,
        site_labels=result.site_labels,
        plot_individual=cfg.plot_individual_bool,
        plot_sum=cfg.plot_sum_bool,
        title='Frequency-swept spectrum; $H_0$ = ' + str(cfg.H0_T) + ' T',
        x_label='Frequency (MHz)',
        y_label='Intensity (arb. units)',
        legend_lines=[],
        legend_width_ratio=cfg.plot_legend_width_ratio,
        x_limits=x_limits,
        y_limits=y_limits,
        exp_x=exp_x,
        exp_y=exp_y,
        exp_label='experiment',
        site_table_header=site_table_header,
        site_table_rows=site_table_rows,
        global_table_header=global_table_header,
        global_table_rows=global_table_rows,
        metadata_entries=metadata_entries,
    )
    rendered = save_spectrum_plot(
        plot_data,
        save_path=save_fig_path,
        html_path=out_base.with_suffix('.html'),
        dpi=dpi,
    )

    if cfg.sim_export_file:
        export_path = Path(cfg.sim_export_file)
        if not export_path.suffix:
            export_path = export_path.with_suffix('.txt')
        export_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(export_path, np.column_stack((freq_axis, total_with_bg)))
        for idx, spec in enumerate(per_site):
            per_site_path = export_path.with_name(f"{export_path.stem}_{cfg.isotope_list[idx]}_{idx}.txt")
            np.savetxt(per_site_path, np.column_stack((freq_axis, spec / max_val)))

    return rendered


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Single-crystal frequency-swept spectra (multisite).')
    parser.add_argument('--config', type=Path, required=True, help='Configuration file (.py/.yaml/.json).')
    parser.add_argument('--out', type=Path, default=Path('output/freq_spectrum_multisite'),
                        help='Output stem for exported data/plots.')
    parser.add_argument('--save-fig', type=Path,
                        help='Optional path to save the figure (png/svg/pdf). Overrides default.')
    parser.add_argument('--no-show', action='store_true', help='Do not open a GUI window for the plot.')
    parser.add_argument('--fig-format', default='png', choices=['png', 'svg', 'pdf'],
                        help='Figure format if --save-fig is not set.')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for saved figures.')
    args = parser.parse_args(argv)

    raw_cfg = _load_config(args.config)
    axis_normalized = normalize_axis_keys(raw_cfg, warn=False)
    if axis_normalized is None:
        axis_normalized = raw_cfg
    normalized_cfg = normalize_freq_config(axis_normalized)
    cfg_model = SingleCrystalFreqConfig(**normalized_cfg)
    result = simulate_single_crystal_freq(cfg_model)
    show_plot = not args.no_show

    save_fig_path = args.save_fig
    if save_fig_path is None:
        save_fig_path = args.out.with_suffix(f'.{args.fig_format}')
    rendered = _plot_and_save(result, cfg_model, args.out, save_fig_path, args.dpi,
                              args.config.parent, args.config.resolve())

    finalize_rendered_figure(rendered, show_plot)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
