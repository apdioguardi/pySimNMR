"""Frequency-spectrum fitting CLI built on the shared simulation helpers."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

import lmfit
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config_loader import load_python_config
from .freq_simulation import SingleCrystalFreqConfig, simulate_single_crystal_freq
from .plotly_utils import save_lines_html
from .plotting import SpectrumPlotData, OverlayTrace, save_spectrum_plot, finalize_rendered_figure
from .plotting_utils import (
    normalize_axis_keys,
    resolve_axis_limits,
    build_parameter_table,
    format_isotope_label,
)
from .progress import ProgressManager


def _load_config(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == '.py':
        return load_python_config(path)
    with open(path, 'r', encoding='utf-8') as fh:
        if suffix in {'.yml', '.yaml'}:
            return yaml.safe_load(fh)
        return json.load(fh)


@dataclass
class ParameterSpec:
    value: Optional[float]
    vary: bool
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    expr: Optional[str] = None


@dataclass
class SiteParameter:
    name: Optional[str]
    default: Optional[float]


@dataclass
class SiteMetadata:
    isotope: str
    amplitude: SiteParameter
    Ka: SiteParameter
    Kb: SiteParameter
    Kc: SiteParameter
    va: SiteParameter
    vb: SiteParameter
    vc: SiteParameter
    eta: SiteParameter
    Hinta: SiteParameter
    Hintb: SiteParameter
    Hintc: SiteParameter
    phi_z_deg: SiteParameter
    theta_xp_deg: SiteParameter
    psi_zp_deg: SiteParameter
    FWHM: SiteParameter
    FWHM_vQ: SiteParameter
    line_shape: str


class SingleCrystalFreqFitConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    exp_data_file: str = ''
    number_of_header_lines: int = 0
    exp_data_delimiter: str = ' '
    missing_values_string: Optional[str] = None
    exp_x_scaling: float = 1.0
    exp_y_scaling: float = 1.0
    exp_x_offset: float = 0.0
    exp_y_offset: float = 0.0

    minimization_algorithm: str = 'leastsq'
    epsilon: Optional[float] = None
    verbose_bool: bool = False
    max_nfev: Optional[int] = None

    isotope_list: List[str]
    H0: List[Any]
    amplitude_list: List[Any]
    Ka_list: List[Any]
    Kb_list: List[Any]
    Kc_list: List[Any]
    va_list: List[Any] = Field(default_factory=list)
    vb_list: List[Any] = Field(default_factory=list)
    vc_list: List[Any]
    eta_list: List[Any]
    Hinta_list: List[Any]
    Hintb_list: List[Any]
    Hintc_list: List[Any]
    phi_z_deg_list: List[Any]
    theta_xp_deg_list: List[Any]
    psi_zp_deg_list: List[Any]
    FWHM_list: List[Any]
    FWHM_vQ_list: List[Any]

    min_freq: float
    max_freq: float
    n_plot_points: int = Field(alias='freq_points', default=1000, ge=2)

    line_shape_func_list: List[str]
    background_list: List[List[Any]] = Field(default_factory=lambda: [[0.0]])

    plot_initial_guess_bool: bool = True
    plot_individual_bool: bool = False
    plot_sum_bool: bool = True
    plot_legend_width_ratio: List[float] = Field(default_factory=lambda: [3.25, 1.0])
    x_axis_min: Optional[float] = None
    x_axis_max: Optional[float] = None
    y_axis_min: Optional[float] = None
    y_axis_max: Optional[float] = None
    exp_plot_style_str: str = 'k-'

    sim_export_file: str = ''

    @field_validator('amplitude_list', 'Ka_list', 'Kb_list', 'Kc_list',
                     'va_list', 'vb_list', 'vc_list', 'eta_list',
                     'Hinta_list', 'Hintb_list', 'Hintc_list',
                     'phi_z_deg_list', 'theta_xp_deg_list', 'psi_zp_deg_list',
                     'FWHM_list', 'FWHM_vQ_list', mode='before')
    @classmethod
    def _ensure_nested(cls, value):
        if isinstance(value, list):
            nested = []
            for item in value:
                if isinstance(item, list):
                    nested.append(item)
                else:
                    nested.append([item])
            return nested
        raise ValueError('Parameter lists must be lists')

    @model_validator(mode='after')
    def _validate_lengths(self) -> 'SingleCrystalFreqFitConfig':
        n_sites = len(self.isotope_list)

        def ensure_length(name: str, allow_empty: bool = False) -> None:
            lst = getattr(self, name, None)
            if not lst:
                if allow_empty:
                    setattr(self, name, [[None] for _ in range(n_sites)])
                    return
                raise ValueError(f"{name} must be provided")
            if len(lst) != n_sites:
                raise ValueError(f"{name} must have {n_sites} entries")

        for name in [
            'amplitude_list', 'Ka_list', 'Kb_list', 'Kc_list',
            'vc_list', 'eta_list', 'Hinta_list', 'Hintb_list',
            'Hintc_list', 'phi_z_deg_list', 'theta_xp_deg_list',
            'psi_zp_deg_list', 'FWHM_list', 'FWHM_vQ_list',
        ]:
            ensure_length(name)

        ensure_length('va_list', allow_empty=True)
        ensure_length('vb_list', allow_empty=True)

        if len(self.line_shape_func_list) == 1 and n_sites > 1:
            self.line_shape_func_list = self.line_shape_func_list * n_sites
        elif len(self.line_shape_func_list) != n_sites:
            raise ValueError("line_shape_func_list must match isotope_list length")

        if len(self.plot_legend_width_ratio) == 1:
            self.plot_legend_width_ratio = [self.plot_legend_width_ratio[0], 1.0]
        elif len(self.plot_legend_width_ratio) != 2:
            raise ValueError("plot_legend_width_ratio must have one or two entries")

        return self


def _parse_parameter_entry(entry: Any, *, allow_none: bool = False) -> ParameterSpec:
    if isinstance(entry, str):
        return ParameterSpec(value=None, vary=False, expr=entry)
    if not isinstance(entry, list):
        entry = [entry]
    if not entry:
        raise ValueError("parameter specification cannot be empty")
    initial = entry[0]
    if initial is None and not allow_none:
        raise ValueError("parameter value cannot be None")
    value = None if initial is None else float(initial)
    if len(entry) == 1:
        return ParameterSpec(value=value, vary=False)
    if len(entry) == 2:
        return ParameterSpec(value=value, vary=bool(entry[1]))
    if len(entry) == 4:
        return ParameterSpec(value=value,
                             vary=bool(entry[1]),
                             minimum=float(entry[2]),
                             maximum=float(entry[3]))
    raise ValueError("parameter specification must have 1, 2, or 4 elements")


def _value_or_default(param: SiteParameter, pars: lmfit.Parameters) -> Optional[float]:
    if param.name and param.name in pars:
        return float(pars[param.name].value)
    return param.default


def _apply_background(freq: np.ndarray, data: np.ndarray, bgd: List[float]) -> np.ndarray:
    out = np.array(data, dtype=float)
    if not bgd:
        return out
    if len(bgd) >= 1:
        denom = np.nanmax(out) + bgd[0] if np.nanmax(out) else 1.0
        out = (out + bgd[0]) / denom
    if len(bgd) >= 2:
        offset, slope = bgd[0], bgd[1]
        out = offset + slope * freq + out
        max_val = np.nanmax(out)
        out = out / max_val if max_val else out
    if len(bgd) >= 3:
        center, width, intensity = bgd[0], bgd[1], bgd[2]
        corr = (1.0 / (np.sqrt(2 * np.pi) * width)) * np.exp(-(freq - center) ** 2 / (2 * width ** 2))
        corr = corr / np.nanmax(corr) * intensity if np.nanmax(corr) else corr
        out = corr + out
        max_val = np.nanmax(out)
        out = out / max_val if max_val else out
    return out


class FrequencyFitRunner:
    def __init__(self,
                 cfg: SingleCrystalFreqFitConfig,
                 exp_x: np.ndarray,
                 exp_y: np.ndarray,
                 progress: ProgressManager | None = None) -> None:
        self.cfg = cfg
        self.exp_x = exp_x
        self.exp_y = exp_y
        self.progress = progress
        self.params = lmfit.Parameters()
        self._fit_bar = None
        self._pending_expr: Dict[str, str] = {}
        self.site_meta: List[SiteMetadata] = []
        self.background_params: List[SiteParameter] = []
        self.H0_param: SiteParameter = SiteParameter(name=None, default=None)
        self._build_parameters()

    def _add_parameter(self, name: str, spec: ParameterSpec, *, default: Optional[float] = None) -> SiteParameter:
        if spec.expr:
            self._pending_expr[name] = spec.expr
            return SiteParameter(name=name, default=None)
        if spec.value is None and not spec.vary:
            return SiteParameter(name=None, default=default)
        kwargs: Dict[str, float] = {}
        if spec.minimum is not None:
            kwargs['min'] = spec.minimum
        if spec.maximum is not None:
            kwargs['max'] = spec.maximum
        self.params.add(name, value=spec.value if spec.value is not None else 0.0,
                        vary=spec.vary, **kwargs)
        return SiteParameter(name=name, default=None)

    def _build_parameters(self) -> None:
        self.H0_param = self._add_parameter('H0', _parse_parameter_entry(self.cfg.H0[0]))
        n_sites = len(self.cfg.isotope_list)
        for idx, isotope in enumerate(self.cfg.isotope_list):
            def make(name: str, lst: List[Any], allow_none: bool = False) -> SiteParameter:
                return self._add_parameter(
                    f"{name}_{isotope}_{idx}",
                    _parse_parameter_entry(lst[idx], allow_none=allow_none)
                )
            site = SiteMetadata(
                isotope=isotope,
                amplitude=make('amplitude', self.cfg.amplitude_list),
                Ka=make('Ka', self.cfg.Ka_list),
                Kb=make('Kb', self.cfg.Kb_list),
                Kc=make('Kc', self.cfg.Kc_list),
                va=make('va', self.cfg.va_list, allow_none=True),
                vb=make('vb', self.cfg.vb_list, allow_none=True),
                vc=make('vc', self.cfg.vc_list),
                eta=make('eta', self.cfg.eta_list),
                Hinta=make('Hinta', self.cfg.Hinta_list),
                Hintb=make('Hintb', self.cfg.Hintb_list),
                Hintc=make('Hintc', self.cfg.Hintc_list),
                phi_z_deg=make('phi_z_deg', self.cfg.phi_z_deg_list),
                theta_xp_deg=make('theta_xp_deg', self.cfg.theta_xp_deg_list),
                psi_zp_deg=make('psi_zp_deg', self.cfg.psi_zp_deg_list),
                FWHM=make('FWHM', self.cfg.FWHM_list),
                FWHM_vQ=make('FWHM_vQ', self.cfg.FWHM_vQ_list),
                line_shape=self.cfg.line_shape_func_list[idx],
            )
            self.site_meta.append(site)
        self._build_background_params()
        for name, expr in self._pending_expr.items():
            self.params.add(name, expr=expr)

    def _build_background_params(self) -> None:
        specs: List[SiteParameter] = []
        entries = self.cfg.background_list
        labels = ['background_offset', 'background_slope', 'background_center',
                  'background_width', 'background_intensity']
        for idx, entry in enumerate(entries[:3]):
            spec = _parse_parameter_entry(entry)
            name = labels[idx]
            specs.append(self._add_parameter(name, spec))
        self.background_params = specs

    def _background_values(self, pars: lmfit.Parameters) -> List[float]:
        values: List[float] = []
        for param in self.background_params:
            val = _value_or_default(param, pars)
            if val is not None:
                values.append(float(val))
        return values

    def _build_freq_config(self, pars: lmfit.Parameters) -> SingleCrystalFreqConfig:
        amplitudes = []
        Ka = []
        Kb = []
        Kc = []
        va = []
        vb = []
        vc = []
        eta = []
        Hinta = []
        Hintb = []
        Hintc = []
        phi = []
        theta = []
        psi = []
        FWHM = []
        FWHM_vQ = []

        for site in self.site_meta:
            amplitudes.append(_value_or_default(site.amplitude, pars) or 1.0)
            Ka.append(_value_or_default(site.Ka, pars))
            Kb.append(_value_or_default(site.Kb, pars))
            Kc.append(_value_or_default(site.Kc, pars))
            va.append(_value_or_default(site.va, pars))
            vb.append(_value_or_default(site.vb, pars))
            vc.append(_value_or_default(site.vc, pars))
            eta.append(_value_or_default(site.eta, pars))
            Hinta.append(_value_or_default(site.Hinta, pars))
            Hintb.append(_value_or_default(site.Hintb, pars))
            Hintc.append(_value_or_default(site.Hintc, pars))
            phi.append(_value_or_default(site.phi_z_deg, pars))
            theta.append(_value_or_default(site.theta_xp_deg, pars))
            psi.append(_value_or_default(site.psi_zp_deg, pars))
            FWHM.append(_value_or_default(site.FWHM, pars))
            FWHM_vQ.append(_value_or_default(site.FWHM_vQ, pars))

        cfg_dict = {
            'isotope_list': self.cfg.isotope_list,
            'site_multiplicity_list': amplitudes,
            'Ka_list': Ka,
            'Kb_list': Kb,
            'Kc_list': Kc,
            'va_list': va,
            'vb_list': vb,
            'vc_list': vc,
            'eta_list': eta,
            'Hinta_list': Hinta,
            'Hintb_list': Hintb,
            'Hintc_list': Hintc,
            'phi_z_deg_list': phi,
            'theta_x_prime_deg_list': theta,
            'psi_z_prime_deg_list': psi,
            'convolution_function_list': [site.line_shape for site in self.site_meta],
            'conv_FWHM_list': FWHM,
            'conv_vQ_FWHM_list': FWHM_vQ,
            'H0': _value_or_default(self.H0_param, pars),
            'min_freq': self.cfg.min_freq,
            'max_freq': self.cfg.max_freq,
            'n_freq_points': self.cfg.n_plot_points,
        }
        return SingleCrystalFreqConfig(**cfg_dict)

    def simulate(self, pars: lmfit.Parameters, *, show_progress: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[str]]:
        freq_cfg = self._build_freq_config(pars)
        sim_progress = None
        if show_progress and self.progress is not None and self.progress.enabled:
            sim_progress = self.progress
        result = simulate_single_crystal_freq(freq_cfg, progress=sim_progress)
        per_site = [spec.copy() for spec in result.per_site]
        stack = np.vstack(per_site) if per_site else np.zeros((1, result.freq_axis.size))
        total = stack.sum(axis=0) if per_site else np.zeros(result.freq_axis.size)
        denom = np.nanmax(total) or 1.0
        per_site_norm = [spec / denom for spec in per_site]
        total_norm = total / denom
        bg_vals = self._background_values(pars)
        total_with_bg = _apply_background(result.freq_axis, total_norm, bg_vals)
        return result.freq_axis, total_with_bg, per_site_norm, result.site_labels

    def residual(self, pars: lmfit.Parameters) -> np.ndarray:
        freq_axis, fitted, _, _ = self.simulate(pars)
        interp = np.interp(self.exp_x, freq_axis, fitted,
                           left=fitted[0], right=fitted[-1])
        diff = interp - self.exp_y
        if self._fit_bar is not None:
            self._fit_bar.update(1)
        return diff

    def fit(self, max_nfev: Optional[int] = None) -> lmfit.MinimizerResult:
        minimizer = lmfit.Minimizer(self.residual, self.params)
        # Use an indeterminate bar (total=None) because the number of function
        # evaluations is not known ahead of time.
        if self.progress is not None and self.progress.enabled:
            self._fit_bar = self.progress.bar(total=None, desc='Fitting (freq)')
        else:
            self._fit_bar = None

        try:
            result = minimizer.minimize(method=self.cfg.minimization_algorithm,
                                        epsfcn=self.cfg.epsilon,
                                        max_nfev=max_nfev or self.cfg.max_nfev)
        finally:
            if self._fit_bar is not None:
                self._fit_bar.complete()
                self._fit_bar.close()
                self._fit_bar = None
        return result

    def save_exports(self,
                     freq_axis: np.ndarray,
                     fitted: np.ndarray,
                     per_site: List[np.ndarray],
                     labels: List[str],
                     out_base: Path) -> None:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_base.with_suffix('.txt'), np.column_stack((freq_axis, fitted)))
        try:
            save_lines_html(freq_axis, fitted, xlab='Frequency (MHz)',
                            ylab='Intensity (arb. units)',
                            title='Frequency fit', out_html=out_base.with_suffix('.html'))
        except Exception:
            pass

        if self.cfg.sim_export_file:
            export_path = Path(self.cfg.sim_export_file)
            if not export_path.suffix:
                export_path = export_path.with_suffix('.txt')
            export_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(export_path, np.column_stack((freq_axis, fitted)))
            for idx, spec in enumerate(per_site):
                site_path = export_path.with_name(f"{export_path.stem}_{labels[idx]}.txt")
                np.savetxt(site_path, np.column_stack((freq_axis, spec)))


def _load_experimental_data(cfg: SingleCrystalFreqFitConfig, config_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not cfg.exp_data_file:
        raise SystemExit("exp_data_file must be provided for fitting")
    data_path = Path(cfg.exp_data_file)
    if not data_path.is_absolute():
        data_path = config_path / data_path
    data = np.genfromtxt(
        data_path,
        delimiter=cfg.exp_data_delimiter,
        skip_header=cfg.number_of_header_lines,
        missing_values=cfg.missing_values_string,
    )
    exp_x = data[:, 0] * cfg.exp_x_scaling + cfg.exp_x_offset
    exp_y = data[:, 1] * cfg.exp_y_scaling + cfg.exp_y_offset
    return exp_x, exp_y


def _plot_fit(exp_x: np.ndarray,
              exp_y: np.ndarray,
              freq_axis: np.ndarray,
              fitted: np.ndarray,
              per_site: List[np.ndarray],
              labels: List[str],
              initial_guess: Optional[Tuple[np.ndarray, np.ndarray]],
              cfg: SingleCrystalFreqFitConfig,
              config_path: Path,
              save_path: Path,
              html_path: Path,
              dpi: int):
    overlays: List[OverlayTrace] = [
        OverlayTrace(x=exp_x, y=exp_y, label='experimental data', mpl_style=cfg.exp_plot_style_str)
    ]
    if initial_guess and cfg.plot_initial_guess_bool:
        init_freq, init_curve = initial_guess
        overlays.append(
            OverlayTrace(
                x=init_freq,
                y=init_curve,
                label='initial guess',
                mpl_style='k--',
                line_width=1.5,
                opacity=0.6,
            )
        )

    x_min, x_max = resolve_axis_limits([freq_axis, exp_x], cfg.x_axis_min, cfg.x_axis_max)
    y_min, y_max = resolve_axis_limits([fitted, exp_y], cfg.y_axis_min, cfg.y_axis_max)

    site_headers = [format_isotope_label(iso) for iso in cfg.isotope_list]
    per_site_params = {
        r'$H_0$ (T)': cfg.H0,
        'Amplitude': cfg.amplitude_list,
        r'$K_a$ (%)': cfg.Ka_list,
        r'$K_b$ (%)': cfg.Kb_list,
        r'$K_c$ (%)': cfg.Kc_list,
        r'$\nu_c$ (MHz)': cfg.vc_list,
        r'$\eta$': cfg.eta_list,
        r'$\phi$ (deg)': cfg.phi_z_deg_list,
        r'$\theta$ (deg)': cfg.theta_xp_deg_list,
        r'$\psi$ (deg)': cfg.psi_zp_deg_list,
        'Line shape': cfg.line_shape_func_list,
        'FWHM (MHz)': cfg.FWHM_list,
        r'$\mathrm{FWHM}_{d\nu_Q}$ (MHz)': cfg.FWHM_vQ_list,
    }
    global_params = {
        'Algorithm': cfg.minimization_algorithm,
        'max nfev': cfg.max_nfev or "auto",
        'Freq range (MHz)': f"{cfg.min_freq} - {cfg.max_freq}",
    }
    site_table_header, site_table_rows, global_table_header, global_table_rows = build_parameter_table(
        site_headers, per_site_params, global_params
    )
    metadata_entries = [
        ("Config file", str(config_path)),
        ("Experimental data", str(Path(cfg.exp_data_file).resolve())),
        ("Figure", str(save_path.resolve())),
        ("HTML report", str(html_path.resolve())),
    ]

    plot_data = SpectrumPlotData(
        x=freq_axis,
        total_with_background=fitted,
        per_site_normalized=per_site,
        site_labels=labels,
        plot_individual=cfg.plot_individual_bool,
        plot_sum=cfg.plot_sum_bool,
        title='Frequency fit',
        x_label='Frequency (MHz)',
        y_label='Intensity (arb. units)',
        legend_lines=[],
        legend_width_ratio=cfg.plot_legend_width_ratio,
        x_limits=(x_min, x_max),
        y_limits=(y_min, y_max),
        overlays=overlays,
        site_table_header=site_table_header,
        site_table_rows=site_table_rows,
        global_table_header=global_table_header,
        global_table_rows=global_table_rows,
        metadata_entries=metadata_entries,
    )
    return save_spectrum_plot(
        plot_data,
        save_path=save_path,
        html_path=html_path,
        dpi=dpi,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Fit frequency-swept multisite spectra using lmfit.')
    parser.add_argument('--config', required=True, type=Path,
                        help='Configuration file (Python module/YAML/JSON).')
    parser.add_argument('--out', type=Path, default=Path('output/freq_fit_multisite'),
                        help='Output stem for exported data/plots (default: %(default)s).')
    parser.add_argument('--save-fig', type=Path,
                        help='Optional figure path; defaults to OUT with fig-format extension.')
    parser.add_argument('--fig-format', default='png', choices=['png', 'svg', 'pdf'],
                        help='Figure format when --save-fig is omitted.')
    parser.add_argument('--dpi', type=int, default=150, help='Figure DPI (default: %(default)s).')
    parser.add_argument('--no-show', action='store_true', help='Disable interactive windows.')
    parser.add_argument('--max-nfev', type=int,
                        help='Optional cap on the number of function evaluations (overrides config).')
    parser.set_defaults(progress=True)
    parser.add_argument('--progress', dest='progress', action='store_true',
                        help='Show tqdm progress bars (default; disabled automatically with --no-show).')
    parser.add_argument('--no-progress', dest='progress', action='store_false',
                        help='Disable progress bars.')
    args = parser.parse_args(argv)

    progress_flag = args.progress and not args.no_show
    progress = ProgressManager(progress_flag)

    raw_cfg = _load_config(args.config)
    axis_normalized = normalize_axis_keys(raw_cfg) or raw_cfg
    cfg = SingleCrystalFreqFitConfig(**axis_normalized)
    exp_x, exp_y = _load_experimental_data(cfg, args.config.parent)
    show_plot = not args.no_show
    runner = FrequencyFitRunner(cfg, exp_x, exp_y, progress=progress)

    initial_guess = None
    if cfg.plot_initial_guess_bool:
        initial_guess = runner.simulate(runner.params, show_progress=True)[:2]

    result = runner.fit(max_nfev=args.max_nfev)
    freq_axis, fitted, per_site, labels = runner.simulate(result.params, show_progress=True)

    save_fig_path = args.save_fig or args.out.with_suffix(f'.{args.fig_format}')
    html_path = args.out.with_suffix('.html')
    rendered = _plot_fit(
        exp_x,
        exp_y,
        freq_axis,
        fitted,
        per_site,
        labels,
        initial_guess,
        cfg,
        args.config.resolve(),
        save_fig_path,
        html_path,
        args.dpi,
    )
    runner.save_exports(freq_axis, fitted, per_site, labels, args.out)

    if cfg.verbose_bool:
        print(lmfit.fit_report(result))
    else:
        print(f"nmr-freq-fit: reduced chi-square = {result.redchi:.4g}")

    finalize_rendered_figure(rendered, show_plot)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
