"""Field-spectrum fitting CLI built on the shared field simulation helper."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import lmfit
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config_loader import load_python_config
from .field_simulation import SingleCrystalFieldConfig, simulate_single_crystal_field
from .plotting import SpectrumPlotData, OverlayTrace, save_spectrum_plot, finalize_rendered_figure
from .plotting_utils import (
    normalize_axis_keys,
    resolve_axis_limits,
    build_parameter_table,
    format_isotope_label,
)
from .plotly_utils import save_lines_html
from .progress import ProgressManager


class ParameterSpec(BaseModel):
    value: Optional[float] = None
    vary: bool = False
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    expr: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def _convert(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            return {'expr': value}
        if not isinstance(value, list):
            return {'value': value, 'vary': False}
        if len(value) == 1:
            if isinstance(value[0], str):
                return {'expr': value[0]}
            return {'value': value[0], 'vary': False}
        if len(value) == 2:
            return {'value': value[0], 'vary': bool(value[1])}
        if len(value) == 4:
            return {'value': value[0],
                    'vary': bool(value[1]),
                    'minimum': value[2],
                    'maximum': value[3]}
        raise ValueError('Parameter specification must have 1, 2, or 4 elements')


class FieldFitConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    exp_data_file: str
    number_of_header_lines: int = 0
    exp_data_delimiter: str = ' '
    missing_values_string: Optional[str] = None
    exp_x_scaling: float = 1.0
    exp_y_scaling: float = 1.0
    exp_y_offset: float = 0.0

    minimization_algorithm: str = 'leastsq'
    epsilon: Optional[float] = None
    verbose_bool: bool = False
    max_nfev: Optional[int] = None

    isotope_list: List[str]
    f0: float
    site_multiplicity_list: List[ParameterSpec] = Field(alias='site_multiplicity_list')
    Ka_list: List[ParameterSpec] = Field(alias='Ka_list')
    Kb_list: List[ParameterSpec] = Field(alias='Kb_list')
    Kc_list: List[ParameterSpec] = Field(alias='Kc_list')
    va_list: List[ParameterSpec] = Field(alias='va_list')
    vb_list: List[ParameterSpec] = Field(alias='vb_list')
    vc_list: List[ParameterSpec] = Field(alias='vc_list')
    eta_list: List[ParameterSpec] = Field(alias='eta_list')
    Hinta_list: List[ParameterSpec] = Field(alias='Hinta_list')
    Hintb_list: List[ParameterSpec] = Field(alias='Hintb_list')
    Hintc_list: List[ParameterSpec] = Field(alias='Hintc_list')
    phi_z_deg_list: List[ParameterSpec] = Field(alias='phi_z_deg_list')
    theta_xp_deg_list: List[ParameterSpec] = Field(alias='theta_x_prime_deg_list')
    psi_zp_deg_list: List[ParameterSpec] = Field(alias='psi_z_prime_deg_list')
    phi_deg_list: List[ParameterSpec] = Field(alias='phi_deg_list')
    theta_deg_list: List[ParameterSpec] = Field(alias='theta_deg_list')
    conv_FWHM_list: List[ParameterSpec] = Field(alias='conv_FWHM_list')
    conv_vQ_FWHM_list: List[ParameterSpec] = Field(alias='conv_vQ_FWHM_list')

    line_shape_func_list: List[str] = Field(alias='convolution_function_list')
    background_list: List[List[Any]] = Field(default_factory=lambda: [[0.0]])
    sim_type: str = 'exact diag'
    min_field: float
    max_field: float
    n_field_points: int = Field(ge=2)
    mtx_elem_min: float = 0.1
    delta_f0: float = 0.001

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

    @model_validator(mode='after')
    def _validate_lengths(self) -> 'FieldFitConfig':
        n_sites = len(self.isotope_list)

        def ensure(name: str):
            lst = getattr(self, name)
            if len(lst) != n_sites:
                raise ValueError(f"{name} must have {n_sites} entries")

        for name in [
            'site_multiplicity_list', 'Ka_list', 'Kb_list', 'Kc_list',
            'va_list', 'vb_list', 'vc_list', 'eta_list',
            'Hinta_list', 'Hintb_list', 'Hintc_list',
            'phi_z_deg_list', 'theta_xp_deg_list', 'psi_zp_deg_list',
            'phi_deg_list', 'theta_deg_list',
            'conv_FWHM_list', 'conv_vQ_FWHM_list',
        ]:
            ensure(name)

        if len(self.line_shape_func_list) == 1 and n_sites > 1:
            self.line_shape_func_list = self.line_shape_func_list * n_sites
        elif len(self.line_shape_func_list) != n_sites:
            raise ValueError("line_shape_func_list must match isotope_list length")

        if len(self.plot_legend_width_ratio) == 1:
            self.plot_legend_width_ratio = [self.plot_legend_width_ratio[0], 1.0]
        elif len(self.plot_legend_width_ratio) != 2:
            raise ValueError("plot_legend_width_ratio must have one or two entries")

        return self


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
    phi_deg: SiteParameter
    theta_deg: SiteParameter
    FWHM: SiteParameter
    FWHM_vQ: SiteParameter
    line_shape: str


class FieldFitRunner:
    def __init__(self,
                 cfg: FieldFitConfig,
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
        self._build_parameters()

    def _add_parameter(self, name: str, spec: ParameterSpec) -> SiteParameter:
        if spec.expr:
            self._pending_expr[name] = spec.expr
            return SiteParameter(name=name, default=None)
        if spec.value is None and not spec.vary:
            return SiteParameter(name=None, default=None)
        kwargs: Dict[str, float] = {}
        if spec.minimum is not None:
            kwargs['min'] = spec.minimum
        if spec.maximum is not None:
            kwargs['max'] = spec.maximum
        self.params.add(name,
                        value=spec.value if spec.value is not None else 0.0,
                        vary=spec.vary,
                        **kwargs)
        return SiteParameter(name=name, default=None)

    def _build_parameters(self) -> None:
        n_sites = len(self.cfg.isotope_list)
        for idx, isotope in enumerate(self.cfg.isotope_list):
            def make(label: str, specs: List[ParameterSpec]) -> SiteParameter:
                return self._add_parameter(f"{label}_{isotope}_{idx}", specs[idx])

            site = SiteMetadata(
                isotope=isotope,
                amplitude=make('amplitude', self.cfg.site_multiplicity_list),
                Ka=make('Ka', self.cfg.Ka_list),
                Kb=make('Kb', self.cfg.Kb_list),
                Kc=make('Kc', self.cfg.Kc_list),
                va=make('va', self.cfg.va_list),
                vb=make('vb', self.cfg.vb_list),
                vc=make('vc', self.cfg.vc_list),
                eta=make('eta', self.cfg.eta_list),
                Hinta=make('Hinta', self.cfg.Hinta_list),
                Hintb=make('Hintb', self.cfg.Hintb_list),
                Hintc=make('Hintc', self.cfg.Hintc_list),
                phi_z_deg=make('phi_z_deg', self.cfg.phi_z_deg_list),
                theta_xp_deg=make('theta_xp_deg', self.cfg.theta_xp_deg_list),
                psi_zp_deg=make('psi_zp_deg', self.cfg.psi_zp_deg_list),
                phi_deg=make('phi_deg', self.cfg.phi_deg_list),
                theta_deg=make('theta_deg', self.cfg.theta_deg_list),
                FWHM=make('FWHM', self.cfg.conv_FWHM_list),
                FWHM_vQ=make('FWHM_vQ', self.cfg.conv_vQ_FWHM_list),
                line_shape=self.cfg.line_shape_func_list[idx],
            )
            self.site_meta.append(site)
        self._build_background_params()
        for name, expr in self._pending_expr.items():
            self.params.add(name, expr=expr)

    def _build_background_params(self) -> None:
        labels = ['bg_offset', 'bg_slope', 'bg_center', 'bg_width', 'bg_intensity']
        params: List[SiteParameter] = []
        for idx, entry in enumerate(self.cfg.background_list):
            spec = ParameterSpec.model_validate(entry)
            params.append(self._add_parameter(labels[idx], spec))
        self.background_params = params

    def _value_or_default(self, param: SiteParameter, pars: lmfit.Parameters) -> Optional[float]:
        if param.name and param.name in pars:
            return float(pars[param.name].value)
        return param.default

    def _background_values(self, pars: lmfit.Parameters) -> List[float]:
        values = []
        for param in self.background_params:
            val = self._value_or_default(param, pars)
            if val is not None:
                values.append(val)
        return values

    def _build_field_config(self, pars: lmfit.Parameters) -> SingleCrystalFieldConfig:
        mult = []
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
        phi_z = []
        theta_xp = []
        psi_zp = []
        phi = []
        theta = []
        FWHM = []
        FWHM_vQ = []

        for site in self.site_meta:
            mult.append(self._value_or_default(site.amplitude, pars) or 1.0)
            Ka.append(self._value_or_default(site.Ka, pars))
            Kb.append(self._value_or_default(site.Kb, pars))
            Kc.append(self._value_or_default(site.Kc, pars))
            va.append(self._value_or_default(site.va, pars))
            vb.append(self._value_or_default(site.vb, pars))
            vc.append(self._value_or_default(site.vc, pars))
            eta.append(self._value_or_default(site.eta, pars))
            Hinta.append(self._value_or_default(site.Hinta, pars))
            Hintb.append(self._value_or_default(site.Hintb, pars))
            Hintc.append(self._value_or_default(site.Hintc, pars))
            phi_z.append(self._value_or_default(site.phi_z_deg, pars))
            theta_xp.append(self._value_or_default(site.theta_xp_deg, pars))
            psi_zp.append(self._value_or_default(site.psi_zp_deg, pars))
            phi.append(self._value_or_default(site.phi_deg, pars))
            theta.append(self._value_or_default(site.theta_deg, pars))
            FWHM.append(self._value_or_default(site.FWHM, pars))
            FWHM_vQ.append(self._value_or_default(site.FWHM_vQ, pars))

        mode = self.cfg.sim_type.strip().lower()
        if mode in {'exact diag', 'exact_diag', 'exact-diag', 'ed'}:
            sim_mode = 'ed'
        elif mode in {'2nd order', '2nd-order', '2nd', 'perturbative'}:
            sim_mode = 'perturbative'
        else:
            sim_mode = 'ed'

        cfg_dict = {
            'isotope_list': self.cfg.isotope_list,
            'site_multiplicity_list': mult,
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
            'phi_z_deg_list': phi_z,
            'theta_x_prime_deg_list': theta_xp,
            'psi_z_prime_deg_list': psi_zp,
            'phi_deg_list': phi,
            'theta_deg_list': theta,
            'conv_FWHM_list': FWHM,
            'conv_vQ_FWHM_list': FWHM_vQ,
            'sim_type': sim_mode,
            'f0': self.cfg.f0,
            'min_field': self.cfg.min_field,
            'max_field': self.cfg.max_field,
            'n_field_points': self.cfg.n_field_points,
            'convolution_function_list': self.cfg.line_shape_func_list,
            'mtx_elem_min': self.cfg.mtx_elem_min,
            'delta_f0': self.cfg.delta_f0,
        }
        return SingleCrystalFieldConfig(**cfg_dict)

    def simulate(self, pars: lmfit.Parameters, *, show_progress: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[str]]:
        cfg = self._build_field_config(pars)
        sim_progress = None
        if show_progress and self.progress is not None and self.progress.enabled:
            sim_progress = self.progress
        result = simulate_single_crystal_field(cfg, progress=sim_progress)
        if not result.per_site:
            raise RuntimeError("No spectra generated during fit")
        stack = np.vstack(result.per_site)
        total = stack.sum(axis=0)
        max_val = np.nanmax(total) or 1.0
        per_site_norm = [spec / max_val for spec in result.per_site]
        total_norm = total / max_val
        bg = self._background_values(pars)
        total_with_bg = self._apply_background(result.field_axis, total_norm, bg)
        return result.field_axis, total_with_bg, per_site_norm, result.site_labels

    @staticmethod
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

    def residual(self, pars: lmfit.Parameters) -> np.ndarray:
        field_axis, fitted, _, _ = self.simulate(pars)
        interp = np.interp(self.exp_x, field_axis, fitted,
                           left=fitted[0], right=fitted[-1])
        diff = interp - self.exp_y
        if self._fit_bar is not None:
            self._fit_bar.update(1)
        return diff

    def fit(self, max_nfev: Optional[int] = None) -> lmfit.MinimizerResult:
        minimizer = lmfit.Minimizer(self.residual, self.params)
        total_evals = max_nfev or self.cfg.max_nfev or 200
        total_int = max(1, int(total_evals))
        if self.progress is not None and self.progress.enabled:
            self._fit_bar = self.progress.bar(total=total_int, desc='Fitting (field)')
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
                     field_axis: np.ndarray,
                     fitted: np.ndarray,
                     per_site: List[np.ndarray],
                     labels: List[str],
                     out_base: Path) -> None:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_base.with_suffix('.txt'), np.column_stack((field_axis, fitted)))
        try:
            save_lines_html(field_axis, fitted, xlab='Field (T)', ylab='Intensity',
                            title='Field fit', out_html=out_base.with_suffix('.html'))
        except Exception:
            pass
        if self.cfg.sim_export_file:
            export_path = Path(self.cfg.sim_export_file)
            if not export_path.suffix:
                export_path = export_path.with_suffix('.txt')
            export_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(export_path, np.column_stack((field_axis, fitted)))
            for idx, spec in enumerate(per_site):
                site_path = export_path.with_name(f"{export_path.stem}_{labels[idx]}.txt")
                np.savetxt(site_path, np.column_stack((field_axis, spec)))


def _load_config(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == '.py':
        return load_python_config(path)
    with open(path, 'r', encoding='utf-8') as fh:
        if suffix in {'.yml', '.yaml'}:
            return yaml.safe_load(fh)
        return json.load(fh)


def _load_experimental_data(cfg: FieldFitConfig, config_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data_path = Path(cfg.exp_data_file)
    if not data_path.is_absolute():
        data_path = config_path / data_path
    data = np.genfromtxt(
        data_path,
        delimiter=cfg.exp_data_delimiter,
        skip_header=cfg.number_of_header_lines,
        missing_values=cfg.missing_values_string,
    )
    exp_x = data[:, 0] * cfg.exp_x_scaling
    exp_y = data[:, 1] * cfg.exp_y_scaling + cfg.exp_y_offset
    return exp_x, exp_y


def _plot_fit(exp_x: np.ndarray,
              exp_y: np.ndarray,
              field_axis: np.ndarray,
              fitted: np.ndarray,
              per_site: List[np.ndarray],
              labels: List[str],
              initial_guess: Optional[Tuple[np.ndarray, np.ndarray]],
              cfg: FieldFitConfig,
              config_path: Path,
              save_path: Path,
              html_path: Path,
              dpi: int):
    overlays: List[OverlayTrace] = [
        OverlayTrace(
            x=exp_x,
            y=exp_y,
            label='experimental',
            mpl_style=cfg.exp_plot_style_str,
        )
    ]
    if initial_guess and cfg.plot_initial_guess_bool:
        overlays.append(
            OverlayTrace(
                x=initial_guess[0],
                y=initial_guess[1],
                label='initial guess',
                mpl_style='k--',
                line_width=1.5,
                opacity=0.6,
            )
        )

    x_min, x_max = resolve_axis_limits([field_axis, exp_x], cfg.x_axis_min, cfg.x_axis_max)
    y_min, y_max = resolve_axis_limits([fitted, exp_y], cfg.y_axis_min, cfg.y_axis_max)

    site_headers = [format_isotope_label(iso) for iso in cfg.isotope_list]
    per_site_params = {
        r'$m_i$': cfg.site_multiplicity_list,
        r'$K_a$ (%)': cfg.Ka_list,
        r'$K_b$ (%)': cfg.Kb_list,
        r'$K_c$ (%)': cfg.Kc_list,
        r'$\nu_c$ (MHz)': cfg.vc_list,
        r'$\eta$': cfg.eta_list,
        'Conv. func.': cfg.line_shape_func_list,
        'FWHM (T)': cfg.conv_FWHM_list,
        r'$\mathrm{FWHM}_{d\nu_Q}$ (T)': cfg.conv_vQ_FWHM_list,
    }
    global_params = {
        r'$f_0$ (MHz)': cfg.f0,
        'Simulation': cfg.sim_type,
        'Field range (T)': f"{cfg.min_field} - {cfg.max_field}",
        'n field points': cfg.n_field_points,
        'Matrix elem. min': cfg.mtx_elem_min,
        r'$\Delta f_0$ (MHz)': cfg.delta_f0,
        'Algorithm': cfg.minimization_algorithm,
        'max nfev': cfg.max_nfev or "auto",
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
        x=field_axis,
        total_with_background=fitted,
        per_site_normalized=per_site,
        site_labels=labels,
        plot_individual=cfg.plot_individual_bool,
        plot_sum=cfg.plot_sum_bool,
        title='Field fit',
        x_label='Field (T)',
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
    parser = argparse.ArgumentParser(description='Fit field-swept multisite spectra using lmfit.')
    parser.add_argument('--config', type=Path, required=True,
                        help='Configuration file (.py/.yml/.json) describing the fit.')
    parser.add_argument('--out', type=Path, default=Path('output/field_fit_multisite'),
                        help='Output stem for exported data/plots (default: %(default)s).')
    parser.add_argument('--save-fig', type=Path,
                        help='Optional explicit figure path; defaults to OUT + fig-format extension.')
    parser.add_argument('--fig-format', default='png', choices=['png', 'svg', 'pdf'],
                        help='Figure format when --save-fig is omitted.')
    parser.add_argument('--dpi', type=int, default=150, help='Figure DPI.')
    parser.add_argument('--no-show', action='store_true', help='Disable interactive windows.')
    parser.add_argument('--max-nfev', type=int,
                        help='Override the maximum number of function evaluations.')
    parser.set_defaults(progress=True)
    parser.add_argument('--progress', dest='progress', action='store_true',
                        help='Show tqdm progress bars (default; disabled automatically with --no-show).')
    parser.add_argument('--no-progress', dest='progress', action='store_false',
                        help='Disable progress bars.')
    args = parser.parse_args(argv)

    progress_flag = args.progress and not args.no_show
    progress = ProgressManager(progress_flag)

    raw_cfg = _load_config(args.config)
    axis_cfg = normalize_axis_keys(raw_cfg, warn=False) or raw_cfg
    cfg = FieldFitConfig(**axis_cfg)
    show_plot = not args.no_show
    exp_x, exp_y = _load_experimental_data(cfg, args.config.parent)
    runner = FieldFitRunner(cfg, exp_x, exp_y, progress=progress)

    initial_guess = None
    if cfg.plot_initial_guess_bool:
        initial_guess = runner.simulate(runner.params, show_progress=True)[:2]

    result = runner.fit(max_nfev=args.max_nfev)
    field_axis, fitted, per_site, labels = runner.simulate(result.params, show_progress=True)

    save_fig = args.save_fig or args.out.with_suffix(f'.{args.fig_format}')
    html_path = args.out.with_suffix('.html')
    rendered = _plot_fit(
        exp_x,
        exp_y,
        field_axis,
        fitted,
        per_site,
        labels,
        initial_guess,
        cfg,
        args.config.resolve(),
        save_fig,
        html_path,
        args.dpi,
    )
    runner.save_exports(field_axis, fitted, per_site, labels, args.out)

    if cfg.verbose_bool:
        print(lmfit.fit_report(result))
    else:
        print(f"nmr-field-fit: reduced chi-square = {result.redchi:.4g}")

    finalize_rendered_figure(rendered, show_plot)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
