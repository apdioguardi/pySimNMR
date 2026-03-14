"""Powder field spectrum fitting CLI built on the shared simulation helper.

NOTE: This module is still under active development. In particular:
  - MCMC-based uncertainty estimation (n_mcmc_steps > 0) is experimental and
    may be slow for large n_samples / many free parameters.
  - The default algorithm ('nelder', Nelder-Mead simplex) is derivative-free
    and recommended for powder ED spectra where finite-difference gradients
    are unreliable due to the stochastic nature of the simulation.
  - Levenberg-Marquardt ('leastsq') is available but not recommended for ED
    powder patterns; use it only with perturbative (2nd-order) simulations or
    when n_samples is very large and the spectrum is smooth.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import lmfit
import numpy as np
import yaml
import pySimNMR
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config_loader import load_python_config
from .powder_field_simulation import PowderFieldConfig, simulate_powder_field
from .plotting import SpectrumPlotData, OverlayTrace, save_spectrum_plot, finalize_rendered_figure
from .plotting_utils import (
    normalize_axis_keys,
    resolve_axis_limits,
    build_parameter_table,
    format_isotope_label,
)
from .plotly_utils import save_lines_html
from .progress import ProgressManager

# Methods that support the epsfcn (finite-difference step) kwarg.
_LM_METHODS = {'leastsq', 'least_squares'}


@dataclass
class ParameterSpec:
    value: Optional[float] = None
    vary: bool = False
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    expr: Optional[str] = None

    @staticmethod
    def parse(entry: Any) -> 'ParameterSpec':
        if isinstance(entry, str):
            return ParameterSpec(expr=entry)
        if not isinstance(entry, list):
            return ParameterSpec(value=float(entry), vary=False)
        if len(entry) == 0:
            raise ValueError("Parameter specification cannot be empty")
        first = entry[0]
        if isinstance(first, str):
            return ParameterSpec(expr=first)
        value = float(first) if first is not None else None
        if len(entry) == 1:
            return ParameterSpec(value=value, vary=False)
        if len(entry) == 2:
            return ParameterSpec(value=value, vary=bool(entry[1]))
        if len(entry) == 4:
            return ParameterSpec(value=value,
                                 vary=bool(entry[1]),
                                 minimum=float(entry[2]),
                                 maximum=float(entry[3]))
        raise ValueError("Parameter specification must have 1, 2, or 4 elements")


class PowderFieldFitConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    exp_data_file: str
    number_of_header_lines: int = 0
    exp_data_delimiter: str = ' '
    missing_values_string: Optional[str] = None
    exp_x_scaling: float = 1.0
    exp_y_scaling: float = 1.0
    exp_y_offset: float = 0.0

    # 'nelder' (Nelder-Mead simplex) is the recommended default for ED powder
    # spectra because the objective function is not smooth enough for gradient-
    # based methods to compute reliable finite-difference Jacobians.
    minimization_algorithm: str = 'nelder'
    epsilon: Optional[float] = None   # only used by leastsq / least_squares
    verbose_bool: bool = False
    max_nfev: Optional[int] = None

    # MCMC uncertainty estimation (experimental).
    # Set n_mcmc_steps > 0 to run an emcee MCMC chain after the primary
    # optimization and report posterior means and standard deviations for each
    # free parameter.  n_mcmc_walkers controls the ensemble size; a value of
    # at least 2 * (number of free parameters) is required by emcee.
    # WARNING: each step calls the full powder ED simulation, so runtimes can
    # be very long for realistic n_samples values.
    n_mcmc_steps: int = 0
    n_mcmc_walkers: int = 100

    isotope_list: List[str]
    f0: float
    site_multiplicity_list: List[Any]
    Ka_list: List[Any]
    Kb_list: List[Any]
    Kc_list: List[Any]
    vc_list: Optional[List[Any]] = None
    vQ_list: Optional[List[Any]] = None
    eta_list: List[Any]
    sim_type: str = 'exact diag'
    min_field: float
    max_field: float
    n_field_points: int = Field(ge=2)
    line_shape_func_list: List[str]
    FWHM_list: List[Any]
    FWHM_vQ_list: List[Any]
    n_samples: float = 1e3
    recalc_random_samples: bool = False
    mtx_elem_min: float = 0.5
    delta_f0: float = 0.015

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

    @model_validator(mode='after')
    def _validate_lengths(self) -> 'PowderFieldFitConfig':
        n_sites = len(self.isotope_list)

        def ensure(name: str) -> None:
            lst = getattr(self, name)
            if len(lst) != n_sites:
                raise ValueError(f"{name} must have {n_sites} entries")

        for key in [
            'site_multiplicity_list', 'Ka_list', 'Kb_list', 'Kc_list',
            'eta_list', 'line_shape_func_list',
            'FWHM_list', 'FWHM_vQ_list',
        ]:
            ensure(key)

        if self.vc_list is None:
            if self.vQ_list is None:
                raise ValueError("Either vc_list or vQ_list must be provided")
            ensure('vQ_list')
        else:
            ensure('vc_list')

        if len(self.plot_legend_width_ratio) == 1:
            self.plot_legend_width_ratio = [self.plot_legend_width_ratio[0], 1.0]
        elif len(self.plot_legend_width_ratio) != 2:
            raise ValueError("plot_legend_width_ratio must contain one or two values")

        return self


@dataclass
class FitParameter:
    name: Optional[str]
    default: Optional[float]


@dataclass
class SiteMetadata:
    isotope: str
    amplitude: FitParameter
    Ka: FitParameter
    Kb: FitParameter
    Kc: FitParameter
    vc: FitParameter
    eta: FitParameter
    FWHM: FitParameter
    FWHM_vQ: FitParameter
    line_shape: str


class PowderFieldFitRunner:
    def __init__(self,
                 cfg: PowderFieldFitConfig,
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
        self.background_params: List[FitParameter] = []
        self.rotation_cache = None
        self._build_parameters()

    def _add_parameter(self, name: str, spec: ParameterSpec) -> FitParameter:
        if spec.expr:
            self._pending_expr[name] = spec.expr
            return FitParameter(name=name, default=None)
        if spec.value is None and not spec.vary:
            return FitParameter(name=None, default=None)
        kwargs: Dict[str, float] = {}
        if spec.minimum is not None:
            kwargs['min'] = spec.minimum
        if spec.maximum is not None:
            kwargs['max'] = spec.maximum
        self.params.add(name,
                        value=spec.value if spec.value is not None else 0.0,
                        vary=spec.vary,
                        **kwargs)
        return FitParameter(name=name, default=None)

    def _build_parameters(self) -> None:
        vc_specs = self.cfg.vc_list or self.cfg.vQ_list
        for idx, isotope in enumerate(self.cfg.isotope_list):
            site = SiteMetadata(
                isotope=isotope,
                amplitude=self._add_parameter(f"amp_{isotope}_{idx}", ParameterSpec.parse(self.cfg.site_multiplicity_list[idx])),
                Ka=self._add_parameter(f"Ka_{isotope}_{idx}", ParameterSpec.parse(self.cfg.Ka_list[idx])),
                Kb=self._add_parameter(f"Kb_{isotope}_{idx}", ParameterSpec.parse(self.cfg.Kb_list[idx])),
                Kc=self._add_parameter(f"Kc_{isotope}_{idx}", ParameterSpec.parse(self.cfg.Kc_list[idx])),
                vc=self._add_parameter(f"vc_{isotope}_{idx}", ParameterSpec.parse(vc_specs[idx])),
                eta=self._add_parameter(f"eta_{isotope}_{idx}", ParameterSpec.parse(self.cfg.eta_list[idx])),
                FWHM=self._add_parameter(f"FWHM_{isotope}_{idx}", ParameterSpec.parse(self.cfg.FWHM_list[idx])),
                FWHM_vQ=self._add_parameter(f"FWHM_vQ_{isotope}_{idx}", ParameterSpec.parse(self.cfg.FWHM_vQ_list[idx])),
                line_shape=self.cfg.line_shape_func_list[idx],
            )
            self.site_meta.append(site)

        self._build_background_params()
        for name, expr in self._pending_expr.items():
            self.params.add(name, expr=expr)

        if self.cfg.sim_type.strip().lower().startswith('exact'):
            cache_sim = pySimNMR.SimNMR('1H')
            self.rotation_cache = cache_sim.random_rotation_matrices(
                self.cfg.isotope_list,
                recalc_random_samples=self.cfg.recalc_random_samples,
                n_samples=int(self.cfg.n_samples),
            )

    def _build_background_params(self) -> None:
        labels = ['bg_offset', 'bg_slope', 'bg_center', 'bg_width', 'bg_intensity']
        params: List[FitParameter] = []
        for idx, entry in enumerate(self.cfg.background_list[:3]):
            spec = ParameterSpec.parse(entry)
            params.append(self._add_parameter(labels[idx], spec))
        self.background_params = params

    def _value(self, param: FitParameter, pars: lmfit.Parameters) -> Optional[float]:
        if param.name and param.name in pars:
            return float(pars[param.name].value)
        return param.default

    def _background_values(self, pars: lmfit.Parameters) -> List[float]:
        vals = []
        for param in self.background_params:
            val = self._value(param, pars)
            if val is not None:
                vals.append(val)
        return vals

    def _build_config(self, pars: lmfit.Parameters) -> PowderFieldConfig:
        def val(site_param: FitParameter) -> float:
            if site_param.name:
                return float(pars[site_param.name].value)
            return site_param.default if site_param.default is not None else 0.0

        mult = [val(site.amplitude) or 1.0 for site in self.site_meta]
        Ka = [val(site.Ka) for site in self.site_meta]
        Kb = [val(site.Kb) for site in self.site_meta]
        Kc = [val(site.Kc) for site in self.site_meta]
        vc = [val(site.vc) for site in self.site_meta]
        eta = [val(site.eta) for site in self.site_meta]
        FWHM = [val(site.FWHM) for site in self.site_meta]
        FWHM_vQ = [val(site.FWHM_vQ) for site in self.site_meta]

        sim_type = self.cfg.sim_type.strip().lower()
        if sim_type in {'exact diag', 'exact_diag', 'exact-diag', 'ed'}:
            mode = 'ed'
        elif sim_type in {'2nd order', '2nd-order', '2nd', 'perturbative'}:
            mode = 'perturbative'
        else:
            mode = 'ed'

        cfg_dict = {
            'isotope_list': self.cfg.isotope_list,
            'site_multiplicity_list': mult,
            'Ka_list': Ka,
            'Kb_list': Kb,
            'Kc_list': Kc,
            'vc_list': vc,
            'eta_list': eta,
            'f0': self.cfg.f0,
            'sim_type': mode,
            'min_field': self.cfg.min_field,
            'max_field': self.cfg.max_field,
            'n_field_points': self.cfg.n_field_points,
            'convolution_function_list': [site.line_shape for site in self.site_meta],
            'conv_FWHM_list': FWHM,
            'conv_vQ_FWHM_list': FWHM_vQ,
            'mtx_elem_min': self.cfg.mtx_elem_min,
            'delta_f0': self.cfg.delta_f0,
            'n_samples': self.cfg.n_samples,
            'recalc_random_samples': self.cfg.recalc_random_samples,
        }
        return PowderFieldConfig(**cfg_dict)

    def simulate(self, pars: lmfit.Parameters, *, show_progress: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[str]]:
        cfg = self._build_config(pars)
        sim_progress = None
        if show_progress and self.progress is not None and self.progress.enabled:
            sim_progress = self.progress
        result = simulate_powder_field(cfg, progress=sim_progress)
        stack = np.vstack(result.per_site)
        total = stack.sum(axis=0)
        max_val = np.nanmax(total) or 1.0
        per_site_norm = [spec / max_val for spec in result.per_site]
        total_norm = total / max_val
        spectrum = self._apply_background(result.field_axis, total_norm, self._background_values(pars))
        return result.field_axis, spectrum, per_site_norm, result.site_labels

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
        """Run the primary minimization.

        Uses the algorithm specified by cfg.minimization_algorithm.
        epsfcn (finite-difference step size) is only forwarded to gradient-
        based LM methods; derivative-free methods ignore it.
        """
        minimizer = lmfit.Minimizer(self.residual, self.params)
        # Use an indeterminate bar (total=None) because the number of function
        # evaluations is not known ahead of time; derivative-free methods like
        # Nelder-Mead routinely exceed a naive 200-iteration estimate.
        if self.progress is not None and self.progress.enabled:
            self._fit_bar = self.progress.bar(total=None, desc='Fitting (powder field)')
        else:
            self._fit_bar = None

        method = self.cfg.minimization_algorithm.strip().lower()

        # epsfcn is a Levenberg-Marquardt-specific kwarg; passing it to
        # derivative-free methods raises a TypeError in scipy.
        extra_kwargs: Dict[str, Any] = {}
        if method in _LM_METHODS and self.cfg.epsilon is not None:
            extra_kwargs['epsfcn'] = self.cfg.epsilon

        try:
            result = minimizer.minimize(method=method,
                                        max_nfev=max_nfev or self.cfg.max_nfev,
                                        **extra_kwargs)
        finally:
            if self._fit_bar is not None:
                self._fit_bar.complete()
                self._fit_bar.close()
                self._fit_bar = None
        return result

    def run_mcmc(self,
                 result: lmfit.MinimizerResult,
                 n_steps: int,
                 n_walkers: int) -> lmfit.MinimizerResult:
        """Run an emcee MCMC chain starting from the best-fit parameters.

        This is an experimental feature intended to provide posterior
        uncertainty estimates for the free NMR parameters.  Each MCMC step
        calls the full powder ED simulation, so total runtime scales as
        n_steps * n_walkers * (time per simulation).  Use small n_steps
        (e.g. 100-300) for exploratory runs.

        Returns an lmfit MinimizerResult whose .params have the median
        posterior values and whose .flatchain can be used for corner plots.
        """
        import sys
        print(
            "[pySimNMR] Starting MCMC uncertainty estimation "
            f"({n_walkers} walkers, {n_steps} steps). "
            "This may take a long time for large n_samples values.",
            file=sys.stderr,
        )
        # emcee requires at least 2 * n_free walkers; enforce a safe minimum.
        n_free = sum(1 for p in result.params.values() if p.vary)
        min_walkers = max(n_walkers, 2 * n_free + 2)
        if min_walkers > n_walkers:
            print(
                f"[pySimNMR] Increasing n_mcmc_walkers from {n_walkers} to "
                f"{min_walkers} (emcee requires >= 2 * n_free_params + 2).",
                file=sys.stderr,
            )

        minimizer = lmfit.Minimizer(self.residual, result.params)
        mcmc_result = minimizer.minimize(
            method='emcee',
            params=result.params,
            nwalkers=min_walkers,
            steps=n_steps,
            burn=max(1, n_steps // 5),   # discard first 20% as burn-in
            thin=1,
            is_weighted=False,
        )
        return mcmc_result

    def save_exports(self,
                     field_axis: np.ndarray,
                     fitted: np.ndarray,
                     per_site: List[np.ndarray],
                     labels: List[str],
                     out_base: Path) -> None:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_base.with_suffix('.txt'), np.column_stack((field_axis, fitted)))
        try:
            save_lines_html(field_axis, fitted, xlab='Field (T)',
                            ylab='Intensity', title='Powder field fit',
                            out_html=out_base.with_suffix('.html'))
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


def _load_experimental_data(cfg: PowderFieldFitConfig, config_path: Path) -> Tuple[np.ndarray, np.ndarray]:
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
              cfg: PowderFieldFitConfig,
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
    vc_values = cfg.vc_list if cfg.vc_list is not None else cfg.vQ_list
    per_site_params = {
        r'$m_i$': cfg.site_multiplicity_list,
        r'$K_a$ (%)': cfg.Ka_list,
        r'$K_b$ (%)': cfg.Kb_list,
        r'$K_c$ (%)': cfg.Kc_list,
        r'$\nu_c$ (MHz)': vc_values,
        r'$\eta$': cfg.eta_list,
        'Line shape': cfg.line_shape_func_list,
        'FWHM (T)': cfg.FWHM_list,
        r'$\mathrm{FWHM}_{d\nu_Q}$ (T)': cfg.FWHM_vQ_list,
    }
    global_params = {
        r'$f_0$ (MHz)': cfg.f0,
        'Simulation': cfg.sim_type,
        'Field range (T)': f"{cfg.min_field} - {cfg.max_field}",
        'n field points': cfg.n_field_points,
        'Matrix elem. min': cfg.mtx_elem_min,
        r'$\Delta f_0$ (MHz)': cfg.delta_f0,
        'n samples': cfg.n_samples,
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
        title='Powder field fit',
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


def _print_fit_summary(result: lmfit.MinimizerResult,
                       mcmc_result: Optional[lmfit.MinimizerResult],
                       cfg: PowderFieldFitConfig,
                       verbose: bool) -> None:
    """Print fit quality and parameter values/uncertainties to stdout."""
    import sys

    method = cfg.minimization_algorithm.strip().lower()
    is_derivative_free = method not in _LM_METHODS

    if verbose:
        print(lmfit.fit_report(result))
    else:
        print(f"nmr-field-powder-fit: reduced chi-square = {result.redchi:.4g}")

    # Warn when the primary optimizer cannot provide covariance-based error bars.
    if is_derivative_free and mcmc_result is None:
        print(
            f"[pySimNMR] Note: '{method}' is a derivative-free method and does not "
            "produce covariance-based parameter uncertainties.\n"
            "  To obtain error bars, set n_mcmc_steps > 0 in your config to run\n"
            "  an MCMC chain after optimization (experimental, slow).",
            file=sys.stderr,
        )

    if mcmc_result is not None:
        _print_mcmc_summary(mcmc_result)


def _print_mcmc_summary(mcmc_result: lmfit.MinimizerResult) -> None:
    """Print posterior means and standard deviations from the MCMC chain."""
    print("\nMCMC posterior parameter estimates (mean +/- std):")
    print("-" * 55)
    # flatchain is a dict of {param_name: array_of_samples}
    if hasattr(mcmc_result, 'flatchain') and mcmc_result.flatchain is not None:
        for name, samples in mcmc_result.flatchain.items():
            mean = float(np.mean(samples))
            std = float(np.std(samples))
            print(f"  {name:<30s}  {mean:>12.6g}  +/-  {std:.6g}")
    else:
        # Fall back to lmfit param stderr if flatchain is unavailable.
        for name, param in mcmc_result.params.items():
            if param.vary:
                stderr = param.stderr if param.stderr is not None else float('nan')
                print(f"  {name:<30s}  {param.value:>12.6g}  +/-  {stderr:.6g}")
    print("-" * 55)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Fit powder field spectra using lmfit. '
                    '(NOTE: this script is under active development.)'
    )
    parser.add_argument('--config', type=Path, required=True,
                        help='Configuration file (.py/.yml/.json).')
    parser.add_argument('--out', type=Path, default=Path('output/field_powder_fit'),
                        help='Output stem for exported data/plots (default: %(default)s).')
    parser.add_argument('--save-fig', type=Path,
                        help='Optional explicit figure path; defaults to OUT with fig-format extension.')
    parser.add_argument('--fig-format', default='png', choices=['png','svg','pdf'],
                        help='Figure format when --save-fig is omitted.')
    parser.add_argument('--dpi', type=int, default=150, help='Figure DPI.')
    parser.add_argument('--no-show', action='store_true', help='Disable GUI windows.')
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
    cfg = PowderFieldFitConfig(**axis_cfg)
    show_plot = not args.no_show
    exp_x, exp_y = _load_experimental_data(cfg, args.config.parent)
    runner = PowderFieldFitRunner(cfg, exp_x, exp_y, progress=progress)

    initial_guess = None
    if cfg.plot_initial_guess_bool:
        initial_guess = runner.simulate(runner.params, show_progress=True)[:2]

    result = runner.fit(max_nfev=args.max_nfev)
    field_axis, fitted, per_site, labels = runner.simulate(result.params, show_progress=True)

    # Optional MCMC uncertainty estimation (experimental).
    mcmc_result = None
    if cfg.n_mcmc_steps > 0:
        mcmc_result = runner.run_mcmc(result, cfg.n_mcmc_steps, cfg.n_mcmc_walkers)

    _print_fit_summary(result, mcmc_result, cfg, cfg.verbose_bool)

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

    finalize_rendered_figure(rendered, show_plot)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
