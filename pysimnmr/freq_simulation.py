from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import pySimNMR
from .progress import ProgressManager


class HintVector(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class SingleCrystalFreqConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    isotope_list: List[str]
    site_multiplicity_list: List[float] = Field(default_factory=list)
    Ka_list: List[float]
    Kb_list: List[float]
    Kc_list: List[float]
    va_list: Optional[List[Optional[float]]] = None
    vb_list: Optional[List[Optional[float]]] = None
    vc_list: List[float]
    eta_list: List[float]
    H0_T: float = Field(alias='H0')
    Hint_list: List[HintVector] = Field(default_factory=list)

    # Orientation controls
    phi_deg_list: Optional[List[float]] = None
    theta_deg_list: Optional[List[float]] = None
    phi_z_deg_list: Optional[List[float]] = None
    theta_x_prime_deg_list: Optional[List[float]] = None
    psi_z_prime_deg_list: Optional[List[float]] = None

    sim_mode: Literal['ed', 'perturbative'] = Field(default='ed', alias='sim_type')
    freq_min_MHz: float = Field(alias='min_freq')
    freq_max_MHz: float = Field(alias='max_freq')
    freq_points: int = Field(alias='n_freq_points', ge=2)
    convolution_function_list: List[str]
    conv_FWHM_list: List[float]
    conv_vQ_FWHM_list: List[float]
    matrix_element_cutoff: float = 0.5

    # Plot/export options kept for CLI convenience
    bgd: List[float] = Field(default_factory=lambda: [0.0])
    exp_data_file: str = ''
    number_of_header_lines: int = 0
    exp_data_delimiter: str = ','
    missing_values_string: Optional[str] = 'nan'
    exp_x_scaling: float = 1.0
    exp_y_scaling: float = 1.0
    exp_x_offset: float = 0.0
    exp_y_offset: float = 0.0

    plot_individual_bool: bool = True
    plot_sum_bool: bool = True
    plot_legend_width_ratio: List[float] = Field(default_factory=lambda: [3.25, 1.0])
    x_axis_min: Optional[float] = None
    x_axis_max: Optional[float] = None
    y_axis_min: Optional[float] = None
    y_axis_max: Optional[float] = None

    sim_export_file: str = ''

    @field_validator('Hint_list', mode='before')
    @classmethod
    def _coerce_hint(cls, value):
        if value is None:
            return value
        coerced = []
        for entry in value:
            if isinstance(entry, dict):
                coerced.append(entry)
            else:
                x, y, z = (list(entry) + [0.0, 0.0, 0.0])[:3]
                coerced.append({'x': float(x), 'y': float(y), 'z': float(z)})
        return coerced

    @model_validator(mode='after')
    def _check_lengths(self) -> 'SingleCrystalFreqConfig':
        n_sites = len(self.isotope_list)

        if not self.site_multiplicity_list:
            self.site_multiplicity_list = [1.0] * n_sites
        if self.va_list is None:
            self.va_list = [None] * n_sites
        if self.vb_list is None:
            self.vb_list = [None] * n_sites
        if not self.Hint_list:
            self.Hint_list = [HintVector(x=0.0, y=0.0, z=0.0) for _ in range(n_sites)]
        if not self.plot_legend_width_ratio:
            self.plot_legend_width_ratio = [3.25, 1.0]
        elif len(self.plot_legend_width_ratio) == 1:
            self.plot_legend_width_ratio = [self.plot_legend_width_ratio[0], 1.0]

        if self.sim_mode == 'perturbative':
            if not self.phi_deg_list:
                self.phi_deg_list = [0.0] * n_sites
            if not self.theta_deg_list:
                self.theta_deg_list = [0.0] * n_sites
        else:
            if not self.phi_z_deg_list:
                self.phi_z_deg_list = [0.0] * n_sites
            if not self.theta_x_prime_deg_list:
                self.theta_x_prime_deg_list = [0.0] * n_sites
            if not self.psi_z_prime_deg_list:
                self.psi_z_prime_deg_list = [0.0] * n_sites

        def _ensure_length(name: str, lst: Optional[List]) -> None:
            if lst is None:
                return
            if len(lst) != n_sites:
                raise ValueError(f"{name} must have {n_sites} entries (got {len(lst)})")

        for name in [
            'site_multiplicity_list',
            'Ka_list',
            'Kb_list',
            'Kc_list',
            'vc_list',
            'eta_list',
            'Hint_list',
            'convolution_function_list',
            'conv_FWHM_list',
            'conv_vQ_FWHM_list',
        ]:
            lst = getattr(self, name)
            _ensure_length(name, lst)

        _ensure_length('va_list', self.va_list)
        _ensure_length('vb_list', self.vb_list)

        if self.sim_mode == 'perturbative':
            _ensure_length('phi_deg_list', self.phi_deg_list)
            _ensure_length('theta_deg_list', self.theta_deg_list)
        else:
            _ensure_length('phi_z_deg_list', self.phi_z_deg_list)
            _ensure_length('theta_x_prime_deg_list', self.theta_x_prime_deg_list)
            _ensure_length('psi_z_prime_deg_list', self.psi_z_prime_deg_list)

        return self


@dataclass
class SingleCrystalFreqResult:
    freq_axis: np.ndarray
    per_site: List[np.ndarray]
    site_labels: List[str]
    gamma_list: List[float]


def _ensure_orientation_array(deg_value: float) -> np.ndarray:
    return np.array([np.deg2rad(deg_value)], dtype=float)


def _start_bar(progress: ProgressManager | None, total: int, desc: str):
    if progress is None or not progress.enabled or total <= 0:
        return None
    return progress.bar(total=total, desc=desc)


def _finish_bar(bar) -> None:
    if bar is None:  # pragma: no cover - trivial guard
        return
    bar.complete()
    bar.close()


def simulate_single_crystal_freq(cfg: SingleCrystalFreqConfig,
                                 progress: ProgressManager | None = None) -> SingleCrystalFreqResult:
    per_site: List[np.ndarray] = []
    labels: List[str] = []
    gamma_vals: List[float] = []
    freq_axis: Optional[np.ndarray] = None
    bar = _start_bar(progress, len(cfg.isotope_list), 'Freq sites')

    for idx, isotope in enumerate(cfg.isotope_list):
        sim = pySimNMR.SimNMR(isotope)
        gamma_vals.append(sim.isotope_data_dict[isotope]["gamma"])
        multiplicity = float(cfg.site_multiplicity_list[idx])
        va = cfg.va_list[idx] if cfg.va_list else None
        vb = cfg.vb_list[idx] if cfg.vb_list else None
        hint_vec = cfg.Hint_list[idx].as_tuple()

        if cfg.sim_mode == 'perturbative':
            theta_deg = cfg.theta_deg_list[idx] if cfg.theta_deg_list else 0.0
            phi_deg = cfg.phi_deg_list[idx] if cfg.phi_deg_list else 0.0
            phi_array = np.array([np.deg2rad(phi_deg)], dtype=float)
            theta_array = np.array([np.deg2rad(theta_deg)], dtype=float)
            spec = sim.sec_ord_freq_spec(
                I0=sim.isotope_data_dict[isotope]["I0"],
                gamma=gamma_vals[-1],
                H0=cfg.H0_T,
                Ka=cfg.Ka_list[idx],
                Kb=cfg.Kb_list[idx],
                Kc=cfg.Kc_list[idx],
                vQ=cfg.vc_list[idx],
                eta=cfg.eta_list[idx],
                theta_array=theta_array,
                phi_array=phi_array,
                nbins=cfg.freq_points,
                min_freq=cfg.freq_min_MHz,
                max_freq=cfg.freq_max_MHz,
                broadening_func=cfg.convolution_function_list[idx],
                FWHM_MHz=cfg.conv_FWHM_list[idx],
            )
        else:
            phi_z = cfg.phi_z_deg_list[idx] if cfg.phi_z_deg_list else 0.0
            theta_x = cfg.theta_x_prime_deg_list[idx] if cfg.theta_x_prime_deg_list else 0.0
            psi_z = cfg.psi_z_prime_deg_list[idx] if cfg.psi_z_prime_deg_list else 0.0
            phi_arr = _ensure_orientation_array(phi_z)
            theta_arr = _ensure_orientation_array(theta_x)
            psi_arr = _ensure_orientation_array(psi_z)
            r, ri = sim.generate_r_matrices(phi_arr, theta_arr, psi_arr)
            SR, SRi = sim.generate_r_spin_matrices(phi_arr, theta_arr, psi_arr)
            rotation_matrices = (r, ri, SR, SRi)
            try:
                spec = sim.freq_spec_edpp(
                    H0=cfg.H0_T,
                    Ka=cfg.Ka_list[idx],
                    Kb=cfg.Kb_list[idx],
                    Kc=cfg.Kc_list[idx],
                    va=va,
                    vb=vb,
                    vc=cfg.vc_list[idx],
                    eta=cfg.eta_list[idx],
                    rotation_matrices=rotation_matrices,
                    Hinta=hint_vec[0],
                    Hintb=hint_vec[1],
                    Hintc=hint_vec[2],
                    mtx_elem_min=cfg.matrix_element_cutoff,
                    FWHM_MHz=cfg.conv_FWHM_list[idx],
                    FWHM_dvQ_MHz=cfg.conv_vQ_FWHM_list[idx],
                    min_freq=cfg.freq_min_MHz,
                    max_freq=cfg.freq_max_MHz,
                    nbins=cfg.freq_points,
                    save_files_bool=False,
                    out_filename='',
                )
            except Exception:
                theta_array = np.array([np.deg2rad(theta_x)], dtype=float)
                phi_array = np.array([np.deg2rad(phi_z)], dtype=float)
                spec = sim.sec_ord_freq_spec(
                    I0=sim.isotope_data_dict[isotope]["I0"],
                    gamma=gamma_vals[-1],
                    H0=cfg.H0_T,
                    Ka=cfg.Ka_list[idx],
                    Kb=cfg.Kb_list[idx],
                    Kc=cfg.Kc_list[idx],
                    vQ=cfg.vc_list[idx],
                    eta=cfg.eta_list[idx],
                    theta_array=theta_array,
                    phi_array=phi_array,
                    nbins=cfg.freq_points,
                    min_freq=cfg.freq_min_MHz,
                    max_freq=cfg.freq_max_MHz,
                    broadening_func=cfg.convolution_function_list[idx],
                    FWHM_MHz=cfg.conv_FWHM_list[idx],
                )
        if freq_axis is None:
            freq_axis = spec[:, 0]
        site_data = spec[:, 1] * multiplicity
        per_site.append(site_data)
        labels.append(f"{isotope}_{idx}")
        if bar is not None:
            bar.update(1)

    if freq_axis is None:
        raise RuntimeError("No spectra were generated; check configuration.")
    _finish_bar(bar)

    return SingleCrystalFreqResult(
        freq_axis=freq_axis,
        per_site=per_site,
        site_labels=labels,
        gamma_list=gamma_vals,
    )


__all__ = [
    'SingleCrystalFreqConfig',
    'SingleCrystalFreqResult',
    'simulate_single_crystal_freq',
]
