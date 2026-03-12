from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import pySimNMR
from .progress import ProgressManager


class SingleCrystalFieldConfig(BaseModel):
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
    Hinta_list: List[float]
    Hintb_list: List[float]
    Hintc_list: List[float]
    phi_z_deg_list: Optional[List[float]] = None
    theta_x_prime_deg_list: Optional[List[float]] = None
    psi_z_prime_deg_list: Optional[List[float]] = None
    phi_deg_list: Optional[List[float]] = None
    theta_deg_list: Optional[List[float]] = None

    sim_mode: Literal['ed', 'perturbative'] = Field(default='ed', alias='sim_type')
    f0_MHz: float = Field(alias='f0')
    field_min_T: float = Field(alias='min_field')
    field_max_T: float = Field(alias='max_field')
    field_points: int = Field(alias='n_field_points', ge=2)
    convolution_function_list: List[str]
    conv_FWHM_list: List[float]
    conv_vQ_FWHM_list: List[float]
    mtx_elem_min: float = 0.1
    delta_f0_MHz: float = Field(default=0.001, alias='delta_f0')

    @model_validator(mode='after')
    def _validate_lengths(self) -> 'SingleCrystalFieldConfig':
        n_sites = len(self.isotope_list)

        def ensure(name: str, default: Optional[float] = None):
            lst = getattr(self, name)
            if lst is None:
                if default is not None:
                    setattr(self, name, [default] * n_sites)
                    return
                raise ValueError(f"{name} must be set")
            if len(lst) != n_sites:
                raise ValueError(f"{name} must have {n_sites} entries")

        for name in [
            'site_multiplicity_list', 'Ka_list', 'Kb_list', 'Kc_list',
            'vc_list', 'eta_list', 'Hinta_list', 'Hintb_list', 'Hintc_list',
            'conv_FWHM_list', 'conv_vQ_FWHM_list', 'convolution_function_list',
        ]:
            ensure(name)

        if self.va_list is None:
            self.va_list = [None] * n_sites
        else:
            ensure('va_list')
        if self.vb_list is None:
            self.vb_list = [None] * n_sites
        else:
            ensure('vb_list')

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

        return self


@dataclass
class SingleCrystalFieldResult:
    field_axis: np.ndarray
    per_site: List[np.ndarray]
    site_labels: List[str]
    gamma_list: List[float]


def _start_bar(progress: ProgressManager | None, total: int, desc: str):
    if progress is None or not progress.enabled or total <= 0:
        return None
    return progress.bar(total=total, desc=desc)


def _finish_bar(bar) -> None:
    if bar is None:  # pragma: no cover - trivial guard
        return
    bar.complete()
    bar.close()


def simulate_single_crystal_field(cfg: SingleCrystalFieldConfig,
                                  progress: ProgressManager | None = None) -> SingleCrystalFieldResult:
    per_site = []
    labels = []
    gamma_vals = []
    field_axis: Optional[np.ndarray] = None

    H0_axis = np.linspace(cfg.field_min_T, cfg.field_max_T, cfg.field_points)
    bar = _start_bar(progress, len(cfg.isotope_list), 'Field sites')

    for idx, isotope in enumerate(cfg.isotope_list):
        sim = pySimNMR.SimNMR(isotope)
        gamma_vals.append(sim.isotope_data_dict[isotope]["gamma"])
        mult = cfg.site_multiplicity_list[idx]

        if cfg.sim_mode == 'perturbative':
            phi = np.deg2rad(cfg.phi_deg_list[idx])
            theta = np.deg2rad(cfg.theta_deg_list[idx])
            spec = sim.sec_ord_field_spec(
                I0=sim.isotope_data_dict[isotope]["I0"],
                gamma=gamma_vals[-1],
                f0=cfg.f0_MHz,
                Ka=cfg.Ka_list[idx],
                Kb=cfg.Kb_list[idx],
                Kc=cfg.Kc_list[idx],
                vQ=cfg.vc_list[idx],
                eta=cfg.eta_list[idx],
                theta_array=np.array([theta]),
                phi_array=np.array([phi]),
                nbins=cfg.field_points,
                min_field=cfg.field_min_T,
                max_field=cfg.field_max_T,
                broadening_func=cfg.convolution_function_list[idx],
                FWHM_T=cfg.conv_FWHM_list[idx],
            )
            spec_axis = spec[:, 0]
            spec_int = spec[:, 1]
        else:
            phi_z = np.deg2rad(cfg.phi_z_deg_list[idx])
            theta_xp = np.deg2rad(cfg.theta_x_prime_deg_list[idx])
            psi_zp = np.deg2rad(cfg.psi_z_prime_deg_list[idx])
            phi_arr = np.full((cfg.field_points,), phi_z)
            theta_arr = np.full((cfg.field_points,), theta_xp)
            psi_arr = np.full((cfg.field_points,), psi_zp)
            r, ri = sim.generate_r_matrices(phi_arr, theta_arr, psi_arr)
            SR, SRi = sim.generate_r_spin_matrices(phi_arr, theta_arr, psi_arr)
            rotation_matrices = (r, ri, SR, SRi)
            spec = sim.field_spec_edpp(
                f0=cfg.f0_MHz,
                H0_array=H0_axis,
                Ka=cfg.Ka_list[idx],
                Kb=cfg.Kb_list[idx],
                Kc=cfg.Kc_list[idx],
                va=cfg.va_list[idx],
                vb=cfg.vb_list[idx],
                vc=cfg.vc_list[idx],
                eta=cfg.eta_list[idx],
                rotation_matrices=rotation_matrices,
                Hinta=cfg.Hinta_list[idx],
                Hintb=cfg.Hintb_list[idx],
                Hintc=cfg.Hintc_list[idx],
                mtx_elem_min=cfg.mtx_elem_min,
                min_field=cfg.field_min_T,
                max_field=cfg.field_max_T,
                delta_f0=cfg.delta_f0_MHz,
                FWHM_T=cfg.conv_FWHM_list[idx],
                FWHM_dvQ_T=cfg.conv_vQ_FWHM_list[idx],
                broadening_func=cfg.convolution_function_list[idx],
                baseline=0.5,
                nbins=cfg.field_points,
                save_files_bool=False,
                out_filename='',
            )
            spec_axis = spec[:, 0]
            spec_int = spec[:, 1]

        if field_axis is None:
            field_axis = spec_axis
        site_data = spec_int * mult
        per_site.append(site_data)
        labels.append(f"{isotope}_{idx}")
        if bar is not None:
            bar.update(1)

    if field_axis is None:
        raise RuntimeError("No spectra generated; check configuration.")
    _finish_bar(bar)

    return SingleCrystalFieldResult(
        field_axis=field_axis,
        per_site=per_site,
        site_labels=labels,
        gamma_list=gamma_vals,
    )


__all__ = [
    'SingleCrystalFieldConfig',
    'SingleCrystalFieldResult',
    'simulate_single_crystal_field',
]
