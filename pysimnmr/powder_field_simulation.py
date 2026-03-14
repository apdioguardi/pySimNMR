from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pySimNMR
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .plotting_utils import load_experimental_data
from .progress import ProgressManager


class PowderFieldConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='allow')

    isotope_list: List[str]
    site_multiplicity_list: List[float]
    Ka_list: List[float]
    Kb_list: List[float]
    Kc_list: List[float]
    vc_list: Optional[List[float]] = None
    vQ_list: Optional[List[float]] = None
    eta_list: Optional[List[Optional[float]]] = None
    va_list: Optional[List[Optional[float]]] = None
    vb_list: Optional[List[Optional[float]]] = None
    f0_MHz: float = Field(alias='f0')

    sim_mode: Literal['ed', 'perturbative'] = Field(default='ed', alias='sim_type')
    min_field: float
    max_field: float
    field_points: int = Field(alias='n_field_points', ge=2)
    convolution_function_list: List[str]
    conv_FWHM_list: List[float]
    conv_vQ_FWHM_list: List[float]
    mtx_elem_min: float = 0.5
    delta_f0: float = Field(default=0.015, alias='delta_f0')
    recalc_random_samples: bool = False
    n_samples: int = 1000
    exp_data_file: Optional[str] = None
    exp_data_delimiter: str = Field(default=' ', alias='exp_data_delimiter')
    number_of_header_lines: int = 0
    exp_x_scaling: float = 1.0
    exp_y_scaling: float = 1.0
    exp_y_offset: float = 0.0

    @model_validator(mode='after')
    def _validate_lengths(self) -> 'PowderFieldConfig':
        n_sites = len(self.isotope_list)

        def ensure(name: str):
            lst = getattr(self, name)
            if len(lst) != n_sites:
                raise ValueError(f"{name} must have {n_sites} entries")

        for key in [
            'site_multiplicity_list', 'Ka_list', 'Kb_list', 'Kc_list',
            'convolution_function_list',
            'conv_FWHM_list', 'conv_vQ_FWHM_list',
        ]:
            ensure(key)

        if self.eta_list is None:
            self.eta_list = [None] * n_sites
        else:
            ensure('eta_list')

        if self.va_list is None:
            self.va_list = [None] * n_sites
        else:
            ensure('va_list')

        if self.vb_list is None:
            self.vb_list = [None] * n_sites
        else:
            ensure('vb_list')

        if self.vQ_list is not None:
            if len(self.vQ_list) != n_sites:
                raise ValueError(f"vQ_list must have {n_sites} entries")

        va_vals = self.va_list or [None] * n_sites
        vb_vals = self.vb_list or [None] * n_sites
        eta_vals = self.eta_list or [None] * n_sites
        derived_vc: List[Optional[float]] = [None] * n_sites
        derived_eta: List[Optional[float]] = [None] * n_sites
        for idx, (va, vb) in enumerate(zip(va_vals, vb_vals)):
            if va is not None and vb is not None:
                vc_from_tensor = -float(va + vb)
                derived_vc[idx] = vc_from_tensor
                if vc_from_tensor:
                    derived_eta[idx] = float(va - vb) / vc_from_tensor

        vc_values = self.vc_list if self.vc_list is not None else [None] * n_sites
        resolved_vc: List[float] = []
        for idx in range(n_sites):
            vc = vc_values[idx] if vc_values else None
            if vc is None:
                if derived_vc[idx] is not None:
                    vc = derived_vc[idx]
                elif self.vQ_list is not None:
                    eta_for_vq = eta_vals[idx] if eta_vals[idx] is not None else derived_eta[idx]
                    if eta_for_vq is None:
                        raise ValueError(
                            f"Site {idx}: eta is required to convert vQ -> vc; "
                            "provide eta_list or va/vb components."
                        )
                    denom = np.sqrt(1 - (eta_for_vq ** 2) / 3)
                    vc = float(self.vQ_list[idx]) / denom if denom else float(self.vQ_list[idx])
                else:
                    raise ValueError(
                        f"Site {idx}: provide vc_list, vQ_list, or (va_list & vb_list) to define the EFG tensor."
                    )
            resolved_vc.append(float(vc))
        self.vc_list = resolved_vc

        resolved_eta: List[float] = []
        for idx in range(n_sites):
            eta = eta_vals[idx]
            if eta is None:
                eta = derived_eta[idx]
            if eta is None:
                raise ValueError(
                    f"Site {idx}: eta is required; provide eta_list or va/vb components."
                )
            resolved_eta.append(float(eta))
        self.eta_list = resolved_eta

        if len(self.convolution_function_list) == 1 and n_sites > 1:
            self.convolution_function_list = self.convolution_function_list * n_sites

        return self


@dataclass
class PowderFieldResult:
    field_axis: np.ndarray
    per_site: List[np.ndarray]
    site_labels: List[str]
    exp_x: Optional[np.ndarray] = None
    exp_y: Optional[np.ndarray] = None


def simulate_powder_field(cfg: PowderFieldConfig,
                          progress: ProgressManager | None = None) -> PowderFieldResult:
    per_site = []
    labels = []

    H0_axis = np.linspace(cfg.min_field, cfg.max_field, cfg.field_points)
    n_sites = len(cfg.isotope_list)

    # Single persistent bar whose total spans all field points across all sites,
    # giving an accurate ETA.  The description is updated as we move between
    # phases (rotation-cache generation, then per-site solving).
    main_bar = None
    if progress is not None:
        if cfg.sim_mode == 'ed':
            bar_total = len(H0_axis) * n_sites
        else:
            bar_total = int(cfg.n_samples) * n_sites
        if bar_total > 0:
            main_bar = progress.bar(total=bar_total, desc='Field powder simulation')

    rotation_cache = None
    try:
        if cfg.sim_mode == 'ed':
            if main_bar is not None:
                main_bar.set_description('Generating rotation matrices')
            cache_sim = pySimNMR.SimNMR('1H')
            rotation_cache = cache_sim.random_rotation_matrices(
                cfg.isotope_list,
                recalc_random_samples=cfg.recalc_random_samples,
                n_samples=cfg.n_samples,
            )

        for idx, isotope in enumerate(cfg.isotope_list):
            if main_bar is not None:
                main_bar.set_description(f'Field powder: {isotope} ({idx + 1}/{n_sites})')
            sim = pySimNMR.SimNMR(isotope)
            va = cfg.va_list[idx] if cfg.va_list else None
            vb = cfg.vb_list[idx] if cfg.vb_list else None
            if cfg.sim_mode == 'ed':
                I0_string = sim.isotope_data_dict[isotope]['I0_string']
                r = rotation_cache['real_space']['r']
                ri = rotation_cache['real_space']['ri']
                r_spin = rotation_cache['spin_space'][f'I0={I0_string}']['r_spin']
                ri_spin = rotation_cache['spin_space'][f'I0={I0_string}']['ri_spin']
                spec = sim.field_spec_edpp_loop_parallel(
                    f0=cfg.f0_MHz,
                    H0_array=H0_axis,
                    Ka=cfg.Ka_list[idx],
                    Kb=cfg.Kb_list[idx],
                    Kc=cfg.Kc_list[idx],
                    va=va,
                    vb=vb,
                    vc=cfg.vc_list[idx],
                    eta=cfg.eta_list[idx],
                    rotation_matrices=(r, ri, r_spin, ri_spin),
                    Hinta=0.0,
                    Hintb=0.0,
                    Hintc=0.0,
                    mtx_elem_min=cfg.mtx_elem_min,
                    min_field=cfg.min_field,
                    max_field=cfg.max_field,
                    delta_f0=cfg.delta_f0,
                    FWHM=cfg.conv_FWHM_list[idx],
                    FWHM_dvQ=cfg.conv_vQ_FWHM_list[idx],
                    broadening_func=cfg.convolution_function_list[idx],
                    save_files_bool=False,
                    out_filename='',
                    progress_bar=main_bar,
                )
            else:
                spec = sim.sec_ord_field_pp(
                    I0=sim.isotope_data_dict[isotope]['I0'],
                    gamma=sim.isotope_data_dict[isotope]['gamma'],
                    f0=cfg.f0_MHz,
                    Ka=cfg.Ka_list[idx],
                    Kb=cfg.Kb_list[idx],
                    Kc=cfg.Kc_list[idx],
                    vQ=cfg.vc_list[idx],
                    eta=cfg.eta_list[idx],
                    nrands=int(cfg.n_samples),
                    nbins=cfg.field_points,
                    min_field=cfg.min_field,
                    max_field=cfg.max_field,
                    broadening_func=cfg.convolution_function_list[idx],
                    FWHM_T=cfg.conv_FWHM_list[idx],
                    progress_bar=main_bar,
                )
            spec[:, 1] *= cfg.site_multiplicity_list[idx]
            per_site.append(spec[:, 1])
            labels.append(f"{isotope}_{idx}")
            field_axis = spec[:, 0]
    finally:
        if main_bar is not None:
            main_bar.complete()
            main_bar.close()

    exp_x = exp_y = None
    if cfg.exp_data_file:
        exp_path = Path(cfg.exp_data_file)
        try:
            data = load_experimental_data(
                exp_path,
                delimiter=cfg.exp_data_delimiter,
                skip_header=cfg.number_of_header_lines,
            )
        except Exception as exc:  # pragma: no cover - guardrail
            raise RuntimeError(f"Failed to load experimental data from '{exp_path}': {exc}") from exc
        exp_x = data[:, 0] * cfg.exp_x_scaling
        exp_y = data[:, 1] * cfg.exp_y_scaling + cfg.exp_y_offset

    return PowderFieldResult(
        field_axis=field_axis,
        per_site=per_site,
        site_labels=labels,
        exp_x=exp_x,
        exp_y=exp_y,
    )


__all__ = [
    'PowderFieldConfig',
    'PowderFieldResult',
    'simulate_powder_field',
]
