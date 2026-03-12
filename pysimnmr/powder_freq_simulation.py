from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

import pySimNMR
from .progress import ProgressManager


class PowderFreqConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    isotope_list: List[str]
    site_multiplicity_list: List[float]
    Ka_list: List[float]
    Kb_list: List[float]
    Kc_list: List[float]
    vQ_list: Optional[List[float]] = None
    vc_list: Optional[List[float]] = None
    eta_list: List[float]
    H0_T: float = Field(alias='H0')

    sim_mode: Literal['ed', 'perturbative'] = Field(default='ed', alias='sim_type')
    min_freq: float
    max_freq: float
    freq_points: int = Field(alias='n_freq_points', ge=2)
    convolution_function_list: List[str]
    conv_FWHM_list: List[float]
    conv_vQ_FWHM_list: List[float]
    mtx_elem_min: float = 0.01
    recalc_random_samples: bool = False
    n_samples: int = 100000

    @model_validator(mode='after')
    def _validate_lengths(self) -> 'PowderFreqConfig':
        n_sites = len(self.isotope_list)

        def ensure(name: str):
            lst = getattr(self, name)
            if len(lst) != n_sites:
                raise ValueError(f"{name} must have {n_sites} entries")

        for name in [
            'site_multiplicity_list', 'Ka_list', 'Kb_list', 'Kc_list',
            'eta_list', 'conv_FWHM_list', 'conv_vQ_FWHM_list',
            'convolution_function_list',
        ]:
            ensure(name)

        if self.vc_list is None:
            if self.vQ_list is None:
                raise ValueError("Either vc_list or vQ_list must be provided")
            vc_computed = []
            for vq, eta in zip(self.vQ_list, self.eta_list):
                denom = np.sqrt(1 - (eta ** 2) / 3)
                vc_computed.append(float(vq) / denom if denom else float(vq))
            self.vc_list = vc_computed
        else:
            ensure('vc_list')

        if len(self.convolution_function_list) == 1 and n_sites > 1:
            self.convolution_function_list = self.convolution_function_list * n_sites

        return self


@dataclass
class PowderFreqResult:
    freq_axis: np.ndarray
    per_site: List[np.ndarray]
    site_labels: List[str]


def _start_bar(progress: ProgressManager | None, total: int, desc: str):
    if progress is None or total <= 0:
        return None
    return progress.bar(total=total, desc=desc)


def _finish_bar(bar):
    if bar is None:  # pragma: no cover - simple guard
        return
    bar.complete()
    bar.close()


def simulate_powder_frequency(cfg: PowderFreqConfig,
                              progress: ProgressManager | None = None) -> PowderFreqResult:
    per_site = []
    labels = []
    freq_axis: Optional[np.ndarray] = None

    rotation_cache = None
    stage_total = len(cfg.isotope_list) + (1 if cfg.sim_mode == 'ed' else 0)
    stage_bar = _start_bar(progress, stage_total, 'Frequency powder simulation')
    if cfg.sim_mode == 'ed':
        cache_sim = pySimNMR.SimNMR('1H')
        rotation_cache = cache_sim.random_rotation_matrices(
            cfg.isotope_list,
            recalc_random_samples=cfg.recalc_random_samples,
            n_samples=cfg.n_samples,
        )
        if stage_bar is not None:
            stage_bar.update(1)

    for idx, isotope in enumerate(cfg.isotope_list):
        sim = pySimNMR.SimNMR(isotope)
        if cfg.sim_mode == 'ed':
            I0_string = sim.isotope_data_dict[isotope]['I0_string']
            r = rotation_cache['real_space']['r']
            ri = rotation_cache['real_space']['ri']
            r_spin = rotation_cache['spin_space'][f'I0={I0_string}']['r_spin']
            ri_spin = rotation_cache['spin_space'][f'I0={I0_string}']['ri_spin']
            spec = sim.freq_spec_edpp(
                H0=cfg.H0_T,
                Ka=cfg.Ka_list[idx],
                Kb=cfg.Kb_list[idx],
                Kc=cfg.Kc_list[idx],
                va=None,
                vb=None,
                vc=cfg.vc_list[idx],
                eta=cfg.eta_list[idx],
                rotation_matrices=(r, ri, r_spin, ri_spin),
                mtx_elem_min=cfg.mtx_elem_min,
                min_freq=cfg.min_freq,
                max_freq=cfg.max_freq,
                FWHM_MHz=cfg.conv_FWHM_list[idx],
                FWHM_dvQ_MHz=cfg.conv_vQ_FWHM_list[idx],
                broadening_func=cfg.convolution_function_list[idx],
                baseline=0.5,
                nbins=cfg.freq_points,
                save_files_bool=False,
                out_filename='',
            )
        else:
            spec = sim.sec_ord_freq_pp(
                H0=cfg.H0_T,
                Ka=cfg.Ka_list[idx],
                Kb=cfg.Kb_list[idx],
                Kc=cfg.Kc_list[idx],
                vQ=cfg.vc_list[idx],
                eta=cfg.eta_list[idx],
                nrands=int(cfg.n_samples),
                nbins=cfg.freq_points,
                min_freq=cfg.min_freq,
                max_freq=cfg.max_freq,
                broadening_func=cfg.convolution_function_list[idx],
                FWHM_MHz=cfg.conv_FWHM_list[idx],
                FWHM_dvQ_MHz=cfg.conv_vQ_FWHM_list[idx],
                save_files_bool=False,
                out_filename='',
                baseline=0.5,
            )
        if freq_axis is None:
            freq_axis = spec[:, 0]
        site_data = spec[:, 1] * cfg.site_multiplicity_list[idx]
        per_site.append(site_data)
        labels.append(f"{isotope}_{idx}")
        if stage_bar is not None:
            stage_bar.update(1)

    if freq_axis is None:
        raise RuntimeError("No spectra generated; check configuration.")

    _finish_bar(stage_bar)
    return PowderFreqResult(freq_axis=freq_axis, per_site=per_site, site_labels=labels)


__all__ = [
    'PowderFreqConfig',
    'PowderFreqResult',
    'simulate_powder_frequency',
]
