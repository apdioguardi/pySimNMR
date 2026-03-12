from __future__ import annotations

import os
import numpy as np

from pysimnmr.core import SimNMR


class _SpyBar:
    def __init__(self) -> None:
        self.updates: int = 0

    def update(self, n: int = 1) -> None:
        self.updates += n

    def complete(self) -> None:  # pragma: no cover - simple stub
        return

    def close(self) -> None:  # pragma: no cover - simple stub
        return


def test_field_spec_edpp_reports_progress(monkeypatch) -> None:
    monkeypatch.setenv("PY_SIMNMR_DISABLE_PARALLEL", "1")
    sim = SimNMR("75As")
    rotation_cache = sim.random_rotation_matrices(["75As"], recalc_random_samples=True, n_samples=16)
    I0_key = f"I0={sim.isotope_data_dict['75As']['I0_string']}"
    bar = _SpyBar()
    H0 = np.linspace(0.5, 0.6, 4)
    sim.field_spec_edpp_loop_parallel(
        f0=45.0,
        H0_array=H0,
        Ka=0.0,
        Kb=0.0,
        Kc=0.0,
        va=None,
        vb=None,
        vc=0.5,
        eta=0.1,
        rotation_matrices=(
            rotation_cache["real_space"]["r"],
            rotation_cache["real_space"]["ri"],
            rotation_cache["spin_space"][I0_key]["r_spin"],
            rotation_cache["spin_space"][I0_key]["ri_spin"],
        ),
        mtx_elem_min=0.1,
        min_field=0.5,
        max_field=0.7,
        delta_f0=0.01,
        FWHM=0.001,
        FWHM_dvQ=0.0,
        broadening_func="gauss",
        progress_bar=bar,
    )
    assert bar.updates >= len(H0)


def test_sec_ord_field_progress_chunks(monkeypatch) -> None:
    monkeypatch.setenv("PY_SIMNMR_DISABLE_PARALLEL", "1")
    sim = SimNMR("75As")
    data = sim.isotope_data_dict["75As"]
    bar = _SpyBar()
    sim.sec_ord_field_pp(
        I0=data["I0"],
        gamma=data["gamma"],
        f0=45.0,
        Ka=0.0,
        Kb=0.0,
        Kc=0.0,
        vQ=5.0,
        eta=0.2,
        nrands=64,
        nbins=32,
        min_field=0.2,
        max_field=2.0,
        progress_bar=bar,
    )
    assert bar.updates >= 2
