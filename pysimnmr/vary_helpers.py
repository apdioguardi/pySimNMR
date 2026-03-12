# -- coding: utf-8 --
"""Single-crystal variations: frequency/levels vs field/eta/angle grids.

These helpers are thin wrappers over the legacy exact-diagonalization API in
`SimNMR`, organizing outputs onto regular grids for plotting/exports.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Literal, TYPE_CHECKING
from .core import SimNMR
from .progress import ProgressManager

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from typing import Any
    BarType = Any
else:  # pragma: no cover - hint placeholder
    BarType = object


def _hint_array(hint_pas: Optional[np.ndarray], count: int) -> np.ndarray:
    arr = np.zeros((count, 3, 1), dtype=float)
    if hint_pas is None:
        return arr
    vec = np.asarray(hint_pas, dtype=float).ravel()
    if vec.size < 3:
        raise ValueError('Hint_pas must have three components (x, y, z)')
    arr[:, 0, 0] = vec[0]
    arr[:, 1, 0] = vec[1]
    arr[:, 2, 0] = vec[2]
    return arr

def _single_orientation_eulers(phi: float, theta: float, psi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.array([phi], float), np.array([theta], float), np.array([psi], float))

def _grid_lin(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(float(lo), float(hi), int(n))

def _start_bar(progress: ProgressManager | None, total: int, desc: str) -> BarType | None:
    if progress is None or total <= 0:
        return None
    return progress.bar(total=total, desc=desc)


def _finish_bar(bar: BarType | None) -> None:
    if bar is None:  # pragma: no cover - trivial guard
        return
    bar.complete()
    bar.close()


def freq_vs_field_sweep(sim: SimNMR, *, B_min_T: float, B_max_T: float, B_points: int,
                        Ka: float, Kb: float, Kc: float, vQ_MHz: float, eta: float,
                        phi: float = 0.0, theta: float = 0.0, psi: float = 0.0,
                        Hint_pas: Optional[np.ndarray] = None, axis: Literal['x','y','z']='z',
                        progress: ProgressManager | None = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Return a field grid and padded transition map for a single orientation.

    We keep a rectangular (B_points x n_transitions) matrix filled with NaN so
    downstream plotting/rasterisation can treat every column as "one line"
    even when the number of allowed transitions changes with field.
    """
    B = _grid_lin(B_min_T, B_max_T, B_points)
    phiK, thK, psK = _single_orientation_eulers(phi, theta, psi)
    r, ri = sim.generate_r_matrices(phiK, thK, psK)
    SR, SRi = sim.generate_r_spin_matrices(phiK, thK, psK)
    rotation_matrices = (r, ri, SR, SRi)
    hint_arr = _hint_array(Hint_pas, rotation_matrices[0].shape[0])
    lines = []
    max_trans = 0
    bar = _start_bar(progress, B.size, 'Field sweep')
    for Bval in B:
        result = sim.exact_diag(
            H0=float(Bval),
            Ka=Ka,
            Kb=Kb,
            Kc=Kc,
            va=None,
            vb=None,
            vc=vQ_MHz,
            eta=eta,
            rotation_matrices=rotation_matrices,
            Hint=hint_arr,
            matrix_element_cutoff=1e-6,
        )
        freqs = np.sort(result['frequency'])
        lines.append(freqs)
        if freqs.size > max_trans:
            max_trans = freqs.size
        if bar is not None:
            bar.update(1)
    # pad with NaN to rectangular
    Y = np.full((B.size, max_trans), np.nan, float)
    for i, arr in enumerate(lines):
        Y[i, :arr.size] = arr
    # Order columns by the last row so that the highest-frequency transition
    # stays in column 0 across the entire variation grid, stabilising plots.
    order = np.argsort(np.nan_to_num(Y[-1], nan=-np.inf))[::-1]
    _finish_bar(bar)
    return B, Y[:, order]

def elevels_vs_field_sweep(sim: SimNMR, *, B_min_T: float, B_max_T: float, B_points: int,
                           Ka: float, Kb: float, Kc: float, vQ_MHz: float, eta: float,
                           phi: float = 0.0, theta: float = 0.0, psi: float = 0.0,
                           axis: Literal['x','y','z']='z', Hint_pas: Optional[np.ndarray]=None,
                           progress: ProgressManager | None = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep B0 and return eigen-levels padded/sorted for easier plotting."""
    stage_bar = _start_bar(progress, 3, 'Energy levels vs field')
    B = _grid_lin(B_min_T, B_max_T, B_points)
    phiK, thK, psK = _single_orientation_eulers(phi, theta, psi)
    r, ri = sim.generate_r_matrices(phiK, thK, psK)
    SR, SRi = sim.generate_r_spin_matrices(phiK, thK, psK)
    rotation_matrices = (r, ri, SR, SRi)
    if stage_bar is not None:
        stage_bar.update(1)
    # Use built-in helper to get levels across B, then reshape
    elevels_fields, _ = sim.elevels_vs_field_ed(H0=B, Ka=Ka, Kb=Kb, Kc=Kc,
                                               va=None, vb=None, vc=vQ_MHz, eta=eta,
                                               rotation_matrices=rotation_matrices,
                                               Hinta=Hint_pas[0] if Hint_pas is not None else 0.0,
                                               Hintb=Hint_pas[1] if Hint_pas is not None else 0.0,
                                               Hintc=Hint_pas[2] if Hint_pas is not None else 0.0)
    if stage_bar is not None:
        stage_bar.update(1)
    # reshape levels to (B_points, dim)
    levels_only = elevels_fields[:, 1]
    dim = levels_only.size // B.size
    Y = levels_only.reshape(B.size, dim)
    # Sort by the last field point so the lowest-energy level is always column 0.
    order = np.argsort(Y[-1])
    _finish_bar(stage_bar)
    return B, Y[:, order]

def freq_vs_eta_sweep(sim: SimNMR, *, eta_min: float, eta_max: float, eta_points: int,
                      Ka: float, Kb: float, Kc: float, vQ_MHz: float,
                      phi: float = 0.0, theta: float = 0.0, psi: float = 0.0,
                      H0_T: float = 9.4, axis: Literal['x','y','z']='z',
                      Hint_pas: Optional[np.ndarray]=None,
                      progress: ProgressManager | None = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Return eta grid + padded transition map sorted for stable visualisation."""
    eta_grid = _grid_lin(eta_min, eta_max, eta_points)
    phiK, thK, psK = _single_orientation_eulers(phi, theta, psi)
    r, ri = sim.generate_r_matrices(phiK, thK, psK)
    SR, SRi = sim.generate_r_spin_matrices(phiK, thK, psK)
    rotation_matrices = (r, ri, SR, SRi)
    hint_arr = _hint_array(Hint_pas, rotation_matrices[0].shape[0])
    lines = []
    max_trans = 0
    bar = _start_bar(progress, eta_grid.size, 'Eta sweep')
    for eta_val in eta_grid:
        result = sim.exact_diag(
            H0=float(H0_T),
            Ka=Ka,
            Kb=Kb,
            Kc=Kc,
            va=None,
            vb=None,
            vc=vQ_MHz,
            eta=float(eta_val),
            rotation_matrices=rotation_matrices,
            Hint=hint_arr,
            matrix_element_cutoff=1e-6,
        )
        freqs = np.sort(result['frequency'])
        lines.append(freqs)
        if freqs.size > max_trans:
            max_trans = freqs.size
        if bar is not None:
            bar.update(1)
    Y = np.full((eta_grid.size, max_trans), np.nan, float)
    for i, arr in enumerate(lines):
        Y[i, :arr.size] = arr
    # Keep transition ordering consistent across eta so contour plots look smooth.
    order = np.argsort(np.nan_to_num(Y[-1], nan=-np.inf))[::-1]
    _finish_bar(bar)
    return eta_grid, Y[:, order]

def freq_vs_angle_sweep(sim: SimNMR, *, angle_name: Literal['phi','theta','psi'],
                        angle_min_deg: float, angle_max_deg: float, angle_points: int,
                        Ka: float, Kb: float, Kc: float, vQ_MHz: float, eta: float,
                        fixed_phi_deg: float = 0.0, fixed_theta_deg: float = 0.0, fixed_psi_deg: float = 0.0,
                        H0_T: float = 9.4, axis: Literal['x','y','z']='z',
                        Hint_pas: Optional[np.ndarray]=None,
                        progress: ProgressManager | None = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Vary one Euler angle while holding the others fixed, padding outputs."""
    ang_grid_deg = _grid_lin(angle_min_deg, angle_max_deg, angle_points)
    lines = []
    max_trans = 0
    bar = _start_bar(progress, ang_grid_deg.size, f"{angle_name} sweep")
    for a_deg in ang_grid_deg:
        phi = np.deg2rad(a_deg) if angle_name == 'phi' else np.deg2rad(fixed_phi_deg)
        theta = np.deg2rad(a_deg) if angle_name == 'theta' else np.deg2rad(fixed_theta_deg)
        psi = np.deg2rad(a_deg) if angle_name == 'psi' else np.deg2rad(fixed_psi_deg)
        phiK, thK, psK = _single_orientation_eulers(phi, theta, psi)
        r, ri = sim.generate_r_matrices(phiK, thK, psK)
        SR, SRi = sim.generate_r_spin_matrices(phiK, thK, psK)
        rotation_matrices = (r, ri, SR, SRi)
        hint_arr = _hint_array(Hint_pas, r.shape[0])
        result = sim.exact_diag(
            H0=float(H0_T),
            Ka=Ka,
            Kb=Kb,
            Kc=Kc,
            va=None,
            vb=None,
            vc=vQ_MHz,
            eta=eta,
            rotation_matrices=rotation_matrices,
            Hint=hint_arr,
            matrix_element_cutoff=1e-6,
        )
        arr = np.sort(result['frequency'])
        lines.append(arr)
        if arr.size > max_trans:
            max_trans = arr.size
        if bar is not None:
            bar.update(1)
    Y = np.full((ang_grid_deg.size, max_trans), np.nan, float)
    for i, arr in enumerate(lines):
        Y[i, :arr.size] = arr
    # Sorting keeps the same transition in the same column as the angle variations.
    order = np.argsort(np.nan_to_num(Y[-1], nan=-np.inf))[::-1]
    _finish_bar(bar)
    return ang_grid_deg, Y[:, order]
