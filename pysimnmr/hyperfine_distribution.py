"""Powder-style simulations driven by internal field distributions."""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

from .core import SimNMR
from .plotly_utils import save_internal_field_distribution_html

__all__ = [
    "freq_spectrum_from_internal_field_distribution",
    "save_internal_field_distribution_html",
    "validate_internal_field_samples",
]


def validate_internal_field_samples(
    hint_pas_samples: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and normalize internal field samples and weights.

    Parameters
    ----------
    hint_pas_samples:
        Array of shape ``(n_samples, 3)`` with internal field components expressed in the
        principal-axis system (PAS) in Tesla.
    weights:
        Optional array of shape ``(n_samples,)`` with per-sample probabilities. Values do not
        need to be normalized. If omitted, equal weights are used.

    Returns
    -------
    tuple
        Tuple ``(samples, weights)`` with ``samples`` as ``float64`` and ``weights`` normalized to
        sum to one.
    """
    samples = np.asarray(hint_pas_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("hint_pas_samples must have shape (n_samples, 3)")
    n_samples = samples.shape[0]
    if n_samples == 0:
        raise ValueError("at least one internal field sample is required")

    if weights is None:
        norm_weights = np.full(n_samples, 1.0 / n_samples, dtype=float)
    else:
        norm_weights = np.asarray(weights, dtype=float)
        if norm_weights.shape != (n_samples,):
            raise ValueError("weights must have shape (n_samples,)")
        weight_sum = np.sum(norm_weights)
        if not np.isfinite(weight_sum) or weight_sum == 0.0:
            raise ValueError("weights must sum to a finite, non-zero value")
        norm_weights = norm_weights / weight_sum

    return samples, norm_weights


def _prepare_rotations(
    sim: SimNMR,
    angles_rad: Optional[np.ndarray],
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate rotation matrices for each sample, defaulting to identity rotations."""
    if angles_rad is None:
        phi = np.zeros(n_samples)
        theta = np.zeros(n_samples)
        psi = np.zeros(n_samples)
    else:
        angle_arr = np.asarray(angles_rad, dtype=float)
        if angle_arr.shape != (n_samples, 3):
            raise ValueError("euler_angles_rad must have shape (n_samples, 3)")
        phi, theta, psi = angle_arr.T
    r, ri = sim.generate_r_matrices(phi, theta, psi)
    SR, SRi = sim.generate_r_spin_matrices(phi, theta, psi)
    return r, ri, SR, SRi


def freq_spectrum_from_internal_field_distribution(
    sim: SimNMR,
    freq_axis_MHz: np.ndarray,
    *,
    hint_pas_samples_T: np.ndarray,
    weights: Optional[np.ndarray] = None,
    Ka: float,
    Kb: float,
    Kc: float,
    vQ_MHz: float,
    eta: float,
    H0_T: float,
    euler_angles_rad: Optional[np.ndarray] = None,
    mtx_elem_min: float = 0.1,
    min_freq_MHz: Optional[float] = None,
    max_freq_MHz: Optional[float] = None,
    FWHM_MHz: float = 0.01,
    FWHM_vQ_MHz: float = 0.0,
    line_shape_func: str = "gauss",
) -> np.ndarray:
    """Simulate a powder-style frequency spectrum over a sampled internal-field distribution.

    Each internal field vector contributes a frequency-domain spectrum that is scaled by its
    probability weight. The vectors are assumed to be defined in the PAS of the hyperfine field;
    supply ``euler_angles_rad`` if the PAS orientation varies per sample.

    Parameters
    ----------
    sim:
        ``SimNMR`` instance configured for the isotope/site under study.
    freq_axis_MHz:
        1D array of frequency points (MHz) at which to evaluate the spectrum.
    hint_pas_samples_T:
        Array of internal field vectors (Tesla) expressed in the PAS; shape ``(n_samples, 3)``.
    weights:
        Optional probability weights for each vector. Defaults to equal weighting.
    Ka, Kb, Kc:
        Components of the shift tensor (percent).
    vQ_MHz, eta:
        Electric field gradient parameters.
    H0_T:
        External magnetic field magnitude (Tesla).
    euler_angles_rad:
        Optional array of ZXZ Euler angles (phi, theta, psi in radians) defining PAS orientation
        per sample. If omitted, the PAS is aligned with the lab frame.
    mtx_elem_min, min_freq_MHz, max_freq_MHz, FWHM_MHz, FWHM_vQ_MHz, line_shape_func:
        Parameters forwarded to the exact-diagonalisation spectrum helper. The optional
        ``min_freq_MHz``/``max_freq_MHz`` values are retained for API compatibility; set the
        desired window directly in ``freq_axis``.

    Returns
    -------
    numpy.ndarray
        Simulated spectrum evaluated at ``freq_axis_MHz``.
    """
    freq_axis = np.asarray(freq_axis_MHz, dtype=float)
    if freq_axis.ndim != 1:
        raise ValueError("freq_axis_MHz must be a 1D array")

    samples, norm_weights = validate_internal_field_samples(hint_pas_samples_T, weights)
    r, ri, SR, SRi = _prepare_rotations(sim, euler_angles_rad, samples.shape[0])

    spectrum = np.zeros_like(freq_axis)
    for idx, (weight, hint_vec) in enumerate(zip(norm_weights, samples)):
        # Keep a length-1 leading axis because SimNMR expects batched rotation stacks.
        rotation_matrices = (
            r[idx:idx + 1],
            ri[idx:idx + 1],
            SR[idx:idx + 1],
            SRi[idx:idx + 1],
        )
        hint_vector = np.asarray(hint_vec, dtype=float)
        _, spectrum_i = sim.freq_spec_ed_mix(
            x=freq_axis,
            H0=H0_T,
            Ka=Ka,
            Kb=Kb,
            Kc=Kc,
            va=None,
            vb=None,
            vc=vQ_MHz,
            eta=eta,
            rotation_matrices=rotation_matrices,
            Hint=hint_vector,
            matrix_element_cutoff=mtx_elem_min,
            FWHM=FWHM_MHz,
            FWHM_vQ=FWHM_vQ_MHz,
            line_shape_func=line_shape_func,
        )
        spectrum += weight * spectrum_i

    return spectrum
