import numpy as np
import pytest

from pysimnmr.core import SimNMR
from pysimnmr.hyperfine_distribution import freq_spectrum_from_internal_field_distribution
from pysimnmr import plotly_utils
from pysimnmr.hyperfine_cli import run_hyperfine_simulation, load_hyperfine_config


def _identity_rotation(sim: SimNMR):
    phi = theta = psi = np.zeros(1)
    r, ri = sim.generate_r_matrices(phi, theta, psi)
    SR, SRi = sim.generate_r_spin_matrices(phi, theta, psi)
    return r, ri, SR, SRi


def test_load_hyperfine_config_python(tmp_path):
    cfg = {
        'isotope': '75As',
        'Ka': 0.0,
        'Kb': 0.0,
        'Kc': 0.0,
        'vQ_MHz': 10.0,
        'eta': 0.0,
        'H0_T': 0.0,
        'hyperfine_distribution': {'vectors': [[0.0, 0.0, 0.0]]},
    }
    cfg_path = tmp_path / "hf_cfg.py"
    cfg_path.write_text(f"CONFIG = {cfg!r}\n", encoding="utf-8")
    loaded = load_hyperfine_config(cfg_path)
    assert loaded == cfg


def test_internal_field_distribution_matches_manual_sum():
    sim = SimNMR('1H')
    freq_axis = np.linspace(18.0, 24.0, 64)
    hint_samples = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.05],
        [0.005, 0.0, 0.02],
    ])
    weights = np.array([0.5, 0.3, 0.2])
    spectrum_powder = freq_spectrum_from_internal_field_distribution(
        sim,
        freq_axis,
        hint_pas_samples_T=hint_samples,
        weights=weights,
        Ka=0.0,
        Kb=0.0,
        Kc=0.0,
        vQ_MHz=0.1,
        eta=0.0,
        H0_T=0.5,
        mtx_elem_min=1e-6,
        FWHM_MHz=0.05,
    )

    rotation_matrices = _identity_rotation(sim)
    spectrum_manual = np.zeros_like(freq_axis)
    for weight, hint_vec in zip(weights / weights.sum(), hint_samples):
        _, spectrum_piece = sim.freq_spec_ed_mix(
            x=freq_axis,
            H0=0.5,
            Ka=0.0,
            Kb=0.0,
            Kc=0.0,
            va=None,
            vb=None,
            vc=0.1,
            eta=0.0,
            rotation_matrices=rotation_matrices,
            Hint=np.asarray(hint_vec, float),
            matrix_element_cutoff=1e-6,
            FWHM=0.05,
        )
        spectrum_manual += weight * spectrum_piece
    assert np.allclose(spectrum_powder, spectrum_manual)


@pytest.mark.skipif(plotly_utils.go is None, reason="plotly is not available")
def test_internal_field_distribution_plot(tmp_path):
    vectors = np.array([
        [0.0, 0.0, 0.01],
        [0.02, 0.0, 0.0],
        [0.0, 0.015, 0.0],
    ])
    weights = np.array([0.2, 0.5, 0.3])
    out_html = tmp_path / "internal_fields.html"
    plotly_utils.save_internal_field_distribution_html(
        vectors,
        weights=weights,
        title="Test Internal Fields",
        out_html=out_html,
    )
    contents = out_html.read_text()
    assert "Test Internal Fields" in contents
    assert "html" in contents.lower()


def test_internal_field_distribution_zero_field():
    sim = SimNMR('75As')
    freq_axis = np.linspace(-40.0, 40.0, 256)
    hint_samples = np.zeros((3, 3))
    hint_samples[:, :2] = np.array([[0.02, 0.0], [0.0, 0.02], [0.015, -0.015]])
    spectrum = freq_spectrum_from_internal_field_distribution(
        sim,
        freq_axis,
        hint_pas_samples_T=hint_samples,
        Ka=0.0,
        Kb=0.0,
        Kc=0.0,
        vQ_MHz=10.0,
        eta=0.0,
        H0_T=0.0,
        FWHM_MHz=0.05,
        mtx_elem_min=1e-6,
    )
    assert np.all(np.isfinite(spectrum))
    assert spectrum.shape == freq_axis.shape


def test_run_hyperfine_simulation_config(tmp_path):
    cfg = {
        'isotope': '1H',
        'Ka': 0.0,
        'Kb': 0.0,
        'Kc': 0.0,
        'vQ_MHz': 0.1,
        'eta': 0.0,
        'H0_T': 0.5,
        'matrix_element_cutoff': 1e-6,
        'FWHM_MHz': 0.05,
        'freq_axis': {'min_MHz': 18.0, 'max_MHz': 24.0, 'points': 64},
        'hyperfine_distribution': {
            'vectors': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]],
            'weights': {'values': [0.6, 0.4]},
        },
        'plotly': {'html': False},
        'spectrum_basename': 'spectrum.txt',
    }
    result = run_hyperfine_simulation(cfg, tmp_path, enable_plotly=False)
    assert result['spectrum_path'].exists()
    data = result['spectrum_path'].read_text().strip().splitlines()
    assert len(data) > 1
    assert result['html_path'] is None


def test_run_hyperfine_simulation_generated(tmp_path):
    cfg = {
        'isotope': '1H',
        'Ka': 0.0,
        'Kb': 0.0,
        'Kc': 0.0,
        'vQ_MHz': 0.1,
        'eta': 0.0,
        'H0_T': 0.5,
        'matrix_element_cutoff': 1e-6,
        'FWHM_MHz': 0.05,
        'freq_axis': {'min_MHz': 18.0, 'max_MHz': 24.0, 'points': 32},
        'hyperfine_distribution': {
            'samples': 8,
            'seed': 12345,
            'plane': 'xy',
            'angle_distribution': {'type': 'uniform'},
            'magnitude_distribution': {'type': 'delta', 'value_T': 0.02},
            'weights': {'type': 'equal'},
        },
        'plotly': {'html': False},
        'spectrum_basename': 'generated.txt',
    }
    result = run_hyperfine_simulation(cfg, tmp_path, enable_plotly=False)
    assert result['spectrum_path'].exists()
    assert result['html_path'] is None
    data = np.loadtxt(result['spectrum_path'])
    assert data.shape[0] == 32


def test_run_hyperfine_simulation_custom_generator(tmp_path):
    cfg = {
        'isotope': '1H',
        'Ka': 0.0,
        'Kb': 0.0,
        'Kc': 0.0,
        'vQ_MHz': 0.1,
        'eta': 0.0,
        'H0_T': 0.5,
        'matrix_element_cutoff': 1e-6,
        'FWHM_MHz': 0.05,
        'freq_axis': {'min_MHz': 18.0, 'max_MHz': 24.0, 'points': 32},
        'allow_custom_generators': True,
        'hyperfine_distribution': {
            'custom_generator': {
                'module': 'tests.custom_generators',
                'function': 'planar_ring',
                'params': {'samples': 8, 'magnitude_T': 0.02, 'seed': 42},
            },
            'weights': {'type': 'equal'},
        },
        'plotly': {'html': False},
        'spectrum_basename': 'custom.txt',
    }
    result = run_hyperfine_simulation(cfg, tmp_path, enable_plotly=False)
    assert result['spectrum_path'].exists()
    data = np.loadtxt(result['spectrum_path'])
    assert data.shape[0] == 32


def test_run_hyperfine_simulation_custom_generator_requires_opt_in(tmp_path):
    cfg = {
        'isotope': '1H',
        'Ka': 0.0,
        'Kb': 0.0,
        'Kc': 0.0,
        'vQ_MHz': 0.1,
        'eta': 0.0,
        'H0_T': 0.5,
        'matrix_element_cutoff': 1e-6,
        'FWHM_MHz': 0.05,
        'freq_axis': {'min_MHz': 18.0, 'max_MHz': 24.0, 'points': 32},
        'hyperfine_distribution': {
            'custom_generator': {
                'module': 'tests.custom_generators',
                'function': 'weighted_pair',
            },
        },
        'plotly': {'html': False},
        'spectrum_basename': 'custom.txt',
    }
    with pytest.raises(ValueError):
        run_hyperfine_simulation(cfg, tmp_path, enable_plotly=False)


def test_run_hyperfine_simulation_aliases(tmp_path):
    cfg = {
        'isotope': '1H',
        'Ka_percent': 0.0,
        'Kb_percent': 0.0,
        'Kc_percent': 0.0,
        'vc': 0.05,
        'eta': 0.0,
        'H0': 0.2,
        'Hinta': 0.01,
        'Hintb': 0.0,
        'Hintc': 0.0,
        'freq_axis': {'min_MHz': 0.0, 'max_MHz': 1.0, 'points': 8},
        'hyperfine_distribution': {
            'vectors': [[0.0, 0.0, 0.0]],
            'weights': {'values': [1.0]},
        },
        'plotly': {'html': False},
        'spectrum_basename': 'alias.txt',
    }
    result = run_hyperfine_simulation(cfg, tmp_path, enable_plotly=False)
    assert result['spectrum_path'].exists()
