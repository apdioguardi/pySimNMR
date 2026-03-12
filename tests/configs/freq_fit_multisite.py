"""Multisite config for `nmr-freq-fit` smoke tests."""

from __future__ import annotations

import os
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[1] / "generated_data"
_DEFAULT_DATA = _DATA_DIR / "fit_test_75As_vc=11p1MHz_H=5T_f=20-55MHz_FWHM=0p4MHz_vQFWHM=0p1MHz_noisy.txt"
_DATA_FILE = Path(os.environ.get("PY_SIMNMR_TEST_FREQ_FIT_DATA", _DEFAULT_DATA))

CONFIG = {
    'exp_data_file': str(_DATA_FILE),
    'number_of_header_lines': 0,
    'exp_data_delimiter': ' ',
    'missing_values_string': None,
    'exp_x_scaling': 1.0,
    'exp_y_scaling': 1.0,
    'exp_x_offset': 0.0,
    'exp_y_offset': 0.0,
    'minimization_algorithm': 'leastsq',
    'epsilon': None,
    'verbose_bool': False,
    'isotope_list': ['75As'],
    'H0': [[5.0]],
    'amplitude_list': [[0.9, True]],
    'Ka_list': [[0.0]],
    'Kb_list': [[0.0]],
    'Kc_list': [[0.0]],
    'vc_list': [[11.0, True]],
    'eta_list': [[0.0]],
    'Hinta_list': [[0.0]],
    'Hintb_list': [[0.0]],
    'Hintc_list': [[0.0]],
    'phi_z_deg_list': [[0.0]],
    'theta_xp_deg_list': [[0.0]],
    'psi_zp_deg_list': [[0.0]],
    'FWHM_list': [[0.3, True]],
    'FWHM_vQ_list': [[0.1]],
    'min_freq': 20.0,
    'max_freq': 55.0,
    'freq_points': 400,
    'line_shape_func_list': ['gauss'],
    'background_list': [[0.0]],
    'plot_initial_guess_bool': True,
    'plot_individual_bool': False,
    'plot_sum_bool': True,
    'plot_legend_width_ratio': [3.25, 1.0],
    'exp_plot_style_str': 'ko-',
    'sim_export_file': '',
}
