"""Exact-diagonalisation config for `nmr-field-powder-fit` smoke tests."""

from __future__ import annotations

import os
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[1] / "generated_data"
_DEFAULT_DATA = _DATA_DIR / "field_fit_multisite_data.txt"
_DATA_FILE = Path(os.environ.get("PY_SIMNMR_TEST_FIELD_POWDER_FIT_DATA", _DEFAULT_DATA))

CONFIG = {
    'exp_data_file': str(_DATA_FILE),
    'number_of_header_lines': 0,
    'exp_data_delimiter': ' ',
    'missing_values_string': None,
    'exp_x_scaling': 1.0,
    'exp_y_scaling': 1.0,

    'minimization_algorithm': 'leastsq',
    'epsilon': None,
    'verbose_bool': False,

    'isotope_list': ['75As'],
    'f0': 45.0,
    'site_multiplicity_list': [[1.0]],
    'Ka_list': [[0.0]],
    'Kb_list': [[0.0]],
    'Kc_list': [[0.0]],
    'vc_list': [[5.0]],
    'eta_list': [[0.2]],

    'min_field': 0.5,
    'max_field': 1.5,
    'n_field_points': 200,
    'sim_type': 'exact diag',
    'line_shape_func_list': ['gauss'],
    'FWHM_list': [[0.01]],
    'FWHM_vQ_list': [[0.005]],
    'background_list': [[0.0]],
    'mtx_elem_min': 0.2,

    'plot_initial_guess_bool': True,
    'plot_individual_bool': False,
    'plot_sum_bool': True,
    'plot_legend_width_ratio': [3.25, 1.0],
    'sim_export_file': '',
}
