"""Compact ED config for nmr-field-powder-fit smoke tests."""
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / 'data' / 'field_powder_fit_ed_sample.txt'

CONFIG = {
    'exp_data_file': str(DATA_FILE),
    'number_of_header_lines': 0,
    'exp_data_delimiter': ' ',
    'missing_values_string': None,
    'exp_x_scaling': 1.0,
    'exp_y_scaling': 1.0,

    'minimization_algorithm': 'leastsq',
    'epsilon': None,
    'verbose_bool': False,

    'isotope_list': ['51V'],
    'f0': 34.75,
    'site_multiplicity_list': [[1.0]],
    'Ka_list': [[1.65]],
    'Kb_list': [[1.0]],
    'Kc_list': [[0.8]],
    'vQ_list': [[0.445]],
    'eta_list': [[0.7]],
    'min_field': 2.85,
    'max_field': 3.25,
    'n_field_points': 250,
    'sim_type': 'exact diag',
    'line_shape_func_list': ['gauss'],
    'FWHM_list': [[0.01]],
    'FWHM_vQ_list': [[0.005]],
    'background_list': [[0.0]],
    'mtx_elem_min': 0.5,

    'plot_initial_guess_bool': True,
    'plot_individual_bool': False,
    'plot_sum_bool': True,
    'plot_legend_width_ratio': [3.25, 1.0],
    'sim_export_file': '',
}
