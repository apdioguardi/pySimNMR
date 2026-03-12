"""Sample configuration for nmr-field-fit (multisite field spectrum)."""
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / 'data' / 'field_fit_sample.txt'

CONFIG = {
    'exp_data_file': str(DATA_FILE),
    'number_of_header_lines': 0,
    'exp_data_delimiter': ' ',
    'missing_values_string': None,
    'exp_x_scaling': 1.0,
    'exp_y_scaling': 1.0,
    'exp_y_offset': 0.0,

    'minimization_algorithm': 'leastsq',
    'epsilon': None,
    'verbose_bool': False,

    'isotope_list': ['75As'],
    'f0': 45.0,
    'site_multiplicity_list': [[1.0]],
    'Ka_list': [[0.0]],
    'Kb_list': [[0.0]],
    'Kc_list': [[0.0]],
    'va_list': [[None]],
    'vb_list': [[None]],
    'vc_list': [[11.0]],
    'eta_list': [[0.0]],
    'Hinta_list': [[0.0]],
    'Hintb_list': [[0.0]],
    'Hintc_list': [[0.0]],
    'phi_z_deg_list': [[0.0]],
    'theta_x_prime_deg_list': [[0.0]],
    'psi_z_prime_deg_list': [[0.0]],
    'phi_deg_list': [[0.0]],
    'theta_deg_list': [[0.0]],

    'min_field': 0.5,
    'max_field': 1.5,
    'n_field_points': 256,
    'sim_type': 'exact diag',
    'convolution_function_list': ['gauss'],
    'conv_FWHM_list': [[0.01]],
    'conv_vQ_FWHM_list': [[0.005]],
    'background_list': [[0.0]],

    'plot_initial_guess_bool': True,
    'plot_individual_bool': False,
    'plot_sum_bool': True,
    'plot_legend_width_ratio': [3.25, 1.0],
    'exp_plot_style_str': 'ko-',
    'sim_export_file': '',
}
