"""CONFIG for nmr-freq-powder-fit smoke tests."""
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / 'data' / 'freq_powder_fit_sample.txt'

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

    'isotope_list': ['75As'],
    'H0': 3.0,
    'amplitude_list': [[1.0]],
    'Ka_list': [[0.0]],
    'Kb_list': [[0.0]],
    'Kc_list': [[0.0]],
    'vc_list': [[5.0]],
    'eta_list': [[0.2]],

    'sim_type': 'exact diag',
    'min_freq': 20.0,
    'max_freq': 25.0,
    'n_plot_points': 256,
    'line_shape_func_list': ['gauss'],
    'FWHM_list': [[0.1]],
    'FWHM_vQ_list': [[0.01]],
    'background_list': [[0.0]],
    'n_samples': 1000,
    'recalc_random_samples': True,

    'plot_initial_guess_bool': True,
    'plot_individual_bool': False,
    'plot_sum_bool': True,
    'plot_legend_width_ratio': [3.25, 1.0],
    'exp_plot_style_str': 'ko-',
    'sim_export_file': '',
}
