"""Sample configuration for nmr-freq-fit (multisite frequency spectrum)."""
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / 'data' / 'freq_fit_sample.txt'

CONFIG = {
    # Experimental data options
    'exp_data_file': str(DATA_FILE),
    'number_of_header_lines': 1,
    'exp_data_delimiter': ' ',
    'missing_values_string': None,
    'exp_x_scaling': 1.0,
    'exp_y_scaling': 1.0,
    'exp_x_offset': 0.0,
    'exp_y_offset': 0.0,

    # Optimisation settings
    'minimization_algorithm': 'leastsq',
    'epsilon': None,
    'verbose_bool': False,

    # Physical parameters and fit controls
    'isotope_list': ['75As'],
    'H0': [[5.0]],  # magnetic field (T)
    'amplitude_list': [[0.9, True]],
    'Ka_list': [[0.0]],
    'Kb_list': [[0.0]],
    'Kc_list': [[0.0]],
    'va_list': [[None]],
    'vb_list': [[None]],
    'vc_list': [[11.0, True]],  # MHz
    'eta_list': [[0.0]],
    'Hinta_list': [[0.0]],
    'Hintb_list': [[0.0]],
    'Hintc_list': [[0.0]],
    'phi_z_deg_list': [[0.0]],
    'theta_xp_deg_list': [[0.0]],
    'psi_zp_deg_list': [[0.0]],
    'FWHM_list': [[0.3, True]],
    'FWHM_vQ_list': [[0.1]],

    # Frequency grid and line-shape control
    'min_freq': 20.0,
    'max_freq': 55.0,
    'freq_points': 1000,
    'line_shape_func_list': ['gauss'],
    'background_list': [[0.0]],

    # Plotting/export behavior
    'plot_initial_guess_bool': True,
    'plot_individual_bool': False,
    'plot_sum_bool': True,
    'plot_legend_width_ratio': [3.25, 1.0],
    'exp_plot_style_str': 'ko-',
    'sim_export_file': '',
}
