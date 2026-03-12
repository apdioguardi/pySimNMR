"""Reference CONFIG for `nmr-freq-fit` based on the original multisite script."""

CONFIG = {
    'exp_data_file': 'path/to/your_multisite_data.txt',
    'number_of_header_lines': 1,
    'exp_data_delimiter': ' ',
    'missing_values_string': None,
    'exp_x_scaling': 1.0,

    'minimization_algorithm': 'leastsq',
    'epsilon': None,
    'verbose_bool': False,

    'H0': [[5.0]],                               # Field per site (Tesla)
    'isotope_list': ['75As'],
    'amplitude_list': [[0.9, True]],
    'Ka_list': [[0.0]],                          # Knight-shift tensor Ka (%)
    'Kb_list': [[0.0]],
    'Kc_list': [[0.0]],
    'va_list': [[None]],                         # Optional EFG components
    'vb_list': [[None]],
    'vc_list': [[11.0, True]],
    'eta_list': [[0.0]],
    'Hinta_list': [[0.0]],                       # Internal field offsets (Tesla)
    'Hintb_list': [[0.0]],
    'Hintc_list': [[0.0]],
    'phi_z_deg_list': [[0.0]],
    'theta_xp_deg_list': [[0.0]],
    'psi_zp_deg_list': [[0.0]],

    'line_shape_func_list': ['gauss'],
    'FWHM_list': [[0.3, True]],                 # Gaussian linewidth (MHz)
    'FWHM_vQ_list': [[0.1]],
    'background_list': [[0.0]],
    'mtx_elem_min': 0.4,

    'min_freq': 20.0,
    'max_freq': 55.0,
    'n_plot_points': 1000,

    'plot_initial_guess_bool': True,
    'plot_individual_bool': False,
    'plot_sum_bool': True,
    'plot_legend_width_ratio': [3.25, 1.0],
    'x_axis_min': 20.0,
    'x_axis_max': 55.0,
    'y_axis_min': 0.0,
    'y_axis_max': 1.1,
    'sim_export_file': 'output/reference/fit_freq_spectrum_multisite.txt',
}
