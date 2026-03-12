"""Reference CONFIG for `nmr-freq-powder-fit` (generic example)."""

CONFIG = {
    # Experimental data handling -----------------------------------------------
    'exp_data_file': 'path/to/your_powder_data.txt',  # Replace with your dataset path
    'number_of_header_lines': 30,
    'exp_data_delimiter': ' ',
    'missing_values_string': 'nan',
    'exp_x_scaling': 1.0,
    'exp_y_scaling': 1.0,

    # Optimisation controls ----------------------------------------------------
    'minimization_algorithm': 'leastsq',
    'epsilon': None,
    'verbose_bool': False,
    'max_nfev': None,

    # Physical parameters / fit variables --------------------------------------
    'H0': [10.0],                               # Static field(s) in Tesla
    'isotope_list': ['63Cu'],
    'amplitude_list': [[1.0, True]],            # [value, vary?, min, max]
    'Ka_list': [[0.01]],                        # Knight-shift tensor Ka (%)
    'Kb_list': [[0.01]],                        # Knight-shift tensor Kb (%)
    'Kc_list': [[0.01]],                        # Knight-shift tensor Kc (%)
    'va_list': [[None]],                        # Optional EFG components
    'vb_list': [[None]],
    'vc_list': [[24.75, True]],                 # Quadrupolar coupling (MHz)
    'eta_list': [[0.01, True]],                 # Asymmetry parameter η
    'Hinta_list': [[0.0]],                      # Internal field offsets (Tesla)
    'Hintb_list': [[0.0]],
    'Hintc_list': [[0.0]],

    # Simulation controls ------------------------------------------------------
    'sim_type': 'exact diag',
    'min_freq': 75.0,
    'max_freq': 150.0,
    'n_plot_points': 1000,
    'line_shape_func_list': ['gauss'],
    'FWHM_list': [[0.4, True, 0.0, 1.0]],
    'FWHM_vQ_list': [[0.6, True, 0.0, 1.0]],
    'recalc_random_samples': False,
    'n_samples': 100_000,
    'mtx_elem_min': 0.1,
    'background_list': [[0.0]],                 # Baseline offset term

    # Plot/export controls -----------------------------------------------------
    'plot_initial_guess_bool': True,
    'plot_individual_bool': False,
    'plot_sum_bool': True,
    'plot_legend_width_ratio': [3.25, 1.0],
    'x_axis_min': 75.0,
    'x_axis_max': 150.0,
    'y_axis_min': 0.0,
    'y_axis_max': 1.1,
    'sim_export_file': 'output/reference/freq_powder_fit.txt',
}
