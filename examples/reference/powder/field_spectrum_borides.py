"""Reference CONFIG for `nmr-field-powder` covering a two-site 11B boride sample.

TODO: These parameters are placeholder values that need to be validated against
published experimental data. The intended reference compound is PuB4 (first
author Blackwell). Obtain the published NMR parameters and data file, update
this config, and add a row to the validation table in doc/pySimNMR_manual.tex.
"""

CONFIG = {
    'isotope_list': ['11B', '11B'],
    'site_multiplicity_list': [1.0, 0.35],      # Site 2 contributes 35% of site 1
    'Ka_list': [-0.055, -0.05],                 # Knight-shift tensors (%)
    'Kb_list': [-0.055, -0.05],
    'Kc_list': [-0.055, -0.05],
    'va_list': [None, None],                    # Optional EFG components not used here
    'vb_list': [None, None],
    'vc_list': [0.579, 0.48],                   # Quadrupolar coupling (MHz)
    'eta_list': [0.0, 0.9],
    'Hinta_list': [0.0, 0.0],                   # Internal field offsets (Tesla)
    'Hintb_list': [0.0, 0.0],
    'Hintc_list': [0.0, 0.0],

    'f0': 40.0,                                 # RF/carrier frequency (MHz)
    'sim_type': 'exact diag',
    'min_field': 2.87,
    'max_field': 3.0,
    'n_field_points': 500,
    'delta_f0': 0.015,
    'convolution_function_list': ['gauss', 'gauss'],
    'conv_FWHM_list': [0.005, 0.005],           # Broadening FWHM (Tesla)
    'conv_vQ_FWHM_list': [1e-8, 1e-8],
    'mtx_elem_min': 0.5,
    'recalc_random_samples': False,
    'n_samples': 1000,                          # Used only for perturbative simulations

    'plot_individual_bool': True,
    'plot_sum_bool': True,
    'plot_legend_width_ratio': [3.25, 1.0],
    'sim_export_file': 'output/reference/borides_field_powder.txt',
}
