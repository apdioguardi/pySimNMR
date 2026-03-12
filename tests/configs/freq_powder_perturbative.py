"""Perturbative config for `nmr-freq-powder` smoke tests."""

CONFIG = {
    'isotope_list': ['75As'],
    'site_multiplicity_list': [1.0],
    'Ka_list': [0.0],
    'Kb_list': [0.0],
    'Kc_list': [0.0],
    'vc_list': [5.0],
    'eta_list': [0.2],
    'H0': 3.0,
    'sim_type': '2nd order',
    'min_freq': 20.0,
    'max_freq': 25.0,
    'n_freq_points': 160,
    'convolution_function_list': ['gauss'],
    'conv_FWHM_list': [0.1],
    'conv_vQ_FWHM_list': [0.01],
    'n_samples': 800,
    'plot_individual_bool': False,
    'plot_sum_bool': True,
    'sim_export_file': '',
}
