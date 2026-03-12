"""Reference CONFIG for `nmr-field-spec` mirroring the original multisite field run."""

CONFIG = {
    # Two inequivalent 11B sites with different multiplicities -----------------
    'isotope_list': ['11B', '11B'],
    'site_multiplicity_list': [1.0, 0.5],       # Relative population (site 2 is 50% of site 1)
    'Ka_list': [0.1, 0.1],                      # Knight-shift Ka (%) for each site
    'Kb_list': [0.2, 0.2],                      # Knight-shift Kb (%) for each site
    'Kc_list': [0.3, 0.3],                      # Knight-shift Kc (%) for each site
    'vc_list': [0.5, 0.5],                      # Quadrupolar coupling (MHz)
    'eta_list': [0.7, 0.7],                     # Asymmetry parameter η per site
    'va_list': [None, None],                    # Optional EFG components not used here
    'vb_list': [None, None],
    'Hinta_list': [0.0, 0.0],                   # Internal field offsets (Tesla)
    'Hintb_list': [0.0, 0.0],
    'Hintc_list': [0.0, 0.0],
    'phi_z_deg_list': [0.0, 0.0],               # Euler φ (deg)
    'theta_x_prime_deg_list': [90.0, 90.0],     # Euler θ (deg)
    'psi_z_prime_deg_list': [0.0, 90.0],        # Euler ψ (deg)

    # Simulation controls ------------------------------------------------------
    'sim_type': 'exact diag',                   # Exact diagonalisation
    'f0': 40.813,                               # RF/carrier frequency (MHz)
    'min_field': 2.87,                          # Sweep from 2.87–3.11 T
    'max_field': 3.11,
    'n_field_points': 2000,
    'convolution_function_list': ['gauss', 'gauss'],
    'conv_FWHM_list': [0.002, 0.002],           # Broadening FWHM (Tesla)
    'conv_vQ_FWHM_list': [0.003, 0.003],        # Quadrupolar broadening FWHM (Tesla)
    'mtx_elem_min': 0.1,                        # Drop transitions weaker than this cutoff

    # Plot/export behaviour ----------------------------------------------------
    'plot_individual_bool': True,
    'plot_sum_bool': True,
    'plot_legend_width_ratio': [4.0, 1.0],
    'x_axis_min': 2.87,
    'x_axis_max': 3.11,
    'y_axis_min': -0.5,
    'y_axis_max': 1.5,
    'sim_export_file': 'output/reference/field_spectrum_multisite.txt',
}
