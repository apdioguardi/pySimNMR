"""Reference CONFIG for `nmr-freq` demonstrating a single-site 75As single-crystal
frequency spectrum.

The parameters (vQ ~ 11.1 MHz, H0 = 5 T) are consistent with 75As NMR in iron
pnictide superconductors such as BaFe2As2. The intended validation reference is
Kitagawa et al. -- obtain the published experimental spectrum and NMR parameters,
update this config, add the experimental data file to examples/example_data/, and
add a row to the validation table in doc/pySimNMR_manual.tex.

TODO: Identify the exact Kitagawa paper, confirm parameters, add experimental
data overlay, and mark as validated.
"""

CONFIG = {
    # Lattice/site definitions -------------------------------------------------
    'isotope_list': ['75As'],                  # One crystallographic site (extend list for multisite)
    'site_multiplicity_list': [1.0],           # Relative population of each site
    'Ka_list': [0.0],                          # Knight-shift tensor Ka (%) per site
    'Kb_list': [0.0],                          # Knight-shift tensor Kb (%) per site
    'Kc_list': [0.0],                          # Knight-shift tensor Kc (%) per site
    'vc_list': [11.1],                         # Quadrupolar coupling (MHz)
    'eta_list': [0.0],                         # Asymmetry parameter η per site
    'Hint_list': [                             # Internal field offsets (Tesla) in PAS components
        {'x': 0.0, 'y': 0.0, 'z': 0.0},
    ],
    'phi_z_deg_list': [0.0],                   # Euler φ for each site (deg)
    'theta_x_prime_deg_list': [0.0],           # Euler θ (deg)
    'psi_z_prime_deg_list': [0.0],             # Euler ψ (deg)

    # Simulation controls ------------------------------------------------------
    'H0': 5.0,                                 # Static magnetic field (Tesla)
    'sim_type': 'exact diag',                  # Use exact diagonalisation
    'min_freq': 20.0,                          # Plot window lower bound (MHz)
    'max_freq': 55.0,                          # Plot window upper bound (MHz)
    'n_freq_points': 1000,                     # Number of bins across the frequency axis
    'convolution_function_list': ['gauss'],    # Line shape per site
    'conv_FWHM_list': [0.4],                   # Broadening FWHM (MHz)
    'conv_vQ_FWHM_list': [0.1],                # Quadrupolar broadening FWHM (MHz)
    'matrix_element_cutoff': 0.5,              # Drop transitions with amplitude below this value

    # Plot/export behaviour ----------------------------------------------------
    'plot_individual_bool': True,              # Fill individual site contributions
    'plot_sum_bool': True,                     # Show the summed spectrum
    'plot_legend_width_ratio': [3.25, 1.0],    # Main plot vs legend width
    'x_axis_min': 20.0,
    'x_axis_max': 55.0,
    'y_axis_min': 0.0,
    'y_axis_max': 1.1,
    'sim_export_file': 'output/reference/freq_spectrum_multisite.txt',
}
