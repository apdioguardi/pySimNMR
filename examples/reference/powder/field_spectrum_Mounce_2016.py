"""Reference CONFIG for `nmr-field-powder` reproducing the PuMeCl NMR data
from Mounce et al. (2016), Nuclear Magnetic Resonance Study of PuMeCl.

Parameters were taken from the published paper. The simulation reproduces the
general lineshape of the 35Cl and 37Cl powder patterns but some small features
did not agree well with the published spectrum. The likely cause is that
`mtx_elem_min` is set too low, allowing weak spurious transitions to appear in
the lineshape. Try increasing `mtx_elem_min` (currently 0.5) and re-running
before treating the discrepancy as a physics mismatch.

TODO: Systematically sweep mtx_elem_min, compare each result against the
published figure, and update this config once agreement is satisfactory.

Reference:
  A. M. Mounce et al., Physical Review B 93, 024419 (2016).
  DOI: 10.1103/PhysRevB.93.024419
"""
from pathlib import Path

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / 'example_data'
    / 'Mounce_2016_NuclearMagneticResonance_PuMeCl_37Cl.txt'
)

# see experimental data header for NMR parameters
Kiso = -0.75
Kaniso = 2.1
Kx = Kiso - Kaniso/2
Ky = Kx
Kz = Kaniso - Kiso


CONFIG = {
    'isotope_list': ['35Cl_Reyes', '37Cl_Reyes'],
    'site_multiplicity_list': [0.758, 0.246],     # natural abundance ratio 35Cl:37Cl
    'Ka_list': [Kx]*2,                            # Knight-shift tensors (%)
    'Kb_list': [Ky]*2,
    'Kc_list': [Kz]*2,
    'va_list': [-4.125/2, -3.259/2],              # EFG tensor principal components (MHz)
    'vb_list': [-4.125/2, -3.259/2],
    'vc_list': [4.125, 3.259],
    'eta_list': [None]*2,                         # derived from va/vb above; must supply either (va & vb) or eta
    'Hint_list': [[0.0, 0.0, 0.0],    # Site 1 (Hinta, Hintb, Hintc) in Tesla
                  [0.0, 0.0, 0.0]],   # Site 2

    'f0': 14.5,                                 # RF/carrier frequency (MHz)
    'sim_type': 'exact diag',
    'min_field': 3.2,
    'max_field': 4.6,
    'n_field_points': 250,
    'delta_f0': 0.015,
    'convolution_function_list': ['gauss']*2,
    'conv_FWHM_list': [0.02]*2,                  # Broadening FWHM (Tesla)
    'conv_vQ_FWHM_list': [1e-8]*2,
    'mtx_elem_min': 0.5,
    'recalc_random_samples': False,
    'n_samples': 500,

    'plot_individual_bool': True,
    'plot_sum_bool': True,
    'plot_legend_width_ratio': [3.25, 1.0],
    'sim_export_file': 'output/Mounce_2016_PuMeCl_HS_sim.txt',

    # experimental data overlay
    'exp_data_file': str(DATA_FILE),
    'number_of_header_lines': 11,
    'exp_data_delimiter': '\t',
    'missing_values_string': None,
    'exp_x_scaling': 1.0,
    'exp_y_scaling': 1.0,
    'exp_x_offset': 0.0,
    'exp_y_offset': 0.0,
}
