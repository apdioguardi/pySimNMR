"""75As internal-field distribution example."""

CONFIG = {
    "isotope": "75As",
    "Ka": 0.0,
    "Kb": 0.0,
    "Kc": 0.0,
    "vQ_MHz": 10.0,
    "eta": 0.0,
    "H0_T": 0.0,
    "matrix_element_cutoff": 1.0e-6,
    "FWHM_MHz": 0.05,
    "FWHM_vQ_MHz": 0.0,
    "line_shape": "gauss",
    "freq_axis": {
        "min_MHz": -40.0,
        "max_MHz": 40.0,
        "points": 4096,
    },
    "hyperfine_distribution": {
        "samples": 1000,
        "seed": 20251007,
        "plane": "xy",
        "angle_distribution": {"type": "uniform"},
        "magnitude_distribution": {
            "type": "gaussian",
            "mean_T": 0.20,
            "sigma_T": 0.04,
        },
        "weights": {"type": "equal"},
    },
    "plotly": {
        "html": True,
        "title": "75As internal fields in the xy-plane",
    },
    "spectrum_basename": "75As_xy_internal_field_spectrum.txt",
    "plotly_basename": "75As_xy_internal_field_vectors.html",
}
