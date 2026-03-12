"""73Ge internal-field distribution tuned for uniform xy-plane sampling."""

CONFIG = {
    "isotope": "73Ge",
    "Ka": 0.0,
    "Kb": 0.0,
    "Kc": 0.0,
    "vQ_MHz": 0.2464,
    "eta": 0.06,
    "H0_T": 0.0,
    "matrix_element_cutoff": 1.0e-6,
    "FWHM_MHz": 1.0e-5,
    "FWHM_vQ_MHz": 0.0,
    "line_shape": "gauss",
    "freq_axis": {
        "min_MHz": 0.737,
        "max_MHz": 0.740,
        "points": 4096,
    },
    "hyperfine_distribution": {
        "samples": 1000,
        "seed": 20251007,
        "plane": "xy",
        "angle_distribution": {"type": "uniform"},
        "magnitude_distribution": {
            "type": "gaussian",
            "mean_T": 0.012,
            "sigma_T": 0.000001,
        },
        "weights": {"type": "equal"},
    },
    "plotly": {
        "html": True,
        "title": "73Ge internal fields in the xy-plane",
    },
    "spectrum_basename": "73Ge_xy_internal_field_spectrum.txt",
    "plotly_basename": "73Ge_xy_internal_field_vectors.html",
}
