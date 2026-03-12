"""Example CONFIG for the freq-vs-eta variation (115In legacy parameters)."""

CONFIG = {
    "isotope": "115In",
    "phi": 0.0,                  # radians (legacy file used degrees = 0)
    "theta": 0.0,
    "psi": 0.0,
    "B0_T": 0.0,                 # Static field (Tesla)
    "B_axis": "z",
    "Ka": 0.0,
    "Kb": 0.0,
    "Kc": 0.0,
    "vQ_MHz": 4.4,               # Quadrupolar coupling (MHz)
    "eta_min": 0.0,
    "eta_max": 1.0,
    "eta_points": 101,
}
