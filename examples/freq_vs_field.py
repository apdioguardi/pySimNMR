"""Example CONFIG for the freq-vs-field variation (converted from the legacy script)."""

CONFIG = {
    "isotope": "75As",
    "phi": 0.5276828460479657,    # 30.234 degrees in radians
    "theta": 1.4895512501145605,  # 85.345 degrees
    "psi": 0.8399135225834912,    # 48.1235 degrees
    "B_axis": "z",                # Sweep the Z component of the magnetic field
    "B_min_T": 0.001,
    "B_max_T": 10.0,
    "B_points": 1000,
    "Ka": 0.25,                   # Knight-shift tensor components (%)
    "Kb": 0.25,
    "Kc": 0.25,
    "vQ_MHz": 21.15,              # Quadrupolar coupling (MHz)
    "eta": 0.0,
}
