"""Example CONFIG for the freq-vs-angle variation (converted from the 73Ge legacy setup)."""

CONFIG = {
    "isotope": "73Ge",
    "angle_name": "theta",          # Legacy script varied theta_x_prime_deg
    "angle_min_deg": 0.0,
    "angle_max_deg": 90.0,
    "angle_points": 150,
    "fixed_phi_deg": 0.0,
    "fixed_theta_deg": 0.0,
    "fixed_psi_deg": 0.0,
    "B0_T": 25.0,
    "B_axis": "z",
    "Ka": 0.0,                      # Knight-shift tensor components (%)
    "Kb": 0.0,
    "Kc": 0.0,
    "vQ_MHz": 0.25,                 # Quadrupolar coupling (MHz)
    "eta": 0.0,
}
