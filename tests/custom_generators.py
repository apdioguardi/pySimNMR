import numpy as np


def planar_ring(samples=6, magnitude_T=0.02, seed=None):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=samples)
    vectors = np.zeros((samples, 3), dtype=float)
    vectors[:, 0] = magnitude_T * np.cos(angles)
    vectors[:, 1] = magnitude_T * np.sin(angles)
    return vectors


def weighted_pair():
    vectors = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.05],
    ])
    weights = np.array([0.25, 0.75])
    return vectors, weights
