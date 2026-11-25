# initial_data.py

import numpy as np


def initialize_field(u, v, v0=12.0, sigma=2.0):
    """Create a field with Gaussian data on the first u row."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if u.ndim != 1 or v.ndim != 1:
        raise ValueError("u and v must be one-dimensional arrays.")

    if u.size < 2 or v.size < 2:
        raise ValueError("Each grid must contain at least two points.")

    if not np.isfinite(v0):
        raise ValueError("Gaussian centre must be finite.")

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian width must be finite and positive.")

    psi = np.zeros((len(u), len(v)), dtype=float)

    psi[0, :] = np.exp(
        -((v - v0)**2) / (2.0 * sigma**2)
    )

    return psi