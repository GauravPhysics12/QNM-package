# qnmtd/evolution.py

import numpy as np

from .coordinates import r1
from .potentials import V_RW


def evolve(u, v, psi, l, s, M):
    """Evolve psi in place with precomputed coordinate and potential values."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if u.ndim != 1 or v.ndim != 1:
        raise ValueError("u and v must be one-dimensional.")

    if len(u) < 2 or len(v) < 2:
        raise ValueError("Each grid requires at least two points.")

    if psi.shape != (len(u), len(v)):
        raise ValueError("psi shape must match (len(u), len(v)).")

    if not np.issubdtype(psi.dtype, np.inexact):
        raise ValueError("psi must have a floating-point or complex dtype.")

    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("Grid coordinates must be finite.")

    du = np.diff(u)
    dv = np.diff(v)

    if np.any(du <= 0) or np.any(dv <= 0):
        raise ValueError("Grid coordinates must increase.")

    if not np.allclose(du, du[0]):
        raise ValueError("u must be uniformly spaced.")

    if not np.allclose(dv, dv[0]):
        raise ValueError("v must be uniformly spaced.")

    if not np.isclose(du[0], dv[0]):
        raise ValueError("This evolution requires du == dv.")

    h = du[0]
    n_u, n_v = len(u), len(v)
    coefficient = h**2 / 8.0

    # For equal grid spacing, r_star depends only on j - i.
    offsets = np.arange(-(n_u - 1), n_v)
    r_star = 0.5 * (v[0] - u[0] + offsets * h)

    # One vectorized mapping call for all required coordinates.
    radii = r1(r_star)
    potential = V_RW(radii, l, s)

    if not np.all(np.isfinite(potential)):
        raise ValueError("Potential contains non-finite values.")

    for i in range(1, n_u):
        previous = psi[i - 1]
        current = psi[i]
        index_offset = n_u - 2 - i

        for j in range(1, n_v):
            S = previous[j - 1]
            W = current[j - 1]
            E = previous[j]

            k = j + index_offset
            V_W = potential[k]
            V_E = potential[k + 2]

            current[j] = (
                W + E - S
                - coefficient * (V_W * W + V_E * E)
            )

    return psi