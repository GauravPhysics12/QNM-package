"""Square-grid characteristic evolution with precomputed potentials."""

import numpy as np

from .coordinates import r1
from .potentials import V_RW


def evolve(u, v, psi, l, s, M, potential=None):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if u.ndim != 1 or v.ndim != 1 or len(u) < 2 or len(v) < 2:
        raise ValueError("Use one-dimensional grids with at least two points.")
    if psi.shape != (len(u), len(v)) or psi.dtype.kind not in "fc":
        raise ValueError("psi must be a floating/complex array matching the grid.")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("Grid must be finite.")
    du, dv = np.diff(u), np.diff(v)
    if np.any(du <= 0) or np.any(dv <= 0):
        raise ValueError("Grid coordinates must increase.")
    if not np.allclose(du, du[0]) or not np.allclose(dv, du[0]):
        raise ValueError("Both grids must be uniform with du == dv.")
    h = du[0]
    n_u, n_v = len(u), len(v)
    offsets = np.arange(-(n_u-1), n_v)
    rstar = 0.5*(v[0]-u[0]+offsets*h)

    if potential is None:
        V = V_RW(r1(rstar), l, s, M=M)
    elif potential.coordinate == "rstar":
        V = potential(rstar)
    else:
        V = potential(rstar, r1(rstar))
    if not np.all(np.isfinite(V)):
        raise ValueError("Potential contains non-finite values.")

    coefficient = h*h/8.0
    for i in range(1, n_u):
        current, previous = psi[i], psi[i-1]
        index_offset = n_u-2-i
        for j in range(1, n_v):
            W, E, S = current[j-1], previous[j], previous[j-1]
            k = j+index_offset
            current[j] = W+E-S-coefficient*(V[k]*W+V[k+2]*E)
        if not np.all(np.isfinite(current)):
            raise FloatingPointError("Evolution became non-finite; check the potential and resolution.")
    return psi
