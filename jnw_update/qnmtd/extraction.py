# extraction.py

import numpy as np


def extract_waveform(u, v, psi, x_obs):
    """Extract using the original first-point-within-h rule."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if len(u) < 2 or len(v) < 2:
        raise ValueError("Each grid needs at least two points.")

    if psi.shape != (len(u), len(v)):
        raise ValueError("psi shape must match the grids.")

    h = u[1] - u[0]

    time = []
    signal = []

    for i in range(len(u)):
        rstar = 0.5 * (v - u[i])
        matches = np.flatnonzero(np.abs(rstar - x_obs) < h)

        if matches.size:
            j = matches[0]
            time.append(0.5 * (u[i] + v[j]))
            signal.append(psi[i, j])

    if len(time) < 2:
        raise ValueError(
            "Too few extraction points. Check x_obs and grid limits."
        )

    return np.asarray(time), np.asarray(signal)