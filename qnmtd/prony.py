# prony.py

import numpy as np


def fit_prony(time, signal, t_min, t_max, order=50):
    """Fit complex exponentials using the convention exp(-i*omega*t)."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal)

    if time.ndim != 1 or signal.ndim != 1:
        raise ValueError("time and signal must be one-dimensional.")

    if time.size != signal.size:
        raise ValueError("time and signal must have equal lengths.")

    if not np.all(np.isfinite(time)):
        raise ValueError("time contains non-finite values.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains non-finite values.")

    if time.size < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("time must contain increasing sample times.")

    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer.")

    if not np.isfinite(t_min) or not np.isfinite(t_max):
        raise ValueError("Fit limits must be finite.")

    if t_max <= t_min:
        raise ValueError("t_max must exceed t_min.")

    mask = (time >= t_min) & (time <= t_max)
    t_fit = time[mask]
    x = signal[mask]

    # Preserve the original sample-count requirement.
    if len(x) < 2 * order + 1:
        raise ValueError(
            f"order={order} requires at least {2 * order + 1} "
            f"samples in the fit window; found {len(x)}. "
            "Reduce order or enlarge the fitting window."
        )

    spacing = np.diff(t_fit)
    dt_sample = spacing[0]

    if not np.allclose(spacing, dt_sample):
        raise ValueError("Prony fitting requires uniform time samples.")

    if not np.any(x):
        raise ValueError("Cannot fit an entirely zero signal.")

    # Linear prediction system.
    X = np.array([
        [x[n - m] for m in range(1, order + 1)]
        for n in range(order, len(x))
    ])

    rhs = -x[order:]

    alpha, _, rank, singular_values = np.linalg.lstsq(
        X, rhs, rcond=None
    )

    # Polynomial roots and complex frequencies.
    z = np.roots(np.concatenate(([1.0], alpha))).astype(complex)

    if np.any(z == 0) or not np.all(np.isfinite(z)):
        raise ValueError(
            "Invalid Prony roots. Try a lower order or another window."
        )

    omega = (1j / dt_sample) * np.log(z)

    # Amplitudes refer to the beginning of the fitting window.
    sample_indices = np.arange(len(x))
    Z = z[None, :] ** sample_indices[:, None]

    if not np.all(np.isfinite(Z)):
        raise ValueError(
            "Amplitude matrix overflowed. Try a lower fitting order."
        )

    amplitudes = np.linalg.lstsq(Z, x, rcond=None)[0]
    reconstruction = Z @ amplitudes

    return {
        "time": t_fit,
        "signal": x,
        "omega": omega,
        "amplitudes": amplitudes,
        "reconstruction": reconstruction,
        "rank": rank,
        "singular_values": singular_values,
    }