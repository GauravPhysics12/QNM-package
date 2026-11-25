# ==============================================================================
# Import Required Libraries
# ==============================================================================
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator, CubicSpline, interp1d
from scipy.optimize import minimize_scalar
from scipy.special import lambertw

import matplotlib.ticker as ticker
from matplotlib import gridspec



# Keep your existing M definition in reference.py.
M = 1

from qnmtd import (
    initialize_field,
    evolve,
    extract_waveform,
    fit_prony,
)

from qnmtd.coordinates import r1
from qnmtd.metric_input import ask_metric_functions
from qnmtd.coordinates import configure_coordinates
from qnmtd.potentials import V_RW, V_ZP, V_PT
from qnmtd.initial_data import initialize_field
from qnmtd.evolution import evolve
from qnmtd.extraction import extract_waveform
from qnmtd.prony import fit_prony

# ==============================================================================
# Waveform Extraction
# ==============================================================================


def ask_float(prompt, default, positive=False):
    while True:
        text = input(f"{prompt} [{default}]: ").strip()

        try:
            value = float(text) if text else float(default)

            if not np.isfinite(value):
                raise ValueError

            if positive and value <= 0:
                raise ValueError

            return value

        except ValueError:
            if positive:
                print("Enter a finite number greater than zero.")
            else:
                print("Enter a finite number.")


def ask_grid_points(default=1000):
    while True:
        text = input(
            f"Number of points on each axis [{default}]: "
        ).strip()

        try:
            value = int(text) if text else default

            if value < 3:
                raise ValueError

            return value

        except ValueError:
            print("Enter an integer of at least 3.")


# ---- User Parameters ----

n_points = ask_grid_points()

grid_max = ask_float(
    "Upper limit for both u and v",
    default=500.0,
    positive=True,
)

v0 = ask_float(
    "Gaussian centre in v",
    default=12.0,
)

sigma = ask_float(
    "Gaussian width in v",
    default=2.0,
    positive=True,
)

b = 2
l = 2
s = 2


# ---- Grid ----

u = np.linspace(0.0, grid_max, n_points)
v = np.linspace(0.0, grid_max, n_points)

du = u[1] - u[0]
dv = v[1] - v[0]
h = du

# ---- Initial Field ----

psi = initialize_field(u, v, v0=v0, sigma=sigma)

print(f"\nGrid: {n_points} × {n_points}")
print(f"Grid spacing: h = {h:.6f}")
print(f"Gaussian: centre v = {v0}, width = {sigma}")
print(f"Field memory: {psi.nbytes / 1024**2:.1f} MiB\n")


# ---- Configure the Coordinate Mapping ----

f, g = ask_metric_functions(M)

print("\nSpecify one reference point to fix the coordinate shift.")
print("For standard Schwarzschild coordinates, r = 4M at r_star = 4M.")

r_anchor = ask_float(
    "Reference radius r",
    default=4.0 * M,
    positive=True,
)

rstar_anchor = ask_float(
    "Corresponding r_star",
    default=4.0 * M,
)

required_min = 0.5 * (v[0] - u[-1])
required_max = 0.5 * (v[-1] - u[0])
margin = u[1] - u[0]

configure_coordinates(
    f,
    g,
    r_anchor=r_anchor,
    rstar_anchor=rstar_anchor,
    rstar_min=min(required_min, rstar_anchor) - margin,
    rstar_max=max(required_max, rstar_anchor) + margin,
)

print("Coordinate mapping ready.")


psi = evolve(u, v, psi, l=l, s=s, M=1)

print("Evolution complete.")



# ==============================================================================
# Waveform Extraction
# ==============================================================================

x_obs = ask_float("Observation position in r_star", default=100.0)

time, signal = extract_waveform(u, v, psi, x_obs=x_obs)

print(
    f"Available waveform: t = {time[0]:.3f} "
    f"to {time[-1]:.3f}"
)

# ==============================================================================
# Prony Fit
# ==============================================================================

t_min = ask_float("Fit start time", default=190.0)
t_max = ask_float("Fit end time", default=300.0)

# Polynomial order, not the number of confirmed physical QNMs.
p = 50

fit = fit_prony(
    time,
    signal,
    t_min=t_min,
    t_max=t_max,
    order=p,
)

t_fit = fit["time"]
omega = fit["omega"]
C = fit["amplitudes"]
x_recon = fit["reconstruction"]

print(f"\nLinear prediction rank: {fit['rank']} / {p}")
print("Fitted positive-frequency, decaying modes:")

for i, w in enumerate(omega):
    if w.real > 0 and w.imag < 0:
        Q = w.real / (2.0 * abs(w.imag))

        print(
            f"Mode {i + 1}: "
            f"omega = {w.real:.6f} - i {abs(w.imag):.6f}"
            f" | Q = {Q:.6f}"
        )

# ==============================================================================
# Plot
# ==============================================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    signal.real,
    label="Original signal",
    color="black",
)

plt.plot(
    t_fit,
    x_recon.real,
    "--",
    label="Prony fit (real part)",
    linewidth=2,
    color="blue",
)

plt.xlabel(r"Time $t$")
plt.ylabel(r"$\psi(t)$")
plt.title("Prony Fit to the Ringing Region")
plt.grid(True, linestyle="--")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================================================================
# Optional Display Filters — These Do Not Validate Physical Modes
# ==============================================================================

max_frequency = 1.0
max_decay = 1.0
min_relative_amplitude = 0.01

max_amp = np.max(np.abs(C))

print("\nModes passing the chosen display filters:")
print(f"{'Mode':<6} {'Real':<12} {'Imag':<12} {'Q':<12}")

for i, w in enumerate(omega):
    if w.real <= 0 or w.imag >= 0:
        continue

    if w.real > max_frequency:
        continue

    if abs(w.imag) > max_decay:
        continue

    if abs(C[i]) < min_relative_amplitude * max_amp:
        continue

    Q = w.real / (2.0 * abs(w.imag))

    print(
        f"{i + 1:<6} "
        f"{w.real:<12.6f} "
        f"{w.imag:<12.6f} "
        f"{Q:<12.6f}"
    )