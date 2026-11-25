import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from qnmtd.coordinates import get_r_from_rstar
from qnmtd.metric_input import FUNCTIONS, ask_metric_functions
from qnmtd.potentials import compile_potential, show_potential_help


# ==============================================================================
# User input helpers
# ==============================================================================

def ask_float(prompt, default=None, positive=False):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        text = input(f"{prompt}{suffix}: ").strip()

        try:
            if not text and default is None:
                raise ValueError

            value = float(text) if text else float(default)

            if not np.isfinite(value) or (positive and value <= 0):
                raise ValueError

            return value

        except ValueError:
            message = "Enter a finite number"
            if positive:
                message += " greater than zero"
            print(message + ".")


def ask_int(prompt, default=1000, minimum=2):
    while True:
        text = input(f"{prompt} [{default}]: ").strip()

        try:
            value = int(text) if text else default
            if value < minimum:
                raise ValueError
            return value

        except ValueError:
            print(f"Enter an integer of at least {minimum}.")


def find_potential_parameters(expression):
    """Find formula symbols that need user-supplied values."""
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", expression))

    known_names = {
        "r",
        "M",
        "pi",
        "e",
        *FUNCTIONS.keys(),
    }

    if "rstar" in names:
        raise ValueError("Enter the potential as V(r), using r rather than rstar.")

    return sorted(names - known_names)


def ask_plot_choice():
    while True:
        choice = input(
            "\nPlot: 1 = V(r) vs r, 2 = V(r(r*)) vs rstar [2]: "
        ).strip()

        if not choice:
            return "rstar"
        if choice == "1":
            return "r"
        if choice == "2":
            return "rstar"

        print("Enter 1 or 2.")


# ==============================================================================
# Ask for mass and metric
# ==============================================================================

M = ask_float("Mass M", 1.0, positive=True)

# Asks whether f(r) and g(r) are the same, then asks for the expression(s).
f, g = ask_metric_functions(M)


# ==============================================================================
# Ask for the potential V(r) and its parameters
# ==============================================================================

print("\nEnter the potential as a function of r. Type help for supported functions.")

while True:
    expression = input("V(r) = ").strip()

    if expression.lower() == "help":
        show_potential_help(variable="r")
        continue

    if not expression:
        print("Enter a potential formula.")
        continue

    try:
        parameter_names = find_potential_parameters(expression)
        parameters = {}

        if parameter_names:
            print("Detected potential parameters:", ", ".join(parameter_names))

        for name in parameter_names:
            default = 2.0 if name in {"l", "ell", "s"} else None
            parameters[name] = ask_float(f"Value for {name}", default)

        potential = compile_potential(
            expression,
            mass=M,
            parameters=parameters,
            variable="r",
        )
        break

    except (ValueError, OverflowError) as error:
        print(f"Could not use that formula: {error}")
        print("Enter the potential formula again.\n")


# ==============================================================================
# Ask which plot to make
# ==============================================================================

plot_mode = ask_plot_choice()
points = ask_int("Number of plot points", default=1000, minimum=2)


# ==============================================================================
# Plot V(r) versus r
# ==============================================================================

if plot_mode == "r":
    r_min = ask_float("Minimum radius r", positive=True)
    r_max = ask_float("Maximum radius r", positive=True)

    if r_min >= r_max:
        raise ValueError("Maximum radius must be greater than minimum radius.")

    r_axis = np.linspace(r_min, r_max, points)
    V_values = potential(np.zeros_like(r_axis), r_axis)

    x_values = r_axis
    x_label = r"$r$"
    title = r"Potential $V(r)$"


# ==============================================================================
# Plot V(r(r*)) versus rstar
# ==============================================================================

else:
    raw_min = ask_float("Lower limit for coordinate integration", -100.0)
    raw_max = ask_float("Upper limit for coordinate integration", 100.0)

    if raw_min >= raw_max:
        raise ValueError("Upper limit must be greater than lower limit.")

    # The mapper starts at the detected inner boundary plus this radial offset.
    r_of_raw_rstar = get_r_from_rstar(
        f,
        g,
        rstar_min=raw_min,
        rstar_max=raw_max,
        mass=M,
        boundary_epsilon=1e-8,
    )

    r_at_min = float(r_of_raw_rstar(raw_min))
    r_at_max = float(r_of_raw_rstar(raw_max))

    if r_at_max <= r_at_min:
        raise ValueError("The generated radius mapping is not increasing.")

    print(
        f"Coordinate map covers approximately "
        f"r = {r_at_min:.8g} to {r_at_max:.8g}."
    )

    print("\nSet the additive rstar shift with a reference point.")
    print("For standard Schwarzschild coordinates, use r = 4M and rstar = 4M.")

    r_anchor = ask_float("Reference radius r", positive=True)
    rstar_anchor = ask_float("Desired rstar at that radius")

    if not r_at_min < r_anchor < r_at_max:
        raise ValueError(
            "The reference radius must be inside the mapped radial interval. "
            "Adjust the integration limits or choose another reference radius."
        )

    raw_anchor = brentq(
        lambda raw_rstar: float(r_of_raw_rstar(raw_rstar)) - r_anchor,
        raw_min,
        raw_max,
    )

    shift = rstar_anchor - raw_anchor
    print(f"Applied additive rstar shift: {shift:+.8f}")

    raw_rstar_axis = np.linspace(raw_min, raw_max, points)
    r_values = np.asarray(r_of_raw_rstar(raw_rstar_axis), dtype=float)
    V_values = potential(raw_rstar_axis, r_values)

    # The chosen reference point shifts the horizontal axis.
    x_values = raw_rstar_axis + shift
    x_label = r"$r_*$"
    title = r"Potential $V(r(r_*))$"


# ==============================================================================
# Draw plot
# ==============================================================================

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_values, V_values, color="navy", linewidth=1.8)
ax.set_xlabel(x_label)
ax.set_ylabel(r"$V$")
ax.set_title(title)
ax.set_ylim(0.0, 0.5)
ax.grid(True, linestyle="--", alpha=0.45)

fig.tight_layout()
plt.show()
