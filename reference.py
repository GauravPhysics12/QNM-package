"""Characteristic QNM evolution with user-selected metric and potential."""


# ==============================================================================
# Imports
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from qnmtd import (
    initialize_field,
    evolve,
    extract_waveform,
    fit_prony,
)
from qnmtd.coordinates import configure_coordinates, r1
from qnmtd.metric_input import ask_metric_functions
from qnmtd.potentials import (
    compile_potential,
    plot_potential,
    show_potential_help,
)


# ==============================================================================
# User input helpers
# ==============================================================================

def ask_float(prompt, default, positive=False):
    while True:
        text = input(f"{prompt} [{default}]: ").strip()

        try:
            value = float(text) if text else float(default)

            if not np.isfinite(value) or (positive and value <= 0):
                raise ValueError

            return value

        except ValueError:
            if positive:
                print("Enter a finite number greater than zero.")
            else:
                print("Enter a finite number.")


def ask_int(prompt, default, minimum=1):
    while True:
        text = input(f"{prompt} [{default}]: ").strip()

        try:
            value = int(text) if text else int(default)

            if value < minimum:
                raise ValueError

            return value

        except ValueError:
            print(f"Enter an integer of at least {minimum}.")


def ask_user_potential(M, l, s, b, nu):
    """Ask for the existing default or compile a custom potential V(r)."""
    while True:
        choice = input(
            "Potential: d = existing default, c = custom V(r) [d/c]: "
        ).strip().lower()

        if choice in ("d", "c"):
            break

        print("Enter d or c.")

    if choice == "d":
        return None

    parameters = {
        "l": l,
        "ell": l,
        "s": s,
        "b": b,
        "nu": nu,
    }

    print("Enter the potential as a function of r. Type help for functions.")

    while True:
        expression = input("V(r): ").strip()

        if expression.lower() == "help":
            show_potential_help(
                variable="r",
                parameters=parameters,
            )
            continue

        try:
            return compile_potential(
                expression,
                mass=M,
                parameters=parameters,
                variable="r",
            )

        except (ValueError, OverflowError) as error:
            print(error)


# ==============================================================================
# Metric and potential setup
# ==============================================================================

def configure_model(M, l, s, nu):
    """Ask for the metric, then ask for the potential."""
    print("\nEnter the metric functions first.")
    f, g = ask_metric_functions(M)

    # For JNW, b = 2M/nu. For a different metric/potential, edit b as needed.
    b = 2.0 * M / nu

    print(f"\nParameters available to the potential: M={M}, l={l}, s={s}, b={b}, nu={nu}")
    potential = ask_user_potential(M, l, s, b=b, nu=nu)

    return f, g, b, potential


# ==============================================================================
# Potential plotting
# ==============================================================================

def plot_selected_potential(potential, b, required_min, required_max):
    """Optionally plot V(r) against r or V(r(rstar)) against rstar."""
    if potential is None:
        print("Using the evolution module's existing default potential.")
        return

    print(f"Using V(r) = {potential.expression}")

    draw = input(
        "Plot this potential before evolution? [y/N]: "
    ).strip().lower()

    if draw not in ("y", "yes"):
        return

    while True:
        plot_axis = input(
            "Plot V(r) versus r, or V(r(rstar)) versus rstar? [r/rstar]: "
        ).strip().lower()

        if plot_axis in ("r", "rstar"):
            break

        print("Enter r or rstar.")

    if plot_axis == "r":
        # Start outside the JNW singular radius when using the JNW example.
        default_r_min = b + 0.01 * max(1.0, abs(b))
        plot_min = ask_float(
            "Minimum r for plot",
            default_r_min,
            positive=True,
        )
        plot_max = ask_float(
            "Maximum r for plot",
            10.0 * b,
            positive=True,
        )
    else:
        plot_min = ask_float(
            "Minimum rstar for plot",
            required_min,
        )
        plot_max = ask_float(
            "Maximum rstar for plot",
            required_max,
        )

    if plot_min >= plot_max:
        raise ValueError(
            "Plot minimum must be smaller than the plot maximum."
        )

    n_plot = ask_int(
        "Number of plot points",
        1000,
        minimum=2,
    )

    plot_potential(
        potential,
        plot_min,
        plot_max,
        points=n_plot,
        against=plot_axis,
        radius_map=r1 if plot_axis == "rstar" else None,
    )


# ==============================================================================
# Main program
# ==============================================================================

def main():
    # Physical parameters.
    M = 1.0
    l = 2
    s = 2

    # JNW example parameter. Change this for a different JNW model.
    nu = 0.8

    # Ask for the metric, then the potential.
    f, g, b, potential = configure_model(M, l, s, nu)

    # Grid and initial Gaussian.
    n_points = ask_int(
        "Number of points on each axis",
        1000,
        minimum=3,
    )
    grid_max = ask_float(
        "Upper limit for both u and v",
        500.0,
        positive=True,
    )
    v0 = ask_float(
        "Gaussian centre in v",
        12.0,
    )
    sigma = ask_float(
        "Gaussian width in v",
        2.0,
        positive=True,
    )

    u = np.linspace(0.0, grid_max, n_points)
    v = np.linspace(0.0, grid_max, n_points)
    h = u[1] - u[0]

    psi = initialize_field(u, v, v0=v0, sigma=sigma)

    print(f"\nGrid: {n_points} × {n_points}")
    print(f"Grid spacing: h = {h:.6f}")
    print(f"Gaussian: centre v = {v0}, width = {sigma}")
    print(f"Field memory: {psi.nbytes / 1024**2:.1f} MiB")

    # Configure r(rstar). The coordinate module automatically chooses
    # the shift and detects the inner metric boundary.
    required_min = 0.5 * (v[0] - u[-1])
    required_max = 0.5 * (v[-1] - u[0])

    rstar_min = required_min - h
    rstar_max = required_max + h

    configure_coordinates(
        f,
        g,
        rstar_min=rstar_min,
        rstar_max=rstar_max,
        mass=M,
    )

    print("Coordinate mapping ready.")

    # Plot the selected potential, if requested.
    plot_selected_potential(
        potential,
        b,
        required_min,
        required_max,
    )

    # Evolve the field.
    if potential is None:
        psi = evolve(
            u,
            v,
            psi,
            l=l,
            s=s,
            M=M,
        )
    else:
        psi = evolve(
            u,
            v,
            psi,
            l=l,
            s=s,
            M=M,
            potential=potential,
        )

    print("Evolution complete.")

    # Extract the waveform.
    x_obs = ask_float(
        "Observation position in rstar",
        100.0,
        positive=True,
    )
    time, signal = extract_waveform(
        u,
        v,
        psi,
        x_obs=x_obs,
    )

    print(
        f"Available waveform: t = {time[0]:.3f} "
        f"to {time[-1]:.3f}"
    )

    # Fit the waveform with Prony's method.
    t_min = ask_float(
        "Fit start time",
        190.0,
    )
    t_max = ask_float(
        "Fit end time",
        300.0,
    )
    order = ask_int(
        "Prony polynomial order",
        50,
        minimum=1,
    )

    fit = fit_prony(
        time,
        signal,
        t_min=t_min,
        t_max=t_max,
        order=order,
    )

    omega = fit["omega"]
    amplitudes = fit["amplitudes"]
    reconstruction = fit["reconstruction"]

    print(f"\nLinear prediction rank: {fit['rank']} / {order}")
    print("Fitted positive-frequency, decaying modes:")

    for index, frequency in enumerate(omega):
        if frequency.real > 0 and frequency.imag < 0:
            quality = frequency.real / (2.0 * abs(frequency.imag))

            print(
                f"Mode {index + 1}: "
                f"omega = {frequency.real:.6f} "
                f"- i {abs(frequency.imag):.6f} "
                f"| Q = {quality:.6f}"
            )

    # Plot the waveform and its Prony reconstruction.
    plt.figure(figsize=(10, 5))
    plt.plot(
        time,
        signal.real,
        color="black",
        label="Original signal",
    )
    plt.plot(
        fit["time"],
        reconstruction.real,
        "--",
        color="blue",
        linewidth=2,
        label="Prony fit (real part)",
    )
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"$\psi(t)$")
    plt.title("Prony Fit to the Ringing Region")
    plt.grid(True, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional display filters. Passing them does not prove a mode is physical.
    max_frequency = 1.0
    max_decay = 1.0
    min_relative_amplitude = 0.01

    max_amp = (
        np.max(np.abs(amplitudes))
        if len(amplitudes)
        else 0.0
    )

    print("\nModes passing the display filters (not physical validation):")
    print(f"{'Mode':<6} {'Real':<12} {'Imag':<12} {'Q':<12}")

    for index, frequency in enumerate(omega):
        if frequency.real <= 0 or frequency.imag >= 0:
            continue

        if (
            frequency.real > max_frequency
            or abs(frequency.imag) > max_decay
        ):
            continue

        relative_amplitude = (
            abs(amplitudes[index]) / max_amp
            if max_amp
            else 0.0
        )

        if relative_amplitude < min_relative_amplitude:
            continue

        quality = frequency.real / (2.0 * abs(frequency.imag))

        print(
            f"{index + 1:<6} "
            f"{frequency.real:<12.6f} "
            f"{frequency.imag:<12.6f} "
            f"{quality:<12.6f}"
        )


# ==============================================================================
# Script entry point
# ==============================================================================

if __name__ == "__main__":
    main()