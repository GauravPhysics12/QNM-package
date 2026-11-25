"""Interactive Schwarzschild/JNW axial evolution and Prony fitting."""

import numpy as np
import matplotlib.pyplot as plt

from qnmtd.initial_data import initialize_field
from qnmtd.evolution import evolve
from qnmtd.extraction import extract_waveform
from qnmtd.prony import fit_prony
from qnmtd.coordinates import configure_coordinates
from qnmtd.jnw import initialize_jnw, evolve_jnw, extract_jnw


# ==============================================================================
# Input Helpers
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
            message = "Enter a finite number"

            if positive:
                message += " greater than zero"

            print(message + ".")


def ask_int(prompt, default, minimum=1):
    while True:
        text = input(f"{prompt} [{default}]: ").strip()

        try:
            value = int(text) if text else default

            if value < minimum:
                raise ValueError

            return value

        except ValueError:
            print(f"Enter an integer >= {minimum}.")


def ask_yes_no(prompt, default=False):
    suffix = "[Y/n]" if default else "[y/N]"

    while True:
        answer = input(f"{prompt} {suffix}: ").strip().lower()

        if not answer:
            return default

        if answer in ("y", "yes"):
            return True

        if answer in ("n", "no"):
            return False

        print("Enter y or n.")


# ==============================================================================
# Plotting
# ==============================================================================

def plot_waveform(time, signal, title, fit=None):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    axes[0].plot(
        time,
        signal.real,
        color="black",
        linewidth=1.2,
        label="Numerical waveform",
    )

    amplitude = np.abs(signal)

    axes[1].semilogy(
        time,
        np.where(amplitude > 0, amplitude, np.nan),
        color="black",
        linewidth=1.2,
        label="Numerical amplitude",
    )

    if fit is not None:
        reconstruction = fit["reconstruction"]
        fit_amplitude = np.abs(reconstruction)

        axes[0].plot(
            fit["time"],
            reconstruction.real,
            "--",
            color="blue",
            linewidth=1.8,
            label="Prony fit",
        )

        axes[1].semilogy(
            fit["time"],
            np.where(fit_amplitude > 0, fit_amplitude, np.nan),
            "--",
            color="blue",
            linewidth=1.8,
            label="Prony fit amplitude",
        )

    axes[0].set_ylabel(r"$\Psi(t)$")
    axes[1].set_ylabel(r"$|\Psi(t)|$")
    axes[1].set_xlabel(r"Time $t$")
    axes[0].set_title(title)

    for ax in axes:
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend()

    fig.tight_layout()

    # Close this window to resume the terminal prompts.
    plt.show(block=True)


# ==============================================================================
# Mode Display
# ==============================================================================

def print_modes(
    fit,
    max_frequency=None,
    max_decay=None,
    min_relative_amplitude=0.0,
):
    """Display fitted positive-frequency, decaying roots."""
    omega = np.asarray(fit["omega"])
    amplitudes = np.asarray(fit["amplitudes"])

    max_amp = np.max(np.abs(amplitudes))
    count = 0

    for index in np.argsort(omega.real):
        w = omega[index]

        if not np.isfinite(w):
            continue

        if w.real <= 0 or w.imag >= 0:
            continue

        relative_amplitude = (
            abs(amplitudes[index]) / max_amp
            if max_amp > 0
            else 0.0
        )

        if max_frequency is not None and w.real > max_frequency:
            continue

        if max_decay is not None and abs(w.imag) > max_decay:
            continue

        if relative_amplitude < min_relative_amplitude:
            continue

        quality = w.real / (2.0 * abs(w.imag))

        print(
            f"Root {index + 1}: "
            f"omega = {w.real:.6f} - i {abs(w.imag):.6f}"
            f" | Q = {quality:.6f}"
            f" | relative amplitude = {relative_amplitude:.6g}",
            flush=True,
        )

        count += 1

    if count == 0:
        print(
            "No fitted roots pass these selections.",
            flush=True,
        )

    return count


def print_all_roots(fit):
    """Show raw fitted roots when no decaying positive-frequency roots exist."""
    print("\nAll fitted roots for diagnosis:")
    print(f"{'Root':<8} {'Real':>15} {'Imag':>15}")
    print("-" * 40)

    for index, w in enumerate(fit["omega"]):
        print(
            f"{index + 1:<8} "
            f"{w.real:>15.8f} "
            f"{w.imag:>15.8f}"
        )


# ==============================================================================
# Main Program
# ==============================================================================

def main():
    # ---- Model Selection ----

    while True:
        model = input(
            "Model: j = JNW axial, s = Schwarzschild axial [j]: "
        ).strip().lower() or "j"

        if model in ("j", "s"):
            break

        print("Enter j or s.")

    M = ask_float("ADM mass M", 1.0, positive=True)
    ell = ask_int("Angular index ell", 2, minimum=2)

    if model == "j":
        while True:
            nu = ask_float(
                "JNW nu (0 < nu < 1)",
                0.78,
                positive=True,
            )

            if nu < 1:
                break

            print("For nu=1, choose the Schwarzschild branch.")

        print(
            f"\nJNW: b = 2*M/nu = {2.0 * M / nu:.9f}"
        )
        print("The singularity is placed at r_star = 0.")

    # ---- Grid and Gaussian Parameters ----

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
        positive=True,
    )

    sigma = ask_float(
        "Gaussian width in v",
        2.0,
        positive=True,
    )

    u = np.linspace(0.0, grid_max, n_points)
    v = u.copy()
    h = u[1] - u[0]

    # ---- Observer ----

    while True:
        x_obs = ask_float(
            "Observer r_star (less than grid_max/2)",
            grid_max / 5.0,
            positive=True,
        )

        if x_obs < grid_max / 2.0:
            break

        print("Observer must lie inside the domain.")

    # ---- Initialization ----

    if model == "j":
        psi = initialize_jnw(
            u,
            v,
            v0=v0,
            sigma=sigma,
        )

        title = f"JNW axial: M={M}, nu={nu}, ell={ell}"

        print(
            "\nUsing Gaussian minus reflected Gaussian "
            "to satisfy the inner boundary."
        )

    else:
        configure_coordinates(
            lambda r: 1.0 - 2.0 * M / r,
            r_anchor=4.0 * M,
            rstar_anchor=4.0 * M,
            rstar_min=min(-grid_max / 2.0, 4.0 * M) - h,
            rstar_max=max(grid_max / 2.0, 4.0 * M) + h,
        )

        psi = initialize_field(
            u,
            v,
            v0=v0,
            sigma=sigma,
        )

        title = f"Schwarzschild axial: M={M}, ell={ell}"

    print(f"\nGrid: {n_points} x {n_points}")
    print(f"Grid spacing: h = {h:.6f}")
    print(f"Field memory: {psi.nbytes / 1024**2:.1f} MiB")
    print("Starting evolution...", flush=True)

    # ---- Evolution and Extraction ----

    if model == "j":
        psi = evolve_jnw(
            u,
            v,
            psi,
            M=M,
            nu=nu,
            ell=ell,
        )

        time, signal = extract_jnw(
            u,
            v,
            psi,
            x_obs=x_obs,
        )

    else:
        psi = evolve(
            u,
            v,
            psi,
            l=ell,
            s=2,
            M=M,
        )

        time, signal = extract_waveform(
            u,
            v,
            psi,
            x_obs=x_obs,
        )

    time = np.asarray(time)
    signal = np.asarray(signal)

    print("\nEvolution complete.", flush=True)
    print(
        f"Available waveform: "
        f"t = {time[0]:.6f} to {time[-1]:.6f}"
    )

    # ---- Inspect Waveform Before Fitting ----

    print(
        "\nInspect the waveform."
        "\nCLOSE THE PLOT WINDOW to continue to Prony extraction.",
        flush=True,
    )

    plot_waveform(time, signal, title)

    # ---- Prony Fit: Always Reached After Closing the Plot ----

    print("\nStarting Prony extraction...", flush=True)

    duration = time[-1] - time[0]

    while True:
        # These are prompt defaults, not an automatic ringdown selection.
        t_min = ask_float(
            "Fit start time (choose from the plot)",
            time[0] + 0.3 * duration,
        )

        t_max = ask_float(
            "Fit end time",
            time[0] + 0.6 * duration,
        )

        order = ask_int(
            "Prony polynomial order",
            8,
            minimum=1,
        )

        if not time[0] <= t_min < t_max <= time[-1]:
            print(
                "Choose an increasing fitting window "
                "inside the available waveform interval."
            )
            continue

        try:
            fit = fit_prony(
                time,
                signal,
                t_min=t_min,
                t_max=t_max,
                order=order,
            )
            break

        except (ValueError, np.linalg.LinAlgError) as error:
            print(f"\nFit failed: {error}")
            print("Choose another window or polynomial order.")

    # ---- Print Extracted Modes ----

    print("\n" + "=" * 72)
    print("PRONY RESULTS")
    print("=" * 72)

    print(f"Fitting interval: [{t_min}, {t_max}]")
    print(f"Polynomial order: {order}")
    print(f"Prediction rank: {fit['rank']} / {order}")

    print(
        "\nExtracted positive-frequency, decaying modes:"
        "\nConvention: omega = omega_R - i*omega_I, omega_I > 0",
        flush=True,
    )

    count = print_modes(fit)

    if count == 0:
        print_all_roots(fit)

    print(
        "\nRoot numbers are polynomial-root labels, "
        "not physical overtone numbers."
    )

    # ---- Plot the Full Prony Reconstruction ----

    print(
        "\nClose the fit plot to continue to optional display filters.",
        flush=True,
    )

    plot_waveform(time, signal, title, fit=fit)

    # ---- Optional Display Filters ----

    if ask_yes_no("Apply optional display filters?"):
        print(
            "\nThese filters only select displayed roots."
            "\nThey do not validate physical QNMs or change the fitted signal."
        )

        max_frequency = ask_float(
            "Maximum real frequency",
            1.0,
            positive=True,
        )

        max_decay = ask_float(
            "Maximum decay magnitude",
            1.0,
            positive=True,
        )

        while True:
            min_relative_amplitude = ask_float(
                "Minimum relative amplitude (0 to 1)",
                0.01,
            )

            if 0.0 <= min_relative_amplitude <= 1.0:
                break

            print("Enter a value between 0 and 1.")

        print("\nModes passing the display filters:")

        print_modes(
            fit,
            max_frequency=max_frequency,
            max_decay=max_decay,
            min_relative_amplitude=min_relative_amplitude,
        )

    print("\nAnalysis complete.", flush=True)


if __name__ == "__main__":
    main()