"""Interactive Schwarzschild/JNW axial time-domain example.

Keep the existing qnmtd modules; add qnmtd/jnw.py from this update.
"""

import numpy as np
import matplotlib.pyplot as plt

from qnmtd.initial_data import initialize_field
from qnmtd.evolution import evolve
from qnmtd.extraction import extract_waveform
from qnmtd.prony import fit_prony
from qnmtd.coordinates import configure_coordinates
from qnmtd.jnw import initialize_jnw, evolve_jnw, extract_jnw
from qnmtd.potentials import ask_potential


def ask_float(prompt, default, positive=False):
    while True:
        try:
            text = input(f"{prompt} [{default}]: ").strip()
            value = float(text) if text else float(default)
            if not np.isfinite(value) or (positive and value <= 0):
                raise ValueError
            return value
        except ValueError:
            print("Enter a finite number" + (" greater than zero." if positive else "."))


def ask_int(prompt, default, minimum=1):
    while True:
        try:
            text = input(f"{prompt} [{default}]: ").strip()
            value = int(text) if text else default
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            print(f"Enter an integer >= {minimum}.")


def plot_waveform(time, signal, title, fit=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time, signal.real, color="black", label="Numerical waveform")
    axes[1].semilogy(time, np.where(np.abs(signal)>0, np.abs(signal), np.nan))
    if fit is not None:
        axes[0].plot(fit["time"], fit["reconstruction"].real, "--", label="Prony fit")
    axes[0].set_ylabel(r"$\Psi$")
    axes[1].set_ylabel(r"$|\Psi|$")
    axes[1].set_xlabel(r"Time $t$")
    axes[0].set_title(title)
    axes[0].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def main():
    while True:
        model = input("Model: j = JNW axial, s = Schwarzschild axial [j]: ").strip().lower() or "j"
        if model in ("j", "s"):
            break
        print("Enter j or s.")

    M = ask_float("ADM mass M", 1.0, positive=True)
    ell = ask_int("Angular index ell", 2, minimum=2)
    if model == "j":
        while True:
            nu = ask_float("JNW nu (0 < nu < 1)", 0.78, positive=True)
            if nu < 1:
                break
            print("For nu=1 choose the Schwarzschild branch.")
        b = 2.0*M/nu
        print(f"JNW: b = 2*M/nu = {b:.9f}; rstar=0 is the singularity.")
        print("JNW evolution uses only rstar > 0 and a zero-field inner boundary.")
    else:
        b = None
        nu = None

    n_points = ask_int("Number of points on each axis", 1000, minimum=3)
    grid_max = ask_float("Upper limit for both u and v", 500.0, positive=True)
    v0 = ask_float("Gaussian centre in v", 12.0, positive=True)
    sigma = ask_float("Gaussian width in v", 2.0, positive=True)
    u = np.linspace(0.0, grid_max, n_points)
    v = u.copy()
    h = u[1]-u[0]

    print("\nChoose the potential used by the evolution equation.")
    print("A radial V(r) uses the selected model's r(rstar) mapping.")
    if model == "j":
        print("For JNW, rstar=0 is the singularity; the evolved domain has rstar>0.")
    else:
        print("For Schwarzschild, rstar is shifted so r=4M is at rstar=4M.")
    potential = ask_potential(M, l=ell, s=2, b=b, nu=nu)
    if potential is None:
        print("Using the model's default axial potential.")
    else:
        print(f"Using {potential.expression} potential, expressed in {potential.coordinate}.")

    while True:
        x_obs = ask_float("Observer r_star (less than grid_max/2)", grid_max/5, positive=True)
        if x_obs < grid_max/2:
            break
        print("Observer must lie inside the domain.")

    if model == "j":
        psi = initialize_jnw(u, v, v0, sigma)
        print("Using Gaussian minus reflected Gaussian to satisfy the inner boundary.")
        print(f"Grid spacing: {h:.6f}; field memory: {psi.nbytes/1024**2:.1f} MiB")
        psi = evolve_jnw(u, v, psi, M=M, nu=nu, ell=ell, potential=potential)
        time, signal = extract_jnw(u, v, psi, x_obs)
        title = f"JNW axial: M={M}, nu={nu}, ell={ell}"
    else:
        configure_coordinates(
            lambda r: 1.0-2.0*M/r,
            r_anchor=4.0*M, rstar_anchor=4.0*M,
            rstar_min=min(-grid_max/2, 4.0*M)-h,
            rstar_max=max(grid_max/2, 4.0*M)+h,
        )
        psi = initialize_field(u, v, v0=v0, sigma=sigma)
        print(f"Grid spacing: {h:.6f}; field memory: {psi.nbytes/1024**2:.1f} MiB")
        psi = evolve(u, v, psi, l=ell, s=2, M=M, potential=potential)
        time, signal = extract_waveform(u, v, psi, x_obs=x_obs)
        title = f"Schwarzschild axial: M={M}, ell={ell}"

    print("Evolution complete.")
    print(f"Waveform interval: [{time[0]:.6f}, {time[-1]:.6f}]")
    print("Inspect the waveform, then close the plot to choose a fitting window.")
    plot_waveform(time, signal, title)

    choice = input("Fit this waveform with Prony? [y/N]: ").strip().lower()
    if choice not in ("y", "yes"):
        return

    # Defaults only populate prompts; they do not identify a ringdown region.
    duration = time[-1]-time[0]
    while True:
        t_min = ask_float("Fit start time (choose from the plot)", time[0]+0.3*duration)
        t_max = ask_float("Fit end time", time[0]+0.6*duration)
        order = ask_int("Prony polynomial order", 8)
        if not time[0] <= t_min < t_max <= time[-1]:
            print("Choose an increasing window inside the waveform interval.")
            continue
        try:
            fit = fit_prony(time, signal, t_min=t_min, t_max=t_max, order=order)
            break
        except (ValueError, np.linalg.LinAlgError) as error:
            print(error)

    print(f"Prediction rank: {fit['rank']} / {order}")
    print("Fitted decaying positive-frequency modes (not automatically physical QNMs):")
    max_amp = np.max(np.abs(fit["amplitudes"]))
    for index in np.argsort(fit["omega"].real):
        w = fit["omega"][index]
        if w.real > 0 and w.imag < 0:
            relative = abs(fit["amplitudes"][index])/max_amp if max_amp else 0.0
            print(
                f"Root {index+1}: omega = {w.real:.6f} - i {abs(w.imag):.6f}"
                f" | Q = {w.real/(2*abs(w.imag)):.6f} | relative amplitude = {relative:.3g}"
            )
    plot_waveform(time, signal, title, fit)


if __name__ == "__main__":
    main()
