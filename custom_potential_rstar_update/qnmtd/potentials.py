"""Built-in potentials and user-defined potential expressions."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .metric_input import parse_metric, show_supported_functions


def V_RW(r, l, s, M=1.0):
    r = np.asarray(r, dtype=float)
    value = (1.0-2.0*M/r) * (l*(l+1)/r**2 + 2.0*(1.0-s*s)*M/r**3)
    return value.item() if value.ndim == 0 else value


def V_ZP(r, l, s=None, *, M=1.0):
    """Schwarzschild Zerilli potential; s is unused."""
    r = np.asarray(r, dtype=float)
    scalar = r.ndim == 0
    r = np.atleast_1d(r)
    value = np.full_like(r, np.nan)
    valid = r > 2.0*M
    radius = r[valid]
    n = 0.5*(l-1)*(l+2)
    f = 1.0-2.0*M/radius
    numerator = 2.0*f*(
        n*n*(n+1)*radius**3 + 3.0*M*n*n*radius**2
        + 9.0*M*M*n*radius + 9.0*M**3
    )
    value[valid] = numerator/(radius**3*(n*radius+3.0*M)**2)
    return value.item() if scalar else value


def V_PT(rstar):
    """Original Poschl-Teller example, evaluated without cosh overflow."""
    x = np.asarray(rstar, dtype=float)
    e = np.exp(-2.0*np.abs(0.1*x))
    value = 0.6*e/(1.0+e)**2
    return value.item() if value.ndim == 0 else value


@dataclass(frozen=True)
class UserPotential:
    expression: str
    coordinate: str
    function: Callable

    def __call__(self, rstar, r=None):
        """Return potential values shaped like the requested rstar grid."""
        x = np.asarray(rstar, dtype=float)
        argument = x if self.coordinate == "rstar" else r
        if argument is None:
            raise ValueError("A radial potential requires the radial mapping.")
        values = np.asarray(self.function(argument), dtype=float)
        values = np.broadcast_to(values, x.shape)
        if not np.all(np.isfinite(values)):
            raise ValueError("Potential must be finite at all sampled points.")
        return values.item() if values.ndim == 0 else values.copy()


def ask_potential(M, l, s=2, b=None, nu=None):
    """Prompt for the model default, one of the examples, or a custom V.

    Radial expressions use the branch's own r(rstar) map. Expressions in
    rstar use the coordinate convention printed by the driver.
    """
    while True:
        print("Potential choices: default, rw, zerilli, pt, custom")
        choice = input("Potential [default]: ").strip().lower() or "default"
        if choice in ("default", "d"):
            return None
        if choice in ("rw", "zerilli", "z", "pt"):
            if choice == "rw":
                expr = "Regge-Wheeler"
                func = lambda r: V_RW(r, l, s, M=M)
                coord = "r"
            elif choice in ("zerilli", "z"):
                expr = "Zerilli"
                func = lambda r: V_ZP(r, l, M=M)
                coord = "r"
            else:
                expr = "Pöschl-Teller"
                func = V_PT
                coord = "rstar"
            return UserPotential(expr, coord, func)
        if choice in ("custom", "c"):
            break
        print("Enter default, rw, zerilli, pt, or custom.")

    while True:
        coordinate = input("Potential coordinate: r or rstar [r]: ").strip().lower() or "r"
        if coordinate in ("r", "rstar"):
            break
        print("Enter r or rstar.")

    parameters = {"l": l, "ell": l, "s": s}
    if b is not None:
        parameters["b"] = b
    if nu is not None:
        parameters["nu"] = nu
    print(f"Variable: {coordinate}; M={M}; " + ", ".join(f"{k}={v}" for k, v in parameters.items()))
    print("Other constants: pi, e. Enter help for supported functions.")
    print("This replaces V only; the selected model still sets the mapping and boundaries.")

    while True:
        expression = input(f"Enter V({coordinate}): ").strip()
        if expression.lower() == "help":
            show_supported_functions()
            continue
        try:
            function = parse_metric(expression, M, parameters=parameters, variable=coordinate)
            return UserPotential(expression, coordinate, function)
        except (ValueError, OverflowError) as error:
            print(error)
