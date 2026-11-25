"""Effective-potential functions, safe expression compilation, and plotting.

This module has no interactive prompts. ``reference.py`` should collect the
user's choices, call :func:`compile_potential`, and pass the returned object
to the evolution function.
"""

import ast
import operator
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

try:
    from .metric_input import FUNCTIONS, show_supported_functions
except ImportError:  # Also permit direct import from inside the qnmtd folder.
    from metric_input import FUNCTIONS, show_supported_functions


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def V_RW(r, l, s, M=1.0):
    """Regge-Wheeler potential as a function of radial coordinate r."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = (1.0 - 2.0*M/r) * (
            l*(l + 1.0)/r**2 + 2.0*(1.0 - s**2)*M/r**3
        )
    return value.item() if value.ndim == 0 else value


def V_ZP(r, l, s=None, M=1.0):
    """Zerilli potential for polar gravitational perturbations."""
    r = np.asarray(r, dtype=float)
    scalar = r.ndim == 0
    r = np.atleast_1d(r)
    value = np.full_like(r, np.nan, dtype=float)
    valid = r > 2.0*M

    if np.any(valid):
        radius = r[valid]
        n = 0.5*(l - 1.0)*(l + 2.0)
        f = 1.0 - 2.0*M/radius
        numerator = 2.0*f* (
            n**2*(n + 1.0)*radius**3
            + 3.0*M*n**2*radius**2
            + 9.0*M**2*n*radius
            + 9.0*M**3
        )
        denominator = radius**3*(n*radius + 3.0*M)**2
        value[valid] = numerator/denominator

    return value.item() if scalar else value


def V_PT(r):
    """Pöschl-Teller example: 0.15/cosh(0.1*r)^2."""
    r = np.asarray(r, dtype=float)
    e = np.exp(-0.2*np.abs(r))
    value = 0.6*e/(1.0 + e)**2
    return value.item() if value.ndim == 0 else value

def show_potential_help(variable="r", parameters=None):
    """Print expression syntax and the functions available to potentials."""
    names = ["M", "pi", "e", "l", "ell", "s"]
    names.extend(name for name in (parameters or {}) if name not in names)
    print("\nPotential expression help")
    print(f"Independent variable: {variable}")
    print(f"Available constants: {', '.join(names)}")
    print("Operators: +  -  *  /  ^  **")
    print("Implicit products supported: 2M, l(l+1), (a+b)(c+d)")
    print("Examples:")
    print("  (1-2M/r)*(l*(l+1)/r**2 - 6*M/r**3)")
    print("  gamma(l+1)/beta(2,3) + 1/r**2")
    print("  0.15/cosh(0.1*rstar)**2")
    print("\nAvailable functions:")
    show_supported_functions()    

def compile_potential(expression, mass=1.0, parameters=None, variable="r"):
    """Compile a mathematical expression into a :class:`PotentialExpression`.

    Parameters
    ----------
    expression : str
        Formula such as ``(1-2M/r)*(l(l+1)/r**2 - 6M/r**3)``.
    mass : float
        Value substituted for ``M``.
    parameters : dict, optional
        Additional constants, commonly ``l``, ``ell``, ``s``, ``b``, ``nu``.
    variable : {``"r"``, ``"rstar"``}
        Independent coordinate used by the entered formula.

    The expression grammar uses arithmetic and the function registry in
    ``metric_input.py``. It does not execute Python statements or attributes.
    """
    if variable not in ("r", "rstar"):
        raise ValueError("variable must be 'r' or 'rstar'.")
    if not np.isscalar(mass) or np.iscomplexobj(mass) or not np.isfinite(mass):
        raise ValueError("M must be a finite real number.")

    constants = {"M": float(mass), "pi": np.pi, "e": np.e}
    for name, value in (parameters or {}).items():
        if not name.isidentifier() or name in FUNCTIONS or name in constants:
            raise ValueError(f"Invalid or reserved potential parameter: {name}")
        if not np.isscalar(value) or np.iscomplexobj(value) or not np.isfinite(value):
            raise ValueError(f"Potential parameter {name} must be finite and real.")
        constants[name] = float(value)

    text = expression.strip().replace("^", "**")
    text = re.sub(r"(?<=\d)\s*(?=(?:M|rstar|r)\b|\()", "*", text)
    text = re.sub(r"\)\s*\(", ")*(", text)

    def separate_parameter_call(match):
        name = match.group(1)
        return match.group(0) if name in FUNCTIONS else f"{name}*("

    text = re.sub(r"\b([A-Za-z_]\w*)\s*\(", separate_parameter_call, text)

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ValueError("Invalid mathematical potential expression.") from error

    def build(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
            if not np.isfinite(value):
                raise ValueError("Numeric constants must be finite.")
            return lambda x: value

        if isinstance(node, ast.Name):
            if node.id == variable:
                return lambda x: x
            if node.id in constants:
                value = constants[node.id]
                return lambda x: value
            allowed = ", ".join((variable, *constants.keys()))
            raise ValueError(f"Unknown name '{node.id}'. Use {allowed}.")

        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            function = _OPERATORS[type(node.op)]
            left, right = build(node.left), build(node.right)
            return lambda x: function(left(x), right(x))

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = build(node.operand)
            sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
            return lambda x: sign*operand(x)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in FUNCTIONS:
                raise ValueError(f"Unknown function '{name}'. Use help for the list.")
            function, allowed_counts = FUNCTIONS[name]
            if node.keywords or len(node.args) not in allowed_counts:
                counts = " or ".join(map(str, allowed_counts))
                raise ValueError(f"{name} requires {counts} positional argument(s).")
            arguments = [build(argument) for argument in node.args]
            return lambda x: function(*(argument(x) for argument in arguments))

        raise ValueError("Unsupported mathematical potential expression.")

    evaluator = build(tree.body)

    def evaluate(x):
        try:
            x = np.asarray(x, dtype=float)
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                result = np.asarray(evaluator(x))
            if np.iscomplexobj(result) or not np.all(np.isfinite(result)):
                raise ValueError("Expression must return finite real values.")
            return result.item() if result.ndim == 0 else result
        except (ArithmeticError, TypeError, ValueError) as error:
            raise ValueError(f"Cannot evaluate potential '{expression}': {error}") from error

    return PotentialExpression(expression, variable, evaluate)

@dataclass(frozen=True)
class PotentialExpression:
    """Compiled potential that can be passed to the evolution code."""

    expression: str
    coordinate: str
    evaluator: object

    def __call__(self, rstar, r=None):
        """Evaluate V on an rstar grid (and use r when the formula is radial)."""
        x = np.asarray(rstar, dtype=float)
        if self.coordinate == "rstar":
            argument = x
        elif self.coordinate == "r":
            if r is None:
                raise ValueError("This V(r) expression requires an r(rstar) mapping.")
            argument = np.asarray(r, dtype=float)
        else:
            raise ValueError("Potential coordinate must be 'r' or 'rstar'.")

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            result = np.asarray(self.evaluator(argument), dtype=float)
        try:
            result = np.broadcast_to(result, x.shape)
        except ValueError as error:
            raise ValueError("Potential result has an incompatible shape.") from error
        if not np.all(np.isfinite(result)):
            raise ValueError("Potential is not finite on the sampled grid.")
        return result.item() if result.ndim == 0 else result.copy()


def plot_potential(potential, x_min, x_max, points=1000, against=None, radius_map=None):
    """Plot against the formula's own coordinate, or optionally convert axes.

    The default plots V(r) versus r or V(rstar) versus rstar and does not need
    ``coordinates.py``. To plot against the other coordinate, set ``against``
    to ``"r"`` or ``"rstar"``; the configured r1 mapping is then used unless
    ``radius_map`` is supplied explicitly.
    """
    if not isinstance(potential, PotentialExpression):
        raise TypeError("potential must be a PotentialExpression.")
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        raise ValueError("Require finite limits with x_min < x_max.")
    if not isinstance(points, (int, np.integer)) or points < 2:
        raise ValueError("points must be an integer of at least 2.")

    against = potential.coordinate if against is None else against.lower()
    if against not in ("r", "rstar"):
        raise ValueError("against must be 'r' or 'rstar'.")

    if against == potential.coordinate == "r":
        x = np.linspace(x_min, x_max, points)
        values = potential(np.zeros_like(x), x)
        xlabel = r"$r$"
    elif against == potential.coordinate == "rstar":
        x = np.linspace(x_min, x_max, points)
        values = potential(x)
        xlabel = r"$r_*$"
    else:
        if radius_map is None:
            try:
                from .coordinates import r1 as radius_map
            except ImportError:
                from coordinates import r1 as radius_map

        if potential.coordinate == "r" and against == "rstar":
            x = np.linspace(x_min, x_max, points)  # x is rstar here
            radius = np.asarray(radius_map(x), dtype=float)
            values = potential(x, radius)
            xlabel = r"$r_*$"
        else:
            rstar = np.linspace(x_min, x_max, points)
            x = np.asarray(radius_map(rstar), dtype=float)
            values = potential(rstar)
            xlabel = r"$r$"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, values, color="navy", linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$V$")
    ax.set_title(f"Potential: {potential.expression}")
    ax.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout()
    plt.show()
    return fig, ax
