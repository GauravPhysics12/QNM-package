# qnmtd/metric_input.py

import ast
import operator
import re

import numpy as np
from scipy import special


# Registry: function name -> (callable, allowed argument counts)
FUNCTIONS = {}


def register(names, module, argument_counts):
    for name in names.split():
        FUNCTIONS[name] = (getattr(module, name), argument_counts)


# Standard mathematical functions.
register(
    "sqrt cbrt exp expm1 log log10 log2 log1p "
    "sin cos tan arcsin arccos arctan "
    "sinh cosh tanh arcsinh arccosh arctanh "
    "abs",
    np,
    (1,),
)

register("arctan2 hypot minimum maximum power", np, (2,))

# Gamma-related functions.
register(
    "gamma gammaln loggamma rgamma digamma",
    special,
    (1,),
)

register(
    "polygamma gammainc gammaincc gammaincinv gammainccinv",
    special,
    (2,),
)

# Beta-related functions.
register("beta betaln", special, (2,))
register("betainc betaincc betaincinv betainccinv", special, (3,))

# Error functions and exponential integrals.
register(
    "erf erfc erfcx erfi erfinv erfcinv dawsn expi exp1",
    special,
    (1,),
)

register("expn", special, (2,))

# Bessel functions: order followed by argument.
register("jv yv iv kv jve yve ive kve", special, (2,))
register("j0 j1 y0 y1 i0 i1 k0 k1", special, (1,))

# Hypergeometric functions.
register("hyp0f1", special, (2,))
register("hyp1f1 hyperu", special, (3,))
register("hyp2f1", special, (4,))

# Elliptic integrals.
register("ellipk ellipe ellipkm1", special, (1,))
register("ellipkinc ellipeinc", special, (2,))

# Other useful special functions.
register("expit logit wrightomega", special, (1,))
register("zeta", special, (1, 2))

# Common aliases.
FUNCTIONS["ln"] = FUNCTIONS["log"]
FUNCTIONS["Gamma"] = FUNCTIONS["gamma"]
FUNCTIONS["Beta"] = FUNCTIONS["beta"]

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def show_supported_functions():
    """Print function names and accepted numbers of arguments."""
    print("\nAvailable functions:")

    for name, (_, counts) in sorted(FUNCTIONS.items()):
        count_text = " or ".join(str(count) for count in counts)
        print(f"  {name:<16} {count_text} argument(s)")

    print("\nVariables/constants: r, M, pi, e")
    print("Operators: + - * / ^ **")
    print("Example: 1-2M/r + 0.01*beta(r, 2)\n")


def parse_metric(expression, mass):
    """Convert a restricted mathematical expression into a function of r."""
    if not np.isfinite(mass):
        raise ValueError("M must be finite.")

    text = expression.strip().replace("^", "**")

    # Support 2M, 2r, and 2(r + M).
    text = re.sub(r"(?<=\d)\s*(?=[Mr(])", "*", text)

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ValueError("Invalid mathematical expression.") from error

    constants = {"M": float(mass), "pi": np.pi, "e": np.e}

    def build(node):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
        ):
            value = float(node.value)

            if not np.isfinite(value):
                raise ValueError("Numeric constants must be finite.")

            return lambda r: value

        if isinstance(node, ast.Name):
            if node.id == "r":
                return lambda r: r

            if node.id in constants:
                value = constants[node.id]
                return lambda r: value

            raise ValueError(
                f"Unknown variable '{node.id}'. Use r, M, pi, or e."
            )

        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            operation = OPERATORS[type(node.op)]
            left = build(node.left)
            right = build(node.right)

            return lambda r: operation(left(r), right(r))

        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
        ):
            operand = build(node.operand)
            sign = -1 if isinstance(node.op, ast.USub) else 1

            return lambda r: sign * operand(r)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id

            if name not in FUNCTIONS:
                raise ValueError(
                    f"Unknown function '{name}'. Enter help to see the list."
                )

            function, allowed_counts = FUNCTIONS[name]

            if node.keywords or len(node.args) not in allowed_counts:
                expected = " or ".join(map(str, allowed_counts))
                raise ValueError(
                    f"{name} requires {expected} positional argument(s)."
                )

            arguments = [build(argument) for argument in node.args]

            return lambda r: function(
                *(argument(r) for argument in arguments)
            )

        raise ValueError("Unsupported mathematical expression.")

    evaluator = build(tree.body)

    def metric(r):
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                result = np.asarray(evaluator(r))

            if np.iscomplexobj(result):
                raise ValueError("The metric must be real-valued.")

            if not np.all(np.isfinite(result)):
                raise ValueError("The metric produced NaN or infinity.")

            return result.item() if result.ndim == 0 else result

        except (
            ArithmeticError,
            ValueError,
            TypeError,
        ) as error:
            raise ValueError(
                f"Cannot evaluate '{expression}' at r={r}: {error}"
            ) from error

    return metric


def ask_metric_functions(mass):
    """Prompt for f and, when different, g."""
    while True:
        answer = input(
            "Are f(r) and g(r) the same? [y/n]: "
        ).strip().lower()

        if answer in ("y", "yes", "n", "no"):
            break

        print("Enter y or n.")

    print("Enter help at an expression prompt to list functions.")

    def ask_expression(name):
        while True:
            expression = input(f"Enter {name}(r): ").strip()

            if expression.lower() == "help":
                show_supported_functions()
                continue

            try:
                return parse_metric(expression, mass)
            except (ValueError, OverflowError) as error:
                print(error)

    f = ask_expression("f")

    if answer in ("y", "yes"):
        return f, None

    return f, ask_expression("g")