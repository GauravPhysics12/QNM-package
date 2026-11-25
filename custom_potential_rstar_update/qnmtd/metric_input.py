"""Restricted numerical expression parser; no eval or Python attribute access."""

import ast
import operator
import re

import numpy as np
from scipy import special


FUNCTIONS = {}


def register(names, module, counts):
    for name in names.split():
        FUNCTIONS[name] = (getattr(module, name), counts)


register("sqrt cbrt exp expm1 log log10 log2 log1p sin cos tan arcsin arccos arctan sinh cosh tanh arcsinh arccosh arctanh abs", np, (1,))
register("arctan2 hypot minimum maximum power", np, (2,))
register("gamma gammaln loggamma rgamma digamma", special, (1,))
register("polygamma gammainc gammaincc gammaincinv gammainccinv beta betaln", special, (2,))
register("betainc betaincc betaincinv betainccinv", special, (3,))
register("erf erfc erfcx erfi erfinv erfcinv dawsn expi exp1", special, (1,))
register("expn jv yv iv kv jve yve ive kve hyp0f1", special, (2,))
register("j0 j1 y0 y1 i0 i1 k0 k1", special, (1,))
register("hyp1f1 hyperu", special, (3,))
register("hyp2f1", special, (4,))
register("ellipk ellipe ellipkm1 expit logit wrightomega", special, (1,))
register("ellipkinc ellipeinc", special, (2,))
register("zeta", special, (1, 2))
FUNCTIONS["ln"] = FUNCTIONS["log"]
FUNCTIONS["Gamma"] = FUNCTIONS["gamma"]
FUNCTIONS["Beta"] = FUNCTIONS["beta"]

OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}


def show_supported_functions():
    for name, (_, counts) in sorted(FUNCTIONS.items()):
        print(f"  {name:<16} {' or '.join(map(str, counts))} argument(s)")
    print("Operators: + - * / ^ **. Use explicit * for products involving functions.")
    print("gammainc, gammaincc, betainc and betaincc are regularized functions.")


def parse_metric(expression, mass, parameters=None, variable="r"):
    """Compile an expression of r (or rstar) with named real constants.

    Existing parse_metric(expression, mass) calls remain supported.
    """
    if variable not in ("r", "rstar"):
        raise ValueError("variable must be r or rstar.")
    constants = {"M": mass, "pi": np.pi, "e": np.e}
    for name, value in (parameters or {}).items():
        if not name.isidentifier() or name in FUNCTIONS or name in ("r", "rstar", "M", "pi", "e"):
            raise ValueError(f"Invalid or reserved constant name: {name}")
        constants[name] = value
    for name, value in constants.items():
        if not np.isscalar(value) or np.iscomplexobj(value) or not np.isfinite(value):
            raise ValueError(f"Constant {name} must be a finite real number.")

    text = expression.strip().replace("^", "**")
    # Accept common mathematical shorthand such as 2M, 2r, and l(l+1),
    # while preserving actual function calls such as gamma(r) and beta(2, 3).
    text = re.sub(r"(?<=\d)\s*(?=[Mr(])", "*", text)

    def separate_constant_from_parenthesis(match):
        name = match.group(1)
        return match.group(0) if name in FUNCTIONS else f"{name}*("

    text = re.sub(
        r"\b([A-Za-z_]\w*)\s*\(",
        separate_constant_from_parenthesis,
        text,
    )
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ValueError("Invalid mathematical expression.") from error

    def build(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
            if not np.isfinite(float(value)):
                raise ValueError("Numeric constants must be finite.")
            return lambda x: value
        if isinstance(node, ast.Name):
            if node.id == variable:
                return lambda x: x
            if node.id in constants:
                value = constants[node.id]
                return lambda x: value
            raise ValueError(f"Unknown variable '{node.id}'. Use {variable} or {', '.join(constants)}.")
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            operation = OPERATORS[type(node.op)]
            left, right = build(node.left), build(node.right)
            return lambda x: operation(left(x), right(x))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = build(node.operand)
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return lambda x: sign * operand(x)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in FUNCTIONS:
                raise ValueError(f"Unknown function '{name}'; enter help for the list.")
            function, counts = FUNCTIONS[name]
            if node.keywords or len(node.args) not in counts:
                raise ValueError(f"{name} requires {' or '.join(map(str, counts))} positional argument(s).")
            arguments = [build(arg) for arg in node.args]
            return lambda x: function(*(arg(x) for arg in arguments))
        raise ValueError("Unsupported expression; use arithmetic and listed functions only.")

    evaluator = build(tree.body)

    def evaluate(x):
        try:
            x = np.asarray(x, dtype=float)
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                result = np.asarray(evaluator(x))
            if np.iscomplexobj(result) or not np.all(np.isfinite(result)):
                raise ValueError("Expression must return finite real values.")
            return result.item() if result.ndim == 0 else result
        except (ArithmeticError, ValueError, TypeError) as error:
            raise ValueError(f"Cannot evaluate '{expression}': {error}") from error

    return evaluate


def ask_metric_functions(mass):
    while True:
        answer = input("Are f(r) and g(r) the same? [y/n]: ").strip().lower()
        if answer in ("y", "yes", "n", "no"):
            break
        print("Enter y or n.")

    def ask_expression(name):
        while True:
            expression = input(f"Enter {name}(r), or help: ").strip()
            if expression.lower() == "help":
                show_supported_functions()
                continue
            try:
                return parse_metric(expression, mass)
            except (ValueError, OverflowError) as error:
                print(error)

    f = ask_expression("f")
    return f, None if answer in ("y", "yes") else ask_expression("g")
