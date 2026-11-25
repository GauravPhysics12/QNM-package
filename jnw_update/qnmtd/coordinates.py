# qnmtd/coordinates.py

import numpy as np
from scipy.integrate import solve_ivp


_radius_mapping = None


def get_r_from_rstar(
    f,
    g=None,
    *,
    r_anchor,
    rstar_anchor,
    rstar_min,
    rstar_max,
    rtol=1e-11,
    atol=1e-12,
    max_step=1.0,
):
    """
    Build r(rstar) from:
        dr/drstar = f(r)              when g is omitted
        dr/drstar = sqrt(f(r)*g(r))   otherwise

    The reference point r(rstar_anchor) = r_anchor fixes the shift.
    """
    if not callable(f) or (g is not None and not callable(g)):
        raise TypeError("f and g must be callable functions.")

    values = [r_anchor, rstar_anchor, rstar_min, rstar_max]

    if not np.all(np.isfinite(values)):
        raise ValueError("Reference point and coordinate limits must be finite.")

    if not rstar_min < rstar_anchor < rstar_max:
        raise ValueError(
            "rstar_anchor must lie strictly inside the integration interval."
        )

    def derivative(rstar, y):
        r = float(y[0])
        f_value = float(f(r))

        if g is None:
            speed = f_value
        else:
            product = f_value * float(g(r))

            if not np.isfinite(product) or product < 0:
                raise ValueError(
                    f"f(r)*g(r) must be finite and nonnegative at r={r}."
                )

            speed = np.sqrt(product)

        if not np.isfinite(speed):
            raise ValueError(f"Non-finite dr/drstar at r={r}.")

        return [speed]

    def integrate(endpoint):
        solution = solve_ivp(
            derivative,
            (rstar_anchor, endpoint),
            [r_anchor],
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True,
        )

        if not solution.success:
            raise RuntimeError(
                f"Coordinate integration failed: {solution.message}"
            )

        return solution

    left = integrate(rstar_min)
    right = integrate(rstar_max)

    def radius(rstar):
        x = np.asarray(rstar, dtype=float)
        flat = x.reshape(-1)

        if not np.all(np.isfinite(flat)):
            raise ValueError("Tortoise coordinates must be finite.")

        if np.any((flat < rstar_min) | (flat > rstar_max)):
            raise ValueError(
                f"rstar must lie within [{rstar_min}, {rstar_max}]."
            )

        result = np.empty_like(flat)
        use_left = flat < rstar_anchor

        if np.any(use_left):
            result[use_left] = left.sol(flat[use_left])[0]

        if np.any(~use_left):
            result[~use_left] = right.sol(flat[~use_left])[0]

        result = result.reshape(x.shape)
        return result.item() if result.ndim == 0 else result

    return radius


def configure_coordinates(f, g=None, **settings):
    """Build the mapping once, before evolution."""
    global _radius_mapping

    _radius_mapping = get_r_from_rstar(f, g, **settings)


def r1(rstar):
    """Evaluate the configured mapping for a scalar or array."""
    if _radius_mapping is None:
        raise RuntimeError(
            "Call configure_coordinates() before using r1()."
        )

    return _radius_mapping(rstar)