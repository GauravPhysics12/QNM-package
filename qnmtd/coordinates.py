"""Build the radial map r(rstar) from user-supplied metric functions."""

import numpy as np
from scipy.integrate import solve_ivp


_radius_mapping = None


def _real_scalar(function, r, name):
    """Evaluate a metric function at one radius and require a real scalar."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = np.asarray(function(float(r)))

    if value.ndim != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name}(r) must be real-valued at r={r}.")

    value = float(value.item())

    if not np.isfinite(value):
        raise ValueError(f"{name}(r) is not finite at r={r}.")

    return value


def _make_speed_function(f, g):
    """Return dr/drstar where the metric is real and in its exterior region."""

    def speed(r):
        try:
            f_value = _real_scalar(f, r, "f")

            if g is None:
                # When f and g are identical, dr/drstar = sqrt(f*g) = f.
                if f_value <= 0.0:
                    return None
                return f_value

            g_value = _real_scalar(g, r, "g")

            # Restrict the mapping to the exterior static region.
            if f_value <= 0.0 or g_value <= 0.0:
                return None

            product = f_value * g_value

            if not np.isfinite(product) or product <= 0.0:
                return None

            return float(np.sqrt(product))

        except (ArithmeticError, TypeError, ValueError, OverflowError):
            # A fractional power below its real-valued domain is treated
            # as the inner edge of the allowed radial region.
            return None

    return speed


def _find_outermost_inner_boundary(speed, mass, rstar_min, rstar_max):
    """
    Search inward for the first invalid radius.

    The search starts at a large radius and moves inward. The first invalid
    point brackets the outermost inner boundary of the exterior region.
    """
    scale = max(float(mass), 1e-12)

    r_upper = max(
        100.0 * scale,
        4.0 * max(abs(rstar_min), abs(rstar_max)) + 10.0 * scale,
    )
    r_lower = max(scale * 1e-12, np.finfo(float).tiny)

    radii = np.geomspace(r_upper, r_lower, 4096)

    previous_valid_radius = None

    for radius in radii:
        if speed(radius) is not None:
            previous_valid_radius = float(radius)
            continue

        if previous_valid_radius is None:
            raise ValueError(
                "Could not find a valid exterior radius while searching "
                "for the metric's inner boundary."
            )

        # `low` is invalid and `high` is valid. Refine their boundary.
        low = float(radius)
        high = previous_valid_radius

        for _ in range(100):
            middle = 0.5 * (low + high)

            if speed(middle) is None:
                low = middle
            else:
                high = middle

        # Return the last valid radius just outside the boundary.
        return high

    return None


def _integrate(speed, x_start, x_end, r_start, *, rtol, atol, max_step):
    """Integrate dr/drstar from one coordinate endpoint to another."""

    def derivative(_rstar, y):
        radius = float(y[0])
        value = speed(radius)

        if value is None:
            raise ValueError(
                f"Metric is not real and positive at r={radius}. "
                "The integration reached the inner boundary."
            )

        return [value]

    solution = solve_ivp(
        derivative,
        (x_start, x_end),
        [r_start],
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


def get_r_from_rstar(
    f,
    g=None,
    *,
    rstar_min,
    rstar_max,
    mass=1.0,
    boundary_epsilon=1e-6,
    rtol=1e-10,
    atol=1e-12,
    max_step=1.0,
):
    """
    Build r(rstar) from user-supplied metric functions.

    Uses:
        dr/drstar = f(r)                  if g is omitted
        dr/drstar = sqrt(f(r) * g(r))     otherwise

    If a finite inner boundary is detected, the mapping starts just outside
    it and assigns that cutoff to rstar_min. This chooses the additive
    coordinate shift automatically.

    If no inner boundary is detected, the fallback convention is
    r=4M at rstar=4M.
    """
    if not callable(f) or (g is not None and not callable(g)):
        raise TypeError("f and g must be callable functions.")

    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be finite and positive.")

    if not np.isfinite(rstar_min) or not np.isfinite(rstar_max):
        raise ValueError("Coordinate limits must be finite.")

    if rstar_min >= rstar_max:
        raise ValueError("Require rstar_min < rstar_max.")

    if not np.isfinite(boundary_epsilon) or boundary_epsilon <= 0.0:
        raise ValueError("boundary_epsilon must be finite and positive.")

    speed = _make_speed_function(f, g)

    boundary = _find_outermost_inner_boundary(
        speed,
        mass,
        rstar_min,
        rstar_max,
    )

    if boundary is not None:
        # Keep away from the root/singularity. Potentials can diverge there.
        offset = max(
            boundary_epsilon * max(float(mass), abs(boundary), 1.0),
            100.0 * np.spacing(max(abs(boundary), 1.0)),
        )
        r_start = boundary + offset

        if speed(r_start) is None:
            raise ValueError(
                "Could not start just outside the detected inner boundary. "
                "Increase boundary_epsilon."
            )

        print(
            f"Detected inner boundary near r={boundary:.12g}; "
            f"starting at r={r_start:.12g} and assigning it to "
            f"rstar_min={rstar_min:.6g}."
        )

        solution = _integrate(
            speed,
            rstar_min,
            rstar_max,
            r_start,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        def radius(rstar):
            x = np.asarray(rstar, dtype=float)

            if not np.all(np.isfinite(x)):
                raise ValueError("Tortoise coordinates must be finite.")

            if np.any((x < rstar_min) | (x > rstar_max)):
                raise ValueError(
                    f"rstar must lie within [{rstar_min}, {rstar_max}]."
                )

            result = np.asarray(solution.sol(x)[0])
            result = result.reshape(x.shape)

            return result.item() if result.ndim == 0 else result

        return radius

    # No inner boundary was found. Use a deterministic fallback convention.
    # The additive shift is arbitrary; this preserves the package's previous
    # Schwarzschild-style reference convention without prompting the user.
    r_anchor = 4.0 * float(mass)
    rstar_anchor = 4.0 * float(mass)

    if not rstar_min < rstar_anchor < rstar_max:
        rstar_anchor = 0.5 * (rstar_min + rstar_max)

    if speed(r_anchor) is None:
        raise ValueError(
            "No inner boundary was detected, and the automatic fallback "
            f"radius r={r_anchor} is outside the valid metric domain."
        )

    left = _integrate(
        speed,
        rstar_anchor,
        rstar_min,
        r_anchor,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    right = _integrate(
        speed,
        rstar_anchor,
        rstar_max,
        r_anchor,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    def radius(rstar):
        x = np.asarray(rstar, dtype=float)

        if not np.all(np.isfinite(x)):
            raise ValueError("Tortoise coordinates must be finite.")

        if np.any((x < rstar_min) | (x > rstar_max)):
            raise ValueError(
                f"rstar must lie within [{rstar_min}, {rstar_max}]."
            )

        flat = x.reshape(-1)
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
    """Evaluate the configured r(rstar) mapping."""
    if _radius_mapping is None:
        raise RuntimeError(
            "Call configure_coordinates() before using r1()."
        )

    return _radius_mapping(rstar)