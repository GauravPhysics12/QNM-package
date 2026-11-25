"""JNW axial evolution; M is ADM mass and rstar=0 is the singularity.

Potential: Chowdhury & Banerjee, Phys. Rev. D 102, 124051 (2020), Eq. 19.
Uses a midpoint diamond update, not the endpoint-potential Eq. 23 stencil.
"""

import numpy as np
from scipy.integrate import solve_ivp


def jnw_potential_grid(rstar, M=1.0, nu=0.78, ell=2, potential=None):
    """Evaluate the axial potential strictly outside the singularity.

    z=((r-b)/b)**(1-nu) obeys a regular ODE starting at z(0)=0.
    Work directly with z to avoid rounding tiny r-b to zero.
    """
    x = np.asarray(rstar, dtype=float)
    if x.size == 0 or np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("JNW potential requires finite rstar > 0.")
    if not np.isfinite(M) or M <= 0 or not 0 < nu < 1:
        raise ValueError("Require M > 0 and 0 < nu < 1.")
    if not isinstance(ell, (int, np.integer)) or ell < 2:
        raise ValueError("Axial gravitational perturbations require integer ell >= 2.")
    if potential is not None and potential.coordinate == "rstar":
        return potential(x)
    b = 2.0 * M / nu
    a = 1.0 - nu

    def derivative(t, state):
        z = float(state[0])
        if z < 0:
            raise ValueError("Negative transformed radial coordinate.")
        log1py = 0.0 if z == 0 else np.logaddexp(0.0, np.log(z) / a)
        return [a / b * np.exp(-nu * log1py)]

    sol = solve_ivp(
        derivative, (0.0, float(np.max(x))), [0.0],
        method="DOP853", rtol=1e-11, atol=1e-13,
        max_step=1.0, dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    z = sol.sol(x.ravel())[0].reshape(x.shape)
    if np.any(z <= 0):
        raise ValueError("Unresolved distance from the JNW singularity.")

    if potential is not None:
        r = b * (1.0 + np.exp(np.log(z) / a))
        if np.any(r <= b):
            raise ValueError(
                "r-b rounded to zero for a custom radial potential. "
                "Use the built-in stable JNW potential or supply V(rstar)."
            )
        return potential(x, r)

    # Algebraically equivalent to Eq. 19, with positive, stable terms.
    log1py = np.logaddexp(0.0, np.log(z) / a)
    inv = np.exp(-log1py)
    ratio = -np.expm1(-log1py)  # y/(1+y), y=(r-b)/b
    L = ell * (ell + 1)
    bracket = (
        3.0 * a*a * inv*inv
        + 4.0 * (L - 3.0*nu) * ratio * inv
        + 4.0 * L * ratio*ratio
    )
    potential = np.exp(-2.0*nu*log1py) * bracket / (4.0*b*b*z*z)
    if not np.all(np.isfinite(potential)):
        raise ValueError("Non-finite JNW potential; check parameters and resolution.")
    return potential.item() if potential.ndim == 0 else potential


def initialize_jnw(u, v, v0=12.0, sigma=2.0):
    """Boundary-compatible Gaussian pulse on u=0; diagonal is zero."""
    if not np.isfinite(v0) or v0 <= 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("For this pulse, v0 and sigma must be finite and positive.")
    v = np.asarray(v, dtype=float)
    psi = np.zeros((len(u), len(v)), dtype=float)
    # Gaussian minus its reflected image, evaluated without cancellation.
    psi[0] = np.exp(-0.5*((v-v0)/sigma)**2) * (
        -np.expm1(-2.0*v*v0/sigma**2)
    )
    psi[0, 0] = 0.0
    return psi


def evolve_jnw(u, v, psi, M=1.0, nu=0.78, ell=2, potential=None):
    """Evolve only j>i. The boundary psi[i,i]=0 is never evolved.

    The midpoint stencil avoids sampling the singular potential at rstar=0.
    Convergence at this singular boundary must be checked numerically.
    """
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if u.ndim != 1 or v.ndim != 1 or len(u) < 3 or u.shape != v.shape:
        raise ValueError("Use equal one-dimensional u,v grids with >=3 points.")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("Grid must be finite.")
    if not np.allclose(u, v, rtol=1e-12, atol=1e-12) or u[0] != 0:
        raise ValueError("JNW branch requires identical u,v grids starting at zero.")
    h = u[1] - u[0]
    if h <= 0 or not np.allclose(np.diff(u), h, rtol=1e-10, atol=1e-12):
        raise ValueError("Grid must be increasing and uniform.")
    if psi.shape != (len(u), len(v)) or psi.dtype.kind not in "fc":
        raise ValueError("psi must be a floating/complex array matching the grid.")

    n = len(u)
    x = 0.5*h*np.arange(1, n)
    weights = np.zeros(n)
    weights[1:] = h*h/8.0 * jnw_potential_grid(x, M, nu, ell, potential=potential)

    for i in range(1, n):
        current, previous = psi[i], psi[i-1]
        current[:i+1] = 0.0
        for j in range(i+1, n):
            W, E, S = current[j-1], previous[j], previous[j-1]
            current[j] = W + E - S - weights[j-i]*(W+E)
        if not np.all(np.isfinite(current[i+1:])):
            raise FloatingPointError("Evolution became non-finite; refine the grid.")
    return psi


def extract_jnw(u, v, psi, x_obs):
    """Extract at exactly x_obs by linear interpolation along each u row."""
    if not np.isfinite(x_obs) or not 0 < x_obs < 0.5*(v[-1]-u[0]):
        raise ValueError("Choose 0 < x_obs < grid_max/2.")
    target = u + 2.0*x_obs
    rows = np.flatnonzero(target <= v[-1])
    if len(rows) < 2:
        raise ValueError("Too few waveform samples; reduce x_obs.")
    values = np.array([np.interp(target[i], v, psi[i]) for i in rows])
    return u[rows]+x_obs, values
