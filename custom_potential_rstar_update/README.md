# JNW axial update

This is an update for your existing qnmtd project, not a complete replacement package.

1. Back up your current reference.py.
2. Replace it with the supplied reference.py.
3. Add jnw.py inside the existing qnmtd folder.
4. Replace potentials.py and metric_input.py with the supplied versions, and
   replace evolution.py with the supplied version. Keep the other modules,
   including __init__.py, coordinates.py, initial_data.py, extraction.py, and prony.py.
5. Run python3 reference.py from the project folder.

Choose j for JNW, M=1 and nu=0.78. Here M means ADM mass and b=2*M/nu.
To reproduce a convention with b=2 at nu=0.78, enter M=0.78.
The Schwarzschild branch uses the existing configure_coordinates, evolve,
initialize_field, extract_waveform, and fit_prony interfaces from this conversation.

The driver asks whether to use the model's default axial potential, one of the
Regge-Wheeler/Zerilli/Pöschl-Teller examples, or a custom expression. Custom
potentials may use `r` or `rstar`, and constants `M`, `l`, `ell`, `s`, and (for
JNW) `b`, `nu`. For example, enter `1/r**2 - 3*M/r**3` as a radial expression,
or `0.15/cosh(0.1*rstar)**2` as an rstar expression. Common shorthand such as
`l(l+1)` is accepted. Enter `help` at the expression prompt for supported
functions, including gamma and beta functions.

## Numerical choices

JNW uses rstar=0 at the singularity, evolves only v>u, and sets the diagonal field
to zero. Thus JNW rstar is positive in the evolved region. Schwarzschild instead
uses the shift rstar=4M at r=4M; its mapping covers negative and positive rstar.
The two branches do not share the same additive rstar origin. A custom V(rstar)
is evaluated in the chosen branch's convention; a custom V(r) is evaluated using
that branch's radius mapping. The rest of the allocated JNW square below the
diagonal is unused.
The axial potential is Eq. (19) of Chowdhury and Banerjee, Phys. Rev. D 102,
124051 (2020): https://doi.org/10.1103/PhysRevD.102.124051.
The potential uses the JNW radial coordinate, not Schwarzschild areal radius.

The variable z=((r-b)/b)^(1-nu) satisfies
dz/drstar=(1-nu)/b * (1+z^(1/(1-nu)))^(-nu), z(0)=0.
This regular forward ODE avoids crossing the singularity or subtracting nearly
equal radii. The potential is computed directly from z.

The JNW update samples potential at the cell midpoint, not at W and E.
It does not evaluate V(0), including for the first interior cell. Accuracy at
the singular boundary must be assessed by convergence, not assumed from the
smooth-region stencil. No full JNW QNM benchmark has been established here.

The initial Gaussian has its reflected image subtracted to match the zero
corner and diagonal boundary. For centre=1 and width=1 this noticeably changes
the pulse; for centre=12 and width=2 the image is negligible around the peak.
The JNW waveform is interpolated at the requested observation coordinate.
The plot opens before optional Prony fitting. Close it to continue.

## Checks performed

- Syntax checks for both Python files.
- Transformed potential compared with the original radius formula at 50 points:
  maximum relative difference approximately 1.6e-12, using an independent
  hypergeometric expression for the forward coordinate integral.
- Near-boundary check: rstar^2*V approaches 0.75.
- Finite evolution and exact zero diagonal at 501, 1001 and 2001 grid points;
  interval [0,200], M=1, nu=0.78, ell=2, centre=12, width=2, observer=40.
  Common-time L2 waveform difference ratio approximately 4.69 on refinement.
- User's 1000-point [0,500] grid with centre=1, width=1 checked for finite evolution.

These checks establish basic numerical consistency, not validated physical QNM
frequencies. Inspect fitting-window, order, and grid-resolution dependence.
