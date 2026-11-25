
# ==============================================================================
# --- Potential ---
# ==============================================================================
M = 1

# Regge-Wheeler potential
def V_RW(r, l, s):   
    return (1 - 2*M/r) * (l*(l+1)/r**2 + 2*(1 - s**2)*M/r**3)

# Zerilli potential
def V_ZP(r, l, s=None): 
    """Zerilli potential V^+(r) for polar gravitational perturbations."""
    valid_r = r > 2 * M
    V_plus = np.full_like(r, np.nan, dtype=np.float64)
    if not np.any(valid_r):
        return V_plus
    r_valid = r[valid_r]
    n = 0.5 * (l - 1) * (l + 2)
    f = 1 - 2 * M / r_valid
    num = 2 * f * (n**2 * (n + 1) * r_valid**3 + 3 * M * n**2 * r_valid**2 + 9 * M**2 * n * r_valid + 9 * M**3)
    den = r_valid**3 * (n * r_valid + 3 * M)**2
    V_plus[valid_r] = num / den
    return V_plus

# Poschl-Teller potential
def V_PT(r): #
    return 0.15 / (np.cosh(0.1* r))**2