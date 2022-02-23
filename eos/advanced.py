# Joe Nyhan, 22 February 2022
# Calculates a table of values for EoS from the advanced QHD parameter sets


from numpy import pi, sqrt
from scipy.integrate import quad
from scipy.optimize import root
from numba import njit

# ========== PARAMETER SET

NL3     = 1
FSUGold = 1

# ========== PARMETERS

if NL3:
    # parameters
    g_s     = sqrt(104.3871)
    g_v     = sqrt(165.5854)
    g_rho   = sqrt(79.6000)
    kappa   = 3.8599e-3 #GeV
    lmbda   = -.01591
    zeta    = 0.00
    Lmbda_v = 0.00

    # masses (GeV)
    M       = .939
    m_s     = .5081940
    m_omega = .7825
    m_rho   = .763

elif FSUGold:
    # parameters
    g_s     = sqrt(112.1996)
    g_v     = sqrt(204.5469)
    g_rho   = sqrt(138.4701)
    kappa   = 1.4203e-3 #GeV
    lmbda   = 0.0238
    zeta    = 0.06
    Lmbda_v = 0.03

    # masses (GeV)
    M       = .939
    m_s     = .4915000
    m_omega = .783
    m_rho   = .763

# ========== EQUATIONS OF MOTION

@njit
def k_from_rho(rho):
    return pi**2 * (3*pi**2 * rho)**(-2/3)

def f_phi0(phi0, rho_p, rho_n):
    """
    see Diener (9.20a), p. 79; used for rootfinding
    """
    mstar = M - g_s*phi0

    kp = k_from_rho(rho_p)
    kn = k_from_rho(rho_n)

    f = lambda k: (k**2*mstar)/sqrt(k**2 + mstar**2)
    p_int, p_err = quad(f,0,kp)
    n_int, n_err = quad(f,0,kn)

    return phi0 -  g_s/(m_s**2) * (1/pi**2 * (p_int + n_int) -kappa/2 * (g_s * phi0)**2 - lmbda/6 * (g_s * phi0)**3)

def f_vectorFields(f, rho_p, rho_n):
    """
    see Diener (9.20b), p. 79; used for rootfinding
    """
    V0, b0 = f

    V0_val = V0 - g_v/m_omega**2 * (rho_p + rho_n - zeta/6 * (g_v * V0)**3 - 2*Lmbda_v*(g_v * V0)*(g_rho * b0)**2)
    b0_val = b0 - g_rho/m_rho**2 * (1/2 * (rho_p - rho_n) - 2*Lmbda_v*(g_v * V0)**2*(g_rho * b0))

    return V0_val, b0_val

def f_eps(phi0, V0, b0, )



    
    