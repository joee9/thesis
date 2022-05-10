# Joe Nyhan, 22 February 2022
# Calculates a table of values for EoS from the advanced QHD parameter sets

#%%

# from numpy import np.pi, np.sqrt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
from numba import njit
import sys
np.set_printoptions(threshold=sys.maxsize)

# ========== PARAMETER SET

NL3      = 0
FSU_GOLD = 1

if NL3:        eos = 'NL3'
elif FSU_GOLD: eos = 'FSUGold'
# ========== PARMETERS

# masses (GeV)
M    = .939
m_e  = 511e-6
m_mu = .1057
rho0 = .01

if NL3:
    # parameters
    g_s     = np.sqrt(104.3871)
    g_v     = np.sqrt(165.5854)
    g_rho   = np.sqrt(79.6000)
    kappa   = 3.8599e-3 #GeV
    lmbda   = -.01591
    zeta    = 0.00
    Lmbda_v = 0.00

    # masses (GeV)
    m_s     = .5081940
    m_omega = .7825
    m_rho   = .763

elif FSU_GOLD:
    # parameters
    g_s     = np.sqrt(112.1996)
    g_v     = np.sqrt(204.5469)
    g_rho   = np.sqrt(138.4701)
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
    # if rho == 0: return 0
    # return np.pi**2 * (3*np.pi**2 * rho)**(-2/3)
    return (3*np.pi**2*rho)**(1/3)

@njit
def rho_from_k(k):
    return k**3/(3*np.pi**2)

def ns_system(vec, rho):
    rho_n, rho_p, rho_e, rho_mu, phi0, V0, b0 = vec

    k_n = k_from_rho(rho_n)
    k_p = k_from_rho(rho_p)
    k_e = k_from_rho(rho_e)
    k_mu = k_from_rho(rho_mu)

    mu_mu = np.sqrt(k_mu**2 + m_mu**2)
    mu_e = np.sqrt(k_e**2 + m_e**2)
    
    mstar = (M - g_s*phi0)

    f = lambda k: (k**2*mstar)/np.sqrt(k**2 + mstar**2)
    p_int, p_err = quad(f,0,k_p)
    n_int, n_err = quad(f,0,k_n)

    return np.array([
        rho - (rho_n + rho_p), # conservation of baryon number density
        np.sqrt(k_n**2 + mstar**2) - (np.sqrt(k_p**2 + mstar**2) + g_rho*b0 + np.sqrt(k_e**2 + m_e**2)), # beta equilibrium condition
        mu_mu - mu_e, # equilibrium of muons and electrons
        k_p - (k_e**3 + k_mu**3)**(1/3), # charge equilibrium
        phi0 -  g_s/(m_s**2) * (1/np.pi**2 * (p_int + n_int) -kappa/2 * (g_s * phi0)**2 - lmbda/6 * (g_s * phi0)**3), # phi0 equation
        V0 - g_v/m_omega**2 * (rho_p + rho_n - zeta/6 * (g_v * V0)**3 - 2*Lmbda_v*(g_v * V0)*(g_rho * b0)**2),
        b0 - g_rho/m_rho**2 * (1/2 * (rho_p - rho_n) - 2*Lmbda_v*(g_v * V0)**2*(g_rho * b0))
    ])

def ns_system_no_muons(vec, rho):
    rho_n, rho_p, rho_e, phi0, V0, b0 = vec

    k_n = k_from_rho(rho_n)
    k_p = k_from_rho(rho_p)
    k_e = k_from_rho(rho_e)

    mstar = (M - g_s*phi0)

    f = lambda k: (k**2*mstar)/np.sqrt(k**2 + mstar**2)
    p_int, p_err = quad(f,0,k_p)
    n_int, n_err = quad(f,0,k_n)

    return np.array([
        rho - (rho_n + rho_p), # conservation of baryon number density
        np.sqrt(k_n**2 + mstar**2) - (np.sqrt(k_p**2 + mstar**2) + g_rho*b0 + np.sqrt(k_e**2 + m_e**2)), # beta equilibrium condition
        k_p - k_e, # charge equilibrium
        phi0 -  g_s/(m_s**2) * (1/np.pi**2 * (p_int + n_int) -kappa/2 * (g_s * phi0)**2 - lmbda/6 * (g_s * phi0)**3), # phi0 equation
        V0 - g_v/m_omega**2 * (rho_p + rho_n - zeta/6 * (g_v * V0)**3 - 2*Lmbda_v*(g_v * V0)*(g_rho * b0)**2),
        b0 - g_rho/m_rho**2 * (1/2 * (rho_p - rho_n) - 2*Lmbda_v*(g_v * V0)**2*(g_rho * b0))
    ])


def calc_initial_guess(rho):
    phi0 = g_s / (m_s)**2 * rho
    V0 = g_v / (m_omega)**2 * rho
    b0 = - g_rho/m_rho**2 * rho/2
    mu_n = M - g_s * phi0 + g_v * V0 + g_rho * b0
    mu_e = .12*M*(rho/rho0)**2

    mu_p = mu_n - mu_e
    mu_mu = mu_e

    mstar = M - g_s*phi0

    # how to implement?
    if mu_mu < m_mu:
        k_mu = 0
    else:
        k_mu = np.sqrt(mu_mu**2 - m_mu**2)

    k_n = np.sqrt((mu_n -g_v*V0 + g_rho*b0/2)**2 - mstar**2)
    k_p = np.sqrt((mu_p -g_v*V0 - g_rho*b0/2)**2 - mstar**2)
    k_e = np.sqrt(mu_e**2 - m_e**2)
    k_mu = np.sqrt(mu_mu**2 - m_mu**2)

    rho_n = rho_from_k(k_n)
    rho_p = rho_from_k(k_p)
    rho_e = rho_from_k(k_e)
    rho_mu = rho_from_k(k_mu)

    return np.array([
        rho_n,
        rho_p,
        rho_e,
        rho_mu,
        phi0,
        V0,
        b0
    ])
    

def eps(vec):
    if len(vec) == 6:
        rho_n, rho_p, rho_e, phi0, V0, b0 = vec
        rho_mu = 0
    else:
        rho_n, rho_p, rho_e, rho_mu, phi0, V0, b0 = vec

    k_n = k_from_rho(rho_n)
    k_p = k_from_rho(rho_p)
    k_e = k_from_rho(rho_e)
    k_mu = k_from_rho(rho_mu)

    mstar = (M - g_s*phi0)

    first_line = 1/2 * (m_s**2 * phi0**2) + kappa/np.math.factorial(3) * (g_s*phi0)**3 + lmbda/np.math.factorial(4) * (g_s*phi0)**4 - 1/2 *m_omega**2*V0**2 - zeta/np.math.factorial(4) * (g_v*V0)**4
    second_line = - 1/2 * m_rho**2*b0**2 - Lmbda_v* (g_v*V0)**2*(g_rho*b0)**2 + g_v*V0*(rho_n + rho_p) + 1/2 * g_rho*b0*(rho_p-rho_n)
    
    integrand = lambda k, m: k**2 * np.sqrt(k**2 + m**2)
    
    first_integral, first_err = quad(integrand, 0,k_p, args=mstar)
    second_integral, second_err = quad(integrand, 0,k_n, args=mstar)
    third_integral, third_err = quad(integrand, 0,k_e, args=m_e)
    fourth_integral, fourth_err = quad(integrand, 0,k_mu, args=m_mu)

    integrals = 1/np.pi**2 * (first_integral + second_integral + third_integral + fourth_integral)

    return first_line + second_line + integrals

def P(vec):
    if len(vec) == 6:
        rho_n, rho_p, rho_e, phi0, V0, b0 = vec
        rho_mu = 0
    else:
        rho_n, rho_p, rho_e, rho_mu, phi0, V0, b0 = vec

    k_n = k_from_rho(rho_n)
    k_p = k_from_rho(rho_p)
    k_e = k_from_rho(rho_e)
    k_mu = k_from_rho(rho_mu)

    mstar = (M - g_s*phi0)

    first_line = - 1/2 * (m_s**2 * phi0**2) - kappa/np.math.factorial(3) * (g_s*phi0)**3 - lmbda/np.math.factorial(4) * (g_s*phi0)**4 + 1/2 *m_omega**2*V0**2 + zeta/np.math.factorial(4) * (g_v*V0)**4
    second_line = + 1/2 * m_rho**2*b0**2 + Lmbda_v* (g_v*V0)**2*(g_rho*b0)**2
    
    integrand = lambda k,m: k**4/np.sqrt(k**2 + m**2)
    integrand_leptons = lambda k, m: k**2 * np.sqrt(k**2 + m**2)
    
    first_integral, first_err = quad(integrand, 0,k_p, args=mstar)
    second_integral, second_err = quad(integrand, 0,k_n, args=mstar)
    third_integral, third_err = quad(integrand_leptons, 0,k_e, args=m_e)
    fourth_integral, fourth_err = quad(integrand_leptons, 0,k_mu, args=m_mu)

    integrals = 1/(3*np.pi**2) * (first_integral + second_integral + third_integral + fourth_integral)

    return first_line + second_line + integrals


#%%

# rhos = np.logspace(-1,-5, 1000, base=10)
upper = -1
lower = -5
density = 1000
rho_exp = np.linspace(upper,lower,(upper-lower)*density)

output = open('./0-NL3_vals.vals', 'w')

# rhos = [.01]
# rhos = [.1,] # look for P = eps = 1.4

# x0 = calc_initial_guess(rhos[0])
x0 = [.2, .2, .2, .2, .2, .2, .2]
# x0 = [.02, .02, .02, .02, .02, .02, .02]
print(x0)
vals = []

n = -1
for i, exp in enumerate(rho_exp):

    if i % 100 == 0: print(i)
    rho = 10**(exp)

    if x0[3] <= 1e-12:
        n = i
        break

    sol = least_squares(ns_system, x0, args=[rho], method='lm')

    x0 = sol.x
    output.write(f'{eps(x0):.16e}, {P(x0):.16e}\n')

    # vals.append([rho, P(x0), eps(x0)])

x0 = x0[0], x0[1], x0[2], x0[4], x0[5], x0[6]

for i in range(n, len(rho_exp)):

    if i % 100 == 0: print(i)

    exp = rho_exp[i]
    rho = 10**(exp)
    sol = least_squares(ns_system_no_muons, x0, args=[rho], method='lm')

    x0 = sol.x
    output.write(f'{eps(x0):.16e}, {P(x0):.16e}\n')

output.close()


#%%

# with open('./test.txt', 'w') as f:
#     for val in vals:
#         f.write(f'{val[2], val[1]}\n')
