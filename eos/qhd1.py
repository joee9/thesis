# Joe Nyhan, 18 January 2021
#%%
from numpy import sqrt, pi, logspace
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import root

gamma = 2

# masses (in GeV)
m_s = .520
m_omega = .783
M = .939

# other constants
g_s = sqrt(109.6)
g_v = sqrt(190.4)


def f_phi0(phi0, kf):
    """see Diener p. 66, eq 8.21a"""
    mstar = M - g_s * phi0
    # integral = 1/2* mstar * (kf * sqrt(kf**2 + mstar**2) - mstar**2 * log((kf + sqrt(kf**2 + mstar**2))/mstar))
    def f(k):
        return k**2 * mstar/sqrt(k**2 + mstar**2)

    integral, err = quad(f,0,kf)


    return phi0 - (g_s/m_s**2) * (gamma/(2*pi**2)) * integral

def f_V0(kf):
    """see Diener p. 66, eq 8.21b"""
    rho = 2 * (gamma/(6*pi**2))*kf**3
    # for rho, see Diener p. 67, eq. 8.25; note that gamma = 2 has been substituted
    # print(f'{rho=}')
    
    return (g_v/m_omega**2) * rho

def f_eps(phi0, V0, kf):
    """see Diener p. 66, eq 8.24a"""
    mstar = M - g_s * phi0
    # integral = 1/8 * (kf * (2*kf**2 + mstar**2) * sqrt(kf**2 + mstar**2) - mstar**4 * log((kf + sqrt(kf**2 + mstar**2))/mstar))
    def f(k):
        return k**2 * sqrt(k**2 + mstar**2)
    integral, err = quad(f,0,kf)

    return 1/2 * m_s**2 * phi0**2 + 1/2 * m_omega**2 * V0**2 + (gamma/(2*pi**2)) * integral

def f_P(phi0, V0, kf):
    """see Diener p. 66, eq 8.24b"""
    mstar = M - g_s * phi0
    # integral = 1/8 * (kf * (2*kf**2 - 3*mstar**2) * sqrt(kf**2 + mstar**2) + 3*mstar**4 * log((kf + sqrt(kf**2 + mstar**2))/mstar))

    def f(k):
        return k**4 / sqrt(k**2 + mstar**2)
    integral, err = quad(f,0,kf)

    return -1/2 * m_s**2 * phi0**2 + 1/2 * m_omega**2 * V0**2 + (1/3) * (gamma/(2*pi**2)) * integral

# kfs = [1e0]
kfs = logspace(-6,0,12000,base=10)

energy_densities = []
pressures = []

phi0_guess = 10**(-1)
# phi0_guess = 0

for kf in kfs:

    res = root(f_phi0, phi0_guess, args=kf)
    # print(res)
    phi0 = res.x[0]
    # print(phi0)

    V0 = f_V0(kf)

    # print(f'{V0=}, {phi0=}')

    eps = f_eps(phi0, V0, kf)
    p = f_P(phi0, V0, kf)

    energy_densities.append(eps)
    pressures.append(p)

    phi0_guess = phi0

with open('./qhd1-vals.txt', 'w') as f:
    for i in range(len(pressures)):
        p = pressures[i]
        eps = energy_densities[i]
        f.write(f'{eps:.16e}, {p:.16e}\n')
