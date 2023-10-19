import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


def g(x):
    return 1 / (np.log(1 + x) - x / (1 + x))


def approx_concentration(delta_c, v):
    c = np.linspace(1, 100, 1000)
    y = c**3 * g(c)
    f = interp1d(y, c)
    return f(3 * delta_c / v)


def density_profile_nfw(r, central_density, scale_radius):

    x = r / scale_radius

    return central_density / (x * (1 + x)**2)


def mass_profile_nfw(r, central_density, scale_radius):

    x = r / scale_radius

    return 4 * np.pi * central_density * scale_radius**3 * \
        (1 / (1 + x) + np.log(1 + x) - 1)


def velocity_dispersion_profile_nfw(x, conc, beta):
    def integrand(s, b, c):
        return ((s**(2 * b - 3) * np.log(1 + c * s)) / (1 + c * s)**2) -\
            ((c * s**(2 * b - 2)) / (1 + c * s)**3)
    dispint = []
    for ri in x:
        dispint.append(quad(integrand, ri, np.inf, args=(beta, conc))[0])
    return np.sqrt(g(conc) * (1 + conc * x)**2 * x**(1 - 2 * beta) *
                   np.array(dispint))
