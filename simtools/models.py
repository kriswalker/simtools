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


def density_profile_nfw(r, scale_radius, central_density):

    x = r / scale_radius

    return central_density / (x * (1 + x)**2)


def log_density_slope_profile_nfw(r, scale_radius):

    x = r / scale_radius

    return -1 - 2 * x / (x + 1)


def mass_profile_nfw(r, scale_radius, central_density):

    x = r / scale_radius

    return 4 * np.pi * central_density * scale_radius**3 * \
        (1 / (1 + x) + np.log(1 + x) - 1)


def circular_velocity_profile_nfw(r, virial_radius, concentration):

    x = r / virial_radius

    return np.sqrt(g(concentration) * (
            np.log(1 + concentration * x) - concentration * x /
            (1 + concentration * x)) / x)


def velocity_dispersion_profile_nfw(r, virial_radius, concentration,
                                    anisotropy, kind='total'):

    x = r / virial_radius

    def integrand(s, b, c):
        return (s**(2 * b - 3) * np.log(1 + c * s) / (1 + c * s)**2) - \
            (c * s**(2 * b - 2) / (1 + c * s)**3)
    dispint = [quad(integrand, xi, np.inf, args=(anisotropy, concentration))[0]
               for xi in x]
    if kind == 'total':
        factor = 3 - 2 * anisotropy
    elif kind == 'radial':
        factor = 1
    elif kind == 'azimuthal':
        factor = 1 - anisotropy
    else:
        raise ValueError(
            'kind not recognized. Must be either `total`, `radial`, or'
            ' `azimuthal`.')
    return np.sqrt(factor * g(concentration) * (1 + concentration * x)**2 *
                   x**(1 - 2 * anisotropy) * np.array(dispint))
