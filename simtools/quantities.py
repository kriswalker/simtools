import numpy as np
from simtools.utils import vector_norm


def hubble_parameter(z, H0, Omega_m, Omega_Lambda, Omega_k):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 +
                        Omega_k * (1 + z)**2 +
                        Omega_Lambda)


def add_hubble_flow(velocity, position, z, H0, **Omega):
    return velocity + position * hubble_parameter(z, H0, **Omega)


def radial_velocity(coords, vels, return_radii=False):
    rads = vector_norm(coords)
    if return_radii:
        return np.einsum('...i,...i', vels, coords) / rads, rads
    else:
        return np.einsum('...i,...i', vels, coords) / rads


def calc_specific_angular_momentum(x, v, npart):
    return np.sum(np.cross(x, v), axis=0) / npart
