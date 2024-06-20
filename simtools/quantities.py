import numpy as np
from simtools.utils import vector_norm


def hubble_parameter(z, H0, Omega_m, Omega_Lambda, Omega_k):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 +
                        Omega_k * (1 + z)**2 +
                        Omega_Lambda)


def add_hubble_flow(velocity, position, z, H0, **Omega):
    return velocity + position * hubble_parameter(z, H0, **Omega)


def radial_velocity(coords, vels, return_radii=False):
    rads, rhat = vector_norm(
        coords, return_magnitude=True, return_unit_vectors=True)
    if return_radii:
        return np.einsum('...i,...i', vels, rhat), rhat, rads
    else:
        return np.einsum('...i,...i', vels, rhat), rhat


def azimuthal_velocity(coords, vels):
    vr, rhat = radial_velocity(coords, vels)
    vaz = vels - vr[:, np.newaxis] * rhat
    return vector_norm(vaz, return_magnitude=True, return_unit_vectors=True)


def velocity_dispersion(vels, masses=None):

    vels_ = vels - np.mean(vels, axis=0)
    if masses is None or not hasattr(masses, "__len__"):
        return np.sqrt(np.mean(np.einsum('...i,...i', vels_, vels_)))
    else:
        return np.sqrt(
            np.mean(masses * np.einsum('...i,...i', vels_, vels_)) *
            len(masses) / np.sum(masses))


def calc_specific_angular_momentum(x, v, npart):
    return np.sum(np.cross(x, v), axis=0) / npart
