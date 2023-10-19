import numpy as np
from scipy.signal import savgol_filter

from simtools.utils import magnitude, simple_derivative, churazov_smooth


def bin_halo(coords, n_radial_bins, n_angular_bins, radius_limits):

    rlim1, rlim2 = np.log10(radius_limits)
    radial_bins = np.linspace(rlim1, rlim2, n_radial_bins+1)

    logr = np.log10(magnitude(coords))
    inside_rlims = np.argwhere((logr >= rlim1) & (logr <= rlim2)).flatten()
    logr = logr[inside_rlims]

    if n_angular_bins > 1:
        angular_bins = np.linspace(-np.pi, np.pi, n_angular_bins+1)
    rad_digits = np.digitize(logr, radial_bins)
    binned_inds = []
    for n in range(1, n_radial_bins+1):
        inside_shell = np.argwhere(rad_digits == n).flatten()
        x, y, z = coords[inside_rlims[inside_shell]].T

        if n_angular_bins > 1:
            binned_inds_shell = []
            for coord_pair in [[y, x], [z, x], [z, y]]:
                phi = np.arctan2(*coord_pair)
                ang_digits = np.digitize(phi, angular_bins)
                for m in range(1, n_angular_bins+1):
                    inside_slice = np.argwhere(ang_digits == m).flatten()
                    binned_inds_shell.append(
                        inside_rlims[inside_shell[inside_slice]])
        else:
            binned_inds_shell = inside_shell
        binned_inds.append(binned_inds_shell)

    logrx_offset = radial_bins + (radial_bins[1] - radial_bins[0]) / 2
    redges = 10**radial_bins
    rcenters = 10**logrx_offset[:-1]

    return binned_inds, redges, rcenters


def calc_density_profile(masses, coords=None, n_radial_bins=None,
                         n_angular_bins=None, radius_limits=None,
                         binned_halo=None):

    if binned_halo is None:
        binds, redges, rcenters = bin_halo(
            coords, n_radial_bins, n_angular_bins, radius_limits)
    else:
        binds, redges, rcenters = binned_halo

    n_angular_bins = int(len(binds[0]) / 3)
    density_profiles = []
    for i in range(3 * n_angular_bins):
        if isinstance(masses, np.ndarray):
            bin_masses = np.array([np.sum(masses[inds[i]]) for inds in binds])
        else:
            bin_masses = np.array([masses * len(inds[i]) for inds in binds])
        shell_vols = ((4 * np.pi / 3) * (redges[1:]**3 - redges[:-1]**3))
        density_profiles.append(bin_masses / (shell_vols / n_angular_bins))

    density_profiles = np.array(density_profiles)

    return rcenters, np.median(density_profiles, axis=0)


def calc_log_density_slope_profile(density_profile, r=None, window_length=1,
                                   apply_filter=False, handle_edges=False,
                                   width=None, **savgol_kwargs):

    if apply_filter:

        dsp = savgol_filter(
            np.log10(density_profile), window_length=window_length, deriv=1,
            **savgol_kwargs)

        if handle_edges:
            nedge = int((window_length - 1) / 2)
            where_zero = np.argwhere(density_profile == 0).flatten()
            edge_inner = np.argwhere(density_profile != 0.0).flatten()[0]
            if len(where_zero) != 0:
                where_zero_after_nonzero = where_zero[where_zero > edge_inner]
                if len(where_zero_after_nonzero) > 0:
                    edge_outer = where_zero_after_nonzero[0]-1
                else:
                    edge_outer = len(density_profile) - 1
            else:
                edge_outer = len(density_profile) - 1
            filter_edge_left = slice(edge_inner, edge_inner+nedge+1)
            filter_edge_right = slice(edge_outer-nedge, edge_outer+1)

            r_edge_left = r[filter_edge_left]
            r_edge_right = r[filter_edge_right]
            profile_edge_left = density_profile[filter_edge_left]
            profile_edge_right = density_profile[filter_edge_right]

            if width is None:
                width = dict(savgol_kwargs)['delta']
            smoothed_left = churazov_smooth(
                r_edge_left, profile_edge_left, width)
            smoothed_right = churazov_smooth(
                r_edge_right, profile_edge_right, width)

            dsp_edge_left = simple_derivative(
                np.log10(r[filter_edge_left]), np.array(smoothed_left), 1)
            dsp_edge_right = simple_derivative(
                np.log10(r[filter_edge_right]), np.array(smoothed_right), 1)
            dsp[slice(edge_inner, edge_inner+nedge)] = dsp_edge_left
            dsp[slice(edge_outer-nedge, edge_outer)] = dsp_edge_right

        return dsp

    else:

        return simple_derivative(
            np.log10(r), np.log10(density_profile), window_length)


def calc_mass_profile(masses, coords=None, n_radial_bins=None,
                      radius_limits=None, binned_halo=None):

    if binned_halo is None:
        binds, _, rcenters = bin_halo(
            coords, n_radial_bins, n_angular_bins=1,
            radius_limits=radius_limits)
    else:
        binds, _, rcenters = binned_halo

    return rcenters, np.array([np.sum(masses[inds]) for inds in binds[1]])
