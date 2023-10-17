import os
import glob
import numpy as np
import h5py
from scipy.spatial import KDTree
import time
import warnings
from pathos.multiprocessing import ProcessingPool as Pool

from simtools.utils import hubble_parameter, magnitude


class GadgetBox:

    def __init__(self, internal_length_unit=None, internal_mass_unit=None,
                 internal_velocity_unit=None):

        # Units for format 1/2 data. Later overwritten if data is in format 3.
        self.unit_length = internal_length_unit
        self.unit_mass = internal_mass_unit
        self.unit_velocity = internal_velocity_unit

    def read_parameters(self, datafile, file_format):

        if file_format == 3:
            if 'Header' in datafile:
                if 'BoxSize' in datafile['Header'].attrs:
                    self.box_size = datafile['Header'].attrs['BoxSize']
                if 'Redshift' in datafile['Header'].attrs:
                    self.redshift = datafile['Header'].attrs['Redshift']
                if 'Time' in datafile['Header'].attrs:
                    self.time = datafile['Header'].attrs['Time']
                if 'NSample' in datafile['Header'].attrs:
                    self.nsample = datafile['Header'].attrs['NSample']
                else:
                    self.nsample = 1
                if 'Omega0' in datafile['Header'].attrs:
                    self.Omega0 = datafile['Header'].attrs['Omega0']
                if 'OmegaBaryon' in datafile['Header'].attrs:
                    self.OmegaBaryon = datafile['Header'].attrs[
                        'OmegaBaryon']
                if 'OmegaLambda' in datafile['Header'].attrs:
                    self.OmegaLambda = datafile['Header'].attrs[
                        'OmegaLambda']
                if 'HubbleParam' in datafile['Header'].attrs:
                    self.h = datafile['Header'].attrs['HubbleParam']
                if 'Hubble' in datafile['Header'].attrs:
                    self.hubble = datafile['Header'].attrs['Hubble']
                if 'UnitLength_in_cm' in datafile['Header'].attrs:
                    self.unit_length = datafile['Header'].attrs[
                        'UnitLength_in_cm']
                if 'UnitMass_in_g' in datafile['Header'].attrs:
                    self.unit_mass = datafile['Header'].attrs[
                        'UnitMass_in_g']
                if 'UnitVelocity_in_cm_per_s' in datafile['Header'].attrs:
                    self.unit_velocity = datafile['Header'].attrs[
                        'UnitVelocity_in_cm_per_s']
            if 'Parameters' in datafile:
                if 'Time' in datafile['Parameters'].attrs:
                    self.time = datafile['Parameters'].attrs['Time']
                if 'ComovingIntegrationOn' in datafile['Parameters'].attrs:
                    if datafile['Parameters'].attrs[
                                'ComovingIntegrationOn'] == 1:
                        self.scale_factor = self.time
                else:
                    print('Assuming comoving integration.')
                    self.scale_factor = self.time
                if 'NSample' in datafile['Parameters'].attrs:
                    self.nsample = datafile['Parameters'].attrs['NSample']
                else:
                    self.nsample = 1
                if 'Omega0' in datafile['Parameters'].attrs:
                    self.Omega0 = datafile['Parameters'].attrs['Omega0']
                if 'OmegaBaryon' in datafile['Parameters'].attrs:
                    self.OmegaBaryon = datafile['Parameters'].attrs[
                        'OmegaBaryon']
                if 'OmegaLambda' in datafile['Parameters'].attrs:
                    self.OmegaLambda = datafile['Parameters'].attrs[
                        'OmegaLambda']
                if 'HubbleParam' in datafile['Parameters'].attrs:
                    self.h = datafile['Parameters'].attrs['HubbleParam']
                if 'Hubble' in datafile['Parameters'].attrs:
                    self.hubble = datafile['Parameters'].attrs['Hubble']
                if 'UnitLength_in_cm' in datafile['Parameters'].attrs:
                    self.unit_length = datafile['Parameters'].attrs[
                        'UnitLength_in_cm']
                if 'UnitMass_in_g' in datafile['Parameters'].attrs:
                    self.unit_mass = datafile['Parameters'].attrs[
                        'UnitMass_in_g']
                if 'UnitVelocity_in_cm_per_s' in datafile['Parameters'].attrs:
                    self.unit_velocity = datafile['Parameters'].attrs[
                        'UnitVelocity_in_cm_per_s']

            self.cm_per_kpc = 3.085678e21
            self.g_per_1e10Msun = 1.989e43
            self.cmps_per_kmps = 1.0e5
            if self.unit_length is None:
                warnings.warn('No value for `UnitLength_in_cm` found!'
                              ' Assuming GADGET-4 default of {}.'.format(
                                self.cm_per_kpc))
                self.unit_length = self.cm_per_kpc
            if self.unit_mass is None:
                warnings.warn('No value for `UnitMass_in_g` found!'
                              ' Assuming GADGET-4 default of {}.'.format(
                                self.g_per_1e10Msun))
                self.unit_mass = self.g_per_1e10Msun
            if self.unit_velocity is None:
                warnings.warn('No value for `UnitVelocity_in_cm_per_s` found!'
                              ' Assuming GADGET-4 default of {}.'.format(
                                self.cmps_per_kmps))
                self.unit_velocity = self.cmps_per_kmps
            length_norm = self.unit_length / self.cm_per_kpc
            mass_norm = self.unit_mass / self.g_per_1e10Msun
            velocity_norm = self.unit_velocity / self.cmps_per_kmps
            self.gravitational_constant = 43009.1727 * mass_norm / \
                (length_norm * velocity_norm**2)

            if not hasattr(self, 'Omega0'):
                self.Omega0 = 0
            if not hasattr(self, 'OmegaBaryon'):
                self.OmegaBaryon = 0
            if not hasattr(self, 'OmegaLambda'):
                self.OmegaLambda = 0
            if not hasattr(self, 'h'):
                warnings.warn('No value for `HubbleParam` found!'
                              ' Assuming GADGET-4 default of 0.7.')
                self.h = 0.7
            if not hasattr(self, 'hubble'):
                self.hubble = 0.1 * length_norm / velocity_norm
                warnings.warn('No value for `Hubble` found!'
                              ' Using Hubble={}.'.format(self.hubble))

            self.hubble_constant = self.h * self.hubble
            if hasattr(self, 'redshift'):
                self.hubble_parameter = hubble_parameter(
                    self.redshift, self.hubble_constant,
                    self.Omega0, self.OmegaLambda, 0)

            self.critical_density = 3 * (self.hubble_parameter / self.h)**2 / \
                (8 * np.pi * self.gravitational_constant)

        else:
            offset = 0
            offset += 16
            offset += 4

            datafile.seek(offset, os.SEEK_SET)
            self.number_of_particles_this_file = np.fromfile(
                datafile, dtype=np.int32, count=6)

            offset += 24
            datafile.seek(offset, os.SEEK_SET)
            self.mass_table = np.fromfile(datafile, dtype=np.float64, count=6)

            offset += 48
            datafile.seek(offset, os.SEEK_SET)
            self.scale_factor = np.fromfile(
                datafile, dtype=np.float64, count=1)[0]

            offset += 8
            datafile.seek(offset, os.SEEK_SET)
            self.redshift = np.fromfile(datafile, dtype=np.float64, count=1)[0]

            offset += 8
            offset += 4  # FlagSfr
            offset += 4  # FlagFeedback
            datafile.seek(offset, os.SEEK_SET)
            self.number_of_particles_by_type = np.fromfile(
                datafile, dtype=np.int32, count=6)

            offset += 24
            offset += 4  # FlagCooling
            datafile.seek(offset, os.SEEK_SET)
            self.number_of_files = np.fromfile(
                datafile, dtype=np.int32, count=1)[0]

            offset += 4
            datafile.seek(offset, os.SEEK_SET)
            self.box_size = np.fromfile(datafile, dtype=np.float64, count=1)[0]

            self.number_of_particles = self.number_of_particles_by_type[
                self.particle_type]

            self.cm_per_kpc = 3.085678e21
            self.g_per_1e10Msun = 1.989e43
            self.cmps_per_kmps = 1.0e5
            if self.unit_length is None:
                warnings.warn('No value for `UnitLength_in_cm` found!'
                              ' Assuming GADGET-4 default of {}.'.format(
                                self.cm_per_kpc))
                self.unit_length = self.cm_per_kpc
            if self.unit_mass is None:
                warnings.warn('No value for `UnitMass_in_g` found!'
                              ' Assuming GADGET-4 default of {}.'.format(
                                self.g_per_1e10Msun))
                self.unit_mass = self.g_per_1e10Msun
            if self.unit_velocity is None:
                warnings.warn('No value for `UnitVelocity_in_cm_per_s` found!'
                              ' Assuming GADGET-4 default of {}.'.format(
                                self.cmps_per_kmps))
                self.unit_velocity = self.cmps_per_kmps
            length_norm = self.unit_length / self.cm_per_kpc
            mass_norm = self.unit_mass / self.g_per_1e10Msun
            velocity_norm = self.unit_velocity / self.cmps_per_kmps
            self.gravitational_constant = 43009.1727 * mass_norm / \
                (length_norm * velocity_norm**2)

            offset += 8
            datafile.seek(offset, os.SEEK_SET)
            self.Omega0 = np.fromfile(datafile, dtype=np.float64, count=1)[0]

            offset += 8
            datafile.seek(offset, os.SEEK_SET)
            self.OmegaLambda = np.fromfile(
                datafile, dtype=np.float64, count=1)[0]

            offset += 8
            datafile.seek(offset, os.SEEK_SET)
            self.h = np.fromfile(datafile, dtype=np.float64, count=1)[0]
            self.hubble = 0.1 * length_norm / velocity_norm
            self.hubble_constant = self.h * self.hubble
            self.hubble_parameter = hubble_parameter(
                self.redshift, self.hubble_constant,
                self.Omega0, self.OmegaLambda, 0)

            self.critical_density = 3 * (self.hubble_parameter / self.h)**2 / \
                (8 * np.pi * self.gravitational_constant)

        return


class GadgetSnap(GadgetBox):

    def __init__(self, path, snapshot_filename, snapshot_number, particle_type,
                 load_ids=True, load_coords=True, load_vels=True,
                 load_masses=True, cutout_positions=None, cutout_radii=None,
                 use_kdtree=True, read_mode=1, npool=None,
                 unit_length_in_cm=None, unit_mass_in_g=None,
                 unit_velocity_in_cm_per_s=None, to_physical=False,
                 number_of_particles=None, buffer=0.0, verbose=True):

        super().__init__(unit_length_in_cm, unit_mass_in_g,
                         unit_velocity_in_cm_per_s)

        self.snapshot_path = path
        self.snapshot_filename = snapshot_filename
        self.snapshot_number = snapshot_number
        self.particle_type = particle_type
        self.cutout_positions = cutout_positions
        self.cutout_radii = cutout_radii
        self.use_kdtree = use_kdtree
        self.read_mode = read_mode
        self.npool = npool
        self.buffer = buffer

        if snapshot_filename.split('.')[-1] == 'hdf5':
            self.snapshot_format = 3
        else:
            self.snapshot_format = 2

        snapshot_files = glob.glob(path + '/{}'.format(
            snapshot_filename.format('%03d' % snapshot_number)))

        nsnap = len(snapshot_files)
        if nsnap > 0:
            self.has_snap = True
            if nsnap > 1:  # order files by sub-file number
                snapshot_files = self.reorder_subfiles(
                    snapshot_filename, snapshot_files)
            if verbose:
                print('Found {} snapshot file(s) for snapshot {} in directory'
                      ' {}'.format(nsnap, snapshot_number, path))
                start = time.time()
            self.read_snap_(snapshot_files, number_of_particles, load_ids,
                            load_coords, load_vels, load_masses,
                            cutout_positions, cutout_radii, read_mode)
            if verbose:
                print("...Loaded in {0} seconds\n".format(
                    round(time.time() - start, 4)))

            if to_physical:
                self.coords *= self.scale_factor
                self.vels = self.to_physical_velocity(self.vels)
                self.scale_phys = self.scale_factor
            else:
                self.scale_phys = 1.0
        else:
            self.has_snap = False
            warnings.warn('No snapshot files found!')

    def reorder_subfiles(self, filename, subfiles):
        subfile_err_msg = "Multiple files consistent with '{}' that " \
                          "aren't sub-files. Enter a more specific " \
                          "filename/wildcard".format(filename)
        try:
            splt = np.array([f.split(".") for f in subfiles])
        except ValueError:
            raise ValueError(subfile_err_msg)
        ii, found_subfiles = 0, False
        while (not found_subfiles) and (ii < len(splt[0])):
            found_subfiles = np.all([x.isdigit() for x in splt[:, ii]])
            ii += 1
        if found_subfiles:
            subnums = [int(x) for x in splt[:, ii - 1]]
            return np.array(subfiles)[np.argsort(subnums)]
        else:
            raise ValueError(subfile_err_msg)

    def read_snap_(self, filenames, number_of_particles, load_ids, load_coords,
                   load_vels, load_masses, cutout_positions, cutout_radii,
                   read_mode):

        def read_snap_legacy(fnames):
            snap = open(fnames[0], 'rb')

            self.read_parameters(snap, self.snapshot_format)
            self.number_of_particles = number_of_particles
            coords_all, vels_all, ids_all, masses_all = [], [], [], []
            if self.particle_type == 0:
                us_all, rhos_all = [], []

            npart_total = np.sum(self.number_of_particles_by_type)
            npart = self.number_of_particles_by_type[self.particle_type]
            ptype_offset = np.sum(np.array(
                self.number_of_particles_by_type[:self.particle_type + 1])) \
                - npart

            idx_with_mass = np.where(self.mass_table == 0)[0]
            npart_by_type_in_mass_block = self.number_of_particles_by_type[
                idx_with_mass]
            npart_total_in_mass_block = np.sum(
                npart_by_type_in_mass_block)
            masses_from_table = True
            if self.particle_type in idx_with_mass:
                masses_from_table = False
                ptype_ind_in_mass_block = np.where(
                    idx_with_mass == self.particle_type)[0][0]
                ptype_offset_in_mass_block = np.sum(np.array(
                    npart_by_type_in_mass_block[:ptype_ind_in_mass_block+1])) \
                    - npart

            for f in fnames:
                snap = open(f, 'rb')

                offset = 264  # includes 2 x 4 byte buffers
                offset += 16
                offset += 4  # 1st 4 byte buffer
                offset += 16
                offset += 3 * ptype_offset * 4
                snap.seek(offset, os.SEEK_SET)
                coords = np.fromfile(
                    snap, dtype=np.float32, count=3 * npart).reshape(
                    npart, 3)
                coords_all.append(coords)
                offset -= 3 * ptype_offset * 4

                # Increment beyond the POS block
                offset += 3 * npart_total * 4
                offset += 4  # 2nd 4 byte buffer
                offset += 4  # 1st 4 byte buffer
                offset += 16
                offset += 3 * ptype_offset * 4
                snap.seek(offset, os.SEEK_SET)
                vels = np.fromfile(
                    snap, dtype=np.float32, count=3 * npart).reshape(
                    npart, 3)
                vels_all.append(vels)
                offset -= 3 * ptype_offset * 4

                # Increment beyond the VEL block
                offset += 3 * npart_total * 4
                offset += 4  # 2nd 4 byte buffer
                offset += 4  # 1st 4 byte buffer
                offset += 16
                offset += ptype_offset * 4
                snap.seek(offset, os.SEEK_SET)
                ids = np.fromfile(snap, dtype=np.uint32, count=npart)
                ids_all.append(ids)
                offset -= ptype_offset * 4

                # Increment beyond the IDS block
                offset += npart_total * 4
                offset += 4  # 2nd 4 byte buffer
                offset += 4  # 1st 4 byte buffer
                offset += 16
                if not masses_from_table:
                    offset += ptype_offset_in_mass_block * 4
                    snap.seek(offset, os.SEEK_SET)
                    masses = np.fromfile(
                        snap, dtype=np.float32, count=npart)
                    masses_all.append(masses)
                    offset -= ptype_offset_in_mass_block * 4

                if self.particle_type == 0:
                    # Increment beyond the mass block
                    offset += npart_total_in_mass_block * 4
                    offset += 4  # 2nd 4 byte buffer
                    offset += 4  # 1st 4 byte buffer
                    offset += 16
                    offset += ptype_offset * 4
                    snap.seek(offset, os.SEEK_SET)
                    u = np.fromfile(snap, dtype=np.float32, count=npart)
                    us_all.append(u)
                    offset -= ptype_offset * 4

                    # Increment beyond the u block
                    offset += npart_total * 4
                    offset += 4  # 2nd 4 byte buffer
                    offset += 4  # 1st 4 byte buffer
                    offset += 16
                    offset += ptype_offset * 4
                    snap.seek(offset, os.SEEK_SET)
                    rho = np.fromfile(snap, dtype=np.float32, count=npart)
                    rhos_all.append(rho)
                    offset -= ptype_offset * 4

            if cutout_positions is not None:
                coords = np.concatenate(coords_all)
                r = magnitude(coords - cutout_positions[0])
                inds = np.argwhere(r < cutout_radii[0]).flatten()

                self.ids = np.concatenate(ids_all)[inds]
                self.coords = np.concatenate(coords_all)[inds]
                self.vels = np.concatenate(vels_all)[inds]
                if masses_from_table:
                    self.masses = self.mass_table[self.particle_type]
                else:
                    masses = np.concatenate(masses_all)[inds]
                    if np.all(masses == masses[0]):
                        self.masses = masses[0]
                    else:
                        self.masses = masses
                if self.particle_type == 0:
                    self.us_all = np.concatenate(us_all)[inds]
                    self.rhos_all = np.concatenate(rhos_all)[inds]
                self.region_offsets = np.array([0, len(inds)])
            else:
                self.ids = np.concatenate(ids_all)
                self.coords = np.concatenate(coords_all)
                self.vels = np.concatenate(vels_all)
                if masses_from_table:
                    self.masses = self.mass_table[self.particle_type]
                else:
                    masses = np.concatenate(masses_all)
                    if np.all(masses == masses[0]):
                        self.masses = masses[0]
                    else:
                        self.masses = masses
                if self.particle_type == 0:
                    self.us_all = np.concatenate(us_all)
                    self.rhos_all = np.concatenate(rhos_all)

            snap.close()

        def read_snap(fnames):

            snap = h5py.File(fnames[0], 'r')

            self.read_parameters(snap, self.snapshot_format)

            def read_files(ii):
                f = fnames[ii]
                if self.snapshot_format == 3:
                    snap = h5py.File(f, 'r')
                else:
                    snap = open(f, 'rb')

                snappt = snap['PartType{}'.format(self.particle_type)]

                if cutout_positions is not None:
                    coords = snappt['Coordinates'][()]
                    if self.use_kdtree:
                        kdtree = KDTree(coords,
                                        boxsize=self.box_size*(1 + self.buffer))
                        cutout_inds = kdtree.query_ball_point(
                            cutout_positions, cutout_radii)
                    else:
                        cutout_inds = []
                        for pos, rad in zip(cutout_positions, cutout_radii):
                            r = magnitude(coords - pos)
                            cutout_inds.append(np.argwhere(r < rad).flatten())
                    region_lens = [len(inds) for inds in cutout_inds]
                    cutout_inds = np.hstack(cutout_inds).astype(int)
                    if len(cutout_inds) == 0:
                        nc = len(cutout_positions)
                        return [np.array([], dtype=np.uint64)]*nc, \
                            [np.array([]).reshape(0, 3)]*nc, \
                            [np.array([]).reshape(0, 3)]*nc, \
                            [np.array([])]*nc
                    coords = coords[cutout_inds]
                    coords = np.split(coords, np.cumsum(region_lens))[:-1]
                else:
                    cutout_inds = None

                if load_ids:
                    if cutout_inds is None:
                        ids = snappt['ParticleIDs'][()]
                    else:
                        if read_mode == 1:
                            ids = snappt['ParticleIDs'][()]
                        elif read_mode == 2:
                            ids = snappt['ParticleIDs']
                        ids = ids[cutout_inds]
                        ids = np.split(ids, np.cumsum(region_lens))[:-1]
                else:
                    ids = None

                if load_coords:
                    if cutout_inds is None:
                        coords = snappt['Coordinates'][()]
                else:
                    coords = None

                if load_vels:
                    if cutout_inds is None:
                        vels = snappt['Velocities'][()]
                    else:
                        if read_mode == 1:
                            vels = snappt['Velocities'][()]
                        elif read_mode == 2:
                            vels = snappt['Velocities']
                        vels = vels[cutout_inds]
                        vels = np.split(vels, np.cumsum(region_lens))[:-1]
                else:
                    vels = None

                if load_masses:
                    if 'Masses' in list(snappt):
                        if cutout_inds is None:
                            masses = snappt['Masses'][()]
                        else:
                            if read_mode == 1:
                                masses = snappt['Masses'][()]
                            elif read_mode == 2:
                                masses = snappt['Masses']
                            masses = masses[cutout_inds]
                            masses = np.split(
                                masses, np.cumsum(region_lens))[:-1]
                    else:
                        masses = (snap['Header'].attrs['MassTable'])[
                            self.particle_type]
                else:
                    masses = None

                if 'Metallicity' in list(snappt):
                    if cutout_inds is None:
                        metallicities = snappt['Metallicity'][()]
                    else:
                        if read_mode == 1:
                            metallicities = snappt['Metallicity'][()]
                        elif read_mode == 2:
                            metallicities = snappt['Metallicity']
                        metallicities = metallicities[cutout_inds]
                        metallicities = np.split(
                            metallicities, np.cumsum(region_lens))[:-1]
                else:
                    metallicities = None

                if 'StellarFormationTime' in list(snappt):
                    if cutout_inds is None:
                        formation_times = snappt['StellarFormationTime'][()]
                    else:
                        if read_mode == 1:
                            formation_times = snappt['StellarFormationTime'][
                                ()]
                        elif read_mode == 2:
                            formation_times = snappt['StellarFormationTime']
                        formation_times = formation_times[cutout_inds]
                        formation_times = np.split(
                            formation_times, np.cumsum(region_lens))[:-1]
                else:
                    formation_times = None

                snap.close()

                return ids, coords, vels, masses, metallicities, \
                    formation_times

            if self.npool is None:
                snapdata = []
                for fi in range(len(fnames)):
                    snapdata.append(read_files(fi))
            else:
                print('Starting multiprocessing pool with {} processes'.format(
                    self.npool))
                snapdata = Pool(self.npool).map(
                    read_files, np.arange(len(fnames)))

            def stack(index):
                if cutout_positions is None:
                    return np.concatenate([x[index] for x in snapdata]), None
                else:
                    regions = []
                    for ri in range(len(cutout_positions)):
                        regions.append(
                            np.concatenate([x[index][ri] for x in snapdata]))
                    lens = [len(region) for region in regions]
                    offsets = np.concatenate([[0], np.cumsum(lens)])
                    return np.concatenate(regions), offsets

            region_offsets = None
            if load_ids:
                self.ids, region_offsets = stack(0)
            if load_coords:
                self.coords, region_offsets = stack(1)
            if load_vels:
                self.vels, region_offsets = stack(2)
            if load_masses:
                if (not isinstance(snapdata[0][3], np.ndarray)) and \
                        (not isinstance(snapdata[0][3], list)):
                    self.masses = snapdata[0][3]
                else:
                    self.masses, region_offsets = stack(3)
            if snapdata[0][4] is not None:
                self.metallicities, region_offsets = stack(4)
            if snapdata[0][5] is not None:
                self.formation_times, region_offsets = stack(5)
            self.region_offsets = region_offsets

            snap.close()

        if self.snapshot_format == 3:
            read_snap(filenames)
        else:
            read_snap_legacy(filenames)

        return

    def to_physical_velocity(self, velocities):
        return velocities * np.sqrt(1 / (1 + self.redshift))


class GadgetCat(GadgetBox):

    def __init__(self, path, catalogue_filename, snapshot_number,
                 particle_type, unit_length_in_cm=None, unit_mass_in_g=None,
                 unit_velocity_in_cm_per_s=None, verbose=True):

        super().__init__(unit_length_in_cm, unit_mass_in_g,
                         unit_velocity_in_cm_per_s)

        self.catalogue_path = path
        self.catalogue_filename = catalogue_filename
        self.snapshot_number = snapshot_number
        self.particle_type = particle_type

        if catalogue_filename.split('.')[-1] == 'hdf5':
            self.catalogue_format = 3
        else:
            self.catalogue_format = 2

        catalogue_files = np.sort(glob.glob(path + '/{}'.format(
            catalogue_filename.format('%03d' % snapshot_number))))
        ncat = len(catalogue_files)
        if ncat > 0:
            self.has_cat = True
            if ncat > 1:  # order files by sub-file number
                catalogue_files = self.reorder_subfiles(
                    catalogue_filename, catalogue_files)
            if verbose:
                print('Found {} halo catalogue file(s) for snapshot {} in '
                      'directory {}'.format(ncat, snapshot_number, path))
                start = time.time()
            self.group, self.halo = self.read_halos(catalogue_files)
            if verbose:
                print("...Loaded in {0} seconds\n".format(
                    round(time.time() - start, 4)))
        if ncat == 0 or self.group is None:
            self.has_cat = False
            warnings.warn('No catalogue files found!')

    def reorder_subfiles(self, filename, subfiles):
        subfile_err_msg = "Multiple files consistent with '{}' that " \
                          "aren't sub-files. Enter a more specific " \
                          "filename/wildcard".format(filename)
        try:
            splt = np.array([f.split(".") for f in subfiles])
        except ValueError:
            raise ValueError(subfile_err_msg)
        ii, found_subfiles = 0, False
        while (not found_subfiles) and (ii < len(splt[0])):
            found_subfiles = np.all([x.isdigit() for x in splt[:, ii]])
            ii += 1
        if found_subfiles:
            subnums = [int(x) for x in splt[:, ii - 1]]
            return np.array(subfiles)[np.argsort(subnums)]
        else:
            raise ValueError(subfile_err_msg)

    def read_halos(self, filenames):

        group = dict()
        halo = dict()

        halo_cat = h5py.File(filenames[0], 'r')
        if not hasattr(self, 'redshift'):
            self.read_parameters(halo_cat, self.catalogue_format)
        self.number_of_groups = halo_cat['Header'].attrs['Ngroups_Total']
        self.number_of_halos = halo_cat['Header'].attrs['Nsubhalos_Total']

        group_keys_float = ['R_200', 'R_500', 'M_200', 'M_500', 'Mass',
                            'V_200', 'V_500', 'A_200', 'A_500', 'Vol_200',
                            'Vol_500']
        group_keys_int = ['Len', 'Offset', 'FirstSub', 'Nsubs']
        halo_keys_float = ['Mass', 'HalfmassRad']
        halo_keys_int = ['Len', 'Offset', 'IDMostbound', 'HaloGroupNr',
                         'HaloRankInGr']
        for gkey in group_keys_float:
            group[gkey] = np.empty(self.number_of_groups)
        for gkey in group_keys_int:
            group[gkey] = np.empty(self.number_of_groups, dtype=np.int64)
        for hkey in halo_keys_float:
            halo[hkey] = np.empty(self.number_of_halos)
        for hkey in halo_keys_int:
            halo[hkey] = np.empty(
                self.number_of_halos, dtype=np.int64)
        group['Pos'] = np.empty((self.number_of_groups, 3))
        group['Vel'] = np.empty((self.number_of_groups, 3))
        halo['Pos'] = np.empty((self.number_of_halos, 3))
        halo['CM'] = np.empty((self.number_of_halos, 3))
        halo['Vel'] = np.empty((self.number_of_halos, 3))

        gidx, hidx = 0, 0
        for filename in filenames:
            halo_cat = h5py.File(filename, 'r')

            ngroups = int(halo_cat['Header'].attrs['Ngroups_ThisFile'])
            nhalos = int(halo_cat['Header'].attrs['Nsubhalos_ThisFile'])
            gslice = slice(gidx, gidx + ngroups)
            gidx += ngroups
            hslice = slice(hidx, hidx + nhalos)
            hidx += nhalos

            if ngroups == 0:
                continue

            config_options = list(halo_cat['Config'].attrs)

            R_200 = halo_cat['Group']['Group_R_Crit200'][()]
            R_500 = halo_cat['Group']['Group_R_Crit500'][()]
            M_200 = halo_cat['Group']['Group_M_Crit200'][()]
            M_500 = halo_cat['Group']['Group_M_Crit500'][()]
            group['R_200'][gslice] = R_200
            group['R_500'][gslice] = R_500
            group['M_200'][gslice] = M_200
            group['M_500'][gslice] = M_500
            group['Mass'][gslice] = halo_cat['Group']['GroupMassType'][()][
                :, self.particle_type]
            group['Pos'][gslice] = halo_cat['Group']['GroupPos'][()]
            group['Vel'][gslice] = halo_cat['Group']['GroupVel'][()]
            group['Len'][gslice] = halo_cat['Group']['GroupLenType'][()][
                :, self.particle_type]
            group['Offset'][gslice] = halo_cat['Group']['GroupOffsetType'][()][
                :, self.particle_type]
            if 'SUBFIND' or 'SUBFIND_HBT' in config_options:
                group['FirstSub'][gslice] = halo_cat['Group'][
                    'GroupFirstSub'][()]
                group['Nsubs'][gslice] = halo_cat['Group']['GroupNsubs'][()]

            np.seterr(divide='ignore', invalid='ignore')
            V_200 = np.sqrt(self.gravitational_constant * M_200 / R_200)
            V_500 = np.sqrt(self.gravitational_constant * M_500 / R_500)
            group['V_200'][gslice] = V_200
            group['V_500'][gslice] = V_500
            group['A_200'][gslice] = V_200**2 / R_200
            group['A_500'][gslice] = V_500**2 / R_500
            group['Vol_200'][gslice] = 4 * np.pi * R_200**3 / 3
            group['Vol_500'][gslice] = 4 * np.pi * R_500**3 / 3
            np.seterr(divide='warn', invalid='warn')

            if 'SUBFIND' or 'SUBFIND_HBT' in config_options:
                halo['Mass'][hslice] = halo_cat[
                    'Subhalo']['SubhaloMassType'][()][:, self.particle_type]
                halo['CM'][hslice] = halo_cat['Subhalo']['SubhaloCM'][()]
                halo['Pos'][hslice] = halo_cat['Subhalo']['SubhaloPos'][()]
                halo['Vel'][hslice] = halo_cat['Subhalo']['SubhaloVel'][()]
                halo['HalfmassRad'][hslice] = halo_cat['Subhalo'][
                    'SubhaloHalfmassRadType'][()][:, self.particle_type]
                halo['Len'][hslice] = halo_cat[
                    'Subhalo']['SubhaloLenType'][()][:, self.particle_type]
                halo['Offset'][hslice] = halo_cat[
                    'Subhalo']['SubhaloOffsetType'][()][:, self.particle_type]
                halo['IDMostbound'][hslice] = halo_cat['Subhalo'][
                    'SubhaloIDMostbound'][()]
                halo['HaloGroupNr'][hslice] = halo_cat['Subhalo'][
                    'SubhaloGroupNr'][()]
                halo['HaloRankInGr'][hslice] = halo_cat['Subhalo'][
                    'SubhaloRankInGr'][()]

            halo_cat.close()

        if gidx == 0:
            return None, None
        else:
            return group, halo


class VelociraptorCat:

    def __init__(self, path, catalogue_filename, snapshot_number,
                 particle_type, thidv=int(1e12), verbose=True):

        self.catalogue_path = path
        self.catalogue_filename = catalogue_filename
        self.snapshot_number = snapshot_number
        self.particle_type = particle_type
        self.thidv = thidv

        catalogue_files = glob.glob(
            path + '/{}'.format(catalogue_filename.format(
                '%03d' % snapshot_number, 'catalog_groups')))
        ncat = len(catalogue_files)
        if ncat > 0:
            self.has_cat = True
            if ncat > 1:
                subnums = [int(catfile.split('.')[-1]) for catfile in
                           catalogue_files]
                catalogue_files = np.array(catalogue_files)[
                    np.argsort(subnums)]
            if verbose:
                print('Found {} halo catalogue file(s) for snapshot {} in '
                      'directory {}'.format(ncat, snapshot_number, path))
                start = time.time()
            self.read_params()
            self.group, self.halo = self.read_halos(catalogue_files)
            if verbose:
                print("...Loaded in {0} seconds\n".format(
                    round(time.time() - start, 4)))
        else:
            self.has_cat = False
            warnings.warn('No catalogue files found!')

    def read_params(self):

        param_data = []
        for fext in ['siminfo', 'units']:
            fname = glob.glob(self.catalogue_path + '/*{}*{}'.format(
                self.snapshot_number, fext))[0]
            fdata = open(fname, 'r')
            data = []
            for line in fdata:
                s = line.split()
                if len(s) > 2:
                    data.append([s[0], s[2]])
            param_data.append(data)

        siminfo, units = param_data
        siminfo, units = np.array(siminfo), np.array(units)
        siminfo = dict(zip(siminfo[:, 0], siminfo[:, 1].astype(float)))
        units = dict(zip(units[:, 0], units[:, 1].astype(float)))

        if siminfo['Cosmological_Sim'] == 1:
            self.scale_factor = siminfo['ScaleFactor']
            self.redshift = (1 / self.scale_factor) - 1
        self.Omega0 = siminfo['Omega_m']
        self.OmegaBaryon = siminfo['Omega_b']
        self.OmegaDM = siminfo['Omega_cdm']
        self.OmegaLambda = siminfo['Omega_Lambda']
        self.h = siminfo['h_val']
        self.hubble = siminfo['Hubble_unit']
        self.hubble_constant = self.h * self.hubble
        self.hubble_parameter = hubble_parameter(
            self.redshift, self.hubble_constant,
            self.Omega0, self.OmegaLambda, 0)
        self.unit_length = units['Length_unit_to_kpc']
        self.unit_mass = units['Mass_unit_to_solarmass']
        self.unit_velocity = units['Velocity_unit_to_kms']

        self.gravitational_constant = siminfo['Gravity']

        self.critical_density = 3 * (self.hubble_parameter / self.h) ** 2 / \
            (8 * np.pi * self.gravitational_constant)

    def read_halos(self, catalogue_files):

        group = dict()
        halo = dict()

        catfile = catalogue_files[0]
        halo_cat = h5py.File(catfile, 'r')
        ngroups = halo_cat['Total_num_of_groups'][()][0]

        group_keys_float = ['R_200', 'R_200m', 'M_200', 'M_200m', 'Mass',
                            'V_200', 'A_200', 'Vol_200']
        group_keys_int = ['GroupID', 'Len', 'FirstSub', 'Nsubs']
        halo_keys_float = ['Mass', 'HalfmassRad']
        halo_keys_int = ['HaloID', 'IDMostbound', 'Offset', 'Len',
                         'HaloGroupID', 'ParentHaloID', 'HaloRankInGr']
        for gkey in group_keys_float:
            group[gkey] = np.empty(ngroups)
        for gkey in group_keys_int:
            group[gkey] = np.empty(ngroups, dtype=np.int64)
        for hkey in halo_keys_float:
            halo[hkey] = np.empty(ngroups)
        for hkey in halo_keys_int:
            halo[hkey] = np.empty(ngroups, dtype=np.int64)
        group['Pos'] = np.empty((ngroups, 3))
        group['CM'] = np.empty((ngroups, 3))
        group['PosMinPot'] = np.empty((ngroups, 3))
        group['Vel'] = np.empty((ngroups, 3))
        group['VelCM'] = np.empty((ngroups, 3))
        halo['Pos'] = np.empty((ngroups, 3))
        halo['CM'] = np.empty((ngroups, 3))
        halo['PosMinPot'] = np.empty((ngroups, 3))
        halo['Vel'] = np.empty((ngroups, 3))
        halo['VelCM'] = np.empty((ngroups, 3))
        halo['ParticleIDs'] = []

        gidx, hidx = 0, 0
        for catfile in catalogue_files:
            cat_groups = h5py.File(catfile, 'r')

            ngroups = int(cat_groups['Num_of_groups'][()][0])
            gslice = slice(gidx, gidx + ngroups)
            gidx += ngroups

            group['Len'][gslice] = cat_groups['Group_Size'][()]
            group['Nsubs'][gslice] = cat_groups[
                'Number_of_substructures_in_halo'][()]

            hoffset = cat_groups['Offset'][()]
            halo['Offset'][gslice] = hoffset

            parents = cat_groups['Parent_halo_ID'][()]
            halo['ParentHaloID'][gslice] = parents
            rankingrp = np.zeros(len(parents), dtype=np.int32)
            counts = np.unique(parents[parents > -1], return_counts=True)[1]
            rankidx = len(np.argwhere(parents == -1))
            for k, c in enumerate(counts):
                rankingrp[rankidx:rankidx+c] = np.arange(1, c+1)
                rankidx += c
            halo['HaloRankInGr'][gslice] = rankingrp

            cat_groups.close()

            catfile_particles = catfile.replace(
                'catalog_groups', 'catalog_particles')
            cat_part = h5py.File(catfile_particles, 'r')
            npart = cat_part['Num_of_particles_in_groups'][()][0]
            if len(hoffset) == 1:
                halo['Len'][gslice] = np.array([npart])
            else:
                hlen = hoffset[1:] - hoffset[:-1]
                halo['Len'][gslice] = np.append(hlen, npart - hoffset[-1])
            for pids in np.split(cat_part['Particle_IDs'],
                                 np.append(hoffset[1:], npart))[:-1]:
                halo['ParticleIDs'].append(pids)

            cat_part.close()

            catfile_props = catfile.replace(
                'catalog_groups', 'properties')
            cat_props = h5py.File(catfile_props, 'r')
            groupids = cat_props['ID'][()]
            group['GroupID'][gslice] = groupids
            group['FirstSub'][gslice] = groupids
            self.snapshot_number = int(groupids[0] / self.thidv)
            R_200 = cat_props['R_200crit'][()] * self.h
            M_200 = cat_props['Mass_200crit'][()] * self.h
            group['R_200'][gslice] = R_200
            group['M_200'][gslice] = M_200
            group['R_200m'][gslice] = cat_props['R_200mean'][()] * self.h
            group['M_200m'][gslice] = cat_props['Mass_200mean'][()] * self.h
            group['Mass'][gslice] = cat_props['Mass_FOF'][()] * self.h
            posx, posy, posz = \
                cat_props['Xcmbp'][()] * self.h / self.scale_factor, \
                cat_props['Ycmbp'][()] * self.h / self.scale_factor, \
                cat_props['Zcmbp'][()] * self.h / self.scale_factor
            group['Pos'][gslice] = np.vstack((posx, posy, posz)).T
            cmx, cmy, cmz = \
                cat_props['Xc'][()] * self.h / self.scale_factor, \
                cat_props['Yc'][()] * self.h / self.scale_factor, \
                cat_props['Zc'][()] * self.h / self.scale_factor
            group['CM'][gslice] = np.vstack((cmx, cmy, cmz)).T
            # mpx, mpy, mpz = \
            #     cat_props['Xcminpot'][()] * self.h / self.scale_factor, \
            #     cat_props['Ycminpot'][()] * self.h / self.scale_factor, \
            #     cat_props['Zcminpot'][()] * self.h / self.scale_factor
            # group['PosMinPot'][gslice] = np.vstack((mpx, mpy, mpz)).T
            velx, vely, velz = cat_props['VXcmbp'][()],\
                cat_props['VYcmbp'][()], cat_props['VZcmbp'][()]
            group['Vel'][gslice] = np.vstack((velx, vely, velz)).T
            velxcm, velycm, velzcm = cat_props['VXc'][()],\
                cat_props['VYc'][()], cat_props['VZc'][()]
            group['VelCM'][gslice] = np.vstack((velxcm, velycm, velzcm)).T

            np.seterr(divide='ignore', invalid='ignore')
            V_200 = np.sqrt(self.gravitational_constant * M_200 / R_200)
            group['V_200'][gslice] = V_200
            group['A_200'][gslice] = V_200**2 / R_200
            group['Vol_200'][gslice] = 4 * np.pi * R_200**3 / 3

            haloids = groupids
            halo['HaloID'][gslice] = haloids
            halo['HaloGroupID'][gslice] = haloids
            halo['IDMostbound'][gslice] = cat_props['ID_mbp'][()]
            halo['Mass'][gslice] = cat_props['Mass_tot'][()] * self.h
            halo['CM'][gslice] = np.vstack((cmx, cmy, cmz)).T
            halo['Pos'][gslice] = np.vstack((posx, posy, posz)).T
            # halo['PosMinPot'][gslice] = np.vstack((mpx, mpy, mpz)).T
            halo['Vel'][gslice] = np.vstack((velx, vely, velz)).T
            halo['VelCM'][gslice] = np.vstack((velxcm, velycm, velzcm)).T
            halo['HalfmassRad'][gslice] = cat_props['R_HalfMass'][()] * self.h

            cat_props.close()

        return group, halo


class AHFCat:

    def __init__(self, path, catalogue_filename, snapshot_number,
                 particle_type, verbose=True):

        self.catalogue_path = path
        self.catalogue_filename = catalogue_filename
        self.snapshot_number = snapshot_number
        self.particle_type = particle_type

        catalogue_files = glob.glob(path + '/{}'.format(
            catalogue_filename.format('%03d' % snapshot_number, 'halos')))
        ncat = len(catalogue_files)
        if ncat > 0:
            self.has_cat = True
            if verbose:
                print('Found {} halo catalogue file(s) for snapshot {} in '
                      'directory {}'.format(ncat, snapshot_number, path))
                start = time.time()
            # TODO: Allow reading of multiple catalogue files per snapshot
            self.halo = self.read_halos(catalogue_files[0])
            if verbose:
                print("...Loaded in {0} seconds\n".format(
                    round(time.time() - start, 4)))
        if ncat == 0 or self.halo is None:
            self.has_cat = False
            warnings.warn('No catalogue files found!')

    def read_halos(self, filename):

        halo = dict()

        with open(filename, 'r') as f:
            linecount = sum(1 for _ in f)
        if linecount == 1:  # empty catalogue
            return

        halo['HaloIDs'] = np.loadtxt(
            filename, dtype=np.uint64, skiprows=0, usecols=0)
        halo['HostIDs'] = np.loadtxt(
            filename, dtype=np.uint64, skiprows=0, usecols=1)

        data = np.loadtxt(filename, skiprows=0)[:, 2:]
        halo['Nsubs'] = data[:, 0].astype(np.int32)
        halo['Mass'] = data[:, 1]
        halo['Len'] = data[:, 2].astype(np.int32)
        halo['Pos'] = data[:, 3:6]
        halo['Vel'] = data[:, 6:9]
        halo['Radius'] = data[:, 9]
        halo['MostBoundParticleOffset'] = data[:, 12]
        halo['CenterofMassOffset'] = data[:, 13]
        halo['AngularMomentum'] = data[:, 19:22]

        filename = filename.replace('halos', 'particles')
        particle_ids = np.loadtxt(filename, dtype=np.uint64,
                                  skiprows=1, usecols=0)
        particle_types = np.loadtxt(filename, dtype=np.uint64,
                                    skiprows=1, usecols=1)

        nhalos = len(halo['HaloIDs'])
        halo['ParticleIDs'] = []
        for i, n in enumerate(range(nhalos-1), 1):
            npart = halo['Len'][n]
            start = np.sum(np.array(halo['Len'][:n+1])) - npart

            ptypes = particle_types[start+i:start+npart+i]
            ptype_inds = np.argwhere(ptypes == self.particle_type).flatten()
            pids = (particle_ids[start+i:start+npart+i])[ptype_inds]
            halo['ParticleIDs'].append(pids)

        return halo
