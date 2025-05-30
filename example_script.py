from simtools.box import Snapshot, Catalogue
from simtools.sim_readers import GadgetSnapshot, GadgetCatalogue

###############################################################################
def read_sim_data(dir_name, snap = 122, particle_type = 1, lean_mode=True, load_ids=False, load_coords=False, load_vels=False, verbose=True):
    """
    particle_type: 1 (Default); 0 -> gas, 1 -> Highres DM, 4 -> Stars
    lean_mode: 
        Default(True): Only load halo catalogue data; else load particle data as well.
    
    load_ids: Load particle IDs. Default: False
    load_coords: Load particle coordinates. Default: False
    load_vels: Load particle velocities. Default: False

    """
    snapshot_filename = dir_name + '/snapshot_{}.hdf5'.format('%03d' % snap)
    catalogue_filename = dir_name + '/fof_subhalo_tab_{}.hdf5'.format('%03d' % snap)
    read_mode = 1

    particle_type = particle_type

    catalogue_data = Catalogue(GadgetCatalogue,
                      {'catalogue_filename': catalogue_filename,
                       'particle_type': particle_type,
                       'verbose': verbose}
                      )
    if lean_mode:
        return catalogue_data

    else:
        particle_data = Snapshot(GadgetSnapshot,
                    {'snapshot_filename': snapshot_filename,
                     'particle_type': particle_type,
                     'read_mode': read_mode,
                     'load_ids':load_ids,
                     'load_coords':load_coords,
                     'load_vels':load_vels,
                     'buffer': 1.0e-7,
                     'verbose': verbose}
                    )
        return particle_data, catalogue_data
