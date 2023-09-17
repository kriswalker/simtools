from simtools.box import Snapshot, Catalogue
from simtools.sim_readers import GadgetSnap, GadgetCat

###############################################################################

snapshot_dir = 'data/DM-L25-N128-eps0.004/snapshots'
catalaogue_dir = 'data/DM-L25-N128-eps0.004/catalogues'
snapshot_filename = 'snapshot_{}.hdf5'
catalogue_filename = 'fof_subhalo_tab_{}.hdf5'
read_mode = 1

particle_type = 1
snapshot_number = 48

###############################################################################

snapshot = Snapshot(GadgetSnap,
                    {'path': snapshot_dir,
                     'snapshot_filename': snapshot_filename,
                     'snapshot_number': snapshot_number,
                     'particle_type': particle_type,
                     'read_mode': read_mode,
                     'to_physical': False,
                     'buffer': 1.0e-7,
                     'verbose': True}
                    )

catalogue = Catalogue(GadgetCat,
                      {'path': catalaogue_dir,
                       'catalogue_filename': catalogue_filename,
                       'snapshot_number': snapshot_number,
                       'particle_type': particle_type,
                       'verbose': True}
                      )
