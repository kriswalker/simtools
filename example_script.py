from simtools.box import Snapshot, Catalogue
from simtools.sim_readers import GadgetSnapshot, GadgetCatalogue

###############################################################################

snapshot_dir = 'path/to/snapshots'
catalaogue_dir = 'path/to/catalogues'
snapshot_filename = 'snapshot_{}.hdf5'
catalogue_filename = 'fof_subhalo_tab_{}.hdf5'
read_mode = 1

particle_type = 1
snapshot_number = 48

###############################################################################

snapshot = Snapshot(GadgetSnapshot,
                    {'path': snapshot_dir,
                     'snapshot_filename': snapshot_filename,
                     'snapshot_number': snapshot_number,
                     'particle_type': particle_type,
                     'read_mode': read_mode,
                     'buffer': 1.0e-7,
                     'verbose': True}
                    )

catalogue = Catalogue(GadgetCatalogue,
                      {'path': catalaogue_dir,
                       'catalogue_filename': catalogue_filename,
                       'snapshot_number': snapshot_number,
                       'particle_type': particle_type,
                       'verbose': True}
                      )
