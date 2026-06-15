"""No-op stub for hdf5plugin.

The cluster's real hdf5plugin (1.4.1, linked against HDF5 1.14.2) is ABI-incompatible
with the cluster h5py (3.16, needs HDF5 1.14.6): importing the real hdf5plugin breaks
h5py with `undefined symbol H5T_IEEE_F16BE_g`. OlmoEarth's data.dataset imports
hdf5plugin only to enable HDF5 compression filters for its *pretrain* HDF5 datasets.
The PASTIS eval path (pastis_dataset / pastis_processor) reads .pt/.npy only and never
uses HDF5, so a no-op stub is safe and lets the package import cleanly.
"""
# Intentionally empty. Provides the FILTERS attr some code probes for, just in case.
FILTERS: dict = {}
