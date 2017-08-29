"""
h5netcdf
========

A Python library for the netCDF4 file-format that directly reads and writes
HDF5 files via h5py, without using the Unidata netCDF library.
"""
from .core import CompatibilityError, File, Group, Variable, __version__
