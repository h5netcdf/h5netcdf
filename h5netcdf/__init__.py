"""
h5netcdf
========

A Python library for the netCDF4 file-format that directly reads and writes
HDF5 files via h5py, without using the Unidata netCDF library.

For more details on the netCDF4 file format, see:
https://www.unidata.ucar.edu/software/netcdf/docs/netcdf/NetCDF_002d4-Format.html
"""
from .core import Dataset, Group, Variable
