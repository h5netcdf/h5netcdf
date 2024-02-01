"""
h5netcdf
========

A Python library for the netCDF4 file-format that directly reads and writes
HDF5 files via h5py, without using the Unidata netCDF library.
"""

try:
    from ._version import version as __version__
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

from .core import CompatibilityError, Dimension, File, Group, Variable  # noqa
