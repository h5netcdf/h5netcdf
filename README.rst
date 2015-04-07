h5netcdf
========

.. image:: https://travis-ci.org/shoyer/h5netcdf.svg?branch=master
    :target: https://travis-ci.org/shoyer/h5netcdf

A Python interface for netCDF4_ files reads and writes HDF5 files API directly
via h5py_, without relying on the Unidata netCDF library.

The API design closely follows netCDF4-python_. It currently passes basic
tests for reading and writing netCDF4 files with Python, but it has not been
tested for compatibility with other netCDF4 interfaces.

License: `3-clause BSD`_

.. _netCDF4: https://www.unidata.ucar.edu/software/netcdf/docs/netcdf/NetCDF_002d4-Format.html
.. _h5py: http://www.h5py.org/
.. _netCDF4-python: https://github.com/Unidata/netcdf4-python
.. _3-clause BSD: https://github.com/shoyer/h5netcdf/blob/master/LICENSE.txt
