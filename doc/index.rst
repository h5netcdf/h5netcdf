.. h5netcdf documentation master file, created by
   sphinx-quickstart on Sat Mar  5 16:30:15 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

h5netcdf: A Python interface for the netCDF4 file-format based on h5py
======================================================================

:Release: |release|
:Date: |today|

**h5netcdf** is an open source project and Python package that provides an interface
for the `netCDF4 file-format`_ that reads and writes local or remote HDF5 files directly
via `h5py`_ or `h5pyd`_, without relying on the Unidata netCDF library.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For users

   Overview <self>
   Feature comparison <feature>
   New API Reference <api>
   Legacy API Reference <legacyapi>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For developers

   Developers Guide <devguide>
   Changelog <changelog>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Community

   GitHub issues <https://github.com/h5netcdf/h5netcdf/issues>

.. include:: ../README.rst
   :start-after: .. why-h5netcdf
   :end-before:  .. changelog

History
-------

The project was started in early 2015. The first commit was made on 7th of April in 2015
by Stephan Hoyer. The first `official` ``h5netcdf`` announcement was made by Stephan on the
`xarray issue tracker`_ only one day later.

The library evolved constantly over the years (fixing bugs and adding enhancements)
and gained contributions from 19 other :ref:`contributors` so far. The library is widely used,
especially as backend within `xarray`_.

Early 2020 Kai Mühlbauer started to add contributions and after some time he volunteered
to help in maintaining ``h5netcdf``. Two years later in January 2022 Stephan handed the
project-lead over to Kai. ``h5netcdf`` version 1.0 was released on 31st of March 2022.

.. _netCDF4 file-format: https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#netcdf_4_spec
.. _h5py: https://www.h5py.org/
.. _h5pyd: https://github.com/HDFGroup/h5pyd
.. _xarray issue tracker: https://github.com/pydata/xarray/issues/23#issuecomment-90780331

.. include:: ../README.rst
   :start-after: .. license

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
