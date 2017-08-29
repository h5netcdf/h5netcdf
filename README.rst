h5netcdf
========

.. image:: https://travis-ci.org/shoyer/h5netcdf.svg?branch=master
    :target: https://travis-ci.org/shoyer/h5netcdf
.. image:: https://badge.fury.io/py/h5netcdf.svg
    :target: https://pypi.python.org/pypi/h5netcdf/

A Python interface for the netCDF4_ file-format that reads and writes HDF5
files API directly via h5py_, without relying on the Unidata netCDF library.

.. _netCDF4: http://www.unidata.ucar.edu/software/netcdf/docs/file_format_specifications.html#netcdf_4_spec
.. _h5py: http://www.h5py.org/

Why h5netcdf?
-------------

- We've seen occasional reports of better performance with h5py than
  netCDF4-python, though in many cases performance is identical. For
  `one workflow`_, h5netcdf was reported to be almost **4x faster** than
  `netCDF4-python`_.
- It has one less massive binary dependency (netCDF C). If you already have h5py
  installed, reading netCDF4 with h5netcdf may be much easier than installing
  netCDF4-Python.
- Anecdotally, HDF5 users seem to be unexcited about switching to netCDF --
  hopefully this will convince them that the netCDF4 is actually quite sane!
- Finally, side-stepping the netCDF C library (and Cython bindings to it)
  gives us an easier way to identify the source of performance issues and
  bugs.

.. _one workflow: https://github.com/Unidata/netcdf4-python/issues/390#issuecomment-93864839
.. _xarray: http://github.com/pydata/xarray/

Install
-------

Ensure you have a recent version of h5py installed (I recommend using conda_).
At least version 2.1 is required (for dimension scales); versions 2.3 and newer
have been verified to work, though some tests only pass on h5py 2.6. Then:
``pip install h5netcdf``

.. _conda: http://conda.io/

Usage
-----

h5netcdf has two APIs, a new API and a legacy API. Both interfaces currently
reproduce most of the features of the netCDF interface, with the notable
exceptions of:

- support for operations the rename or delete existing objects.
- support for creating unlimited dimensions.

We simply haven't gotten around to implementing these features yet. Patches
would be very welcome.

New API
~~~~~~~

The new API supports direct hierarchical access of variables and groups. Its
design is an adaptation of h5py to the netCDF data model. For example:

.. code-block:: python

    import h5netcdf
    import numpy as np

    with h5netcdf.File('mydata.nc', 'w') as f:
        # set dimensions with a dictionary
        f.dimensions = {'x': 5}
        # and update them with a dict-like interface
        # f.dimensions['x'] = 5
        # f.dimensions.update({'x': 5})

        v = f.create_variable('hello', ('x',), float)
        v[:] = np.ones(5)

        # you don't need to create groups first
        # you also don't need to create dimensions first if you supply data
        # with the new variable
        v = f.create_variable('/grouped/data', ('y',), data=np.arange(10))

        # access and modify attributes with a dict-like interface
        v.attrs['foo'] = 'bar'

        # you can access variables and groups directly using a hierarchical
        # keys like h5py
        print(f['/grouped/data'])

Legacy API
~~~~~~~~~~

The legacy API is designed for compatibility with netCDF4-python_. To use it, import
``h5netcdf.legacyapi``:

.. _netCDF4-python: https://github.com/Unidata/netcdf4-python

.. code-block:: python

    import h5netcdf.legacyapi as netCDF4
    # everything here would also work with this instead:
    # import netCDF4
    import numpy as np

    with netCDF4.Dataset('mydata.nc', 'w') as ds:
        ds.createDimension('x', 5)
        v = ds.createVariable('hello', float, ('x',))
        v[:] = np.ones(5)

        g = ds.createGroup('grouped')
        g.createDimension('y', 10)
        g.createVariable('data', 'i8', ('y',))
        v = g['data']
        v[:] = np.arange(10)
        v.foo = 'bar'
        print(ds.groups['grouped'].variables['data'])

The legacy API is designed to be easy to try-out for netCDF4-python users, but it is not an
exact match. Here is an incomplete list of functionality we don't include:

- Utility functions ``chartostring``, ``num2date``, etc., that are not directly necessary
  for writing netCDF files.
- We don't support the ``endian`` argument to ``createVariable`` yet (see `GitHub issue`_).
- h5netcdf variables do not support automatic masking or scaling (e.g., of values matching
  the ``_FillValue`` attribute). We prefer to leave this functionality to client libraries
  (e.g., xarray_), which can implement their exact desired scaling behavior.

.. _GitHub issue: https://github.com/shoyer/h5netcdf/issues/15

Invalid netCDF files
~~~~~~~~~~~~~~~~~~~~

h5py implements some features that do not (yet) result in valid netCDF files:

- Data types:
    - Booleans
    - Complex values
    - Non-string variable length types
    - Enum types
    - Reference types
- Compression algorithms:
    - Algorithms other than gzip
    - Scale-offset filters

By default, h5netcdf does not allow writing files using any of these features,
as files with such features are not readable by other netCDF tools.
(For backwards compatibility, this is currently only a warning, but in h5netcdf
v0.5 we will raise ``h5netcdf.CompatibilityError``. Use ``invalid_netcdf=False``
to switch to the new behavior.)

However, these are still valid HDF5 files. If you don't care about netCDF
compatibility, you can use these features by setting ``invalid_netcdf=True``
when creating a file:

.. code-block:: python

  # avoid the .nc extension for non-netcdf files
  f = h5netcdf.File('mydata.h5', invalid_netcdf=True)
  ...

  # works with the legacy API, too, though compression options are not exposed
  ds = h5netcdf.legacyapi.Dataset('mydata.h5', invalid_netcdf=True)
  ...

Change Log
----------

Version 0.4 (Aug 29, 2017):

- Add ``invalid_netcdf`` argument. Warnings are now issued by default when
  writing an invalid NetCDF file. See the "Invalid netCDF files" section of the
  README for full details.

Version 0.3.1 (Sep 2, 2016):

- Fix garbage collection issue.
- Add missing ``.flush()`` method for groups.
- Allow creating dimensions of size 0.

Version 0.3.0 (Aug 7, 2016):

- Datasets are now loaded lazily. This should increase performance when opening
  files with a large number of groups and/or variables.
- Support for writing arrays of variable length unicode strings with `dtype=str`
  via the legacy API.
- h5netcdf now writes the _NCProperties attribute for identifying netCDF4 files.

License
-------

`3-clause BSD`_

.. _3-clause BSD: https://github.com/shoyer/h5netcdf/blob/master/LICENSE
