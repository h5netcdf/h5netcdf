h5netcdf
========

.. image:: https://github.com/h5netcdf/h5netcdf/workflows/CI/badge.svg
    :target: https://github.com/h5netcdf/h5netcdf/actions
.. image:: https://badge.fury.io/py/h5netcdf.svg
    :target: https://pypi.python.org/pypi/h5netcdf/

A Python interface for the netCDF4_ file-format that reads and writes local or
remote HDF5 files directly via h5py_ or h5pyd_, without relying on the Unidata
netCDF library.

.. _netCDF4: http://www.unidata.ucar.edu/software/netcdf/docs/file_format_specifications.html#netcdf_4_spec
.. _h5py: http://www.h5py.org/
.. _h5pyd: https://github.com/HDFGroup/h5pyd

Why h5netcdf?
-------------

- It has one less binary dependency (netCDF C). If you already have h5py
  installed, reading netCDF4 with h5netcdf may be much easier than installing
  netCDF4-Python.
- We've seen occasional reports of better performance with h5py than
  netCDF4-python, though in many cases performance is identical. For
  `one workflow`_, h5netcdf was reported to be almost **4x faster** than
  `netCDF4-python`_.
- Anecdotally, HDF5 users seem to be unexcited about switching to netCDF --
  hopefully this will convince them that netCDF4 is actually quite sane!
- Finally, side-stepping the netCDF C library (and Cython bindings to it)
  gives us an easier way to identify the source of performance issues and
  bugs in the netCDF libraries/specification.

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
exception of support for operations the rename or delete existing objects.
We simply haven't gotten around to implementing this yet. Patches
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

        # add an unlimited dimension
        f.dimensions['z'] = None
        # explicitly resize a dimension and all variables using it
        f.resize_dimension('z', 3)

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
- No support yet for automatic resizing of unlimited dimensions with array
  indexing. This would be a welcome pull request. For now, dimensions can be
  manually resized with ``Group.resize_dimension(dimension, size)``.

.. _GitHub issue: https://github.com/h5netcdf/h5netcdf/issues/15

Invalid netCDF files
~~~~~~~~~~~~~~~~~~~~

h5py implements some features that do not (yet) result in valid netCDF files:

- Data types:
    - Booleans
    - Complex values
    - Non-string variable length types
    - Enum types
    - Reference types
- Arbitrary filters:
    - Scale-offset filters

By default [*]_, h5netcdf will not allow writing files using any of these features,
as files with such features are not readable by other netCDF tools.

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

.. [*] Currently, we only issue a warning, but in a future version of h5netcdf,
       we will raise ``h5netcdf.CompatibilityError``. Use
       ``invalid_netcdf=False`` to switch to the new behavior now.

Decoding variable length strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

h5py 3.0 introduced `new behavior`_ for handling variable length string.
Instead of being automatically decoded with UTF-8 into NumPy arrays of ``str``,
they are required as arrays of ``bytes``.

The legacy API preserves the old behavior of h5py (which matches netCDF4),
and automatically decodes strings.

The new API *also* currently preserves the old behavior of h5py, but issues a
warning that it will change in the future to match h5py. Explicitly set
``decode_vlen_strings=False`` in the ``h5netcdf.File`` constructor to opt-in to
the new behavior early, or set ``decode_vlen_strings=True`` to opt-in to
automatic decoding.

.. _new behavior: https://docs.h5py.org/en/stable/strings.html

Datasets with missing dimension scales
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default [*]_ h5netcdf raises a ``ValueError`` if variables with no dimension
scale associated with one of their axes are accessed.
You can set ``phony_dims='sort'`` when opening a file to let h5netcdf invent
phony dimensions according to `netCDF`_ behaviour.

.. code-block:: python

  # mimic netCDF-behaviour for non-netcdf files
  f = h5netcdf.File('mydata.h5', mode='r', phony_dims='sort')
  ...

Note, that this iterates once over the whole group-hierarchy. This has affects
on performance in case you rely on lazyness of group access.
You can set ``phony_dims='access'`` instead to defer phony dimension creation
to group access time. The created phony dimension naming will differ from
`netCDF`_ behaviour.

.. code-block:: python

  f = h5netcdf.File('mydata.h5', mode='r', phony_dims='access')
  ...

.. _netCDF: https://www.unidata.ucar.edu/software/netcdf/docs/interoperability_hdf5.html
.. [*] Keyword default setting ``phony_dims=None`` for backwards compatibility.

Change Log
----------

Version 0.10.0 (February 11, 2021):

- Replaced ``decode_strings`` with ``decode_vlen_strings``.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Version 0.9.0 (February 7, 2021):

- Special thanks to `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ for
  stepping up as a co-maintainer!
- Support for ``decode_strings``, to restore old behavior with h5py 3.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Version 0.8.1 (July 17, 2020):

- Fix h5py deprecation warning in test suite.

Version 0.8.0 (February 4, 2020):

- Support for reading Datasets with missing dimension scales.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Fixed a bug where ``Datatype`` objects were treated as ``Datasets``.
- Fixed several issues with upstream deprecations.

Version 0.7.4 (June 1, 2019):

- Fixed a flakey test on Python 2.7 and 3.4.

Version 0.7.3 (May 20, 2019):

- Fixed another bug that could result in reusing dimension IDs, when modifying
  existing files.

Version 0.7.1 (Mar 16, 2019):

- Fixed a bug where h5netcdf could write invalid netCDF files with reused
  dimension IDs when dimensions are written in multiple groups.
  netCDF-C 4.6.2 will crash when reading these files, but you can still read
  these files with older versions of the netcdf library (or h5netcdf).
- Updated to use version 2 of ``_NCProperties`` attribute.

Version 0.7 (Feb 26, 2019):

- Support for reading and writing file-like objects (requires h5py 2.9 or
  newer).
  By `Scott Henderson <https://github.com/scottyhq>`_.

Version 0.6.2 (Aug 19, 2018):

- Fixed a bug that prevented creating variables with the same name as
  previously created dimensions in reopened files.

Version 0.6.1 (Jun 8, 2018):

- Compression with arbitrary filters no longer triggers warnings about invalid
  netCDF files, because this is now
  `supported by netCDF <https://github.com/Unidata/netcdf-c/pull/399>`__.

Version 0.6 (Jun 7, 2018):

- Support for reading and writing data to remote HDF5 files via the HDF5 REST
  API using the h5pyd_ package. Any file "path" starting with either
  ``http://``, ``https://``, or ``hdf5://`` will automatically trigger the use
  of this package.
  By `Aleksandar Jelenak <https://github.com/ajelenak-thg>`_.

Version 0.5.1 (Apr 11, 2018):

- Bug fix for files with an unlimited dimension with no associated variables.
  By `Aleksandar Jelenak <https://github.com/ajelenak-thg>`_.

Version 0.5 (Oct 17, 2017):

- Support for creating unlimited dimensions.
  By `Lion Krischer <https://github.com/krischer>`_.

Version 0.4.3 (Oct 10, 2017):

- Fix test suite failure with recent versions of netCDF4-Python.

Version 0.4.2 (Sep 12, 2017):

- Raise ``AttributeError`` rather than ``KeyError`` when attributes are not
  found using the legacy API. This fixes an issue that prevented writing to
  h5netcdf with dask.

Version 0.4.1 (Sep 6, 2017):

- Include tests in source distribution on pypi.

Version 0.4 (Aug 30, 2017):

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
- Support for writing arrays of variable length unicode strings with
  ``dtype=str`` via the legacy API.
- h5netcdf now writes the ``_NCProperties`` attribute for identifying netCDF4
  files.

License
-------

`3-clause BSD`_

.. _3-clause BSD: https://github.com/h5netcdf/h5netcdf/blob/master/LICENSE
