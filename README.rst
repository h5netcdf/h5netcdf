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

.. [*] h5netcdf we will raise ``h5netcdf.CompatibilityError``.

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

Track Order
~~~~~~~~~~~

In h5netcdf version 0.12.0 and earlier, `order tracking`_ was disabled in
HDF5 file. As this is a requirement for the current netCDF4 standard,
it has been enabled without deprecation as of version 0.13.0 [*]_.

Datasets created with h5netcdf version 0.12.0 that are opened with
newer versions of h5netcdf will continue to disable order tracker.

.. _order tracking: https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#creation_order
.. [*] https://github.com/h5netcdf/h5netcdf/issues/128

Changelog
---------

`Changelog`_

.. _Changelog: https://github.com/h5netcdf/h5netcdf/blob/master/CHANGELOG.rst

License
-------

`3-clause BSD`_

.. _3-clause BSD: https://github.com/h5netcdf/h5netcdf/blob/master/LICENSE
