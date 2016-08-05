h5netcdf
========

.. image:: https://travis-ci.org/shoyer/h5netcdf.svg?branch=master
    :target: https://travis-ci.org/shoyer/h5netcdf
.. image:: https://badge.fury.io/py/h5netcdf.svg
    :target: https://pypi.python.org/pypi/h5netcdf/

A Python interface for the netCDF4_ file-format that reads and writes HDF5
files API directly via h5py_, without relying on the Unidata netCDF library.

.. _netCDF4: https://www.unidata.ucar.edu/software/netcdf/docs/netcdf/NetCDF_002d4-Format.html
.. _h5py: http://www.h5py.org/

**This is an experimental project.** It currently passes basic tests for
reading and writing netCDF4 files with Python, but it has not been tested for
compatibility with other netCDF4 interfaces.

Motivations
-----------

Why did I write h5netcdf? Well, here are a few reasons:

- To prove it could be done (it seemed like an obvious thing to do) and that
  netCDF4 is not actually that complicated.
- We've seen occasional reports of better performance with h5py than
  netCDF4-python that I wanted to be able to verify. For at least `one workflow`_,
  h5netcdf was reported to be almost **4x faster** than `netCDF4-python`_. If
  you give it a try, I would love to hear how it works for you!
- h5py seems to have thought through multi-threading pretty carefully, so this
  in particular seems like a case where things could make a difference. I've
  started to care about this because I recently hooked up a multi-threaded
  backend to xray_.
- It's one less massive binary dependency (netCDF C). Anecdotally, HDF5 users
  seem to be unexcited about switching to netCDF -- hopefully this will
  convince them that they are really the same thing!
- Finally, side-stepping the netCDF C library (and Cython bindings to it)
  gives us an easier way to identify the source of performance issues and
  bugs.

.. _one workflow: https://github.com/Unidata/netcdf4-python/issues/390#issuecomment-93864839
.. _xray: http://github.com/xray/xray/

Install
-------

Ensure you have a recent version of h5py installed (I recommend using conda_).
At least version 2.1 is required (for dimension scales); versions 2.3 and newer
have been verified to work. Then: ``pip install h5netcdf``

.. _conda: http://conda.io/

Usage
-----

h5netcdf has two APIs, a new API and a legacy API. Both interfaces currently
reproduce most of the features of the netCDF interface, with the noteable
exceptions of:

- support for operations the rename or delete existing objects.
- suport for creating unlimited dimensions.

We simply haven't gotten around to implementing these features yet.

ChangeLog
---------

A new feature is the lazy opening of datasets. Unlike
the netCDF interface where attributes for all variables and groups are read at
the creation of the Dataset object, h5netcdf loads these attributes only
when accessed.


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

**Warning: The design of the new API is *not yet finished*.** I only
recommended using it for experiments. Please share your feedback in `this
GitHub issue`_.

.. _this GitHub issue: https://github.com/shoyer/h5netcdf/issues/6

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
- We don't support the ``endian`` argument to ``createVariable``. The h5py API does not
  appear to offer this feature.
- h5netcdf variables do not support automatic masking or scaling (e.g., of values matching
  the ``_FillValue`` attribute). We prefer to leave this functionality to client libraries
  (e.g., xray_), which can implement their exact desired scaling behavior.

License
-------

`3-clause BSD`_

.. _3-clause BSD: https://github.com/shoyer/h5netcdf/blob/master/LICENSE.txt
