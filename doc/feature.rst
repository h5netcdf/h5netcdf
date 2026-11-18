legacyapi vs new API feature comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In general both API should be comparable in handling netCDF4 files. The
:ref:`legacyapi <legacyapi>` is more in line with `netCDF4-python`_ , whereas the
:ref:`new API <api>` aligns to `h5py`_. Still, there are some
differences which are outlined in the following table.

.. _netCDF4-python: https://unidata.github.io/netcdf4-python/
.. _h5py: https://www.h5py.org/

.. include:: <isopub.txt>

+---------------------+-------------------------+----------------+------------------+
| feature             | legacyapi               | new api        | type             |
+=====================+=========================+================+==================+
| 1D boolean indexer  | |check|                 | |check|        | Variable/Dataset |
+---------------------+-------------------------+----------------+------------------+
| resize on write     | |check|                 | |cross|        | Dimension        |
|                     |                         |                | Variable/Dataset |
+---------------------+-------------------------+----------------+------------------+
| resize dimension    | only current dimension  | dimension and  | Dimension        |
|                     |                         | all connected  | Variable/Dataset |
|                     |                         | variables      |                  |
+---------------------+-------------------------+----------------+------------------+
| group name          | name only               | full path      | Group            |
+---------------------+-------------------------+----------------+------------------+
| phony_dims          | kwarg                   | kwarg          | Dimension        |
+---------------------+-------------------------+----------------+------------------+
| decode_vlen_strings | |check|                 | kwarg          | Variable/Dataset |
+---------------------+-------------------------+----------------+------------------+
| chunk sizes         | ``h5netcdf``-style      | kwarg          | Variable/Dataset |
+---------------------+-------------------------+----------------+------------------+
| dimension ``.size`` | max size dimension      | size dimension | Dimension        |
|                     | and connected variables |                |                  |
+---------------------+-------------------------+----------------+------------------+
|                     |                         |                | Attribute        |
| valid netcdf        | kwarg                   | kwarg          | Variable/Dataset |
+---------------------+-------------------------+----------------+------------------+
| ``h5py.Empty``      |                         |                |                  |
| string attrs        | ``b""``                 | ``b""``        | Attribute        |
+---------------------+-------------------------+----------------+------------------+
| endian              | |check|                 | |cross|        | Variable/Dataset |
+---------------------+-------------------------+----------------+------------------+
| track order         | |cross|                 | |check|        | File/Group       |
|                     |                         |                | Dataset          |
+---------------------+-------------------------+----------------+------------------+

Numeric attributes read through h5py
------------------------------------

With the h5py backend, h5netcdf writes a numeric scalar attribute and a
one-element numeric sequence as the same one-element array on disk. When
reading the file through h5netcdf, both attributes come back as scalars.
Reading them directly with h5py instead returns arrays with shape ``(1,)``.
The stored attributes do not distinguish the original scalar and sequence.
This matches how netCDF4-Python and netCDF-C handle one-element numeric
attributes.
