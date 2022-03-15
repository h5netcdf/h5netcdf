legacyapi vs new api feature comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..    include:: <isopub.txt>

+---------------------+-------------------------+----------------+------------------+
| feature             | legacyapi               | new api        | type             |
+=====================+=========================+================+==================+
| 1D boolean indexer  | |check|                 | |cross|        | Variable/Dataset |
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
| track order         | |cross|                 | |cross|        | File/Group       |
|                     |                         |                | Dataset          |
+---------------------+-------------------------+----------------+------------------+
