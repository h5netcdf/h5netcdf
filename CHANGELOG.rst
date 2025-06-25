Change Log
----------

Development Version (unreleased):

- Codespell fixes ({pull}`261`).
  By `Kurt Schwehr <https://github.com/schwehr>`_
- Fix hsds/h5pyd test fixture spinup issues ({pull}`265`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_
- Fix and add circular referrer tests for Python 3.14 ({pull}`264`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_
- Avoid opening h5pyd file to check if there is a preexisting file, check existence by REST API instead ({issue}`262`, {pull}`266`).
  By `Jonas Grönberg <https://github.com/JonasGronberg>`_ and `Kai Mühlbauer <https://github.com/kmuehlbauer>`_

Version 1.6.1 (March 7th, 2025):

- Let Variable.chunks return None for scalar variables, independent of what the underlying
  h5ds object returns ({pull}`259`).
  By `Rickard Holmberg <https://github.com/rho-novatron>`_

Version 1.6.0 (March 7th, 2025):

- Allow specifying `h5netcdf.File(driver="h5pyd")` to force the use of h5pyd ({issue}`255`, {pull}`256`).
  By `Rickard Holmberg <https://github.com/rho-novatron>`_
- Add pytest-mypy-plugins for xarray nightly test ({pull}`257`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_

Version 1.5.0 (January 26th, 2025):

- Update CI to new versions (Python 3.13, 3.14 alpha), remove numpy 1 from h5pyd runs ({pull}`250`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_
- Update CI and reinstate h5pyd/hsds test runs ({pull}`247`).
  By `John Readey  <https://github.com/jreadey>`_
- Allow ``zlib`` to be used as an alias for ``gzip`` for enhanced compatibility with h5netcdf's API and xarray.
  By `Mark Harfouche <https://github.com/hmaarrfk>`_

Version 1.4.1 (November 13th, 2024):

- Add CI run for hdf5 1.10.6, fix complex tests, fix enum/user type tests ({pull}`244`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_


Version 1.4.0 (October 7th, 2024):

- Add UserType class, add EnumType ({pull}`229`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_
- Refactor fillvalue and dtype handling for user types, enhance sanity checks and tests ({pull}`230`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_
- Add VLType and CompoundType, commit complex compound type to file. Align with nc-complex ({pull}`227`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_
- Update h5pyd testing.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_
- CI and lint maintenance ({pull}`235`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_
- Support wrapping an h5py ``File`` object. Closing the h5netcdf file object
  does not close the h5py file ({pull}`238`).
  By `Thomas Kluyver <https://github.com/takluyver>`_
- CI and lint maintenance (format README.rst, use more f-strings, change Python 3.9 to 3.10 in CI) ({pull}`239`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_

Version 1.3.0 (November 7th, 2023):

- Add ros3 support by checking `driver`-kwarg.
  By `Ezequiel Cimadevilla Alvarez <https://github.com/zequihg50>`_
- Code and CI maintenance.
  By `Mark Harfouche <https://github.com/hmaarrfk>`_ and
  `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Version 1.2.0 (June 2nd, 2023):

- Remove h5py2 compatibility code, remove h5py2 CI runs, mention NEP29 as
  upstream dependency support strategy.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ and
  `Mark Harfouche <https://github.com/hmaarrfk>`_.
- Update to pyproject.toml-only build process, adapt CI, use `ruff` for linting, add .pre-commit-config.yaml.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Maintenance CI (use setup-micromamba), fix hsds, fix tests, fix license.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Raise early with h5py-error.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Add simple test to ensure that the shape is stored in the coordinates.
  By `Mark Harfouche <https://github.com/hmaarrfk>`_.

Version 1.1.0 (November 23rd, 2022):

- Rework adding _FillValue-attribute, add tests.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Add special add_phony method for creating phony dimensions, add test.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Rewrite _unlabeled_dimension_mix (labeled/unlabeled), add tests.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Add default netcdf fillvalues, pad only if necessary, adapt tests.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Fix regression in padding algorithm, add test.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Set ``track_order=True`` by default in created files if h5py 3.7.0 or
  greater is detected to help compatibility with netCDF4-c programs.
  By `Mark Harfouche <https://github.com/hmaarrfk>`_.

Version 1.0.2 (August 2nd, 2022):

- Adapt boolean indexing as h5py 3.7.0 started supporting it.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Fix several tests to work with new h5py 3.7.0.
  By `Mark Harfouche <https://github.com/hmaarrfk>`_ and `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Version 1.0.1 (June 27th, 2022):

- Fix failing tests when using netCDF4 4.9.0.
  Reported and patch submitted by `Bas Couwenberg <https://github.com/sebastic>`_.

Version 1.0.0 (March 31st, 2022):

- Add HSDS pytest-fixture, make tests work with h5ypd.
  By `Aleksandar Jelenak <https://github.com/ajelenak>`_.
- Remove `_NCProperties` from existing file if writing invalid netcdf features.
  Warn users if `.nc` file extension is used writing invalid netcdf features.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Remove deprecated code (eg. remove deprecated code (eg. handling mode,
  chunking_heuristics, decode_vlen_strings), adapt LICENSE/AUTHOR.txt,
  prepare repository for release 1.0.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Version 0.15.0 (March 18th, 2022):

- Add documentation to ``h5netcdf``, merging current available documentation
  available as ``.rst``-files, in the repo-wiki and new API-docs into one document
  using ``sphinx-doc`` and ``sphinx-book-theme``.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Version 0.14.1 (March 2nd, 2022):

- Directly return non-string ``Empty``-type attributes as empty numpy-ndarray.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Version 0.14.0 (February 25, 2022):

- Add ``chunking_heuristic`` keyword and custom heuristic ``chunking_heuristic="h5netcdf"``
  with better handling of unlimited dimensions.
  By `Dion Häfner <https://github.com/dionhaefner>`_.
- Return group name instead of full group path for legacy API.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Add ``endian`` keyword argument ``legacyapi.Dataset.createVariable``.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Resize Dimensions when writing to variables (legacy API only), return padded arrays.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Allow 1D boolean indexers in legacy API.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Revert order tracking by default to avoid a bug in ``h5py`` (Closes Issue
  #136). By `Mark Harfouche <https://github.com/hmaarrfk>`_.
- Implement Dimension-class.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Return items from 0-dim and one-element 1-dim array attributes. Return multi-element
  attributes as lists. Return string attributes as Python strings decoded from their respective
  encoding (`utf-8`, `ascii`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Version 0.13.0 (January 12, 2022):

- Assign dimensions at creation time, instead of at sync/flush (file-close).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Create/attach dimension scales on the fly, instead of at sync/flush (file-close).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Ensure order tracking is true for newly created netcdf4 files as required
  by the netcdf4 standard. This enables files created by h5netcdf to be
  appended to by netCDF4 library users (Closes Issue #128).
  By `Mark Harfouche <https://github.com/hmaarrfk>`_.

Version 0.12.0 (December 20, 2021):

- Added ``FutureWarning`` to use ``mode='r'`` as default when opening files.
  By `Ryan Grout <https://github.com/groutr>`_.
- Moved handling of ``_nc4_non_coord_`` to ``h5netcdf.BaseVariable``.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Write ``_NCProperties`` when overwriting existing files.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Create/Attach dimension scales on append (``mode="r+"``)
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Create/Attach/Detach dimension scales only if necessary.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Switch warning into error when using invalid netCDF features.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Avoid circular references to objects referencing h5py objects.
  By `Tom Augspurger <https://github.com/TomAugspurger>`_.

Version 0.11.0 (April 20, 2021):

- Included ``h5pyd.Dataset`` objects as netCDF variables.
  By `Aleksandar Jelenak <https://github.com/ajelenak>`_.
- Added automatic PyPI upload on creation of github release.
- Moved Changelog to CHANGELOG.rst.
- Updated ``decode_vlen_strings`` ``FutureWarning``.
- Support for ``h5py.Empty`` strings.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

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
  API using the ``h5pyd`` package. Any file "path" starting with either
  ``http://``, ``https://``, or ``hdf5://`` will automatically trigger the use
  of this package.
  By `Aleksandar Jelenak <https://github.com/ajelenak>`_.

Version 0.5.1 (Apr 11, 2018):

- Bug fix for files with an unlimited dimension with no associated variables.
  By `Aleksandar Jelenak <https://github.com/ajelenak>`_.

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
