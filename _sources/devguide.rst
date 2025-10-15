Developers Guide
================

Team
----

- `Kai Mühlbauer <https://github.com/kmuehlbauer>`_
- `Stephan Hoyer <https://github.com/shoyer>`_

Contributors
------------

- `Aleksandar Jelenak <https://github.com/ajelenak>`_
- `Bas Couwenberg <https://github.com/sebastic>`_.
- `Brett Naul <https://github.com/bnaul>`_
- `David Huard <https://github.com/huard>`_
- `Dion Häfner <https://github.com/dionhaefner>`_
- `Drew Parsons <https://github.com/drew-parsons>`_
- `Ezequiel Cimadevilla Alvarez <https://github.com/zequihg50>`_
- `Frédéric Laliberté <https://github.com/laliberte>`_
- `Ghislain Vaillant <https://github.com/ghisvail>`_
- `John Readey <https://github.com/jreadey>`_
- `Jonas Grönberg <https://github.com/JonasGronberg>`_
- `Kurt Schwehr <https://github.com/schwehr>`_
- `Lion Krischer <https://github.com/krischer>`_
- `Mark Harfouche <https://github.com/hmaarrfk>`_
- `Martin Raspaud <https://github.com/mraspaud>`_
- `Pierre Augier <https://github.com/paugier>`_
- `Rickard Holmberg <https://github.com/rho-novatron>`_
- `Ryan Grout <https://github.com/groutr>`_
- `Scott Henderson <https://github.com/scottyhq>`_
- `Thomas Kluyver <https://github.com/takluyver>`_
- `Tom Augspurger <https://github.com/TomAugspurger>`_

If you are interested to contribute, just let us know by creating an issue or pull request on github.

Contribution Guidelines
-----------------------

- New features and changes should be added via Pull Requests from forks for contributors as well as maintainers.
- Pull Requests should have at least one approval (once the maintainer count has increased).
- Self merges without approval are allowed for repository maintenance, hotfixes and if the code changes do not affect functionality.
- Directly pushing to the repository main branch should only be used as a last resort.
- Releases should be introduced via Pull Request and approved. Exception: Patch release after hotfix.

Continuous Integration
----------------------

``h5netcdf`` uses GitHub Actions for Continuous Integration (CI). On every ``push`` to a repository branch
or a PullRequest branch several checks are performed:

- Lint and style checks (``ruff``, ``black``)
- Unit tests with latest ``h5py3`` (and Python versions) facilitating GitHub Ubuntu worker
- Documentation build, artifacts are made available to download
- On release, source-tarball and universal wheel is uploaded to PyPI and documentation is made available
  on `h5netcdf GitHub Pages`_

.. _h5netcdf GitHub Pages: https://h5netcdf.github.io/h5netcdf

Documentation
-------------

The documentation, located in ``doc``-folder, can be created using ``sphinx-doc`` and the ``sphinx-book_theme``::

    $ cd doc
    $ make html

The rendered documentation is then available in the subfolder ``_build``.

Due to the history several documents, eg. `README.rst`_ and `CHANGELOG.rst`_, are located in the project's root folder.
They are linked into the documentation via ``.. include``-directive. Links and cross-references originating from these files
should be hardcoded to maintain operation also in non-rendered format.

.. _README.rst: https://github.com/h5netcdf/h5netcdf/blob/main/README.rst
.. _CHANGELOG.rst: https://github.com/h5netcdf/h5netcdf/blob/main/CHANGELOG.rst

Release Workflow
----------------

1. Create release commit (can be done per PullRequest for more visibility)
    * versioning is done via `setuptools_scm`
    * update CHANGELOG.rst if necessary
    * add/update sections to README.rst (or documentation) if necessary
    * check all needed dependencies are listed in setup.py
2. Create release
    * draft `new github release`_
    * tag version (eg `v1.2.0`) `@ Target: main`
    * set release title (eg. `release 1.2.0`)
    * add release description (eg. `bugfix-release`), tbd.

This will start the CI workflow once again. The workflow creates `sdist` and universal `wheel` and uploads it to PyPI.

.. _new github release: https://github.com/h5netcdf/h5netcdf/releases/new

References
----------

This section contains links to material how ``netCDF4`` facilitates ``HDF5``.

Some valuable links on dimension scales:

- `HDF5 Dimension Scales`_
- `HDF5 Dimension Scales Part 2`_
- `HDF5 Dimension Scales Part 3`_
- `NetCDF-4 Dimensions and HDF5 Dimension Scales`_
- `NetCDF-4 use of dimension scales`_

Other resources

- `NetCDF-4 performance`_
- `String **NULLTERM**  vs. **NULLPAD**`_

netCDF4-python quirks:

- ``_Netcdf4Dimid`` gets attached to all data variables if a 2D coordinate variable is created  and any variable is written/file is reopened for append, see `issue 1104`_
- unlimited variable dimensions are reported as current size of the dimension scale, even if the variable's underlying ``DATASPACE`` dimension is smaller (eg. 0)

.. _HDF5 Dimension Scales: https://www.unidata.ucar.edu/blogs/developer/en/entry/dimensions_scales
.. _HDF5 Dimension Scales Part 2: https://www.unidata.ucar.edu/blogs/developer/en/entry/dimension_scale2
.. _HDF5 Dimension Scales Part 3: https://www.unidata.ucar.edu/blogs/developer/en/entry/dimension_scales_part_3
.. _NetCDF-4 Dimensions and HDF5 Dimension Scales: https://www.unidata.ucar.edu/blogs/developer/en/entry/netcdf4_shared_dimensions
.. _NetCDF-4 use of dimension scales: https://www.unidata.ucar.edu/blogs/developer/en/entry/netcdf4_use_of_dimension_scales
.. _NetCDF-4 performance: https://www.researchgate.net/publication/330347054_2A5_NETCDF-4_PERFORMANCE_IMPROVEMENTS_OPENING_COMPLEX_DATA_FILES
.. _String **NULLTERM**  vs. **NULLPAD**: https://github.com/PyTables/PyTables/issues/264
.. _issue 1104: https://github.com/Unidata/netcdf4-python/issues/1104
