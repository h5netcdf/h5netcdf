Developers Guide
================

Team
----

- `Kai Mühlbauer <https://github.com/kmuehlbauer>`_
- `Stephan Hoyer <https://github.com/shoyer>`_

Contributors
------------

- `Aleksandar Jelenak <https://github.com/ajelenak>`_
- `Brett Naul <https://github.com/bnaul>`_
- `Dion Häfner <https://github.com/dionhaefner>`_
- `Drew Parsons <https://github.com/drew-parsons>`_
- `Frédéric Laliberté <https://github.com/laliberte>`_
- `Ghislain Vaillant <https://github.com/ghisvail>`_
- `Lion Krischer <https://github.com/krischer>`_
- `Mark Harfouche <https://github.com/hmaarrfk>`_
- `Martin Raspaud <https://github.com/mraspaud>`_
- `Pierre Augier <https://github.com/paugier>`_
- `Ryan Grout <https://github.com/groutr>`_
- `Scott Henderson <https://github.com/scottyhq>`_
- `Tom Augspurger <https://github.com/TomAugspurger>`_

If you are interested to contribute, just let us know by creating an issue or pull request on github.

Contribution Guidelines
-----------------------

- New features and changes should be added via Pull Requests from forks for contributors as well as maintainers.
- Pull Requests should have at least one approval (once the maintainer count has increased).
- Self merges without approval are allowed for repository maintenance, hotfixes and if the code changes do not affect functionality.
- Directly pushing to the repository main branch should only be used as a last resort.
- Releases should be introduced via Pull Request and approved. Exception: Patch release after hotfix.

Release Workflow
----------------

1. Create release commit (can be done per PullRequest for more visibility)
    * versioning is done via `setuptools_scm`
    * update CHANGELOG.rst if necessary
    * add/update sections to README.rst (or documentation) if necessary
    * check all needed dependencies are listed in setup.py
2. Create release
    * draft `new github release`_
    * tag version (eg `v0.11.0`) `@ Target: main`
    * set release title (eg. `release 0.11.0`)
    * add release description (eg. `bugfix-release`), tbd.

This will start the CI workflow once again. The workflow creates `sdist` and universal `wheel` and uploads it to PyPI.

.. _new github release: https://github.com/h5netcdf/h5netcdf/releases/new
