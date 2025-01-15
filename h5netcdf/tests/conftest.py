import os
import tempfile
from pathlib import Path
from shutil import rmtree

import pytest

try:
    from h5pyd import Folder
    from hsds.hsds_app import HsdsApp

    with_reqd_pkgs = True
except ImportError:
    with_reqd_pkgs = False


@pytest.fixture(scope="session")
def hsds_up():
    """Provide HDF Highly Scalabale Data Service (HSDS) for h5pyd testing."""
    if with_reqd_pkgs:
        root_dir = Path(tempfile.mkdtemp(prefix="tmp-hsds-root-"))
        bucket_name = "pytest"
        os.environ["BUCKET_NAME"] = bucket_name
        os.mkdir(
            f"{root_dir}/{bucket_name}"
        )  # need to create a directory for our bucket

        hs_username = "h5netcdf-pytest"
        hs_password = "TestEarlyTestEverything"

        kwargs = {}
        kwargs["username"] = hs_username
        kwargs["password"] = hs_password
        kwargs["root_dir"] = str(root_dir)
        kwargs["logfile"] = f"{root_dir}/hsds.log"
        kwargs["log_level"] = "DEBUG"
        kwargs["host"] = "localhost"
        kwargs["sn_port"] = 5101

        try:
            hsds = HsdsApp(**kwargs)

            hsds.run()
            is_up = hsds.ready

            if is_up:
                os.environ["HS_ENDPOINT"] = hsds.endpoint
                os.environ["HS_USERNAME"] = hs_username
                os.environ["HS_PASSWORD"] = hs_password
                # make folders expected by pytest
                # pytest/home/h5netcdf-pytest
                # Folder("/pytest/", mode='w')
                Folder("/home/", mode="w")
                Folder("/home/h5netcdf-pytest/", mode="w")
        except Exception:
            is_up = False

        yield is_up
        hsds.check_processes()  # this will capture hsds log output
        hsds.stop()

        rmtree(root_dir, ignore_errors=True)

    else:
        yield False
