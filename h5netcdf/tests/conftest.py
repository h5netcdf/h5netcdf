import os
import tempfile
import time
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
    """Provide HDF Highly Scalable Data Service (HSDS) for h5pyd testing."""
    if not with_reqd_pkgs:
        pytest.skip("Required packages h5pyd and hsds not available")

    root_dir = Path(tempfile.mkdtemp(prefix="tmp-hsds-root-"))
    bucket_name = "pytest"
    os.environ["BUCKET_NAME"] = bucket_name
    # need to create a directory for our bucket
    (root_dir / bucket_name).mkdir()

    kwargs = {
        "username": "h5netcdf-pytest",
        "password": "TestEarlyTestEverything",
        "root_dir": str(root_dir),
        "logfile": str(root_dir / "hsds.log"),
        "log_level": "DEBUG",
        "host": "localhost",
        "sn_port": 5101,
    }

    os.environ.update(
        {
            "BUCKET_NAME": bucket_name,
            "HS_USERNAME": kwargs["username"],
            "HS_PASSWORD": kwargs["password"],
            "HS_USE_HTTPS": "False",
        }
    )

    hsds = HsdsApp(**kwargs)

    try:
        hsds.run()
        timeout = time.time() + 60
        while not hsds.ready:
            if time.time() > timeout:
                raise TimeoutError("HSDS server did not become ready in time")
            time.sleep(1)

        os.environ["HS_ENDPOINT"] = hsds.endpoint
        # make folders expected by pytest
        Folder("/home/", mode="w")
        Folder("/home/h5netcdf-pytest/", mode="w")

        yield True

    except Exception as err:
        log_path = kwargs["logfile"]
        if os.path.exists(log_path):
            with open(log_path) as f:
                print("\n=== HSDS Log ===")
                print(f.read())
        else:
            print(f"HSDS log not found at: {log_path}")
        raise err

    finally:
        try:
            hsds.check_processes()
            hsds.stop()
        except Exception:
            pass

    rmtree(root_dir, ignore_errors=True)
