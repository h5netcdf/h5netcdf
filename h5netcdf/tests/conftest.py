import importlib.util
import os
import tempfile
import time
from pathlib import Path
from shutil import rmtree

import netCDF4
import pytest

import h5netcdf
from h5netcdf import legacyapi

try:
    from h5pyd import Folder
    from hsds.hsds_app import HsdsApp

    with_reqd_pkgs = True
except ImportError:
    with_reqd_pkgs = False


def module_available(name: str) -> bool:
    """Return True if a module can be imported."""
    return importlib.util.find_spec(name) is not None


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


# generates test matrix for test_dimensions function,
# maybe a blueprint for a redesign of the testsuite
def pytest_generate_tests(metafunc):
    # nly run if both fixtures are requested
    if {"read_write_matrix", "backend_module"} <= set(metafunc.fixturenames):

        # read/write matrix definition
        matrix_raw = [
            [netCDF4, netCDF4],
            [legacyapi, legacyapi],
            [h5netcdf, h5netcdf],
            [legacyapi, netCDF4],
            [netCDF4, legacyapi],
            [h5netcdf, netCDF4],
            [netCDF4, h5netcdf],
            [legacyapi, h5netcdf],
            [h5netcdf, legacyapi],
        ]

        # filter out any pairs where the modules aren't importable
        matrix_filtered = [
            pair
            for pair in matrix_raw
            if module_available(pair[0].__name__) and module_available(pair[1].__name__)
        ]

        # available backend modules
        backends = [b for b in ["h5py", "pyfive", "h5pyd"] if module_available(b)]

        cases = []
        ids = []
        for rw in matrix_filtered:
            write_module_name = rw[0].__name__
            read_module_name = rw[1].__name__

            if read_module_name == "netCDF4":
                backend_choice = None
                cases.append((rw, backend_choice))
                ids.append(f"{write_module_name}->{read_module_name}::no-backend")
            else:
                for backend in backends:
                    cases.append((rw, backend))
                    ids.append(f"{write_module_name}->{read_module_name}::{backend}")

        metafunc.parametrize("read_write_matrix,backend_module", cases, ids=ids)
