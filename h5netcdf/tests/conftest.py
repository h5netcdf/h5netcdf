import importlib.util
import itertools
import os
import tempfile
import time
from pathlib import Path
from shutil import rmtree

import pytest

from h5netcdf.tests import has_h5pyd, has_hsds


def module_available(name: str) -> bool:
    """Return True if a module can be imported."""
    return importlib.util.find_spec(name) is not None


@pytest.fixture(scope="session")
def hsds_up():
    """Provide HDF Highly Scalable Data Service (HSDS) for h5pyd testing."""
    if not has_h5pyd or not has_hsds:
        pytest.skip("Required packages h5pyd and/or hsds not available")

    from h5pyd import Folder
    from hsds.hsds_app import HsdsApp

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


def pytest_generate_tests(metafunc):
    # read/write matrix definition
    read_write_mod = ["netCDF4", "h5netcdf", "h5netcdf.legacyapi"]

    # filter out modules which aren't importable
    read_write_mod = [
        importlib.import_module(mod) for mod in read_write_mod if module_available(mod)
    ]

    # available backend modules
    backends = [b for b in ["h5py", "pyfive"] if module_available(b)]

    rw_matrix = list(itertools.product(read_write_mod, read_write_mod))

    # generates test matrix for test_dimensions function
    if {"read_write_matrix", "backend_module"} <= set(metafunc.fixturenames):
        cases = []
        ids = []
        for wmod, rmod in rw_matrix:
            if rmod.__name__ == "netCDF4":
                cases.append(((wmod, rmod), None))
                ids.append(f"{wmod.__name__}->{rmod.__name__}::no-backend")
            else:
                for backend in backends:
                    cases.append(((wmod, rmod), backend))
                    ids.append(f"{wmod.__name__}->{rmod.__name__}::{backend}")
        metafunc.parametrize("read_write_matrix, backend_module", cases, ids=ids)

    # generate test_roundtrip_local tests
    if {"tmp_local_netcdf", "wmod", "rmod", "bmod", "decode_vlen"} <= set(
        metafunc.fixturenames
    ):
        cases = []
        ids = []

        # build test matrix
        for wmod, rmod in rw_matrix:
            if rmod.__name__ == "netCDF4":
                cases.append((wmod, rmod, None, False))
                ids.append(f"{wmod.__name__}->{rmod.__name__}::no-backend")
            else:
                for backend in backends:
                    # decode_vlen True/False only for h5netcdf reads, others False
                    decode_values = (
                        [True, False]
                        if rmod.__name__ == "h5netcdf" and backend == "h5py"
                        else [False]
                    )
                    for dec in decode_values:
                        cases.append(
                            (
                                wmod,
                                rmod,
                                backend,
                                dict(decode_vlen_strings=dec),
                            )
                        )
                        ids.append(
                            f"{wmod.__name__}->{rmod.__name__}::{backend}, dec-vl::{dec}"
                        )

        metafunc.parametrize("wmod, rmod, bmod, decode_vlen", cases, ids=ids)
