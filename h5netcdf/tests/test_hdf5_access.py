"""Tests on how much HDF5 work h5netcdf does to serve metadata requests.

Metadata access patterns (e.g. xarray opening a file lazily) query the same
objects many times, so re-doing HDF5 work on every query is costly for files
with many variables and groups.
"""

import shutil

import h5py
import numpy as np
import pytest

import h5netcdf


class _CallCounter:
    def __init__(self, func):
        self._func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self._func(*args, **kwargs)


@pytest.fixture
def hdf5_object_opens(monkeypatch):
    """Count the HDF5 objects opened by name (``H5Oopen``)."""
    counter = _CallCounter(h5py.h5o.open)
    monkeypatch.setattr(h5py.h5o, "open", counter)
    return counter


@pytest.fixture
def netcdf_file(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, "w") as f:
        f.dimensions = {"x": 4, "y": 5}
        f.create_variable("foo", ("x", "y"), float, chunks=(2, 5))
    return tmp_local_netcdf


def test_repeated_variable_metadata_access_opens_no_hdf5_objects_in_read_mode(
    netcdf_file, hdf5_object_opens
):
    with h5netcdf.File(netcdf_file, "r") as f:
        foo = f["foo"]
        foo.shape, foo.dtype, foo.chunks
        opens_after_first_access = hdf5_object_opens.count

        foo.shape, foo.dtype, foo.chunks

        assert hdf5_object_opens.count == opens_after_first_access


def test_file_rewritten_on_disk_shows_new_content_when_reopened(tmpdir):
    # an open (read-only) file must not keep serving stale HDF5 state to a
    # new handle on the same path, see pydata/xarray#4862
    path = str(tmpdir.join("data.nc"))
    rewritten_path = str(tmpdir.join("rewritten.nc"))
    for filename, value in ((path, 1.0), (rewritten_path, 2.0)):
        with h5netcdf.File(filename, "w") as f:
            f.dimensions = {"x": 3}
            f.create_variable("foo", ("x",), float, data=np.full(3, value))

    with h5netcdf.File(path, "r") as old:
        old_foo = old["foo"]
        old_foo[:]

        shutil.copy(rewritten_path, path)

        with h5netcdf.File(path, "r") as new:
            np.testing.assert_array_equal(new["foo"][:], np.full(3, 2.0))
