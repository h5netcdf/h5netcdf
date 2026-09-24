"""Tests on how much HDF5 work h5netcdf does to serve metadata requests.

Metadata access patterns (e.g. xarray opening a file lazily) query the same
objects many times, so re-doing HDF5 work on every query is costly for files
with many variables and groups.
"""

import gc
import shutil
import weakref

import h5py
import numpy as np
import pytest

import h5netcdf


class _CallCounter:
    """Count the calls made to the functions it wraps."""

    def __init__(self):
        self.count = 0

    def wrap(self, func):
        def counted(*args, **kwargs):
            self.count += 1
            return func(*args, **kwargs)

        return counted


@pytest.fixture
def hdf5_object_opens(monkeypatch):
    """Count the HDF5 objects opened by name (``H5Oopen``)."""
    counter = _CallCounter()
    monkeypatch.setattr(h5py.h5o, "open", counter.wrap(h5py.h5o.open))
    return counter


@pytest.fixture
def hdf5_attribute_opens(monkeypatch):
    """Count the HDF5 attributes opened (``H5Aopen``)."""
    counter = _CallCounter()
    monkeypatch.setattr(h5py.h5a, "open", counter.wrap(h5py.h5a.open))
    return counter


@pytest.fixture
def hdf5_name_lookups(monkeypatch):
    """Count the HDF5 lookups of an object's path name.

    Finding the name of an object obtained from a reference requires HDF5 to
    search the file, which is slow for large files.
    """
    counter = _CallCounter()
    monkeypatch.setattr(h5py.h5i, "get_name", counter.wrap(h5py.h5i.get_name))
    monkeypatch.setattr(h5py.h5r, "get_name", counter.wrap(h5py.h5r.get_name))
    return counter


def _variables_sharing_dimensions(path, n_variables):
    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"x": 4, "y": 5}
        for i in range(n_variables):
            f.create_variable(f"var{i}", ("x", "y"), float)


@pytest.fixture
def netcdf_file(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, "w") as f:
        f.dimensions = {"x": 4, "y": 5}
        f.create_variable(
            "foo",
            ("x", "y"),
            float,
            chunks=(2, 5),
            compression="gzip",
            shuffle=True,
            fletcher32=True,
        )
    return tmp_local_netcdf


@pytest.mark.parametrize(
    "metadata",
    [
        "shape",
        "dtype",
        "chunks",
        "compression",
        "compression_opts",
        "shuffle",
        "fletcher32",
    ],
)
def test_repeated_variable_metadata_access_opens_no_hdf5_objects_in_read_mode(
    netcdf_file, hdf5_object_opens, metadata
):
    with h5netcdf.File(netcdf_file, "r") as f:
        foo = f["foo"]
        expected = getattr(foo, metadata)
        before = hdf5_object_opens.count

        assert getattr(foo, metadata) == expected
        assert hdf5_object_opens.count == before


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


def test_dimension_name_lookups_do_not_grow_with_variables_sharing_them(
    tmpdir, hdf5_name_lookups
):
    lookups = {}
    for n_variables in (1, 10):
        path = str(tmpdir.join(f"{n_variables}.nc"))
        _variables_sharing_dimensions(path, n_variables)
        with h5netcdf.File(path, "r") as f:
            variables = list(f.variables.values())
            before = hdf5_name_lookups.count
            dimensions = [var.dimensions for var in variables]
            lookups[n_variables] = hdf5_name_lookups.count - before

        assert dimensions == [("x", "y")] * n_variables

    assert lookups[10] == lookups[1]


def test_dimensions_from_own_or_ancestor_groups_need_no_name_lookups(
    tmp_local_netcdf, hdf5_name_lookups
):
    with h5netcdf.File(tmp_local_netcdf, "w") as f:
        f.dimensions = {"x": 4}
        sub = f.create_group("sub")
        sub.dimensions = {"y": 5}
        sub.create_variable("foo", ("x", "y"), float)

    with h5netcdf.File(tmp_local_netcdf, "r") as f:
        foo = f["sub"]["foo"]
        before = hdf5_name_lookups.count

        assert foo.dimensions == ("x", "y")
        assert hdf5_name_lookups.count == before


def test_resolving_dimensions_opens_the_variable_at_most_once(
    netcdf_file, hdf5_object_opens
):
    with h5netcdf.File(netcdf_file, "r") as f:
        foo = f["foo"]
        before = hdf5_object_opens.count

        assert foo.dimensions == ("x", "y")
        assert hdf5_object_opens.count - before <= 1


@pytest.mark.parametrize(
    "value",
    [np.float64(1.5), np.array([1, 2, 3]), "a string", np.bytes_(b"fixed")],
    ids=["scalar", "array", "vlen string", "fixed string"],
)
def test_reading_an_attribute_opens_it_once(
    tmp_local_netcdf, hdf5_attribute_opens, value
):
    with h5netcdf.File(tmp_local_netcdf, "w") as f:
        f.attrs["foo"] = value
    with h5netcdf.File(tmp_local_netcdf, "r") as f:
        expected = f.attrs["foo"]
        before = hdf5_attribute_opens.count

        np.testing.assert_equal(f.attrs["foo"], expected)
        assert hdf5_attribute_opens.count - before == 1


def test_variables_are_freed_once_their_file_is_closed_and_released(netcdf_file):
    with h5netcdf.File(netcdf_file, "r") as f:
        foo = f["foo"]
        foo.shape, foo.dtype, foo.chunks, foo.compression
        foo_ref = weakref.ref(foo)
    del f, foo
    gc.collect()

    assert foo_ref() is None
