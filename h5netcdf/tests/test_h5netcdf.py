import gc
import io
import random
import re
import string
import tempfile
from os import environ as env

import h5py
import netCDF4
import numpy as np
import pytest
from packaging import version
from pytest import raises

import h5netcdf
from h5netcdf import legacyapi
from h5netcdf.core import NOT_A_VARIABLE, CompatibilityError

try:
    import h5pyd

    without_h5pyd = False
except ImportError:
    without_h5pyd = True


remote_h5 = ("http:", "hdf5:")


@pytest.fixture
def tmp_local_netcdf(tmpdir):
    return str(tmpdir.join("testfile.nc"))


@pytest.fixture(params=["testfile.nc", "hdf5://testfile"])
def tmp_local_or_remote_netcdf(request, tmpdir, hsds_up):
    if request.param.startswith(remote_h5):
        if without_h5pyd:
            pytest.skip("h5pyd package not available")
        elif not hsds_up:
            pytest.skip("HSDS service not running")
        rnd = "".join(random.choice(string.ascii_uppercase) for _ in range(5))
        return (
            "hdf5://"
            + "home"
            + "/"
            + env["HS_USERNAME"]
            + "/"
            + "testfile"
            + rnd
            + ".nc"
        )
    else:
        return str(tmpdir.join(request.param))


@pytest.fixture(params=[True, False])
def decode_vlen_strings(request):
    if version.parse(h5py.__version__) >= version.parse("3.0.0"):
        return dict(decode_vlen_strings=request.param)
    else:
        return {}


@pytest.fixture(params=[netCDF4, legacyapi])
def netcdf_write_module(request):
    return request.param


def get_hdf5_module(resource):
    """Return the correct h5py module based on the input resource."""
    if isinstance(resource, str) and resource.startswith(remote_h5):
        return h5pyd
    else:
        return h5py


def string_to_char(arr):
    """Like nc4.stringtochar, but faster and more flexible."""
    # ensure the array is contiguous
    arr = np.array(arr, copy=False, order="C")
    kind = arr.dtype.kind
    if kind not in ["U", "S"]:
        raise ValueError("argument must be a string")
    return arr.reshape(arr.shape + (1,)).view(kind + "1")


def array_equal(a, b):
    a, b = map(np.array, (a[...], b[...]))
    if a.shape != b.shape:
        return False
    try:
        return np.allclose(a, b)
    except TypeError:
        return (a == b).all()


_char_array = string_to_char(np.array(["a", "b", "c", "foo", "bar", "baz"], dtype="S"))

_string_array = np.array(
    [["foobar0", "foobar1", "foobar3"], ["foofoofoo", "foofoobar", "foobarbar"]]
)

_vlen_string = "foo"


def is_h5py_char_working(tmp_netcdf, name):
    h5 = get_hdf5_module(tmp_netcdf)
    # https://github.com/Unidata/netcdf-c/issues/298
    with h5.File(tmp_netcdf, "r") as ds:
        v = ds[name]
        try:
            assert array_equal(v, _char_array)
            return True
        except Exception as e:
            if re.match("^Can't read data", e.args[0]):
                return False
            else:
                raise


def write_legacy_netcdf(tmp_netcdf, write_module):
    ds = write_module.Dataset(tmp_netcdf, "w")
    ds.setncattr("global", 42)
    ds.other_attr = "yes"
    ds.createDimension("x", 4)
    ds.createDimension("y", 5)
    ds.createDimension("z", 6)
    ds.createDimension("empty", 0)
    ds.createDimension("string3", 3)
    ds.createDimension("unlimited", None)

    v = ds.createVariable("foo", float, ("x", "y"), chunksizes=(4, 5), zlib=True)
    v[...] = 1
    v.setncattr("units", "meters")

    v = ds.createVariable("y", int, ("y",), fill_value=-1)
    v[:4] = np.arange(4)

    v = ds.createVariable("z", "S1", ("z", "string3"), fill_value=b"X")
    v[...] = _char_array

    v = ds.createVariable("scalar", np.float32, ())
    v[...] = 2.0

    # test creating a scalar with compression option (with should be ignored)
    v = ds.createVariable("intscalar", np.int64, (), zlib=6, fill_value=None)
    v[...] = 2

    v = ds.createVariable("foo_unlimited", float, ("x", "unlimited"))
    v[...] = 1

    with raises((h5netcdf.CompatibilityError, TypeError)):
        ds.createVariable("boolean", np.bool_, ("x"))

    g = ds.createGroup("subgroup")
    v = g.createVariable("subvar", np.int32, ("x",))
    v[...] = np.arange(4.0)

    g.createDimension("y", 10)
    g.createVariable("y_var", float, ("y",))

    ds.createDimension("mismatched_dim", 1)
    ds.createVariable("mismatched_dim", int, ())

    v = ds.createVariable("var_len_str", str, ("x"))
    v[0] = "foo"

    ds.close()


def write_h5netcdf(tmp_netcdf):
    ds = h5netcdf.File(tmp_netcdf, "w")
    ds.attrs["global"] = 42
    ds.attrs["other_attr"] = "yes"
    ds.dimensions = {"x": 4, "y": 5, "z": 6, "empty": 0, "unlimited": None}

    v = ds.create_variable(
        "foo", ("x", "y"), float, chunks=(4, 5), compression="gzip", shuffle=True
    )
    v[...] = 1
    v.attrs["units"] = "meters"

    remote_file = isinstance(tmp_netcdf, str) and tmp_netcdf.startswith(remote_h5)
    if not remote_file:
        v = ds.create_variable("y", ("y",), int, fillvalue=-1)
        v[:4] = np.arange(4)

    v = ds.create_variable("z", ("z", "string3"), data=_char_array, fillvalue=b"X")

    v = ds.create_variable("scalar", data=np.float32(2.0))

    v = ds.create_variable("intscalar", data=np.int64(2))

    v = ds.create_variable("foo_unlimited", ("x", "unlimited"), float)
    v[...] = 1

    with raises((h5netcdf.CompatibilityError, TypeError)):
        ds.create_variable("boolean", data=True)

    g = ds.create_group("subgroup")
    v = g.create_variable("subvar", ("x",), np.int32)
    v[...] = np.arange(4.0)
    with raises(AttributeError):
        v.attrs["_Netcdf4Dimid"] = -1

    g.dimensions["y"] = 10
    g.create_variable("y_var", ("y",), float)
    g.flush()

    ds.dimensions["mismatched_dim"] = 1
    ds.create_variable("mismatched_dim", dtype=int)
    ds.flush()

    dt = h5py.special_dtype(vlen=str)
    v = ds.create_variable("var_len_str", ("x",), dtype=dt)
    v[0] = _vlen_string

    ds.close()


def read_legacy_netcdf(tmp_netcdf, read_module, write_module):
    ds = read_module.Dataset(tmp_netcdf, "r")
    assert ds.ncattrs() == ["global", "other_attr"]
    assert ds.getncattr("global") == 42
    if write_module is not netCDF4:
        # skip for now: https://github.com/Unidata/netcdf4-python/issues/388
        assert ds.other_attr == "yes"
    with pytest.raises(AttributeError):
        ds.does_not_exist
    assert set(ds.dimensions) == set(
        ["x", "y", "z", "empty", "string3", "mismatched_dim", "unlimited"]
    )
    assert set(ds.variables) == set(
        [
            "foo",
            "y",
            "z",
            "intscalar",
            "scalar",
            "var_len_str",
            "mismatched_dim",
            "foo_unlimited",
        ]
    )

    assert set(ds.groups) == set(["subgroup"])
    assert ds.parent is None
    v = ds.variables["foo"]
    assert array_equal(v, np.ones((4, 5)))
    assert v.dtype == float
    assert v.dimensions == ("x", "y")
    assert v.ndim == 2
    assert v.ncattrs() == ["units"]
    if write_module is not netCDF4:
        assert v.getncattr("units") == "meters"
    assert tuple(v.chunking()) == (4, 5)

    # check for dict items separately
    # see https://github.com/h5netcdf/h5netcdf/issues/171
    filters = v.filters()
    assert filters["complevel"] == 4
    assert filters["fletcher32"] is False
    assert filters["shuffle"] is True
    assert filters["zlib"] is True

    v = ds.variables["y"]
    assert array_equal(v, np.r_[np.arange(4), [-1]])
    assert v.dtype == int
    assert v.dimensions == ("y",)
    assert v.ndim == 1
    assert v.ncattrs() == ["_FillValue"]
    assert v.getncattr("_FillValue") == -1
    assert v.chunking() == "contiguous"

    # check for dict items separately
    # see https://github.com/h5netcdf/h5netcdf/issues/171
    filters = v.filters()
    assert filters["complevel"] == 0
    assert filters["fletcher32"] is False
    assert filters["shuffle"] is False
    assert filters["zlib"] is False

    ds.close()

    # Check the behavior if h5py. Cannot expect h5netcdf to overcome these
    # errors:
    if is_h5py_char_working(tmp_netcdf, "z"):
        ds = read_module.Dataset(tmp_netcdf, "r")
        v = ds.variables["z"]
        assert array_equal(v, _char_array)
        assert v.dtype == "S1"
        assert v.ndim == 2
        assert v.dimensions == ("z", "string3")
        assert v.ncattrs() == ["_FillValue"]
        assert v.getncattr("_FillValue") == b"X"
    else:
        ds = read_module.Dataset(tmp_netcdf, "r")

    v = ds.variables["scalar"]
    assert array_equal(v, np.array(2.0))
    assert v.dtype == "float32"
    assert v.ndim == 0
    assert v.dimensions == ()
    assert v.ncattrs() == []

    v = ds.variables["intscalar"]
    assert array_equal(v, np.array(2))
    assert v.dtype == "int64"
    assert v.ndim == 0
    assert v.dimensions == ()
    assert v.ncattrs() == []

    v = ds.variables["var_len_str"]
    assert v.dtype == str
    assert v[0] == _vlen_string

    v = ds.groups["subgroup"].variables["subvar"]
    assert ds.groups["subgroup"].parent is ds
    assert array_equal(v, np.arange(4.0))
    assert v.dtype == "int32"
    assert v.ndim == 1
    assert v.dimensions == ("x",)
    assert v.ncattrs() == []

    v = ds.groups["subgroup"].variables["y_var"]
    assert v.shape == (10,)
    assert "y" in ds.groups["subgroup"].dimensions

    ds.close()


def read_h5netcdf(tmp_netcdf, write_module, decode_vlen_strings):
    remote_file = isinstance(tmp_netcdf, str) and tmp_netcdf.startswith(remote_h5)
    ds = h5netcdf.File(tmp_netcdf, "r", **decode_vlen_strings)
    assert ds.name == "/"
    assert list(ds.attrs) == ["global", "other_attr"]
    assert ds.attrs["global"] == 42
    if write_module is not netCDF4:
        # skip for now: https://github.com/Unidata/netcdf4-python/issues/388
        assert ds.attrs["other_attr"] == "yes"
    assert set(ds.dimensions) == set(
        ["x", "y", "z", "empty", "string3", "mismatched_dim", "unlimited"]
    )
    variables = set(
        [
            "foo",
            "z",
            "intscalar",
            "scalar",
            "var_len_str",
            "mismatched_dim",
            "foo_unlimited",
        ]
    )
    # fix current failure of hsds/h5pyd
    if not remote_file:
        variables |= set(["y"])
    assert set(ds.variables) == variables

    assert set(ds.groups) == set(["subgroup"])
    assert ds.parent is None

    v = ds["foo"]
    assert v.name == "/foo"
    assert array_equal(v, np.ones((4, 5)))
    assert v.dtype == float
    assert v.dimensions == ("x", "y")
    assert v.ndim == 2
    assert list(v.attrs) == ["units"]
    if write_module is not netCDF4:
        assert v.attrs["units"] == "meters"
    assert v.chunks == (4, 5)
    assert v.compression == "gzip"
    assert v.compression_opts == 4
    assert not v.fletcher32
    assert v.shuffle

    # fix current failure of hsds/h5pyd
    if not remote_file:
        v = ds["y"]
        assert array_equal(v, np.r_[np.arange(4), [-1]])
        assert v.dtype == int
        assert v.dimensions == ("y",)
        assert v.ndim == 1
        assert list(v.attrs) == ["_FillValue"]
        assert v.attrs["_FillValue"] == -1
        if not remote_file:
            assert v.chunks is None
        assert v.compression is None
        assert v.compression_opts is None
        assert not v.fletcher32
        assert not v.shuffle
    ds.close()

    if is_h5py_char_working(tmp_netcdf, "z"):
        ds = h5netcdf.File(tmp_netcdf, "r")
        v = ds["z"]
        assert array_equal(v, _char_array)
        assert v.dtype == "S1"
        assert v.ndim == 2
        assert v.dimensions == ("z", "string3")
        assert list(v.attrs) == ["_FillValue"]
        assert v.attrs["_FillValue"] == b"X"
    else:
        ds = h5netcdf.File(tmp_netcdf, "r", **decode_vlen_strings)

    v = ds["scalar"]
    assert array_equal(v, np.array(2.0))
    assert v.dtype == "float32"
    assert v.ndim == 0
    assert v.dimensions == ()
    assert list(v.attrs) == []

    v = ds.variables["intscalar"]
    assert array_equal(v, np.array(2))
    assert v.dtype == "int64"
    assert v.ndim == 0
    assert v.dimensions == ()
    assert list(v.attrs) == []

    v = ds["var_len_str"]
    assert h5py.check_dtype(vlen=v.dtype) == str
    if getattr(ds, "decode_vlen_strings", True):
        assert v[0] == _vlen_string
    else:
        assert v[0] == _vlen_string.encode("utf_8")

    v = ds["/subgroup/subvar"]
    assert v is ds["subgroup"]["subvar"]
    assert v is ds["subgroup/subvar"]
    assert v is ds["subgroup"]["/subgroup/subvar"]
    assert v.name == "/subgroup/subvar"
    assert ds["subgroup"].name == "/subgroup"
    assert ds["subgroup"].parent is ds
    assert array_equal(v, np.arange(4.0))
    assert v.dtype == "int32"
    assert v.ndim == 1
    assert v.dimensions == ("x",)
    assert list(v.attrs) == []

    assert ds["/subgroup/y_var"].shape == (10,)
    assert ds["/subgroup"].dimensions["y"].size == 10

    ds.close()


def roundtrip_legacy_netcdf(tmp_netcdf, read_module, write_module):
    write_legacy_netcdf(tmp_netcdf, write_module)
    read_legacy_netcdf(tmp_netcdf, read_module, write_module)


def test_write_legacyapi_read_netCDF4(tmp_local_netcdf):
    roundtrip_legacy_netcdf(tmp_local_netcdf, netCDF4, legacyapi)


def test_roundtrip_h5netcdf_legacyapi(tmp_local_netcdf):
    roundtrip_legacy_netcdf(tmp_local_netcdf, legacyapi, legacyapi)


def test_write_netCDF4_read_legacyapi(tmp_local_netcdf):
    roundtrip_legacy_netcdf(tmp_local_netcdf, legacyapi, netCDF4)


def test_write_h5netcdf_read_legacyapi(tmp_local_netcdf):
    write_h5netcdf(tmp_local_netcdf)
    read_legacy_netcdf(tmp_local_netcdf, legacyapi, h5netcdf)


def test_write_h5netcdf_read_netCDF4(tmp_local_netcdf):
    write_h5netcdf(tmp_local_netcdf)
    read_legacy_netcdf(tmp_local_netcdf, netCDF4, h5netcdf)


def test_roundtrip_h5netcdf(tmp_local_or_remote_netcdf, decode_vlen_strings):
    write_h5netcdf(tmp_local_or_remote_netcdf)
    read_h5netcdf(tmp_local_or_remote_netcdf, h5netcdf, decode_vlen_strings)


def test_write_netCDF4_read_h5netcdf(tmp_local_netcdf, decode_vlen_strings):
    write_legacy_netcdf(tmp_local_netcdf, netCDF4)
    read_h5netcdf(tmp_local_netcdf, netCDF4, decode_vlen_strings)


def test_write_legacyapi_read_h5netcdf(tmp_local_netcdf, decode_vlen_strings):
    write_legacy_netcdf(tmp_local_netcdf, legacyapi)
    read_h5netcdf(tmp_local_netcdf, legacyapi, decode_vlen_strings)


def test_fileobj(decode_vlen_strings):
    if version.parse(h5py.__version__) < version.parse("2.9.0"):
        pytest.skip("h5py > 2.9.0 required to test file-like objects")
    fileobj = tempfile.TemporaryFile()
    write_h5netcdf(fileobj)
    read_h5netcdf(fileobj, h5netcdf, decode_vlen_strings)
    fileobj = io.BytesIO()
    write_h5netcdf(fileobj)
    read_h5netcdf(fileobj, h5netcdf, decode_vlen_strings)


def test_repr(tmp_local_or_remote_netcdf):
    write_h5netcdf(tmp_local_or_remote_netcdf)
    f = h5netcdf.File(tmp_local_or_remote_netcdf, "a")
    assert "h5netcdf.File" in repr(f)
    assert "subgroup" in repr(f)
    assert "foo" in repr(f)
    assert "other_attr" in repr(f)

    assert "h5netcdf.attrs.Attributes" in repr(f.attrs)
    assert "global" in repr(f.attrs)

    d = f.dimensions
    assert "h5netcdf.Dimensions" in repr(d)
    assert "x=<h5netcdf.Dimension 'x': size 4>" in repr(d)

    g = f["subgroup"]
    assert "h5netcdf.Group" in repr(g)
    assert "subvar" in repr(g)

    v = f["foo"]
    assert "h5netcdf.Variable" in repr(v)
    assert "float" in repr(v)
    assert "units" in repr(v)

    f.dimensions["temp"] = None
    assert "temp: <h5netcdf.Dimension 'temp': size 0 (unlimited)>" in repr(f)
    f.resize_dimension("temp", 5)
    assert "temp: <h5netcdf.Dimension 'temp': size 5 (unlimited)>" in repr(f)

    f.close()

    assert "Closed" in repr(f)
    assert "Closed" in repr(d)
    assert "Closed" in repr(g)
    assert "Closed" in repr(v)


def test_attrs_api(tmp_local_or_remote_netcdf):
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5netcdf.File(tmp_local_or_remote_netcdf, "w") as ds:
        ds.attrs["conventions"] = "CF"
        ds.attrs["empty_string"] = h5.Empty(dtype=np.dtype("|S1"))
        ds.dimensions["x"] = 1
        v = ds.create_variable("x", ("x",), "i4")
        v.attrs.update({"units": "meters", "foo": "bar"})
    assert ds._closed
    with h5netcdf.File(tmp_local_or_remote_netcdf, "r") as ds:
        assert len(ds.attrs) == 2
        assert dict(ds.attrs) == {"conventions": "CF", "empty_string": b""}
        assert list(ds.attrs) == ["conventions", "empty_string"]
        assert dict(ds["x"].attrs) == {"units": "meters", "foo": "bar"}
        assert len(ds["x"].attrs) == 2
        assert sorted(ds["x"].attrs) == ["foo", "units"]


def test_optional_netcdf4_attrs(tmp_local_or_remote_netcdf):
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, "w") as f:
        foo_data = np.arange(50).reshape(5, 10)
        f.create_dataset("foo", data=foo_data)
        f.create_dataset("x", data=np.arange(5))
        f.create_dataset("y", data=np.arange(10))
        if version.parse(h5py.__version__) < version.parse("2.10.0"):
            f["foo"].dims.create_scale(f["x"])
            f["foo"].dims.create_scale(f["y"])
        else:
            f["x"].make_scale()
            f["y"].make_scale()
        f["foo"].dims[0].attach_scale(f["x"])
        f["foo"].dims[1].attach_scale(f["y"])
    with h5netcdf.File(tmp_local_or_remote_netcdf, "r") as ds:
        assert ds["foo"].dimensions == ("x", "y")
        assert ds.dimensions.keys() == {"x", "y"}
        assert ds.dimensions["x"].size == 5
        assert ds.dimensions["y"].size == 10
        assert array_equal(ds["foo"], foo_data)


def test_error_handling(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf, "w") as ds:
        ds.dimensions["x"] = 1
        with raises(ValueError):
            ds.dimensions["x"] = 2
        with raises(ValueError):
            ds.dimensions = {"x": 2}
        with raises(ValueError):
            ds.dimensions = {"y": 3}
        ds.create_variable("x", ("x",), dtype=float)
        with raises(ValueError):
            ds.create_variable("x", ("x",), dtype=float)
        ds.create_group("subgroup")
        with raises(ValueError):
            ds.create_group("subgroup")


@pytest.mark.skipif(
    version.parse(h5py.__version__) < version.parse("3.0.0"),
    reason="not needed with h5py < 3.0",
)
def test_decode_string_error(tmp_local_or_remote_netcdf):
    write_h5netcdf(tmp_local_or_remote_netcdf)
    with pytest.raises(TypeError):
        with h5netcdf.legacyapi.Dataset(
            tmp_local_or_remote_netcdf, "r", decode_vlen_strings=True
        ) as ds:
            assert ds.name == "/"


def create_invalid_netcdf_data():
    foo_data = np.arange(125).reshape(5, 5, 5)
    bar_data = np.arange(625).reshape(25, 5, 5)
    var = {"foo1": foo_data, "foo2": bar_data, "foo3": foo_data, "foo4": bar_data}
    var2 = {"x": 5, "y": 5, "z": 5, "x1": 25, "y1": 5, "z1": 5}
    return var, var2


def check_invalid_netcdf4(var, i):
    pdim = "phony_dim_{}"
    assert var["foo1"].dimensions[0] == pdim.format(i * 4)
    assert var["foo1"].dimensions[1] == pdim.format(1 + i * 4)
    assert var["foo1"].dimensions[2] == pdim.format(2 + i * 4)
    assert var["foo2"].dimensions[0] == pdim.format(3 + i * 4)
    assert var["foo2"].dimensions[1] == pdim.format(0 + i * 4)
    assert var["foo2"].dimensions[2] == pdim.format(1 + i * 4)
    assert var["foo3"].dimensions[0] == pdim.format(i * 4)
    assert var["foo3"].dimensions[1] == pdim.format(1 + i * 4)
    assert var["foo3"].dimensions[2] == pdim.format(2 + i * 4)
    assert var["foo4"].dimensions[0] == pdim.format(3 + i * 4)
    assert var["foo4"].dimensions[1] == pdim.format(i * 4)
    assert var["foo4"].dimensions[2] == pdim.format(1 + i * 4)
    assert var["x"].dimensions[0] == pdim.format(i * 4)
    assert var["y"].dimensions[0] == pdim.format(i * 4)
    assert var["z"].dimensions[0] == pdim.format(i * 4)
    assert var["x1"].dimensions[0] == pdim.format(3 + i * 4)
    assert var["y1"].dimensions[0] == pdim.format(i * 4)
    assert var["z1"].dimensions[0] == pdim.format(i * 4)


def test_invalid_netcdf4(tmp_local_or_remote_netcdf):
    if tmp_local_or_remote_netcdf.startswith(remote_h5):
        pytest.skip("netCDF4 package does not work with remote HDF5 files")
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, "w") as f:
        var, var2 = create_invalid_netcdf_data()
        grps = ["bar", "baz"]
        for grp in grps:
            fx = f.create_group(grp)
            for k, v in var.items():
                fx.create_dataset(k, data=v)
            for k, v in var2.items():
                fx.create_dataset(k, data=np.arange(v))

    with h5netcdf.File(tmp_local_or_remote_netcdf, "r", phony_dims="sort") as dsr:
        for i, grp in enumerate(grps):
            var = dsr[grp].variables
            check_invalid_netcdf4(var, i)

    with h5netcdf.File(tmp_local_or_remote_netcdf, "r", phony_dims="access") as dsr:
        for i, grp in enumerate(grps):
            var = dsr[grp].variables
            check_invalid_netcdf4(var, i)

    if not tmp_local_or_remote_netcdf.startswith(remote_h5):
        # netcdf4 package does not work with remote HDF5 files
        with netCDF4.Dataset(tmp_local_or_remote_netcdf, "r") as dsr:
            for i, grp in enumerate(grps):
                var = dsr[grp].variables
                check_invalid_netcdf4(var, i)

    with h5netcdf.File(tmp_local_or_remote_netcdf, "r") as ds:
        with raises(ValueError):
            ds["bar"].variables["foo1"].dimensions

    with raises(ValueError):
        with h5netcdf.File(tmp_local_or_remote_netcdf, "r", phony_dims="srt") as ds:
            pass


def test_fake_phony_dims(tmp_local_or_remote_netcdf):
    # tests writing of dimension with phony naming scheme
    # see https://github.com/h5netcdf/h5netcdf/issues/178
    with h5netcdf.File(tmp_local_or_remote_netcdf, mode="w") as ds:
        ds.dimensions["phony_dim_0"] = 3


def check_invalid_netcdf4_mixed(var, i):
    pdim = "phony_dim_{}".format(i)
    assert var["foo1"].dimensions[0] == "y1"
    assert var["foo1"].dimensions[1] == "z1"
    assert var["foo1"].dimensions[2] == pdim
    assert var["foo2"].dimensions[0] == "x1"
    assert var["foo2"].dimensions[1] == "y1"
    assert var["foo2"].dimensions[2] == "z1"
    assert var["foo3"].dimensions[0] == "y1"
    assert var["foo3"].dimensions[1] == "z1"
    assert var["foo3"].dimensions[2] == pdim
    assert var["foo4"].dimensions[0] == "x1"
    assert var["foo4"].dimensions[1] == "y1"
    assert var["foo4"].dimensions[2] == "z1"
    assert var["x"].dimensions[0] == "y1"
    assert var["y"].dimensions[0] == "y1"
    assert var["z"].dimensions[0] == "y1"
    assert var["x1"].dimensions[0] == "x1"
    assert var["y1"].dimensions[0] == "y1"
    assert var["z1"].dimensions[0] == "z1"


def test_invalid_netcdf4_mixed(tmp_local_or_remote_netcdf):
    if tmp_local_or_remote_netcdf.startswith(remote_h5):
        pytest.skip("netCDF4 package does not work with remote HDF5 files")
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, "w") as f:
        var, var2 = create_invalid_netcdf_data()
        for k, v in var.items():
            f.create_dataset(k, data=v)
        for k, v in var2.items():
            f.create_dataset(k, data=np.arange(v))

        if version.parse(h5py.__version__) < version.parse("2.10.0"):
            f["foo2"].dims.create_scale(f["x1"])
            f["foo2"].dims.create_scale(f["y1"])
            f["foo2"].dims.create_scale(f["z1"])
        else:
            f["x1"].make_scale()
            f["y1"].make_scale()
            f["z1"].make_scale()
        f["foo2"].dims[0].attach_scale(f["x1"])
        f["foo2"].dims[1].attach_scale(f["y1"])
        f["foo2"].dims[2].attach_scale(f["z1"])

    with h5netcdf.File(tmp_local_or_remote_netcdf, "r", phony_dims="sort") as ds:
        var = ds.variables
        check_invalid_netcdf4_mixed(var, 3)

    with h5netcdf.File(tmp_local_or_remote_netcdf, "r", phony_dims="access") as ds:
        var = ds.variables
        check_invalid_netcdf4_mixed(var, 0)

    if not tmp_local_or_remote_netcdf.startswith(remote_h5):
        # netcdf4 package does not work with remote HDF5 files
        with netCDF4.Dataset(tmp_local_or_remote_netcdf, "r") as ds:
            var = ds.variables
            check_invalid_netcdf4_mixed(var, 3)

    with h5netcdf.File(tmp_local_or_remote_netcdf, "r") as ds:
        with raises(ValueError):
            ds.variables["foo1"].dimensions


def test_invalid_netcdf_malformed_dimension_scales(tmp_local_or_remote_netcdf):
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, "w") as f:
        foo_data = np.arange(125).reshape(5, 5, 5)
        f.create_dataset("foo1", data=foo_data)
        f.create_dataset("x", data=np.arange(5))
        f.create_dataset("y", data=np.arange(5))
        f.create_dataset("z", data=np.arange(5))

        if version.parse(h5py.__version__) < version.parse("2.10.0"):
            f["foo1"].dims.create_scale(f["x"])
            f["foo1"].dims.create_scale(f["y"])
            f["foo1"].dims.create_scale(f["z"])
        else:
            f["x"].make_scale()
            f["y"].make_scale()
            f["z"].make_scale()
        f["foo1"].dims[0].attach_scale(f["x"])

    with raises(ValueError):
        with h5netcdf.File(tmp_local_or_remote_netcdf, "r") as ds:
            assert ds
            print(ds)

    with raises(ValueError):
        with h5netcdf.File(tmp_local_or_remote_netcdf, "r", phony_dims="sort") as ds:
            assert ds
            print(ds)


def test_hierarchical_access_auto_create(tmp_local_or_remote_netcdf):
    ds = h5netcdf.File(tmp_local_or_remote_netcdf, "w")
    ds.create_variable("/foo/bar", data=1)
    g = ds.create_group("foo/baz")
    g.create_variable("/foo/hello", data=2)
    assert set(ds) == set(["foo"])
    assert set(ds["foo"]) == set(["bar", "baz", "hello"])
    ds.close()

    ds = h5netcdf.File(tmp_local_or_remote_netcdf, "r")
    assert set(ds) == set(["foo"])
    assert set(ds["foo"]) == set(["bar", "baz", "hello"])
    ds.close()


def test_Netcdf4Dimid(tmp_local_netcdf):
    # regression test for https://github.com/h5netcdf/h5netcdf/issues/53
    with h5netcdf.File(tmp_local_netcdf, "w") as f:
        f.dimensions["x"] = 1
        g = f.create_group("foo")
        g.dimensions["x"] = 2
        g.dimensions["y"] = 3

    with h5py.File(tmp_local_netcdf, "r") as f:
        # all dimension IDs should be present exactly once
        dim_ids = {f[name].attrs["_Netcdf4Dimid"] for name in ["x", "foo/x", "foo/y"]}
        assert dim_ids == {0, 1, 2}


def test_reading_str_array_from_netCDF4(tmp_local_netcdf, decode_vlen_strings):
    # This tests reading string variables created by netCDF4
    with netCDF4.Dataset(tmp_local_netcdf, "w") as ds:
        ds.createDimension("foo1", _string_array.shape[0])
        ds.createDimension("foo2", _string_array.shape[1])
        ds.createVariable("bar", str, ("foo1", "foo2"))
        ds.variables["bar"][:] = _string_array

    ds = h5netcdf.File(tmp_local_netcdf, "r", **decode_vlen_strings)

    v = ds.variables["bar"]
    if getattr(ds, "decode_vlen_strings", True):
        assert array_equal(v, _string_array)
    else:
        assert array_equal(v, np.char.encode(_string_array))

    ds.close()


def test_nc_properties_new(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf, "w"):
        pass
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, "r") as f:
        assert b"h5netcdf" in f.attrs["_NCProperties"]


def test_failed_read_open_and_clean_delete(tmpdir):
    # A file that does not exist but is opened for
    # reading should only raise an IOError and
    # no AttributeError at garbage collection.
    path = str(tmpdir.join("this_file_does_not_exist.nc"))
    try:
        with h5netcdf.File(path, "r") as ds:
            assert ds
    except IOError:
        pass

    # Look at garbage collection:
    # A simple gc.collect() does not raise an exception.
    # Must seek the File object and imitate its del command
    # by forcing it to close.
    obj_list = gc.get_objects()
    for obj in obj_list:
        try:
            is_h5netcdf_File = isinstance(obj, h5netcdf.File)
        except AttributeError:
            is_h5netcdf_File = False
        if is_h5netcdf_File:
            obj.close()


def test_create_variable_matching_saved_dimension(tmp_local_or_remote_netcdf):
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)

    # if h5 is not h5py:
    #     pytest.xfail("https://github.com/h5netcdf/h5netcdf/issues/48")

    with h5netcdf.File(tmp_local_or_remote_netcdf, "w") as f:
        f.dimensions["x"] = 2
        f.create_variable("y", data=[1, 2], dimensions=("x",))

    with h5.File(tmp_local_or_remote_netcdf, "r") as f:
        dimlen = f"{f['y'].dims[0].values()[0].size:10}"
        assert f["y"].dims[0].keys() == [NOT_A_VARIABLE.decode("ascii") + dimlen]

    with h5netcdf.File(tmp_local_or_remote_netcdf, "a") as f:
        f.create_variable("x", data=[0, 1], dimensions=("x",))

    with h5.File(tmp_local_or_remote_netcdf, "r") as f:
        assert f["y"].dims[0].keys() == ["x"]


def test_invalid_netcdf_error(tmp_local_or_remote_netcdf):
    if tmp_local_or_remote_netcdf.startswith(remote_h5):
        pytest.skip("Remote HDF5 does not yet support LZF compression")
    with h5netcdf.File(tmp_local_or_remote_netcdf, "w", invalid_netcdf=False) as f:
        # valid
        f.create_variable(
            "lzf_compressed", data=[1], dimensions=("x"), compression="lzf"
        )
        # invalid
        with pytest.raises(h5netcdf.CompatibilityError):
            f.create_variable("complex", data=1j)
        with pytest.raises(h5netcdf.CompatibilityError):
            f.attrs["complex_attr"] = 1j
        with pytest.raises(h5netcdf.CompatibilityError):
            f.create_variable("scaleoffset", data=[1], dimensions=("x",), scaleoffset=0)


def test_invalid_netcdf_okay(tmp_local_or_remote_netcdf):
    if tmp_local_or_remote_netcdf.startswith(remote_h5):
        pytest.skip("h5pyd does not support NumPy complex dtype yet")
    with pytest.warns(UserWarning, match="invalid netcdf features"):
        with h5netcdf.File(tmp_local_or_remote_netcdf, "w", invalid_netcdf=True) as f:
            f.create_variable(
                "lzf_compressed", data=[1], dimensions=("x"), compression="lzf"
            )
            f.create_variable("complex", data=1j)
            f.attrs["complex_attr"] = 1j
            f.create_variable("scaleoffset", data=[1], dimensions=("x",), scaleoffset=0)
    with h5netcdf.File(tmp_local_or_remote_netcdf, "r") as f:
        np.testing.assert_equal(f["lzf_compressed"][:], [1])
        assert f["complex"][...] == 1j
        assert f.attrs["complex_attr"] == 1j
        np.testing.assert_equal(f["scaleoffset"][:], [1])
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, "r") as f:
        assert "_NCProperties" not in f.attrs


def test_invalid_netcdf_overwrite_valid(tmp_local_netcdf):
    # https://github.com/h5netcdf/h5netcdf/issues/165
    with netCDF4.Dataset(tmp_local_netcdf, mode="w"):
        pass
    with pytest.warns(UserWarning):
        with h5netcdf.File(tmp_local_netcdf, "a", invalid_netcdf=True) as f:
            f.create_variable(
                "lzf_compressed", data=[1], dimensions=("x"), compression="lzf"
            )
            f.create_variable("complex", data=1j)
            f.attrs["complex_attr"] = 1j
            f.create_variable("scaleoffset", data=[1], dimensions=("x",), scaleoffset=0)
    with h5netcdf.File(tmp_local_netcdf, "r") as f:
        np.testing.assert_equal(f["lzf_compressed"][:], [1])
        assert f["complex"][...] == 1j
        assert f.attrs["complex_attr"] == 1j
        np.testing.assert_equal(f["scaleoffset"][:], [1])
    h5 = get_hdf5_module(tmp_local_netcdf)
    with h5.File(tmp_local_netcdf, "r") as f:
        assert "_NCProperties" not in f.attrs


def test_reopen_file_different_dimension_sizes(tmp_local_netcdf):
    # regression test for https://github.com/h5netcdf/h5netcdf/issues/55
    with h5netcdf.File(tmp_local_netcdf, "w") as f:
        f.create_variable("/one/foo", data=[1], dimensions=("x",))
    with h5netcdf.File(tmp_local_netcdf, "a") as f:
        f.create_variable("/two/foo", data=[1, 2], dimensions=("x",))
    with netCDF4.Dataset(tmp_local_netcdf, "r") as f:
        assert f.groups["one"].variables["foo"][...].shape == (1,)


def test_invalid_then_valid_no_ncproperties(tmp_local_or_remote_netcdf):
    with pytest.warns(UserWarning, match="invalid netcdf features"):
        with h5netcdf.File(tmp_local_or_remote_netcdf, "w", invalid_netcdf=True):
            pass
    with h5netcdf.File(tmp_local_or_remote_netcdf, "a"):
        pass
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, "r") as f:
        # still not a valid netcdf file
        assert "_NCProperties" not in f.attrs


def test_creating_and_resizing_unlimited_dimensions(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf, "w") as f:
        f.dimensions["x"] = None
        f.dimensions["y"] = 15
        f.dimensions["z"] = None
        f.resize_dimension("z", 20)

        with pytest.raises(ValueError) as e:
            f.resize_dimension("y", 20)
        assert e.value.args[0] == (
            "Dimension 'y' is not unlimited and thus cannot be resized."
        )

    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    # Assert some behavior observed by using the C netCDF bindings.
    with h5.File(tmp_local_or_remote_netcdf, "r") as f:
        assert f["x"].shape == (0,)
        assert f["x"].maxshape == (None,)
        assert f["y"].shape == (15,)
        assert f["y"].maxshape == (15,)
        assert f["z"].shape == (20,)
        assert f["z"].maxshape == (None,)


def test_creating_variables_with_unlimited_dimensions(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf, "w") as f:
        f.dimensions["x"] = None
        f.dimensions["y"] = 2

        # Creating a variable without data will initialize an array with zero
        # length.
        f.create_variable("dummy", dimensions=("x", "y"), dtype=np.int64)
        assert f.variables["dummy"].shape == (0, 2)
        assert f.variables["dummy"]._h5ds.maxshape == (None, 2)

        # Trying to create a variable while the current size of the dimension
        # is still zero will fail.
        with pytest.raises(ValueError) as e:
            f.create_variable(
                "dummy2", data=np.array([[1, 2], [3, 4]]), dimensions=("x", "y")
            )
        assert e.value.args[0] == "Shape tuple is incompatible with data"

        # Creating a coordinate variable
        f.create_variable("x", dimensions=("x",), dtype=np.int64)

        # Resize data.
        assert f.variables["dummy"].shape == (0, 2)
        f.resize_dimension("x", 3)
        # This will also force a resize of the existing variables and it will
        # be padded with zeros.
        assert f.dimensions["x"].size == 3
        np.testing.assert_allclose(f.variables["dummy"], np.zeros((3, 2)))

        # Creating another variable with no data will now also take the shape
        # of the current dimensions.
        f.create_variable("dummy3", dimensions=("x", "y"), dtype=np.int64)
        assert f.variables["dummy3"].shape == (3, 2)
        assert f.variables["dummy3"]._h5ds.maxshape == (None, 2)
        np.testing.assert_allclose(f.variables["dummy3"], np.zeros((3, 2)))

        # Writing to a variable with an unlimited dimension raises
        with pytest.raises(TypeError) as e:
            f.variables["dummy3"][:] = np.ones((5, 2))
        assert e.value.args[0] == "Can't broadcast (5, 2) -> (3, 2)"
        assert f.variables["dummy3"].shape == (3, 2)
        assert f.variables["dummy3"]._h5ds.maxshape == (None, 2)
        assert f["x"].shape == (3,)
        assert f.dimensions["x"].size == 3
        np.testing.assert_allclose(f.variables["dummy3"], np.zeros((3, 2)))

    # Close and read again to also test correct parsing of unlimited
    # dimensions.
    with h5netcdf.File(tmp_local_or_remote_netcdf, "r") as f:
        assert f.dimensions["x"].isunlimited()
        assert f.dimensions["x"].size == 3
        assert f._h5file["x"].maxshape == (None,)
        assert f._h5file["x"].shape == (3,)

        assert f.dimensions["y"].size == 2
        assert f._h5file["y"].maxshape == (2,)
        assert f._h5file["y"].shape == (2,)


def test_writing_to_an_unlimited_dimension(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf, "w") as f:
        # Two dimensions, only one is unlimited.
        f.dimensions["x"] = None
        f.dimensions["y"] = 3
        f.dimensions["z"] = None

        # Cannot create it without first resizing it.
        with pytest.raises(ValueError) as e:
            f.create_variable(
                "dummy1", data=np.array([[1, 2, 3]]), dimensions=("x", "y")
            )
            assert e.value.args[0] == "Shape tuple is incompatible with data"

        # Without data.
        f.create_variable("dummy1", dimensions=("x", "y"), dtype=np.int64)
        f.create_variable("dummy2", dimensions=("x", "y"), dtype=np.int64)
        f.create_variable("dummy3", dimensions=("x", "y"), dtype=np.int64)
        f.create_variable("dummyX", dimensions=("x", "y", "z"), dtype=np.int64)
        g = f.create_group("test")
        g.create_variable("dummy4", dimensions=("y", "x", "x"), dtype=np.int64)
        g.create_variable("dummy5", dimensions=("y", "y"), dtype=np.int64)

        assert f.variables["dummy1"].shape == (0, 3)
        assert f.variables["dummy2"].shape == (0, 3)
        assert f.variables["dummy3"].shape == (0, 3)
        assert f.variables["dummyX"].shape == (0, 3, 0)
        assert g.variables["dummy4"].shape == (3, 0, 0)
        assert g.variables["dummy5"].shape == (3, 3)

        # resize dimensions and all connected variables
        f.resize_dimension("x", 2)
        assert f.variables["dummy1"].shape == (2, 3)
        assert f.variables["dummy2"].shape == (2, 3)
        assert f.variables["dummy3"].shape == (2, 3)
        assert f.variables["dummyX"].shape == (2, 3, 0)
        assert g.variables["dummy4"].shape == (3, 2, 2)
        assert g.variables["dummy5"].shape == (3, 3)

        # broadcast writing
        f.variables["dummy3"][...] = [[1, 2, 3]]
        np.testing.assert_allclose(f.variables["dummy3"], [[1, 2, 3], [1, 2, 3]])


def test_c_api_can_read_unlimited_dimensions(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, "w") as f:
        # Three dimensions, only one is limited.
        f.dimensions["x"] = None
        f.dimensions["y"] = 3
        f.dimensions["z"] = None
        f.create_variable("dummy1", dimensions=("x", "y"), dtype=np.int64)
        f.create_variable("dummy2", dimensions=("y", "x", "x"), dtype=np.int64)
        g = f.create_group("test")
        g.create_variable("dummy3", dimensions=("y", "y"), dtype=np.int64)
        g.create_variable("dummy4", dimensions=("z", "z"), dtype=np.int64)
        f.resize_dimension("x", 2)

    with netCDF4.Dataset(tmp_local_netcdf, "r") as f:
        assert f.dimensions["x"].size == 2
        assert f.dimensions["x"].isunlimited() is True
        assert f.dimensions["y"].size == 3
        assert f.dimensions["y"].isunlimited() is False
        assert f.dimensions["z"].size == 0
        assert f.dimensions["z"].isunlimited() is True

        assert f.variables["dummy1"].shape == (2, 3)
        assert f.variables["dummy2"].shape == (3, 2, 2)
        g = f.groups["test"]
        assert g.variables["dummy3"].shape == (3, 3)
        assert g.variables["dummy4"].shape == (0, 0)


def test_reading_unlimited_dimensions_created_with_c_api(tmp_local_netcdf):
    with netCDF4.Dataset(tmp_local_netcdf, "w") as f:
        f.createDimension("x", None)
        f.createDimension("y", 3)
        f.createDimension("z", None)

        dummy1 = f.createVariable("dummy1", float, ("x", "y"))
        f.createVariable("dummy2", float, ("y", "x", "x"))
        g = f.createGroup("test")
        g.createVariable("dummy3", float, ("y", "y"))
        g.createVariable("dummy4", float, ("z", "z"))

        # Assign something to trigger a resize.
        dummy1[:] = [[1, 2, 3], [4, 5, 6]]

        # Create another variable with same dimensions
        f.createVariable("dummy5", float, ("x", "y"))

    with h5netcdf.File(tmp_local_netcdf, "r") as f:
        assert f.dimensions["x"].isunlimited()
        assert f.dimensions["y"].size == 3
        assert f.dimensions["z"].isunlimited()

        # This is parsed correctly due to h5netcdf's init trickery.
        assert f.dimensions["x"].size == 2
        assert f.dimensions["y"].size == 3
        assert f.dimensions["z"].size == 0

        # But the actual data-set and arrays are not correct.
        # assert f["dummy1"].shape == (2, 3)
        # XXX: This array has some data with dimension x - netcdf does not
        # appear to keep dimensions consistent.
        # With https://github.com/h5netcdf/h5netcdf/pull/103 h5netcdf will
        # return a padded array
        assert f["dummy2"].shape == (3, 2, 2)
        f.groups["test"]["dummy3"].shape == (3, 3)
        f.groups["test"]["dummy4"].shape == (0, 0)
        assert f["dummy5"].shape == (2, 3)


def test_reading_unused_unlimited_dimension(tmp_local_or_remote_netcdf):
    """Test reading a file with unused dimension of unlimited size"""
    with h5netcdf.File(tmp_local_or_remote_netcdf, "w") as f:
        f.dimensions = {"x": None}
        f.resize_dimension("x", 5)
        assert f.dimensions["x"].isunlimited()
        assert f.dimensions["x"].size == 5


def test_reading_special_datatype_created_with_c_api(tmp_local_netcdf):
    """Test reading a file with unsupported Datatype"""
    with netCDF4.Dataset(tmp_local_netcdf, "w") as f:
        complex128 = np.dtype([("real", np.float64), ("imag", np.float64)])
        f.createCompoundType(complex128, "complex128")
    with h5netcdf.File(tmp_local_netcdf, "r") as f:
        pass


def test_nc4_non_coord(tmp_local_netcdf):
    # Here we generate a few variables and coordinates
    # The default should be to track the order of creation
    # Thus, on reopening the file, the order in which
    # the variables are listed should be maintained
    # y   --   refers to the coordinate y
    # _nc4_non_coord_y  --  refers to the data y
    with h5netcdf.File(tmp_local_netcdf, "w") as f:
        f.dimensions = {"x": None, "y": 2}
        f.create_variable("test", dimensions=("x",), dtype=np.int64)
        f.create_variable("y", dimensions=("x",), dtype=np.int64)

    with h5netcdf.File(tmp_local_netcdf, "r") as f:
        assert list(f.dimensions) == ["x", "y"]
        assert f.dimensions["x"].size == 0
        assert f.dimensions["x"].isunlimited()
        assert f.dimensions["y"].size == 2
        if version.parse(h5py.__version__) >= version.parse("3.7.0"):
            assert list(f.variables) == ["test", "y"]
            assert list(f._h5group.keys()) == ["x", "y", "test", "_nc4_non_coord_y"]

    with h5netcdf.File(tmp_local_netcdf, "w") as f:
        f.dimensions = {"x": None, "y": 2}
        f.create_variable("y", dimensions=("x",), dtype=np.int64)
        f.create_variable("test", dimensions=("x",), dtype=np.int64)

    with h5netcdf.File(tmp_local_netcdf, "r") as f:
        assert list(f.dimensions) == ["x", "y"]
        assert f.dimensions["x"].size == 0
        assert f.dimensions["x"].isunlimited()
        assert f.dimensions["y"].size == 2
        if version.parse(h5py.__version__) >= version.parse("3.7.0"):
            assert list(f.variables) == ["y", "test"]
            assert list(f._h5group.keys()) == ["x", "y", "_nc4_non_coord_y", "test"]


def test_overwrite_existing_file(tmp_local_netcdf):
    # create file with _NCProperties attribute
    with netCDF4.Dataset(tmp_local_netcdf, "w") as ds:
        ds.createDimension("x", 10)

    # check attribute
    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        assert ds.attrs._h5attrs.get("_NCProperties", False)

    # overwrite file with legacyapi
    with legacyapi.Dataset(tmp_local_netcdf, "w") as ds:
        ds.createDimension("x", 10)

    # check attribute
    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        assert ds.attrs._h5attrs.get("_NCProperties", False)

    # overwrite file with new api
    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        ds.dimensions["x"] = 10

    # check attribute
    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        assert ds.attrs._h5attrs.get("_NCProperties", False)


def test_scales_on_append(tmp_local_netcdf):
    # create file with _NCProperties attribute
    with netCDF4.Dataset(tmp_local_netcdf, "w") as ds:
        ds.createDimension("x", 10)

    # append file with netCDF4
    with netCDF4.Dataset(tmp_local_netcdf, "r+") as ds:
        ds.createVariable("test", "i4", ("x",))

    # check scales
    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        assert ds.variables["test"].attrs._h5attrs.get("DIMENSION_LIST", False)

    # append file with legacyapi
    with legacyapi.Dataset(tmp_local_netcdf, "r+") as ds:
        ds.createVariable("test1", "i4", ("x",))

    # check scales
    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        assert ds.variables["test1"].attrs._h5attrs.get("DIMENSION_LIST", False)


def create_attach_scales(filename, append_module):
    # create file with netCDF4
    with netCDF4.Dataset(filename, "w") as ds:
        ds.createDimension("x", 0)
        ds.createDimension("y", 1)
        ds.createVariable("test", "i4", ("x",))
        ds.variables["test"] = np.ones((10,))

    # append file with netCDF4
    with append_module.Dataset(filename, "a") as ds:
        ds.createVariable("test1", "i4", ("x",))
        ds.createVariable("y", "i4", ("x", "y"))

    # check scales
    with h5netcdf.File(filename, "r") as ds:
        refs = ds._h5group["x"].attrs.get("REFERENCE_LIST", False)
        assert len(refs) == 3
        for (ref, dim), name in zip(refs, ["/test", "/test1", "/_nc4_non_coord_y"]):
            assert dim == 0
            assert ds._root._h5file[ref].name == name


def test_create_attach_scales_netcdf4(tmp_local_netcdf):
    create_attach_scales(tmp_local_netcdf, netCDF4)


def test_create_attach_scales_legacyapi(tmp_local_netcdf):
    create_attach_scales(tmp_local_netcdf, legacyapi)


def test_detach_scale(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        ds.dimensions["x"] = 2
        ds.dimensions["y"] = 2

    with h5netcdf.File(tmp_local_netcdf, "a") as ds:
        ds.create_variable("test", dimensions=("x",), dtype=np.int64)

    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        refs = ds._h5group["x"].attrs.get("REFERENCE_LIST", False)
        assert len(refs) == 1
        for (ref, dim), name in zip(refs, ["/test"]):
            assert dim == 0
            assert ds._root._h5file[ref].name == name

    with h5netcdf.File(tmp_local_netcdf, "a") as ds:
        ds.dimensions["x"]._detach_scale()

    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        refs = ds._h5group["x"].attrs.get("REFERENCE_LIST", False)
        assert not refs


def test_is_scale(tmp_local_netcdf):
    with legacyapi.Dataset(tmp_local_netcdf, "w") as ds:
        ds.createDimension("x", 10)
    with legacyapi.Dataset(tmp_local_netcdf, "r") as ds:
        assert ds.dimensions["x"]._isscale


def test_get_dim_scale_refs(tmp_local_netcdf):
    with legacyapi.Dataset(tmp_local_netcdf, "w") as ds:
        ds.createDimension("x", 10)
        ds.createVariable("test0", "i8", ("x",))
        ds.createVariable("test1", "i8", ("x",))
    with legacyapi.Dataset(tmp_local_netcdf, "r") as ds:
        refs = ds.dimensions["x"]._scale_refs
        assert ds._h5file[refs[0][0]] == ds["test0"]._h5ds
        assert ds._h5file[refs[1][0]] == ds["test1"]._h5ds


def create_netcdf_dimensions(ds, idx):
    # dimension and variable setup is adapted from the blogpost at
    # https://www.unidata.ucar.edu/blogs/developer/en/entry/netcdf4_shared_dimensions
    g = ds.createGroup("dimtest" + str(idx))
    g.createDimension("time", 0)  # time
    g.createDimension("nvec", 5 + idx)  # nvec
    g.createDimension("sample", 2 + idx)  # sample
    g.createDimension("ship", 3 + idx)  # ship
    g.createDimension("ship_strlen", 10)  # ship_strlen
    g.createDimension("collide", 7 + idx)  # collide

    time = g.createVariable("time", "f8", ("time",))
    data = g.createVariable("data", "i8", ("ship", "sample", "time", "nvec"))
    collide = g.createVariable("collide", "i8", ("nvec",))
    non_collide = g.createVariable("non_collide", "i8", ("nvec",))
    ship = g.createVariable("ship", "S1", ("ship", "ship_strlen"))
    sample = g.createVariable("sample", "i8", ("time", "sample"))

    time[:] = np.arange(10 + idx)
    data[:] = np.ones((3 + idx, 2 + idx, 10 + idx, 5 + idx)) * 12.0
    collide[...] = np.arange(5 + idx)
    non_collide[...] = np.arange(5 + idx) + 10
    sample[0 : 2 + idx, : 2 + idx] = np.ones((2 + idx, 2 + idx))
    if version.parse(h5py.__version__) >= version.parse("3.0.0"):
        ship[0] = list("Skiff     ")
    else:
        ship[0] = string_to_char(np.array("Skiff     ", dtype="|S1"))


def create_h5netcdf_dimensions(ds, idx):
    # dimension and variable setup is adapted from the blogpost at
    # https://www.unidata.ucar.edu/blogs/developer/en/entry/netcdf4_shared_dimensions
    g = ds.create_group("dimtest" + str(idx))
    g.dimensions["time"] = 0  # time
    g.dimensions["nvec"] = 5 + idx  # nvec
    g.dimensions["sample"] = 2 + idx  # sample
    g.dimensions["ship"] = 3 + idx  # ship
    g.dimensions["ship_strlen"] = 10  # ship_strlen
    g.dimensions["collide"] = 7 + idx  # collide

    g.create_variable("time", dimensions=("time",), dtype=np.float64)
    g.create_variable(
        "data", dimensions=("ship", "sample", "time", "nvec"), dtype=np.int64
    )
    g.create_variable("collide", dimensions=("nvec",), dtype=np.int64)
    g.create_variable("non_collide", dimensions=("nvec",), dtype=np.int64)
    g.create_variable("sample", dimensions=("time", "sample"), dtype=np.int64)
    g.create_variable("ship", dimensions=("ship", "ship_strlen"), dtype="S1")

    g.resize_dimension("time", 10 + idx)
    g.variables["time"][:] = np.arange(10 + idx)
    g.variables["data"][:] = np.ones((3 + idx, 2 + idx, 10 + idx, 5 + idx)) * 12.0
    g.variables["collide"][...] = np.arange(5 + idx)
    g.variables["non_collide"][...] = np.arange(5 + idx) + 10
    g.variables["sample"][0 : 2 + idx, : 2 + idx] = np.ones((2 + idx, 2 + idx))
    if version.parse(h5py.__version__) >= version.parse("3.0.0"):
        g.variables["ship"][0] = list("Skiff     ")
    else:
        g.variables["ship"][0] = string_to_char(np.array("Skiff     ", dtype="|S1"))


def check_netcdf_dimensions(tmp_netcdf, write_module, read_module):
    if read_module in [legacyapi, netCDF4]:
        opener = read_module.Dataset
    else:
        opener = h5netcdf.File
    with opener(tmp_netcdf, "r") as ds:
        for i, grp in enumerate(["dimtest0", "dimtest1"]):
            g = ds.groups[grp]
            assert set(g.dimensions) == {
                "collide",
                "ship_strlen",
                "time",
                "nvec",
                "ship",
                "sample",
            }
            if read_module in [legacyapi, h5netcdf]:
                assert g.dimensions["time"].isunlimited()
                assert g.dimensions["time"].size == 10 + i
                assert not g.dimensions["nvec"].isunlimited()
                assert g.dimensions["nvec"].size == 5 + i
                assert not g.dimensions["sample"].isunlimited()
                assert g.dimensions["sample"].size == 2 + i
                assert not g.dimensions["collide"].isunlimited()
                assert g.dimensions["collide"].size == 7 + i
                assert not g.dimensions["ship"].isunlimited()
                assert g.dimensions["ship"].size == 3 + i
                assert not g.dimensions["ship_strlen"].isunlimited()
                assert g.dimensions["ship_strlen"].size == 10
            else:
                assert g.dimensions["time"].isunlimited()
                assert g.dimensions["time"].size == 10 + i
                assert not g.dimensions["nvec"].isunlimited()
                assert g.dimensions["nvec"].size == 5 + i
                assert not g.dimensions["sample"].isunlimited()
                assert g.dimensions["sample"].size == 2 + i
                assert not g.dimensions["ship"].isunlimited()
                assert g.dimensions["ship"].size == 3 + i
                assert not g.dimensions["ship_strlen"].isunlimited()
                assert g.dimensions["ship_strlen"].size == 10
                assert not g.dimensions["collide"].isunlimited()
                assert g.dimensions["collide"].size == 7 + i

            assert set(g.variables) == {
                "data",
                "collide",
                "non_collide",
                "time",
                "sample",
                "ship",
            }
            assert g.variables["time"].shape == (10 + i,)
            assert g.variables["data"].shape == (3 + i, 2 + i, 10 + i, 5 + i)
            assert g.variables["collide"].shape == (5 + i,)
            assert g.variables["non_collide"].shape == (5 + i,)
            assert g.variables["sample"].shape == (10 + i, 2 + i)
            assert g.variables["ship"].shape == (3 + i, 10)


def write_dimensions(tmp_netcdf, write_module):
    if write_module in [legacyapi, netCDF4]:
        with write_module.Dataset(tmp_netcdf, "w") as ds:
            create_netcdf_dimensions(ds, 0)
            create_netcdf_dimensions(ds, 1)
    else:
        with write_module.File(tmp_netcdf, "w") as ds:
            create_h5netcdf_dimensions(ds, 0)
            create_h5netcdf_dimensions(ds, 1)


@pytest.fixture(
    params=[
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
)
def read_write_matrix(request):
    print("write module:", request.param[0].__name__)
    print("read_module:", request.param[1].__name__)
    return request.param


def test_dimensions(tmp_local_netcdf, read_write_matrix):
    write_dimensions(tmp_local_netcdf, read_write_matrix[0])
    check_netcdf_dimensions(
        tmp_local_netcdf, read_write_matrix[0], read_write_matrix[1]
    )


def test_no_circular_references(tmp_local_netcdf):
    # https://github.com/h5py/h5py/issues/2019
    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        ds.dimensions["x"] = 2
        ds.dimensions["y"] = 2

    gc.collect()
    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        refs = gc.get_referrers(ds)
        for ref in refs:
            print(ref)
        assert len(refs) == 1


def test_expanded_variables_netcdf4(tmp_local_netcdf, netcdf_write_module):
    # partially reimplemented due to performance reason in edge cases
    # https://github.com/h5netcdf/h5netcdf/issues/182

    with netcdf_write_module.Dataset(tmp_local_netcdf, "w") as ds:
        f = ds.createGroup("test")
        f.createDimension("x", None)
        f.createDimension("y", 3)

        dummy1 = f.createVariable("dummy1", float, ("x", "y"))
        dummy2 = f.createVariable("dummy2", float, ("x", "y"))
        dummy3 = f.createVariable("dummy3", float, ("x", "y"))
        dummy4 = f.createVariable("dummy4", float, ("x", "y"))

        dummy1[:] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        dummy2[1, :] = [4, 5, 6]
        dummy3[0:2, :] = [[1, 2, 3], [4, 5, 6]]

        # don't mask, since h5netcdf doesn't do masking
        if netcdf_write_module == netCDF4:
            ds.set_auto_mask(False)

        res1 = dummy1[:]
        res2 = dummy2[:]
        res3 = dummy3[:]
        res4 = dummy4[:]

    with netCDF4.Dataset(tmp_local_netcdf, "r") as ds:
        # don't mask, since h5netcdf doesn't do masking
        if netcdf_write_module == netCDF4:
            ds.set_auto_mask(False)

        f = ds["test"]

        np.testing.assert_allclose(f.variables["dummy1"][:], res1)
        np.testing.assert_allclose(f.variables["dummy1"][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables["dummy1"][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables["dummy1"].shape == (3, 3)
        np.testing.assert_allclose(f.variables["dummy2"][:], res2)
        np.testing.assert_allclose(f.variables["dummy2"][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables["dummy2"][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables["dummy2"].shape == (3, 3)
        np.testing.assert_allclose(f.variables["dummy3"][:], res3)
        np.testing.assert_allclose(f.variables["dummy3"][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables["dummy3"][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables["dummy3"].shape == (3, 3)
        np.testing.assert_allclose(f.variables["dummy4"][:], res4)
        assert f.variables["dummy4"].shape == (3, 3)

    with legacyapi.Dataset(tmp_local_netcdf, "r") as ds:
        f = ds["test"]
        np.testing.assert_allclose(f.variables["dummy1"][:], res1)
        np.testing.assert_allclose(f.variables["dummy1"][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables["dummy1"][1:2, :], [[4.0, 5.0, 6.0]])
        np.testing.assert_allclose(f.variables["dummy1"]._h5ds[1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(
            f.variables["dummy1"]._h5ds[1:2, :], [[4.0, 5.0, 6.0]]
        )
        assert f.variables["dummy1"].shape == (3, 3)
        assert f.variables["dummy1"]._h5ds.shape == (3, 3)
        np.testing.assert_allclose(f.variables["dummy2"][:], res2)
        np.testing.assert_allclose(f.variables["dummy2"][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables["dummy2"][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables["dummy2"].shape == (3, 3)
        assert f.variables["dummy2"]._h5ds.shape == (2, 3)
        np.testing.assert_allclose(f.variables["dummy3"][:], res3)
        np.testing.assert_allclose(f.variables["dummy3"][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables["dummy3"][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables["dummy3"].shape == (3, 3)
        assert f.variables["dummy3"]._h5ds.shape == (2, 3)
        np.testing.assert_allclose(f.variables["dummy4"][:], res4)
        assert f.variables["dummy4"].shape == (3, 3)
        assert f.variables["dummy4"]._h5ds.shape == (0, 3)

    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        f = ds["test"]
        np.testing.assert_allclose(f.variables["dummy1"][:], res1)
        np.testing.assert_allclose(f.variables["dummy1"][:, :], res1)
        np.testing.assert_allclose(f.variables["dummy1"][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables["dummy1"][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables["dummy1"].shape == (3, 3)
        assert f.variables["dummy1"]._h5ds.shape == (3, 3)
        np.testing.assert_allclose(f.variables["dummy2"][:], res2)
        np.testing.assert_allclose(f.variables["dummy2"][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables["dummy2"][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables["dummy2"].shape == (3, 3)
        assert f.variables["dummy2"]._h5ds.shape == (2, 3)
        np.testing.assert_allclose(f.variables["dummy3"][:], res3)
        np.testing.assert_allclose(f.variables["dummy3"][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables["dummy3"][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables["dummy3"].shape == (3, 3)
        assert f.variables["dummy3"]._h5ds.shape == (2, 3)
        np.testing.assert_allclose(f.variables["dummy4"][:], res4)
        assert f.variables["dummy4"].shape == (3, 3)
        assert f.variables["dummy4"]._h5ds.shape == (0, 3)


# https://github.com/h5netcdf/h5netcdf/issues/136
@pytest.mark.skip(reason="h5py bug with track_order prevents editing with netCDF4")
def test_creation_with_h5netcdf_edit_with_netcdf4(tmp_local_netcdf):
    # In version 0.12.0, the wrong file creation attributes were used
    # making netcdf4 unable to open files created by h5netcdf
    # https://github.com/h5netcdf/h5netcdf/issues/128
    with h5netcdf.File(tmp_local_netcdf, "w") as the_file:
        the_file.dimensions = {"x": 5}
        variable = the_file.create_variable("hello", ("x",), float)
        variable[...] = 5

    with netCDF4.Dataset(tmp_local_netcdf, mode="a") as the_file:
        variable = the_file["hello"]
        np.testing.assert_array_equal(variable[...].data, 5)
        # Edit an existing variable
        variable[:3] = 2

        # Create a new variable
        variable = the_file.createVariable("goodbye", float, ("x",))
        variable[...] = 10

    with h5netcdf.File(tmp_local_netcdf, "a") as the_file:
        # Ensure edited variable is consistent with the expected data
        variable = the_file["hello"]
        np.testing.assert_array_equal(variable[...].data, [2, 2, 2, 5, 5])

        # Ensure new variable is accessible
        variable = the_file["goodbye"]
        np.testing.assert_array_equal(variable[...].data, 10)


def test_track_order_specification(tmp_local_netcdf):
    # While netcdf4-c has historically only allowed track_order to be True
    # There doesn't seem to be a good reason for this
    # https://github.com/Unidata/netcdf-c/issues/2054 historically, h5netcdf
    # has not specified this parameter (leaving it implicitely as False)
    # We want to make sure we allow both here
    with h5netcdf.File(tmp_local_netcdf, "w", track_order=False):
        pass

    with h5netcdf.File(tmp_local_netcdf, "w", track_order=True):
        pass


# This should always work with the default file opening settings
# https://github.com/h5netcdf/h5netcdf/issues/136#issuecomment-1017457067
def test_more_than_7_attr_creation(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, "w") as h5file:
        for i in range(100):
            h5file.attrs[f"key{i}"] = i
            h5file.attrs[f"key{i}"] = 0


# Add a test that is supposed to fail in relation to issue #136
# We choose to monitor when h5py will have fixed their issue in our test suite
# to enhance maintainability
# https://github.com/h5netcdf/h5netcdf/issues/136#issuecomment-1017457067
@pytest.mark.parametrize("track_order", [False, True])
def test_more_than_7_attr_creation_track_order(tmp_local_netcdf, track_order):
    h5py_version = version.parse(h5py.__version__)
    if track_order and h5py_version < version.parse("3.7.0"):
        expected_errors = pytest.raises(KeyError)
    else:
        # We don't expect any errors. This is effectively a void context manager
        expected_errors = memoryview(b"")

    with h5netcdf.File(tmp_local_netcdf, "w", track_order=track_order) as h5file:
        with expected_errors:
            for i in range(100):
                h5file.attrs[f"key{i}"] = i
                h5file.attrs[f"key{i}"] = 0


def test_group_names(tmp_local_netcdf):
    # https://github.com/h5netcdf/h5netcdf/issues/68
    with netCDF4.Dataset(tmp_local_netcdf, mode="w") as ds:
        for i in range(10):
            ds = ds.createGroup(f"group{i:02d}")

    with netCDF4.Dataset(tmp_local_netcdf, "r") as ds:
        assert ds.name == "/"
        name = ""
        for i in range(10):
            name = "/".join([name, f"group{i:02d}"])
            assert ds[name].name == name.split("/")[-1]

    with legacyapi.Dataset(tmp_local_netcdf, "r") as ds:
        assert ds.name == "/"
        name = ""
        for i in range(10):
            name = "/".join([name, f"group{i:02d}"])
            assert ds[name].name == name.split("/")[-1]

    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        assert ds.name == "/"
        name = ""
        for i in range(10):
            name = "/".join([name, f"group{i:02d}"])
            assert ds[name].name == name


def test_legacyapi_endianess(tmp_local_netcdf):
    # https://github.com/h5netcdf/h5netcdf/issues/15
    big = legacyapi._check_return_dtype_endianess("big")
    little = legacyapi._check_return_dtype_endianess("little")
    native = legacyapi._check_return_dtype_endianess("native")

    with legacyapi.Dataset(tmp_local_netcdf, "w") as ds:
        ds.createDimension("x", 4)
        # test creating variable using endian keyword argument
        v = ds.createVariable("big", int, ("x"), endian="big")
        v[...] = 65533
        v = ds.createVariable("little", int, ("x"), endian="little")
        v[...] = 65533
        v = ds.createVariable("native", int, ("x"), endian="native")
        v[...] = 65535

    with h5py.File(tmp_local_netcdf, "r") as ds:
        assert ds["big"].dtype.byteorder == big
        assert ds["little"].dtype.byteorder == little
        assert ds["native"].dtype.byteorder == native

    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        assert ds["big"].dtype.byteorder == big
        assert ds["little"].dtype.byteorder == little
        assert ds["native"].dtype.byteorder == native

    with legacyapi.Dataset(tmp_local_netcdf, "r") as ds:
        assert ds["big"].dtype.byteorder == big
        assert ds["little"].dtype.byteorder == little
        assert ds["native"].dtype.byteorder == native

    with netCDF4.Dataset(tmp_local_netcdf, "r") as ds:
        assert ds["big"].dtype.byteorder == big
        assert ds["little"].dtype.byteorder == little
        assert ds["native"].dtype.byteorder == native


def test_bool_slicing_length_one_dim(tmp_local_netcdf):
    # see https://github.com/h5netcdf/h5netcdf/issues/23
    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        ds.dimensions = {"x": 1, "y": 2}
        v = ds.create_variable("hello", ("x", "y"), "float")
        v[:] = np.ones((1, 2))

    bool_slice = np.array([1], dtype=bool)

    # works for legacy API
    with legacyapi.Dataset(tmp_local_netcdf, "a") as ds:
        data = ds["hello"][bool_slice, :]
        np.testing.assert_equal(data, np.ones((1, 2)))
        ds["hello"][bool_slice, :] = np.zeros((1, 2))
        data = ds["hello"][bool_slice, :]
        np.testing.assert_equal(data, np.zeros((1, 2)))

    # should raise for h5py >= 3.0.0 and h5py < 3.7.0
    # https://github.com/h5py/h5py/pull/2079
    # https://github.com/h5netcdf/h5netcdf/pull/125/
    with h5netcdf.File(tmp_local_netcdf, "r") as ds:
        h5py_version = version.parse(h5py.__version__)
        if version.parse("3.0.0") <= h5py_version < version.parse("3.7.0"):
            error = "Indexing arrays must have integer dtypes"
            with pytest.raises(TypeError) as e:
                ds["hello"][bool_slice, :]
            assert error == str(e.value)
        else:
            ds["hello"][bool_slice, :]


def test_fancy_indexing(tmp_local_netcdf):
    # regression test for https://github.com/pydata/xarray/issues/7154
    with h5netcdf.legacyapi.Dataset(tmp_local_netcdf, "w") as ds:
        ds.createDimension("x", None)
        ds.createDimension("y", None)
        ds.createVariable("hello", int, ("x", "y"), fill_value=0)
        ds["hello"][:5, :10] = np.arange(5 * 10, dtype="int").reshape((5, 10))
        ds.createVariable("hello2", int, ("x", "y"))
        ds["hello2"][:10, :20] = np.arange(10 * 20, dtype="int").reshape((10, 20))

    with legacyapi.Dataset(tmp_local_netcdf, "a") as ds:
        np.testing.assert_array_equal(ds["hello"][1, [7, 8, 9]], [17, 18, 19])
        np.testing.assert_array_equal(ds["hello"][1, [9, 10, 11]], [19, 0, 0])
        np.testing.assert_array_equal(ds["hello"][1, slice(9, 12)], [19, 0, 0])
        np.testing.assert_array_equal(ds["hello"][[2, 3, 4], 1], [21, 31, 41])
        np.testing.assert_array_equal(ds["hello"][[4, 5, 6], 1], [41, 0, 0])
        np.testing.assert_array_equal(ds["hello"][slice(4, 7), 1], [41, 0, 0])


def test_h5py_chunking(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        ds.dimensions = {"x": 10, "y": 10, "z": 10, "t": None}

        v = ds.create_variable(
            "hello", ("x", "y", "z", "t"), "float", chunking_heuristic="h5py"
        )
        chunks_h5py = v.chunks

        ds.resize_dimension("t", 4)
        v = ds.create_variable(
            "hello3", ("x", "y", "z", "t"), "float", chunking_heuristic="h5py"
        )
        chunks_resized = v.chunks

    # cases above should be equivalent to a fixed dimension with appropriate size
    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        ds.dimensions = {"x": 10, "y": 10, "z": 10, "t": 1024}

        v = ds.create_variable(
            "hello",
            ("x", "y", "z", "t"),
            "float",
            chunks=True,
            chunking_heuristic="h5py",
        )
        chunks_true = v.chunks

    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        ds.dimensions = {"x": 10, "y": 10, "z": 10, "t": 4}

        v = ds.create_variable(
            "hello",
            ("x", "y", "z", "t"),
            "float",
            chunks=True,
            chunking_heuristic="h5py",
        )
        chunks_true_resized = v.chunks

    assert chunks_h5py == chunks_true
    assert chunks_resized == chunks_true_resized


def test_h5netcdf_chunking(tmp_local_netcdf):
    # produces much smaller chunks for unsized dimensions
    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        ds.dimensions = {"x": 10, "y": 10, "z": 10, "t": None}
        v = ds.create_variable(
            "hello", ("x", "y", "z", "t"), "float", chunking_heuristic="h5netcdf"
        )
        chunks_h5netcdf = v.chunks

    assert chunks_h5netcdf == (10, 10, 10, 1)

    # should produce chunks > 1 for small fixed dims
    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        ds.dimensions = {"x": 10, "t": None}
        v = ds.create_variable(
            "hello", ("x", "t"), "float", chunking_heuristic="h5netcdf"
        )
        chunks_h5netcdf = v.chunks

    assert chunks_h5netcdf == (10, 128)

    # resized unlimited dimensions should be treated like fixed dims
    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        ds.dimensions = {"x": 10, "y": 10, "z": 10, "t": None}
        ds.resize_dimension("t", 10)
        v = ds.create_variable(
            "hello", ("x", "y", "z", "t"), "float", chunking_heuristic="h5netcdf"
        )
        chunks_h5netcdf = v.chunks

    assert chunks_h5netcdf == (5, 5, 5, 10)


def test_create_invalid_netcdf_catch_error(tmp_local_netcdf):
    # see https://github.com/h5netcdf/h5netcdf/issues/138
    with h5netcdf.File(tmp_local_netcdf, "w") as f:
        try:
            f.create_variable("test", ("x", "y"), data=np.ones((10, 10), dtype="bool"))
        except CompatibilityError:
            pass
        assert repr(f.dimensions) == "<h5netcdf.Dimensions: >"


def test_dimensions_in_parent_groups(tmpdir):
    with netCDF4.Dataset(tmpdir.join("test_netcdf.nc"), mode="w") as ds:
        ds0 = ds
        for i in range(10):
            ds = ds.createGroup(f"group{i:02d}")
        ds0.createDimension("x", 10)
        ds0.createDimension("y", 20)
        ds0["group00"].createVariable("test", float, ("x", "y"))
        var = ds0["group00"].createVariable("x", float, ("x", "y"))
        var[:] = np.ones((10, 20))

    with legacyapi.Dataset(tmpdir.join("test_legacy.nc"), mode="w") as ds:
        ds0 = ds
        for i in range(10):
            ds = ds.createGroup(f"group{i:02d}")
        ds0.createDimension("x", 10)
        ds0.createDimension("y", 20)
        ds0["group00"].createVariable("test", float, ("x", "y"))
        var = ds0["group00"].createVariable("x", float, ("x", "y"))
        var[:] = np.ones((10, 20))

    with h5netcdf.File(tmpdir.join("test_netcdf.nc"), mode="r") as ds0:
        with h5netcdf.File(tmpdir.join("test_legacy.nc"), mode="r") as ds1:
            assert repr(ds0.dimensions["x"]) == repr(ds1.dimensions["x"])
            assert repr(ds0.dimensions["y"]) == repr(ds1.dimensions["y"])
            assert repr(ds0["group00"]) == repr(ds1["group00"])
            assert repr(ds0["group00"]["test"]) == repr(ds1["group00"]["test"])
            assert repr(ds0["group00"]["x"]) == repr(ds1["group00"]["x"])


def test_array_attributes(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        dt = h5py.string_dtype("utf-8")
        unicode = "unicodé"
        ds.attrs["unicode"] = unicode
        ds.attrs["unicode_0dim"] = np.array(unicode, dtype=dt)
        ds.attrs["unicode_1dim"] = np.array([unicode], dtype=dt)
        ds.attrs["unicode_arrary"] = np.array([unicode, "foobár"], dtype=dt)
        ds.attrs["unicode_list"] = [unicode]

        dt = h5py.string_dtype("ascii")
        # if dtype is ascii it's irrelevant if the data is provided as bytes or string
        ascii = "ascii"
        ds.attrs["ascii"] = ascii
        ds.attrs["ascii_0dim"] = np.array(ascii, dtype=dt)
        ds.attrs["ascii_1dim"] = np.array([ascii], dtype=dt)
        ds.attrs["ascii_array"] = np.array([ascii, "foobar"], dtype=dt)
        ds.attrs["ascii_list"] = [ascii]

        ascii = b"ascii"
        ds.attrs["bytes"] = ascii
        ds.attrs["bytes_0dim"] = np.array(ascii, dtype=dt)
        ds.attrs["bytes_1dim"] = np.array([ascii], dtype=dt)
        ds.attrs["bytes_array"] = np.array([ascii, b"foobar"], dtype=dt)
        ds.attrs["bytes_list"] = [ascii]

        dt = h5py.string_dtype("utf-8", 10)
        # unicode needs to be encoded properly for fixed size string type
        ds.attrs["unicode_fixed"] = np.array(unicode.encode("utf-8"), dtype=dt)
        ds.attrs["unicode_fixed_0dim"] = np.array(unicode.encode("utf-8"), dtype=dt)
        ds.attrs["unicode_fixed_1dim"] = np.array([unicode.encode("utf-8")], dtype=dt)
        ds.attrs["unicode_fixed_arrary"] = np.array(
            [unicode.encode("utf-8"), "foobár".encode("utf-8")], dtype=dt
        )

        dt = h5py.string_dtype("ascii", 10)
        ascii = "ascii"
        ds.attrs["ascii_fixed"] = np.array(ascii, dtype=dt)
        ds.attrs["ascii_fixed_0dim"] = np.array(ascii, dtype=dt)
        ds.attrs["ascii_fixed_1dim"] = np.array([ascii], dtype=dt)
        ds.attrs["ascii_fixed_array"] = np.array([ascii, "foobar"], dtype=dt)

        ascii = b"ascii"
        ds.attrs["bytes_fixed"] = np.array(ascii, dtype=dt)
        ds.attrs["bytes_fixed_0dim"] = np.array(ascii, dtype=dt)
        ds.attrs["bytes_fixed_1dim"] = np.array([ascii], dtype=dt)
        ds.attrs["bytes_fixed_array"] = np.array([ascii, b"foobar"], dtype=dt)

        ds.attrs["int"] = 1
        ds.attrs["intlist"] = [1]
        ds.attrs["int_array"] = np.arange(10)
        ds.attrs["empty_list"] = []
        ds.attrs["empty_array"] = np.array([])

    with h5netcdf.File(tmp_local_netcdf, mode="r") as ds:
        assert ds.attrs["unicode"] == unicode
        assert ds.attrs["unicode_0dim"] == unicode
        assert ds.attrs["unicode_1dim"] == unicode
        assert ds.attrs["unicode_arrary"] == [unicode, "foobár"]
        assert ds.attrs["unicode_list"] == unicode

        # bytes and strings are received as strings for h5py3
        if version.parse(h5py.__version__) >= version.parse("3.0.0"):
            ascii = "ascii"
            foobar = "foobar"
        # and bytes for h5py2
        else:
            ascii = b"ascii"
            foobar = b"foobar"
        assert ds.attrs["ascii"] == "ascii"
        assert ds.attrs["ascii_0dim"] == ascii
        assert ds.attrs["ascii_1dim"] == ascii
        assert ds.attrs["ascii_array"] == [ascii, foobar]
        # list is decoded for h5py2
        assert ds.attrs["ascii_list"] == "ascii"

        assert ds.attrs["bytes"] == ascii
        assert ds.attrs["bytes_0dim"] == ascii
        assert ds.attrs["bytes_1dim"] == ascii
        assert ds.attrs["bytes_array"] == [ascii, foobar]
        # list is decoded for h5py2
        assert ds.attrs["bytes_list"] == "ascii"

        assert ds.attrs["unicode_fixed"] == unicode
        assert ds.attrs["unicode_fixed_0dim"] == unicode
        assert ds.attrs["unicode_fixed_1dim"] == unicode
        assert ds.attrs["unicode_fixed_arrary"] == [unicode, "foobár"]

        ascii = "ascii"
        assert ds.attrs["ascii_fixed"] == ascii
        assert ds.attrs["ascii_fixed_0dim"] == ascii
        assert ds.attrs["ascii_fixed_1dim"] == ascii
        assert ds.attrs["ascii_fixed_array"] == [ascii, "foobar"]

        assert ds.attrs["bytes_fixed"] == ascii
        assert ds.attrs["bytes_fixed_0dim"] == ascii
        assert ds.attrs["bytes_fixed_1dim"] == ascii
        assert ds.attrs["bytes_fixed_array"] == [ascii, "foobar"]

        assert ds.attrs["int"] == 1
        assert ds.attrs["intlist"] == 1
        np.testing.assert_equal(ds.attrs["int_array"], np.arange(10))
        np.testing.assert_equal(ds.attrs["empty_list"], np.array([]))
        np.testing.assert_equal(ds.attrs["empty_array"], np.array([]))

    with legacyapi.Dataset(tmp_local_netcdf, mode="r") as ds:
        assert ds.unicode == unicode
        assert ds.unicode_0dim == unicode
        assert ds.unicode_1dim == unicode
        assert ds.unicode_arrary == [unicode, "foobár"]
        assert ds.unicode_list == unicode

        # bytes and strings are received as strings for h5py3
        if version.parse(h5py.__version__) >= version.parse("3.0.0"):
            ascii = "ascii"
            foobar = "foobar"
        # and bytes for h5py2
        else:
            ascii = b"ascii"
            foobar = b"foobar"
        assert ds.ascii == "ascii"
        assert ds.ascii_0dim == ascii
        assert ds.ascii_1dim == ascii
        assert ds.ascii_array == [ascii, foobar]
        # list is decoded for h5py2
        assert ds.ascii_list == "ascii"

        assert ds.bytes == ascii
        assert ds.bytes_0dim == ascii
        assert ds.bytes_1dim == ascii
        assert ds.bytes_array == [ascii, foobar]
        # list is decoded for h5py2
        assert ds.bytes_list == "ascii"

        assert ds.unicode_fixed == unicode
        assert ds.unicode_fixed_0dim == unicode
        assert ds.unicode_fixed_1dim == unicode
        assert ds.unicode_fixed_arrary == [unicode, "foobár"]

        ascii = "ascii"
        assert ds.ascii_fixed == ascii
        assert ds.ascii_fixed_0dim == ascii
        assert ds.ascii_fixed_1dim == ascii
        assert ds.ascii_fixed_array == [ascii, "foobar"]

        assert ds.bytes_fixed == ascii
        assert ds.bytes_fixed_0dim == ascii
        assert ds.bytes_fixed_1dim == ascii
        assert ds.bytes_fixed_array == [ascii, "foobar"]

        assert ds.int == 1
        assert ds.intlist == 1
        np.testing.assert_equal(ds.int_array, np.arange(10))
        np.testing.assert_equal(ds.attrs["empty_list"], np.array([]))
        np.testing.assert_equal(ds.attrs["empty_array"], np.array([]))

    with netCDF4.Dataset(tmp_local_netcdf, mode="r") as ds:
        assert ds.unicode == unicode
        assert ds.unicode_0dim == unicode
        assert ds.unicode_1dim == unicode
        assert ds.unicode_arrary == [unicode, "foobár"]
        assert ds.unicode_list == unicode

        ascii = "ascii"
        assert ds.ascii == ascii
        assert ds.ascii_0dim == ascii
        assert ds.ascii_1dim == ascii
        assert ds.ascii_array == [ascii, "foobar"]
        assert ds.ascii_list == ascii

        assert ds.bytes == ascii
        assert ds.bytes_0dim == ascii
        assert ds.bytes_1dim == ascii
        assert ds.bytes_array == [ascii, "foobar"]
        # writing/reading lists is broken with h5py2/netCDF4
        if version.parse(h5py.__version__) >= version.parse("3.0.0"):
            assert ds.bytes_list == ascii

        assert ds.unicode_fixed == unicode
        assert ds.unicode_fixed_0dim == unicode
        assert ds.unicode_fixed_1dim == unicode
        assert ds.unicode_fixed_arrary == [unicode, "foobár"]

        assert ds.ascii_fixed == ascii
        assert ds.ascii_fixed_0dim == ascii
        assert ds.ascii_fixed_1dim == ascii
        assert ds.ascii_fixed_array == [ascii, "foobar"]

        assert ds.bytes_fixed == ascii
        assert ds.bytes_fixed_0dim == ascii
        assert ds.bytes_fixed_1dim == ascii
        assert ds.bytes_fixed_array == [ascii, "foobar"]

        assert ds.int == 1
        assert ds.intlist == 1
        np.testing.assert_equal(ds.int_array, np.arange(10))
        np.testing.assert_equal(ds.empty_list, np.array([]))
        np.testing.assert_equal(ds.empty_array, np.array([]))


@pytest.mark.skipif(
    version.parse(h5py.__version__) < version.parse("3.7.0"),
    reason="does not work with h5py < 3.7.0",
)
def test_vlen_string_dataset_fillvalue(tmp_local_netcdf, decode_vlen_strings):
    # check _FillValue for VLEN string datasets
    # only works for h5py >= 3.7.0

    # first with new API
    with h5netcdf.File(tmp_local_netcdf, "w") as ds:
        ds.dimensions = {"string": 10}
        dt0 = h5py.string_dtype()
        fill_value0 = "bár"
        ds.create_variable("x0", ("string",), dtype=dt0, fillvalue=fill_value0)
        dt1 = h5py.string_dtype("ascii")
        fill_value1 = "bar"
        ds.create_variable("x1", ("string",), dtype=dt1, fillvalue=fill_value1)

    # check, if new API can read them
    with h5netcdf.File(tmp_local_netcdf, "r", **decode_vlen_strings) as ds:
        decode_vlen = decode_vlen_strings["decode_vlen_strings"]
        fvalue0 = fill_value0 if decode_vlen else fill_value0.encode("utf-8")
        fvalue1 = fill_value1 if decode_vlen else fill_value1.encode("utf-8")
        assert ds["x0"][0] == fvalue0
        assert ds["x0"].attrs["_FillValue"] == fill_value0
        assert ds["x1"][0] == fvalue1
        assert ds["x1"].attrs["_FillValue"] == fill_value1

    # check if legacyapi can read them
    with legacyapi.Dataset(tmp_local_netcdf, "r") as ds:
        assert ds["x0"][0] == fill_value0
        assert ds["x0"]._FillValue == fill_value0
        assert ds["x1"][0] == fill_value1
        assert ds["x1"]._FillValue == fill_value1

    # check if netCDF4-python can read them
    with netCDF4.Dataset(tmp_local_netcdf, "r") as ds:
        assert ds["x0"][0] == fill_value0
        assert ds["x0"]._FillValue == fill_value0
        assert ds["x1"][0] == fill_value1
        assert ds["x1"]._FillValue == fill_value1

    # second with legacyapi
    with legacyapi.Dataset(tmp_local_netcdf, "w") as ds:
        ds.createDimension("string", 10)
        fill_value0 = "bár"
        ds.createVariable("x0", str, ("string",), fill_value=fill_value0)
        fill_value1 = "bar"
        ds.createVariable("x1", str, ("string",), fill_value=fill_value1)

    # check if new API can read them
    with h5netcdf.File(tmp_local_netcdf, "r", **decode_vlen_strings) as ds:
        decode_vlen = decode_vlen_strings["decode_vlen_strings"]
        fvalue0 = fill_value0 if decode_vlen else fill_value0.encode("utf-8")
        fvalue1 = fill_value1 if decode_vlen else fill_value1.encode("utf-8")
        assert ds["x0"][0] == fvalue0
        assert ds["x0"].attrs["_FillValue"] == fill_value0
        assert ds["x1"][0] == fvalue1
        assert ds["x1"].attrs["_FillValue"] == fill_value1

    # check if legacyapi can read them
    with legacyapi.Dataset(tmp_local_netcdf, "r") as ds:
        assert ds["x0"][0] == fill_value0
        assert ds["x0"]._FillValue == fill_value0
        assert ds["x1"][0] == fill_value1
        assert ds["x1"]._FillValue == fill_value1

    # check if netCDF4-python can read them
    with netCDF4.Dataset(tmp_local_netcdf, "r") as ds:
        assert ds["x0"][0] == fill_value0
        assert ds["x0"]._FillValue == fill_value0
        assert ds["x1"][0] == fill_value1
        assert ds["x1"]._FillValue == fill_value1
