import gc
import io
import random
import re
import string
import tempfile
from distutils.version import LooseVersion
from os import environ as env

import h5py
import netCDF4
import numpy as np
import pytest
from pytest import raises

import h5netcdf
from h5netcdf import legacyapi
from h5netcdf.core import NOT_A_VARIABLE

try:
    import h5pyd

    without_h5pyd = False
except ImportError:
    without_h5pyd = True


remote_h5 = ("http:", "hdf5:")


@pytest.fixture()
def restapi(pytestconfig):
    return pytestconfig.getoption("restapi")


@pytest.fixture
def tmp_local_netcdf(tmpdir):
    return str(tmpdir.join("testfile.nc"))


@pytest.fixture(params=["testfile.nc", "hdf5://testfile"])
def tmp_local_or_remote_netcdf(request, tmpdir, restapi):
    if request.param.startswith(remote_h5):
        if not restapi:
            pytest.skip("Do not test with HDF5 REST API")
        elif without_h5pyd:
            pytest.skip("h5pyd package not available")
        if any([env.get(v) is None for v in ("HS_USERNAME", "HS_PASSWORD")]):
            pytest.skip("HSDS username and/or password missing")
        rnd = "".join(random.choice(string.ascii_uppercase) for _ in range(5))
        return (
            env["HS_ENDPOINT"]
            + env["H5PYD_TEST_FOLDER"]
            + "/"
            + "testfile"
            + rnd
            + ".nc"
        )
    else:
        return str(tmpdir.join(request.param))


@pytest.fixture(params=[True, False])
def decode_vlen_strings(request):
    if h5py.__version__ >= LooseVersion("3.0.0"):
        return dict(decode_vlen_strings=request.param)
    else:
        return {}


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
    ds.dimensions = {"x": 4, "y": 5, "z": 6, "empty": 0}

    v = ds.create_variable(
        "foo", ("x", "y"), float, chunks=(4, 5), compression="gzip", shuffle=True
    )
    v[...] = 1
    v.attrs["units"] = "meters"

    v = ds.create_variable("y", ("y",), int, fillvalue=-1)
    v[:4] = np.arange(4)

    v = ds.create_variable("z", ("z", "string3"), data=_char_array, fillvalue=b"X")

    v = ds.create_variable("scalar", data=np.float32(2.0))

    v = ds.create_variable("intscalar", data=np.int64(2))

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
        ["x", "y", "z", "empty", "string3", "mismatched_dim"]
    )
    assert set(ds.variables) == set(
        ["foo", "y", "z", "intscalar", "scalar", "var_len_str", "mismatched_dim"]
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
    assert v.filters() == {
        "complevel": 4,
        "fletcher32": False,
        "shuffle": True,
        "zlib": True,
    }

    v = ds.variables["y"]
    assert array_equal(v, np.r_[np.arange(4), [-1]])
    assert v.dtype == int
    assert v.dimensions == ("y",)
    assert v.ndim == 1
    assert v.ncattrs() == ["_FillValue"]
    assert v.getncattr("_FillValue") == -1
    assert v.chunking() == "contiguous"
    assert v.filters() == {
        "complevel": 0,
        "fletcher32": False,
        "shuffle": False,
        "zlib": False,
    }
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
        ["x", "y", "z", "empty", "string3", "mismatched_dim"]
    )
    assert set(ds.variables) == set(
        ["foo", "y", "z", "intscalar", "scalar", "var_len_str", "mismatched_dim"]
    )
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
    assert ds["/subgroup"].dimensions["y"] == 10

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
    if h5py.__version__ < LooseVersion("2.9.0"):
        pytest.skip("h5py > 2.9.0 required to test file-like objects")
    fileobj = tempfile.TemporaryFile()
    write_h5netcdf(fileobj)
    read_h5netcdf(fileobj, h5netcdf, decode_vlen_strings)
    fileobj = io.BytesIO()
    write_h5netcdf(fileobj)
    read_h5netcdf(fileobj, h5netcdf, decode_vlen_strings)


def test_repr(tmp_local_or_remote_netcdf):
    write_h5netcdf(tmp_local_or_remote_netcdf)
    f = h5netcdf.File(tmp_local_or_remote_netcdf, "r")
    assert "h5netcdf.File" in repr(f)
    assert "subgroup" in repr(f)
    assert "foo" in repr(f)
    assert "other_attr" in repr(f)

    assert "h5netcdf.attrs.Attributes" in repr(f.attrs)
    assert "global" in repr(f.attrs)

    d = f.dimensions
    assert "h5netcdf.Dimensions" in repr(d)
    assert "x=4" in repr(d)

    g = f["subgroup"]
    assert "h5netcdf.Group" in repr(g)
    assert "subvar" in repr(g)

    v = f["foo"]
    assert "h5netcdf.Variable" in repr(v)
    assert "float" in repr(v)
    assert "units" in repr(v)

    f.dimensions["temp"] = None
    assert "temp: Unlimited (current: 0)" in repr(f)
    f.resize_dimension("temp", 5)
    assert "temp: Unlimited (current: 5)" in repr(f)

    f.close()

    assert "Closed" in repr(f)
    assert "Closed" in repr(d)
    assert "Closed" in repr(g)
    assert "Closed" in repr(v)


def test_attrs_api(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf) as ds:
        ds.attrs["conventions"] = "CF"
        ds.attrs["empty_string"] = h5py.Empty(dtype=np.dtype("|S1"))
        ds.dimensions["x"] = 1
        v = ds.create_variable("x", ("x",), "i4")
        v.attrs.update({"units": "meters", "foo": "bar"})
    assert ds._closed
    with h5netcdf.File(tmp_local_or_remote_netcdf) as ds:
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
        if h5py.__version__ < LooseVersion("2.10.0"):
            f["foo"].dims.create_scale(f["x"])
            f["foo"].dims.create_scale(f["y"])
        else:
            f["x"].make_scale()
            f["y"].make_scale()
        f["foo"].dims[0].attach_scale(f["x"])
        f["foo"].dims[1].attach_scale(f["y"])
    with h5netcdf.File(tmp_local_or_remote_netcdf, "r") as ds:
        assert ds["foo"].dimensions == ("x", "y")
        assert ds.dimensions == {"x": 5, "y": 10}
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
    h5py.__version__ < LooseVersion("3.0.0"), reason="not needed with h5py < 3.0"
)
def test_decode_string_warning(tmp_local_or_remote_netcdf):
    write_h5netcdf(tmp_local_or_remote_netcdf)
    with pytest.warns(FutureWarning):
        with h5netcdf.File(tmp_local_or_remote_netcdf, "r") as ds:
            assert ds.name == "/"


@pytest.mark.skipif(
    h5py.__version__ < LooseVersion("3.0.0"), reason="not needed with h5py < 3.0"
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
        i = len(grps) - 1
        for grp in grps[::-1]:
            var = dsr[grp].variables
            check_invalid_netcdf4(var, i)
            i -= 1

    with h5netcdf.File(tmp_local_or_remote_netcdf, "r", phony_dims="access") as dsr:
        for i, grp in enumerate(grps[::-1]):
            print(dsr[grp])
            var = dsr[grp].variables
            check_invalid_netcdf4(var, i)

    with netCDF4.Dataset(tmp_local_or_remote_netcdf, mode="r") as dsr:
        for i, grp in enumerate(grps):
            print(dsr[grp])
            var = dsr[grp].variables
            check_invalid_netcdf4(var, i)

    with h5netcdf.File(tmp_local_or_remote_netcdf, "r") as ds:
        with raises(ValueError):
            ds["bar"].variables["foo1"].dimensions

    with raises(ValueError):
        with h5netcdf.File(tmp_local_or_remote_netcdf, "r", phony_dims="srt") as ds:
            pass


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
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, "w") as f:
        var, var2 = create_invalid_netcdf_data()
        for k, v in var.items():
            f.create_dataset(k, data=v)
        for k, v in var2.items():
            f.create_dataset(k, data=np.arange(v))

        if h5py.__version__ < LooseVersion("2.10.0"):
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

        if h5py.__version__ < LooseVersion("2.10.0"):
            f["foo1"].dims.create_scale(f["x"])
            f["foo1"].dims.create_scale(f["y"])
            f["foo1"].dims.create_scale(f["z"])
        else:
            f["x"].make_scale()
            f["y"].make_scale()
            f["z"].make_scale()
        f["foo1"].dims[0].attach_scale(f["x"])

    with raises(ValueError):
        with h5netcdf.File(tmp_local_or_remote_netcdf, "r", phony_dims="sort") as ds:
            assert ds


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

    # adding new dimensions after reopening for append will continue with last dimid + 1
    with h5netcdf.File(tmp_local_netcdf, "a") as f:
        f.dimensions["x1"] = 1
        g = f["foo"]
        g.dimensions["x1"] = 2
        g.dimensions["y1"] = 3

    with h5py.File(tmp_local_netcdf, "r") as f:
        # all dimension IDs should be present exactly once
        dim_ids = {
            f[name].attrs["_Netcdf4Dimid"]
            for name in ["x", "foo/x", "foo/y", "x1", "foo/x1", "foo/y1"]
        }
        assert dim_ids == {0, 1, 2, 3, 4, 5}


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
    with h5netcdf.File(tmp_local_or_remote_netcdf):
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

    if h5 is not h5py:
        pytest.xfail("https://github.com/h5netcdf/h5netcdf/issues/48")

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


def test_invalid_netcdf_warns(tmp_local_or_remote_netcdf):
    if tmp_local_or_remote_netcdf.startswith(remote_h5):
        pytest.skip("h5pyd does not support NumPy complex dtype yet")
    with h5netcdf.File(tmp_local_or_remote_netcdf, "w") as f:
        # valid
        with pytest.warns(None) as record:
            f.create_variable(
                "lzf_compressed", data=[1], dimensions=("x"), compression="lzf"
            )
        assert not record.list
        # invalid
        with pytest.warns(FutureWarning):
            f.create_variable("complex", data=1j)
        with pytest.warns(FutureWarning):
            f.attrs["complex_attr"] = 1j
        with pytest.warns(FutureWarning):
            f.create_variable("scaleoffset", data=[1], dimensions=("x",), scaleoffset=0)
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, "r") as f:
        assert "_NCProperties" not in f.attrs


def test_invalid_netcdf_error(tmp_local_or_remote_netcdf):
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


def test_reopen_file_different_dimension_sizes(tmp_local_netcdf):
    # regression test for https://github.com/h5netcdf/h5netcdf/issues/55
    with h5netcdf.File(tmp_local_netcdf, mode="w") as f:
        f.create_variable("/one/foo", data=[1], dimensions=("x",))
    with h5netcdf.File(tmp_local_netcdf, mode="a") as f:
        f.create_variable("/two/foo", data=[1, 2], dimensions=("x",))
    with netCDF4.Dataset(tmp_local_netcdf, mode="r") as f:
        assert f.groups["one"].variables["foo"][...].shape == (1,)


def test_invalid_then_valid_no_ncproperties(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf, "w", invalid_netcdf=True):
        pass
    with h5netcdf.File(tmp_local_or_remote_netcdf, "r"):
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
    with h5netcdf.File(tmp_local_or_remote_netcdf) as f:
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

        # Resize data.
        assert f.variables["dummy"].shape == (0, 2)
        f.resize_dimension("x", 3)
        # This will also force a resize of the existing variables and it will
        # be padded with zeros..
        np.testing.assert_allclose(f.variables["dummy"], np.zeros((3, 2)))

        # Creating another variable with no data will now also take the shape
        # of the current dimensions.
        f.create_variable("dummy3", dimensions=("x", "y"), dtype=np.int64)
        assert f.variables["dummy3"].shape == (3, 2)
        assert f.variables["dummy3"]._h5ds.maxshape == (None, 2)

    # Close and read again to also test correct parsing of unlimited
    # dimensions.
    with h5netcdf.File(tmp_local_or_remote_netcdf) as f:
        assert f.dimensions["x"] is None
        assert f._h5file["x"].maxshape == (None,)
        assert f._h5file["x"].shape == (3,)

        assert f.dimensions["y"] == 2
        assert f._h5file["y"].maxshape == (2,)
        assert f._h5file["y"].shape == (2,)


def test_writing_to_an_unlimited_dimension(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf) as f:
        # Two dimensions, only one is unlimited.
        f.dimensions["x"] = None
        f.dimensions["y"] = 3

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
        g = f.create_group("test")
        g.create_variable("dummy4", dimensions=("y", "x", "x"), dtype=np.int64)
        g.create_variable("dummy5", dimensions=("y", "y"), dtype=np.int64)

        assert f.variables["dummy1"].shape == (0, 3)
        assert f.variables["dummy2"].shape == (0, 3)
        assert f.variables["dummy3"].shape == (0, 3)
        assert g.variables["dummy4"].shape == (3, 0, 0)
        assert g.variables["dummy5"].shape == (3, 3)
        f.resize_dimension("x", 2)
        assert f.variables["dummy1"].shape == (2, 3)
        assert f.variables["dummy2"].shape == (2, 3)
        assert f.variables["dummy3"].shape == (2, 3)
        assert g.variables["dummy4"].shape == (3, 2, 2)
        assert g.variables["dummy5"].shape == (3, 3)

        f.variables["dummy2"][:] = [[1, 2, 3], [5, 6, 7]]
        np.testing.assert_allclose(f.variables["dummy2"], [[1, 2, 3], [5, 6, 7]])

        f.variables["dummy3"][...] = [[1, 2, 3], [5, 6, 7]]
        np.testing.assert_allclose(f.variables["dummy3"], [[1, 2, 3], [5, 6, 7]])


def test_writing_to_an_unlimited_dimension_2(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf) as f:
        # Two dimensions, only one is unlimited.
        f.dimensions["x"] = None
        f.dimensions["y"] = 3

        # Without data.
        f.create_variable("dummy1", dimensions=("x", "y"), dtype=np.int64)
        f.create_variable("dummy2", dimensions=("x", "y"), dtype=np.int64)
        f.create_variable("dummy3", dimensions=("x", "y"), dtype=np.int64)
        g = f.create_group("test")
        g.create_variable("dummy4", dimensions=("y", "x", "x"), dtype=np.int64)
        g.create_variable("dummy5", dimensions=("y", "y"), dtype=np.int64)

        assert f.variables["dummy1"].shape == (0, 3)
        assert f.variables["dummy2"].shape == (0, 3)
        assert f.variables["dummy3"].shape == (0, 3)
        assert g.variables["dummy4"].shape == (3, 0, 0)
        assert g.variables["dummy5"].shape == (3, 3)

        # variables and their unlimited dimensions are resized on the fly
        f.variables["dummy2"][:] = [[1, 2, 3], [5, 6, 7]]
        np.testing.assert_allclose(f.variables["dummy2"], [[1, 2, 3], [5, 6, 7]])

        f.variables["dummy3"][...] = [[1, 2, 3], [5, 6, 7]]
        np.testing.assert_allclose(f.variables["dummy3"], [[1, 2, 3], [5, 6, 7]])

        # other variables are not affected
        assert f.variables["dummy1"].shape == (0, 3)
        assert f.variables["dummy2"].shape == (2, 3)
        assert f.variables["dummy3"].shape == (2, 3)
        assert g.variables["dummy4"].shape == (3, 0, 0)
        assert g.variables["dummy5"].shape == (3, 3)


def test_resize_dimensions(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf) as f:
        f.dimensions["x"] = None
        f.dimensions["y"] = 2

        # Creating a variable without data will initialize an array with zero
        # length.
        f.create_variable("dummy", dimensions=("x", "y"), dtype=np.int64)
        assert f.variables["dummy"].shape == (0, 2)
        assert f.variables["dummy"]._h5ds.maxshape == (None, 2)

        # Resize dimension but no variables
        assert f.variables["dummy"].shape == (0, 2)
        f.resize_dimension("x", 3, resize_vars=False)

        # This will only resize the dimension, but variables keep untouched.
        assert f.dimensions["x"] == None
        assert f._current_dim_sizes["x"] == 3
        assert f.variables["dummy"].shape == (0, 2)
        assert f.variables["dummy"]._h5ds.maxshape == (None, 2)

        # Creating another variable with no data will now take the shape
        # of the current dimensions.
        f.create_variable("dummy3", dimensions=("x", "y"), dtype=np.int64)
        assert f.variables["dummy3"].shape == (3, 2)
        assert f.variables["dummy3"]._h5ds.maxshape == (None, 2)

        # writing to a variable with an unlimited dimension
        # will resize dimension and variable if necessary
        # but will not change any other variables
        f.variables["dummy"][3:5] = np.ones((2, 2))
        assert f._current_dim_sizes["x"] == 5
        assert f.variables["dummy"].shape == (5, 2)
        assert f.variables["dummy"]._h5ds.maxshape == (None, 2)
        assert f.variables["dummy3"].shape == (3, 2)
        assert f.variables["dummy3"]._h5ds.maxshape == (None, 2)


def test_c_api_can_read_unlimited_dimensions(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf) as f:
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

    with netCDF4.Dataset(tmp_local_netcdf) as f:
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

    with h5netcdf.File(tmp_local_netcdf) as f:
        assert f.dimensions["x"] is None
        assert f.dimensions["y"] == 3
        assert f.dimensions["z"] is None

        # This is parsed correctly due to h5netcdf's init trickery.
        assert f._current_dim_sizes["x"] == 2
        assert f._current_dim_sizes["y"] == 3
        assert f._current_dim_sizes["z"] == 0

        # But the actual data-set and arrays are not correct.
        assert f["dummy1"].shape == (2, 3)
        # XXX: This array has some data with dimension x - netcdf does not
        # appear to keep dimensions consistent.
        assert f["dummy2"].shape == (3, 0, 0)
        f.groups["test"]["dummy3"].shape == (3, 3)
        f.groups["test"]["dummy4"].shape == (0, 0)


def test_reading_unused_unlimited_dimension(tmp_local_or_remote_netcdf):
    """Test reading a file with unused dimension of unlimited size"""
    with h5netcdf.File(tmp_local_or_remote_netcdf, "w") as f:
        f.dimensions = {"x": None}
        f.resize_dimension("x", 5)
        assert f.dimensions == {"x": None}

    f = h5netcdf.File(tmp_local_or_remote_netcdf, "r")


def test_reading_special_datatype_created_with_c_api(tmp_local_netcdf):
    """Test reading a file with unsupported Datatype"""
    with netCDF4.Dataset(tmp_local_netcdf, "w") as f:
        complex128 = np.dtype([("real", np.float64), ("imag", np.float64)])
        f.createCompoundType(complex128, "complex128")
    with h5netcdf.File(tmp_local_netcdf) as f:
        pass
