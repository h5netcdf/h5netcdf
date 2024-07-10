import tempfile
from test_h5netcdf import write_h5netcdf, array_equal, _char_array, _string_array, _vlen_string

import h5netcdf
import netCDF4
import numpy as np
import pyfive
import pytest
import re
import io

remote_h5 = ("http:", "hdf5:")


@pytest.fixture(params=[True, False])
def decode_vlen_strings(request):
    return dict(decode_vlen_strings=request.param)

def is_h5py_char_working(tmp_netcdf, name):
    # https://github.com/Unidata/netcdf-c/issues/298
    with pyfive.File(tmp_netcdf, "r") as ds:
        v = ds[name]
        try:
            assert array_equal(v, _char_array)
            return True
        except Exception as e:
            if re.match("^Can't read data", e.args[0]):
                return False
            else:
                raise


def read_h5netcdf_pyfive(tmp_netcdf, write_module, decode_vlen_strings):
    remote_file = isinstance(tmp_netcdf, str) and tmp_netcdf.startswith(remote_h5)
    ds = h5netcdf.File(tmp_netcdf, "r", backend='pyfive', **decode_vlen_strings)
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
            "enum_var",
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
    assert pyfive.check_dtype(vlen=v.dtype) is str
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

    enum_dict = dict(one=1, two=2, three=3, missing=255)
    enum_type = ds.enumtypes["enum_t"]
    assert enum_type.enum_dict == enum_dict
    v = ds.variables["enum_var"]
    assert array_equal(v, np.ma.masked_equal([1, 2, 3, 255], 255))

    ds.close()


def test_fileobj(decode_vlen_strings):
    fileobj = tempfile.TemporaryFile()
    write_h5netcdf(fileobj)
    read_h5netcdf_pyfive(fileobj, h5netcdf, decode_vlen_strings)
    fileobj = io.BytesIO()
    write_h5netcdf(fileobj)
    read_h5netcdf_pyfive(fileobj, h5netcdf, decode_vlen_strings)