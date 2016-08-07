import netCDF4
import numpy as np
import sys

import h5netcdf
from h5netcdf import legacyapi
from h5netcdf.compat import PY2, unicode
import h5py
import pytest

from pytest import fixture, raises


@pytest.fixture
def tmp_netcdf(tmpdir):
    return str(tmpdir.join('testfile.nc'))


def string_to_char(arr):
    """Like nc4.stringtochar, but faster and more flexible.
    """
    # ensure the array is contiguous
    arr = np.array(arr, copy=False, order='C')
    kind = arr.dtype.kind
    if kind not in ['U', 'S']:
        raise ValueError('argument must be a string')
    return arr.reshape(arr.shape + (1,)).view(kind + '1')


def array_equal(a, b):
    a, b = map(np.array, (a[...], b[...]))
    if a.shape != b.shape:
        return False
    try:
        return np.allclose(a, b)
    except TypeError:
        return (a == b).all()


_char_array = string_to_char(np.array(['a', 'b', 'c', 'foo', 'bar', 'baz'],
                                      dtype='S'))

_string_array = np.array([['foobar0', 'foobar1', 'foobar3'],
                          ['foofoofoo', 'foofoobar', 'foobarbar']])

def is_h5py_char_working(tmp_netcdf, name):
    # https://github.com/Unidata/netcdf-c/issues/298
    with h5py.File(tmp_netcdf, 'r') as ds:
        v = ds[name]
        try:
            assert array_equal(v, _char_array)
            return True
        except Exception as e:
            if e.args[0] == "Can't read data (No appropriate function for conversion path)":
                return False
            else:
                raise

def write_legacy_netcdf(tmp_netcdf, write_module):
    ds = write_module.Dataset(tmp_netcdf, 'w')
    ds.setncattr('global', 42)
    ds.other_attr = 'yes'
    ds.createDimension('x', 4)
    ds.createDimension('y', 5)
    ds.createDimension('z', 6)
    ds.createDimension('string3', 3)

    v = ds.createVariable('foo', float, ('x', 'y'), chunksizes=(4, 5),
                          zlib=True)
    v[...] = 1
    v.setncattr('units', 'meters')

    v = ds.createVariable('y', int, ('y',), fill_value=-1)
    v[:4] = np.arange(4)

    v = ds.createVariable('z', 'S1', ('z', 'string3'), fill_value=b'X')
    v[...] = _char_array

    v = ds.createVariable('scalar', np.float32, ())
    v[...] = 2.0

    # test creating a scalar with compression option (with should be ignored)
    v = ds.createVariable('intscalar', np.int64, (), zlib=6, fill_value=None)
    v[...] = 2

    with raises(TypeError):
        ds.createVariable('boolean', np.bool_, ('x'))

    g = ds.createGroup('subgroup')
    v = g.createVariable('subvar', np.int32, ('x',))
    v[...] = np.arange(4.0)

    g.createDimension('y', 10)
    g.createVariable('y_var', float, ('y',))

    ds.createDimension('mismatched_dim', 1)
    ds.createVariable('mismatched_dim', int, ())

    v = ds.createVariable('var_len_str', str, ('x'))
    v[0] = u'foo'

    ds.close()


def write_h5netcdf(tmp_netcdf):
    ds = h5netcdf.File(tmp_netcdf, 'w')
    ds.attrs['global'] = 42
    ds.attrs['other_attr'] = 'yes'
    ds.dimensions = {'x': 4, 'y': 5, 'z': 6}

    v = ds.create_variable('foo', ('x', 'y'), float, chunks=(4, 5),
                           compression='gzip', shuffle=True)
    v[...] = 1
    v.attrs['units'] = 'meters'

    v = ds.create_variable('y', ('y',), int, fillvalue=-1)
    v[:4] = np.arange(4)

    v = ds.create_variable('z', ('z', 'string3'), data=_char_array,
                           fillvalue=b'X')

    v = ds.create_variable('scalar', data=np.float32(2.0))

    v = ds.create_variable('intscalar', data=np.int64(2))

    with raises(TypeError):
        ds.create_variable('boolean', data=True)

    g = ds.create_group('subgroup')
    v = g.create_variable('subvar', ('x',), np.int32)
    v[...] = np.arange(4.0)
    with raises(AttributeError):
        v.attrs['_Netcdf4Dimid'] = -1

    g.dimensions['y'] = 10
    g.create_variable('y_var', ('y',), float)

    ds.dimensions['mismatched_dim'] = 1
    ds.create_variable('mismatched_dim', dtype=int)

    dt = h5py.special_dtype(vlen=unicode)
    v = ds.create_variable('var_len_str', ('x',), dtype=dt)
    v[0] = u'foo'

    ds.close()


def read_legacy_netcdf(tmp_netcdf, read_module, write_module):
    ds = read_module.Dataset(tmp_netcdf, 'r')
    # ignore _NCProperties for now: https://github.com/shoyer/h5netcdf/issues/18
    attr_names = [k for k in ds.ncattrs() if k != '_NCProperties']
    assert attr_names == ['global', 'other_attr']
    assert ds.getncattr('global') == 42
    if not PY2 and write_module is not netCDF4:
        # skip for now: https://github.com/Unidata/netcdf4-python/issues/388
        assert ds.other_attr == 'yes'
    assert set(ds.dimensions) == set(['x', 'y', 'z', 'string3', 'mismatched_dim'])
    assert set(ds.variables) == set(['foo', 'y', 'z', 'intscalar', 'scalar',
                                     'var_len_str', 'mismatched_dim'])
    assert set(ds.groups) == set(['subgroup'])
    assert ds.parent is None

    v = ds.variables['foo']
    assert array_equal(v, np.ones((4, 5)))
    assert v.dtype == float
    assert v.dimensions == ('x', 'y')
    assert v.ndim == 2
    assert v.ncattrs() == ['units']
    if not PY2 and write_module is not netCDF4:
        assert v.getncattr('units') == 'meters'
    assert tuple(v.chunking()) == (4, 5)
    assert v.filters() == {'complevel': 4, 'fletcher32': False,
                           'shuffle': True, 'zlib': True}

    v = ds.variables['y']
    assert array_equal(v, np.r_[np.arange(4), [-1]])
    assert v.dtype == int
    assert v.dimensions == ('y',)
    assert v.ndim == 1
    assert v.ncattrs() == ['_FillValue']
    assert v.getncattr('_FillValue') == -1
    assert v.chunking() == 'contiguous'
    assert v.filters() == {'complevel': 0, 'fletcher32': False,
                           'shuffle': False, 'zlib': False}
    ds.close()

    #Check the behavior if h5py. Cannot expect h5netcdf to overcome these errors:
    if is_h5py_char_working(tmp_netcdf, 'z'):
        ds = read_module.Dataset(tmp_netcdf, 'r')
        v = ds.variables['z']
        assert array_equal(v, _char_array)
        assert v.dtype == 'S1'
        assert v.ndim == 2
        assert v.dimensions == ('z', 'string3')
        assert v.ncattrs() == ['_FillValue']
        assert v.getncattr('_FillValue') == b'X'
    else:
        ds = read_module.Dataset(tmp_netcdf, 'r')

    v = ds.variables['scalar']
    assert array_equal(v, np.array(2.0))
    assert v.dtype == 'float32'
    assert v.ndim == 0
    assert v.dimensions == ()
    assert v.ncattrs() == []

    v = ds.variables['intscalar']
    assert array_equal(v, np.array(2))
    assert v.dtype == 'int64'
    assert v.ndim == 0
    assert v.dimensions == ()
    assert v.ncattrs() == []

    v = ds.variables['var_len_str']
    assert v.dtype == str
    assert v[0] == u'foo'

    v = ds.groups['subgroup'].variables['subvar']
    assert ds.groups['subgroup'].parent is ds
    assert array_equal(v, np.arange(4.0))
    assert v.dtype == 'int32'
    assert v.ndim == 1
    assert v.dimensions == ('x',)
    assert v.ncattrs() == []

    v = ds.groups['subgroup'].variables['y_var']
    assert v.shape == (10,)
    assert 'y' in ds.groups['subgroup'].dimensions

    ds.close()


def read_h5netcdf(tmp_netcdf, write_module):
    ds = h5netcdf.File(tmp_netcdf, 'r')
    assert ds.name == '/'
    # ignore _NCProperties for now: https://github.com/shoyer/h5netcdf/issues/18
    attr_names = [k for k in list(ds.attrs) if k != '_NCProperties']
    assert attr_names == ['global', 'other_attr']
    assert ds.attrs['global'] == 42
    if not PY2 and write_module is not netCDF4:
        # skip for now: https://github.com/Unidata/netcdf4-python/issues/388
        assert ds.attrs['other_attr'] == 'yes'
    assert set(ds.dimensions) == set(['x', 'y', 'z', 'string3', 'mismatched_dim'])
    assert set(ds.variables) == set(['foo', 'y', 'z', 'intscalar', 'scalar',
                                     'var_len_str', 'mismatched_dim'])
    assert set(ds.groups) == set(['subgroup'])
    assert ds.parent is None

    v = ds['foo']
    assert v.name == '/foo'
    assert array_equal(v, np.ones((4, 5)))
    assert v.dtype == float
    assert v.dimensions == ('x', 'y')
    assert v.ndim == 2
    assert list(v.attrs) == ['units']
    if not PY2 and write_module is not netCDF4:
        assert v.attrs['units'] == 'meters'
    assert v.chunks == (4, 5)
    assert v.compression == 'gzip'
    assert v.compression_opts == 4
    assert not v.fletcher32
    assert v.shuffle

    v = ds['y']
    assert array_equal(v, np.r_[np.arange(4), [-1]])
    assert v.dtype == int
    assert v.dimensions == ('y',)
    assert v.ndim == 1
    assert list(v.attrs) == ['_FillValue']
    assert v.attrs['_FillValue'] == -1
    assert v.chunks == None
    assert v.compression == None
    assert v.compression_opts == None
    assert not v.fletcher32
    assert not v.shuffle
    ds.close()

    if is_h5py_char_working(tmp_netcdf, 'z'):
        ds = h5netcdf.File(tmp_netcdf, 'r')
        v = ds['z']
        assert v.dtype == 'S1'
        assert v.ndim == 2
        assert v.dimensions == ('z', 'string3')
        assert list(v.attrs) == ['_FillValue']
        assert v.attrs['_FillValue'] == b'X'
    else:
        ds = h5netcdf.File(tmp_netcdf, 'r')

    v = ds['scalar']
    assert array_equal(v, np.array(2.0))
    assert v.dtype == 'float32'
    assert v.ndim == 0
    assert v.dimensions == ()
    assert list(v.attrs) == []

    v = ds.variables['intscalar']
    assert array_equal(v, np.array(2))
    assert v.dtype == 'int64'
    assert v.ndim == 0
    assert v.dimensions == ()
    assert list(v.attrs) == []

    v = ds['var_len_str']
    assert h5py.check_dtype(vlen=v.dtype) == unicode
    assert v[0] == u'foo'

    v = ds['/subgroup/subvar']
    assert v is ds['subgroup']['subvar']
    assert v is ds['subgroup/subvar']
    assert v is ds['subgroup']['/subgroup/subvar']
    assert v.name == '/subgroup/subvar'
    assert ds['subgroup'].name == '/subgroup'
    assert ds['subgroup'].parent is ds
    assert array_equal(v, np.arange(4.0))
    assert v.dtype == 'int32'
    assert v.ndim == 1
    assert v.dimensions == ('x',)
    assert list(v.attrs) == []

    assert ds['/subgroup/y_var'].shape == (10,)
    assert ds['/subgroup'].dimensions['y'] == 10

    ds.close()


def roundtrip_legacy_netcdf(tmp_netcdf, read_module, write_module):
    write_legacy_netcdf(tmp_netcdf, write_module)
    read_legacy_netcdf(tmp_netcdf, read_module, write_module)


def test_write_legacyapi_read_netCDF4(tmp_netcdf):
    roundtrip_legacy_netcdf(tmp_netcdf, netCDF4, legacyapi)


def test_roundtrip_h5netcdf_legacyapi(tmp_netcdf):
    roundtrip_legacy_netcdf(tmp_netcdf, legacyapi, legacyapi)


def test_write_netCDF4_read_legacyapi(tmp_netcdf):
    roundtrip_legacy_netcdf(tmp_netcdf, legacyapi, netCDF4)


def test_write_h5netcdf_read_legacyapi(tmp_netcdf):
    write_h5netcdf(tmp_netcdf)
    read_legacy_netcdf(tmp_netcdf, legacyapi, h5netcdf)


def test_write_h5netcdf_read_netCDF4(tmp_netcdf):
    write_h5netcdf(tmp_netcdf)
    read_legacy_netcdf(tmp_netcdf, netCDF4, h5netcdf)


def test_roundtrip_h5netcdf(tmp_netcdf):
    write_h5netcdf(tmp_netcdf)
    read_h5netcdf(tmp_netcdf, h5netcdf)


def test_write_netCDF4_read_h5netcdf(tmp_netcdf):
    write_legacy_netcdf(tmp_netcdf, netCDF4)
    read_h5netcdf(tmp_netcdf, netCDF4)


def test_write_legacyapi_read_h5netcdf(tmp_netcdf):
    write_legacy_netcdf(tmp_netcdf, legacyapi)
    read_h5netcdf(tmp_netcdf, legacyapi)


def test_repr(tmp_netcdf):
    write_h5netcdf(tmp_netcdf)
    f = h5netcdf.File(tmp_netcdf, 'r')
    assert 'h5netcdf.File' in repr(f)
    assert 'subgroup' in repr(f)
    assert 'foo' in repr(f)
    assert 'other_attr' in repr(f)

    assert 'h5netcdf.attrs.Attributes' in repr(f.attrs)
    assert 'global' in repr(f.attrs)

    d = f.dimensions
    assert 'h5netcdf.Dimensions' in repr(d)
    assert 'x=4' in repr(d)

    g = f['subgroup']
    assert 'h5netcdf.Group' in repr(g)
    assert 'subvar' in repr(g)

    v = f['foo']
    assert 'h5netcdf.Variable' in repr(v)
    assert 'float' in repr(v)
    assert 'units' in repr(v)
    f.close()

    assert 'Closed' in repr(f)
    assert 'Closed' in repr(d)
    assert 'Closed' in repr(g)
    assert 'Closed' in repr(v)


def test_attrs_api(tmp_netcdf):
    with h5netcdf.File(tmp_netcdf) as ds:
        ds.attrs['conventions'] = 'CF'
        ds.dimensions['x'] = 1
        v = ds.create_variable('x', ('x',), 'i4')
        v.attrs.update({'units': 'meters', 'foo': 'bar'})
    assert ds._closed
    with h5netcdf.File(tmp_netcdf) as ds:
        assert len(ds.attrs) == 1
        assert dict(ds.attrs) == {'conventions': 'CF'}
        assert list(ds.attrs) == ['conventions']
        assert dict(ds['x'].attrs) == {'units': 'meters', 'foo': 'bar'}
        assert len(ds['x'].attrs) == 2
        assert sorted(ds['x'].attrs) == ['foo', 'units']


def test_optional_netcdf4_attrs(tmp_netcdf):
    with h5py.File(tmp_netcdf) as f:
        foo_data = np.arange(50).reshape(5, 10)
        f.create_dataset('foo', data=foo_data)
        f.create_dataset('x', data=np.arange(5))
        f.create_dataset('y', data=np.arange(10))
        f['foo'].dims.create_scale(f['x'])
        f['foo'].dims.create_scale(f['y'])
        f['foo'].dims[0].attach_scale(f['x'])
        f['foo'].dims[1].attach_scale(f['y'])
    with h5netcdf.File(tmp_netcdf, 'r') as ds:
        assert ds['foo'].dimensions == ('x', 'y')
        assert ds.dimensions == {'x': 5, 'y': 10}
        assert array_equal(ds['foo'], foo_data)


def test_error_handling(tmp_netcdf):
    with h5netcdf.File(tmp_netcdf, 'w') as ds:
        with raises(NotImplementedError):
            ds.dimensions['x'] = None
        ds.dimensions['x'] = 1
        with raises(ValueError):
            ds.dimensions['x'] = 2
        with raises(ValueError):
            ds.dimensions = {'x': 2}
        with raises(ValueError):
            ds.dimensions = {'y': 3}
        ds.create_variable('x', ('x',), dtype=float)
        with raises(ValueError):
            ds.create_variable('x', ('x',), dtype=float)
        ds.create_group('subgroup')
        with raises(ValueError):
            ds.create_group('subgroup')


def test_invalid_netcdf4(tmp_netcdf):
    with h5py.File(tmp_netcdf) as f:
        f.create_dataset('foo', data=np.arange(5))
        # labeled dimensions but no dimension scales
        f['foo'].dims[0].label = 'x'
    with h5netcdf.File(tmp_netcdf, 'r') as ds:
        with raises(ValueError):
            ds.variables['foo'].dimensions


def test_hierarchical_access_auto_create(tmp_netcdf):
    ds = h5netcdf.File(tmp_netcdf, 'w')
    ds.create_variable('/foo/bar', data=1)
    g = ds.create_group('foo/baz')
    g.create_variable('/foo/hello', data=2)
    assert set(ds) == set(['foo'])
    assert set(ds['foo']) == set(['bar', 'baz', 'hello'])
    ds.close()

    ds = h5netcdf.File(tmp_netcdf, 'r')
    assert set(ds) == set(['foo'])
    assert set(ds['foo']) == set(['bar', 'baz', 'hello'])
    ds.close()

def test_reading_str_array_from_netCDF4(tmp_netcdf):
    #This tests reading string variables created by netCDF4
    with netCDF4.Dataset(tmp_netcdf, 'w') as ds:
        ds.createDimension('foo1', _string_array.shape[0])
        ds.createDimension('foo2', _string_array.shape[1])
        ds.createVariable('bar', str, ('foo1', 'foo2'))
        ds.variables['bar'][:] = _string_array

    ds = h5netcdf.File(tmp_netcdf, 'r')

    v = ds.variables['bar']
    assert array_equal(v, _string_array)
    ds.close()
