import netCDF4
import numpy as np
import gc
import re

import h5netcdf
from h5netcdf import legacyapi
from h5netcdf.compat import PY2, unicode
import h5py
import pytest

from pytest import raises


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
            if re.match("^Can't read data", e.args[0]):
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
    ds.createDimension('empty', 0)
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

    with raises((h5netcdf.CompatibilityError, TypeError)):
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
    ds.dimensions = {'x': 4, 'y': 5, 'z': 6, 'empty': 0}

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

    with raises((h5netcdf.CompatibilityError, TypeError)):
        ds.create_variable('boolean', data=True)

    g = ds.create_group('subgroup')
    v = g.create_variable('subvar', ('x',), np.int32)
    v[...] = np.arange(4.0)
    with raises(AttributeError):
        v.attrs['_Netcdf4Dimid'] = -1

    g.dimensions['y'] = 10
    g.create_variable('y_var', ('y',), float)
    g.flush()

    ds.dimensions['mismatched_dim'] = 1
    ds.create_variable('mismatched_dim', dtype=int)
    ds.flush()

    dt = h5py.special_dtype(vlen=unicode)
    v = ds.create_variable('var_len_str', ('x',), dtype=dt)
    v[0] = u'foo'

    ds.close()


def read_legacy_netcdf(tmp_netcdf, read_module, write_module):
    ds = read_module.Dataset(tmp_netcdf, 'r')
    assert ds.ncattrs() == ['global', 'other_attr']
    assert ds.getncattr('global') == 42
    if not PY2 and write_module is not netCDF4:
        # skip for now: https://github.com/Unidata/netcdf4-python/issues/388
        assert ds.other_attr == 'yes'
    with pytest.raises(AttributeError):
        ds.does_not_exist
    assert set(ds.dimensions) == set(['x', 'y', 'z', 'empty', 'string3',
                                      'mismatched_dim'])
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
    assert list(ds.attrs) == ['global', 'other_attr']
    assert ds.attrs['global'] == 42
    if not PY2 and write_module is not netCDF4:
        # skip for now: https://github.com/Unidata/netcdf4-python/issues/388
        assert ds.attrs['other_attr'] == 'yes'
    assert set(ds.dimensions) == set(['x', 'y', 'z', 'empty', 'string3', 'mismatched_dim'])
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

    f.dimensions['temp'] = None
    assert 'temp: Unlimited (current: 0)' in repr(f)
    f.resize_dimension('temp', 5)
    assert 'temp: Unlimited (current: 5)' in repr(f)

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
    # This tests reading string variables created by netCDF4
    with netCDF4.Dataset(tmp_netcdf, 'w') as ds:
        ds.createDimension('foo1', _string_array.shape[0])
        ds.createDimension('foo2', _string_array.shape[1])
        ds.createVariable('bar', str, ('foo1', 'foo2'))
        ds.variables['bar'][:] = _string_array

    ds = h5netcdf.File(tmp_netcdf, 'r')

    v = ds.variables['bar']
    assert array_equal(v, _string_array)
    ds.close()

def test_nc_properties_new(tmp_netcdf):
    with h5netcdf.File(tmp_netcdf, 'w'):
        pass
    with h5py.File(tmp_netcdf, 'r') as f:
        assert 'h5netcdf' in f.attrs['_NCProperties']

def test_failed_read_open_and_clean_delete(tmpdir):
    # A file that does not exist but is opened for
    # reading should only raise an IOError and
    # no AttributeError at garbage collection.
    path = str(tmpdir.join('this_file_does_not_exist.nc'))
    try:
        with h5netcdf.File(path, 'r') as ds:
            pass
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
        except AttributeError as e:
            is_h5netcdf_File = False
        if is_h5netcdf_File:
            obj.close()

def test_invalid_netcdf_warns(tmp_netcdf):
    with h5netcdf.File(tmp_netcdf, 'w') as f:
        with pytest.warns(FutureWarning):
            f.create_variable('complex', data=1j)
        with pytest.warns(FutureWarning):
            f.attrs['complex_attr'] = 1j
        with pytest.warns(FutureWarning):
            f.create_variable('lzf_compressed', data=[1], dimensions=('x'),
                              compression='lzf')
        with pytest.warns(FutureWarning):
            f.create_variable('scaleoffset', data=[1], dimensions=('x',),
                              scaleoffset=0)
    with h5py.File(tmp_netcdf) as f:
        assert '_NCProperties' not in f.attrs

def test_invalid_netcdf_error(tmp_netcdf):
    with h5netcdf.File(tmp_netcdf, 'w', invalid_netcdf=False) as f:
        with pytest.raises(h5netcdf.CompatibilityError):
            f.create_variable('complex', data=1j)
        with pytest.raises(h5netcdf.CompatibilityError):
            f.attrs['complex_attr'] = 1j
        with pytest.raises(h5netcdf.CompatibilityError):
            f.create_variable('lzf_compressed', data=[1], dimensions=('x'),
                              compression='lzf')
        with pytest.raises(h5netcdf.CompatibilityError):
            f.create_variable('scaleoffset', data=[1], dimensions=('x',),
                              scaleoffset=0)

def test_invalid_netcdf_okay(tmp_netcdf):
    with h5netcdf.File(tmp_netcdf, 'w', invalid_netcdf=True) as f:
        f.create_variable('complex', data=1j)
        f.attrs['complex_attr'] = 1j
        f.create_variable('lzf_compressed', data=[1], dimensions=('x'),
                          compression='lzf')
        f.create_variable('scaleoffset', data=[1], dimensions=('x',),
                          scaleoffset=0)
    with h5netcdf.File(tmp_netcdf) as f:
        assert f['complex'][...] == 1j
        assert f.attrs['complex_attr'] == 1j
        np.testing.assert_equal(f['lzf_compressed'][:], [1])
        np.testing.assert_equal(f['scaleoffset'][:], [1])
    with h5py.File(tmp_netcdf) as f:
        assert '_NCProperties' not in f.attrs

def test_invalid_then_valid_no_ncproperties(tmp_netcdf):
    with h5netcdf.File(tmp_netcdf, invalid_netcdf=True):
        pass
    with h5netcdf.File(tmp_netcdf):
        pass
    with h5py.File(tmp_netcdf) as f:
        # still not a valid netcdf file
        assert '_NCProperties' not in f.attrs

def test_creating_and_resizing_unlimited_dimensions(tmp_netcdf):
    with h5netcdf.File(tmp_netcdf) as f:
        f.dimensions['x'] = None
        f.dimensions['y'] = 15
        f.dimensions['z'] = None
        f.resize_dimension('z', 20)

        with pytest.raises(ValueError) as e:
            f.resize_dimension('y', 20)
        assert e.value.args[0] == "Only unlimited dimensions can be resized."

    # Assert some behavior observed by using the C netCDF bindings.
    with h5py.File(tmp_netcdf) as f:
        assert f["x"].shape == (0,)
        assert f["x"].maxshape == (None,)
        assert f["y"].shape == (15,)
        assert f["y"].maxshape == (15,)
        assert f["z"].shape == (20,)
        assert f["z"].maxshape == (None,)

def test_creating_variables_with_unlimited_dimensions(tmp_netcdf):
    with h5netcdf.File(tmp_netcdf) as f:
        f.dimensions['x'] = None
        f.dimensions['y'] = 2

        # Creating a variable without data will initialize an array with zero
        # length.
        f.create_variable('dummy', dimensions=('x', 'y'), dtype=np.int64)
        assert f.variables["dummy"].shape == (0, 2)
        assert f.variables["dummy"]._h5ds.maxshape == (None, 2)

        # Trying to create a variable while the current size of the dimension
        # is still zero will fail.
        with pytest.raises(ValueError) as e:
            f.create_variable('dummy2', data=np.array([[1, 2], [3, 4]]),
                              dimensions=('x', 'y'))
        assert e.value.args[0] == "Shape tuple is incompatible with data"

        # Resize data.
        assert f.variables["dummy"].shape == (0, 2)
        f.resize_dimension('x', 3)
        # This will also force a resize of the existing variables and it will
        # be padded with zeros..
        np.testing.assert_allclose(f.variables["dummy"], np.zeros((3, 2)))

        # Creating another variable with no data will now also take the shape
        # of the current dimensions.
        f.create_variable('dummy3', dimensions=('x', 'y'), dtype=np.int64)
        assert f.variables["dummy3"].shape == (3, 2)
        assert f.variables["dummy3"]._h5ds.maxshape == (None, 2)

    # Close and read again to also test correct parsing of unlimited
    # dimensions.
    with h5netcdf.File(tmp_netcdf) as f:
        assert f.dimensions['x'] is None
        assert f._h5file['x'].maxshape == (None,)
        assert f._h5file['x'].shape == (3,)

        assert f.dimensions['y'] == 2
        assert f._h5file['y'].maxshape == (2,)
        assert f._h5file['y'].shape == (2,)

def test_writing_to_an_unlimited_dimension(tmp_netcdf):
    with h5netcdf.File(tmp_netcdf) as f:
        # Two dimensions, only one is unlimited.
        f.dimensions['x'] = None
        f.dimensions['y'] = 3

        # Cannot create it without first resizing it.
        with pytest.raises(ValueError) as e:
                f.create_variable('dummy1', data=np.array([[1, 2, 3]]),
                                  dimensions=('x', 'y'))
                assert e.value.args[0] == \
                    "Shape tuple is incompatible with data"

        # Without data.
        f.create_variable('dummy1', dimensions=('x', 'y'), dtype=np.int64)
        f.create_variable('dummy2', dimensions=('x', 'y'), dtype=np.int64)
        f.create_variable('dummy3', dimensions=('x', 'y'), dtype=np.int64)
        g = f.create_group('test')
        g.create_variable('dummy4', dimensions=('y', 'x', 'x'), dtype=np.int64)
        g.create_variable('dummy5', dimensions=('y', 'y'), dtype=np.int64)

        assert f.variables['dummy1'].shape == (0, 3)
        assert f.variables['dummy2'].shape == (0, 3)
        assert f.variables['dummy3'].shape == (0, 3)
        assert g.variables['dummy4'].shape == (3, 0, 0)
        assert g.variables['dummy5'].shape == (3, 3)
        f.resize_dimension("x", 2)
        assert f.variables['dummy1'].shape == (2, 3)
        assert f.variables['dummy2'].shape == (2, 3)
        assert f.variables['dummy3'].shape == (2, 3)
        assert g.variables['dummy4'].shape == (3, 2, 2)
        assert g.variables['dummy5'].shape == (3, 3)

        f.variables['dummy2'][:] = [[1, 2, 3], [5, 6, 7]]
        np.testing.assert_allclose(f.variables['dummy2'],
                                   [[1, 2, 3], [5, 6, 7]])

        f.variables['dummy3'][...] = [[1, 2, 3], [5, 6, 7]]
        np.testing.assert_allclose(f.variables['dummy3'],
                                   [[1, 2, 3], [5, 6, 7]])

def test_c_api_can_read_unlimited_dimensions(tmp_netcdf):
    with h5netcdf.File(tmp_netcdf) as f:
        # Two dimensions, only one is unlimited.
        f.dimensions['x'] = None
        f.dimensions['y'] = 3
        f.dimensions['z'] = None
        f.create_variable('dummy1', dimensions=('x', 'y'), dtype=np.int64)
        f.create_variable('dummy2', dimensions=('y', 'x', 'x'), dtype=np.int64)
        g = f.create_group('test')
        g.create_variable('dummy3', dimensions=('y', 'y'), dtype=np.int64)
        g.create_variable('dummy4', dimensions=('z', 'z'), dtype=np.int64)
        f.resize_dimension('x', 2)

    with netCDF4.Dataset(tmp_netcdf) as f:
        assert f.dimensions['x'].size == 2
        assert f.dimensions['x'].isunlimited() is True
        assert f.dimensions['y'].size == 3
        assert f.dimensions['y'].isunlimited() is False
        assert f.dimensions['z'].size == 0
        assert f.dimensions['z'].isunlimited() is True

        assert f.variables['dummy1'].shape == (2, 3)
        assert f.variables['dummy2'].shape == (3, 2, 2)
        g = f.groups["test"]
        assert g.variables['dummy3'].shape == (3, 3)
        assert g.variables['dummy4'].shape == (0, 0)

def test_reading_unlimited_dimensions_created_with_c_api(tmp_netcdf):
    with netCDF4.Dataset(tmp_netcdf, "w") as f:
        f.createDimension('x', None)
        f.createDimension('y', 3)
        f.createDimension('z', None)

        dummy1 = f.createVariable('dummy1', float, ('x', 'y'))
        f.createVariable('dummy2', float, ('y', 'x', 'x'))
        g = f.createGroup('test')
        g.createVariable('dummy3', float, ('y', 'y'))
        g.createVariable('dummy4', float, ('z', 'z'))

        # Assign something to trigger a resize.
        dummy1[:] = [[1, 2, 3], [4, 5, 6]]

    with h5netcdf.File(tmp_netcdf) as f:
        assert f.dimensions['x'] is None
        assert f.dimensions['y'] == 3
        assert f.dimensions['z'] is None

        # This is parsed correctly due to h5netcdf's init trickery.
        assert f._current_dim_sizes['x'] == 2
        assert f._current_dim_sizes['y'] == 3
        assert f._current_dim_sizes['z'] == 0

        # But the actual data-set and arrays are not correct.
        assert f['dummy1'].shape == (2, 3)
        # XXX: This array has some data with dimension x - netcdf does not
        # appear to keep dimensions consistent.
        assert f['dummy2'].shape == (3, 0, 0)
        f.groups['test']['dummy3'].shape == (3, 3)
        f.groups['test']['dummy4'].shape == (0, 0)
