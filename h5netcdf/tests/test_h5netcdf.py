import netCDF4
import numpy as np

import h5netcdf
import pytest


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


def roundtrip_netcdf(tmp_netcdf, read_module, write_module):
    ds = write_module.Dataset(tmp_netcdf, 'w')
    ds.setncattr('global', 42)
    ds.createDimension('x', 4)
    ds.createDimension('y', 5)
    ds.createDimension('z', 6)
    ds.createDimension('string3', 3)

    v = ds.createVariable('foo', float, ('x', 'y'))
    v[...] = 1
    v.setncattr('units', 'meters')

    v = ds.createVariable('y', int, ('y',))
    v[...] = np.arange(5)

    v = ds.createVariable('z', 'S1', ('z', 'string3'))
    char_array = string_to_char(np.array(['a', 'b', 'c', 'foo', 'bar', 'baz'],
                                         dtype='S'))
    v[...] = char_array

    v = ds.createVariable('scalar', np.float32, ())
    v[...] = 2.0

    g = ds.createGroup('subgroup')
    v = g.createVariable('subvar', np.int32, ('x',))
    v[...] = np.arange(4.0)

    ds.createDimension('mismatched_dim', 1)
    ds.createVariable('mismatched_dim', int, ())

    ds.close()

    ds = read_module.Dataset(tmp_netcdf, 'r')
    assert ds.ncattrs() == ['global']
    assert ds.getncattr('global') == 42
    assert set(ds.dimensions) == set(['x', 'y', 'z', 'string3', 'mismatched_dim'])
    assert set(ds.variables) == set(['foo', 'y', 'z', 'scalar', 'mismatched_dim'])
    assert set(ds.groups) == set(['subgroup'])

    assert array_equal(ds.variables['foo'], np.ones((4, 5)))
    assert ds.variables['foo'].dtype == float
    assert ds.variables['foo'].dimensions == ('x', 'y')
    assert ds.variables['foo'].ncattrs() == ['units']
    assert ds.variables['foo'].getncattr('units') == 'meters'

    assert array_equal(ds.variables['y'], np.arange(5))
    assert ds.variables['y'].dtype == int
    assert ds.variables['y'].dimensions == ('y',)
    assert ds.variables['y'].ncattrs() == []

    assert array_equal(ds.variables['z'], char_array)
    assert ds.variables['z'].dtype == 'S1'
    assert ds.variables['z'].dimensions == ('z', 'string3')
    assert ds.variables['z'].ncattrs() == []

    assert array_equal(ds.variables['scalar'], np.array(2.0))
    assert ds.variables['scalar'].dtype == 'float32'
    assert ds.variables['scalar'].dimensions == ()
    assert ds.variables['scalar'].ncattrs() == []

    v = ds.groups['subgroup'].variables['subvar']
    assert array_equal(v, np.arange(4.0))
    assert v.dtype == 'int32'
    assert v.dimensions == ('x',)
    assert v.ncattrs() == []

    ds.close()


def test_write_h5netcdf_read_netCDF4(tmp_netcdf):
    roundtrip_netcdf(tmp_netcdf, netCDF4, h5netcdf)


def test_roundtrip_h5netcdf(tmp_netcdf):
    roundtrip_netcdf(tmp_netcdf, h5netcdf, h5netcdf)


def test_write_netCDF4_read_h5netcdf(tmp_netcdf):
    roundtrip_netcdf(tmp_netcdf, h5netcdf, netCDF4)


def test_attrs_api(tmp_netcdf):
    with h5netcdf.Dataset(tmp_netcdf) as ds:
        ds.attrs['conventions'] = 'CF'
        ds.createDimension('x', 1)
        v = ds.createVariable('x', 'i4', ('x',))
        v.attrs.update({'units': 'meters', 'foo': 'bar'})
    assert ds._closed
    with h5netcdf.Dataset(tmp_netcdf) as ds:
        assert len(ds.attrs) == 1
        assert dict(ds.attrs) == {'conventions': 'CF'}
        assert list(ds.attrs) == ['conventions']
        assert dict(ds.variables['x'].attrs) == {'units': 'meters', 'foo': 'bar'}
        assert len(ds.variables['x'].attrs) == 2
        assert sorted(ds.variables['x'].attrs) == ['foo', 'units']
