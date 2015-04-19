# For details on how netCDF4 builds on HDF5:
# https://www.unidata.ucar.edu/software/netcdf/docs/netcdf/NetCDF_002d4-Format.html
from collections import Mapping

import h5py
import numpy as np

from .compat import OrderedDict
from .attrs import Attributes
from .utils import Frozen


def _reverse_dict(dict_):
    return dict(zip(dict_.values(), dict_.keys()))


class Variable(object):
    def __init__(self, root, h5ds, name, dimensions=None):
        self._root = root
        self._h5ds = h5ds
        self._name = name
        self._dimensions = dimensions
        self._initialized = True

    def _lookup_dimensions(self):
        attrs = self._h5ds.attrs
        if '_Netcdf4Coordinates' in attrs:
            order_dim = _reverse_dict(self._root._dim_order)
            return tuple(order_dim[coord_id]
                         for coord_id in attrs['_Netcdf4Coordinates'])
        elif self._name in self._root.dimensions:
            return (self._name,)
        else:
            dims = []
            for axis, dim in enumerate(self._h5ds.dims):
                # TODO: read dimension labels even if there is no associated
                # scale? it's not netCDF4 spec, but it is unambiguous...
                # Also: the netCDF lib can read HDF5 datasets with unlabeled
                # dimensions.
                if len(dim) == 0:
                    raise ValueError('variable %s has no dimension scale '
                                     'associated with axis %s'
                                     % (self._name, axis))
                name = dim[0].name[1:]
                dims.append(name)
            return tuple(dims)

    @property
    def dimensions(self):
        if self._dimensions is None:
            self._dimensions = self._lookup_dimensions()
        return self._dimensions

    @property
    def shape(self):
        return self._h5ds.shape

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0]

    @property
    def dtype(self):
        return self._h5ds.dtype

    def __array__(self, *args, **kwargs):
        return self._h5ds.__array__(*args, **kwargs)

    def __getitem__(self, key):
        return self._h5ds[key]

    def __setitem__(self, key, value):
        self._h5ds[key] = value

    @property
    def attrs(self):
        return Attributes(self._h5ds.attrs)

    def __repr__(self):
        return '\n'.join(['%r' % type(self)] +
                         ['Dtype:      %s' % self.dtype] +
                         ['Dimensions: %r' % (self.dimensions,)] +
                         ['Shape:      %r' % (self.shape,)] +
                         ['Attributes:'] +
                         ['    %s: %r' % (k, v)
                          for k, v in self.attrs.items()])


NOT_A_VARIABLE = b'This is a netCDF dimension but not a netCDF variable.'


class Group(Mapping):
    _variable_cls = Variable

    @property
    def _group_cls(self):
        return Group

    def __init__(self, parent, h5group):
        self._parent = parent
        self._root = parent._root
        self._h5group = h5group

        self._variables = OrderedDict()
        self._groups = OrderedDict()
        for k, v in h5group.items():
            if isinstance(v, h5py.Group):
                self._groups[k] = self._group_cls(self, v)
            else:
                if v.attrs.get('CLASS') == b'DIMENSION_SCALE':
                    dim_id = v.attrs.get('_Netcdf4Dimid')
                    if '_Netcdf4Coordinates' in v.attrs:
                        assert dim_id is not None
                        coord_ids = v.attrs['_Netcdf4Coordinates']
                        size = v.shape[list(coord_ids).index(dim_id)]
                    else:
                        assert len(v.shape) == 1
                        size = v.size
                    self._root._dim_sizes[k] = size
                    if dim_id is None:
                        dim_id = len(self._root._dim_order)
                    self._root._dim_order[k] = dim_id
                if NOT_A_VARIABLE not in v.attrs.get('NAME', b''):
                    name = k
                    if k.startswith('_nc4_non_coord_'):
                        name = k[len('_nc4_non_coord_'):]
                    self._variables[name] = self._variable_cls(self._root, v, k)
        self._initialized = True

    def create_group(self, name):
        if name in self._groups:
            raise IOError('group %r already exists' % name)
        h5group = self._h5group.create_group(name)
        group = self._group_cls(self, h5group)
        self._groups[name] = group
        return group

    def create_variable(self, name, dimensions=(), dtype=None, data=None,
                        fillvalue=None, **kwargs):
        if name in self._variables:
            raise IOError('variable %r already exists' % name)

        if data is not None:
            data = np.asarray(data)
            for d, s in zip(dimensions, data.shape):
                if d not in self._root.dimensions:
                    self.create_dimension(d, s)

        shape = tuple(self._root.dimensions[d] for d in dimensions)
        if name in self._root.dimensions and name not in dimensions:
            h5name = '_nc4_non_coord_' + name
        else:
            h5name = name

        h5ds = self._h5group.create_dataset(h5name, shape, dtype=dtype,
                                            data=data, fillvalue=fillvalue,
                                            **kwargs)
        variable = self._variable_cls(self._root, h5ds, h5name, dimensions)
        if fillvalue is not None:
            value = variable.dtype.type(fillvalue)
            variable.attrs._h5attrs['_FillValue'] = value
        self._variables[name] = variable
        return variable

    def __getitem__(self, key):
        # TODO: handle relative and absolute paths like '/group/variable'
        try:
            return self.variables[key]
        except KeyError:
            return self.groups[key]

    def __iter__(self):
        for name in self.groups:
            yield name
        for name in self.variables:
            yield name

    def __len__(self):
        return len(self.variables) + len(self.groups)

    def _attach_dim_scales(self):
        for name, var in self.variables.items():
            if self._root is not self or name not in self.dimensions:
                for n, dim in enumerate(var.dimensions):
                    var._h5ds.dims[n].attach_scale(self._root._file[dim])

        for subgroup in self.groups.values():
            subgroup._attach_dim_scales()

    @property
    def parent(self):
        return self._parent

    @property
    def groups(self):
        return Frozen(self._groups)

    @property
    def variables(self):
        return Frozen(self._variables)

    @property
    def attrs(self):
        return Attributes(self._h5group.attrs)

    def __repr__(self):
        return '\n'.join(['%r' % type(self)] +
                         ['Groups:'] + ['    %s' % g for g in self.groups] +
                         ['Variables:'] +
                         ['    %s: %r %s' % (k, v.dimensions, v.dtype)
                          for k, v in self.variables.items()] +
                         ['Attributes:'] +
                         ['    %s: %r' % (k, v)
                          for k, v in self.attrs.items()])


class Dataset(Group):
    def __init__(self, path, mode='a', **kwargs):
        self._file = h5py.File(path, mode, **kwargs)
        self._dim_sizes = {}
        self._dim_order = {}
        self._mode = mode
        self._root = self
        self._closed = False
        super(Dataset, self).__init__(self, self._file)

    @property
    def parent(self):
        return None

    def create_dimension(self, name, size=None):
        if name in self._dim_sizes:
            raise IOError('dimension %r already exists' % name)
        if not size:
            raise NotImplementedError('h5netcdf does not yet support '
                                      'unlimited dimensions')
        self._dim_sizes[name] = size
        self._dim_order[name] = len(self._dim_order)

    @property
    def dimensions(self):
        return Frozen(self._dim_sizes)

    def _create_dim_scales(self):
        for dim in sorted(self._dim_order, key=lambda d: self._dim_order[d]):
            if dim not in self._file:
                size = self.dimensions[dim]
                self._file.create_dataset(dim, (size,), 'S1')

            h5ds = self._file[dim]
            h5ds.attrs['_Netcdf4Dimid'] = self._dim_order[dim]

            if len(h5ds.shape) > 1:
                dims = self._variables[dim].dimensions
                coord_ids = np.array([self._dim_order[d] for d in dims])
                h5ds.attrs['_Netcdf4Coordinates'] = coord_ids.astype('int32')

            scale_name = dim if dim in self.variables else NOT_A_VARIABLE
            h5ds.dims.create_scale(h5ds, scale_name)

    def flush(self):
        if 'r' not in self._mode:
            self._create_dim_scales()
            self._attach_dim_scales()
    sync = flush

    def close(self):
        if not self._closed:
            self.flush()
            self._file.close()
            self._closed = True
    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        return '\n'.join(['%r' % type(self)] +
                         ['Dimensions:'] +
                         ['    %s: %s' % (k, v)
                          for k, v in self.dimensions.items()] +
                         ['Groups:'] + ['    %s' % g for g in self.groups] +
                         ['Variables:'] +
                         ['    %s: %r %s' % (k, v.dimensions, v.dtype)
                          for k, v in self.variables.items()] +
                         ['Attributes:'] +
                         ['    %s: %r' % (k, v)
                          for k, v in self.attrs.items()])
