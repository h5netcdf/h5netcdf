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
    def __init__(self, root, h5ds, dimensions=None):
        self._root = root
        self._h5ds = h5ds
        self._dimensions = dimensions
        self._initialized = True

    @property
    def name(self):
        return self._h5ds.name

    def _lookup_dimensions(self):
        attrs = self._h5ds.attrs
        if '_Netcdf4Coordinates' in attrs:
            order_dim = _reverse_dict(self._root._dim_order)
            return tuple(order_dim[coord_id]
                         for coord_id in attrs['_Netcdf4Coordinates'])

        child_name = self.name.split('/')[-1]
        if child_name in self._root.dimensions:
            return (child_name,)

        dims = []
        for axis, dim in enumerate(self._h5ds.dims):
            # TODO: read dimension labels even if there is no associated
            # scale? it's not netCDF4 spec, but it is unambiguous...
            # Also: the netCDF lib can read HDF5 datasets with unlabeled
            # dimensions.
            if len(dim) == 0:
                raise ValueError('variable %r has no dimension scale '
                                 'associated with axis %s'
                                 % (self.name, axis))
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

    _cls_name = 'h5netcdf.Variable'

    def __repr__(self):
        if self._root._closed:
            return '<Closed %s>' % self._cls_name
        header = ('<%s %r: dimensions %s, shape %s, dtype %s>' %
                  (self._cls_name, self.name, self.dimensions, self.shape,
                   self.dtype))
        return '\n'.join([header] +
                         # ['Dtype:      %s' % self.dtype] +
                         # ['Dimensions: %r' % (self.dimensions,)] +
                         # ['Shape:      %r' % (self.shape,)] +
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
                    self._variables[name] = self._variable_cls(self._root, v)
        self._initialized = True

    @property
    def name(self):
        return self._h5group.name

    def _create_child_group(self, name):
        if name in self._groups:
            raise IOError('group %r already exists' % name)
        h5group = self._h5group.create_group(name)
        group = self._group_cls(self, h5group)
        self._groups[name] = group
        return group

    def _require_child_group(self, name):
        try:
            return self.groups[name]
        except KeyError:
            return self._create_child_group(name)

    def create_group(self, name):
        if name.startswith('/'):
            return self._root.create_group(name[1:])
        keys = name.split('/')
        group = self
        for k in keys[:-1]:
            group = group._require_child_group(k)
        return group._create_child_group(keys[-1])

    def _create_child_variable(self, name, dimensions, dtype, data, fillvalue,
                               **kwargs):
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
        variable = self._variable_cls(self._root, h5ds, dimensions)
        if fillvalue is not None:
            value = variable.dtype.type(fillvalue)
            variable.attrs._h5attrs['_FillValue'] = value
        self._variables[name] = variable
        return variable

    def create_variable(self, name, dimensions=(), dtype=None, data=None,
                        fillvalue=None, **kwargs):
        if name.startswith('/'):
            return self._root.create_variable(name[1:], dimensions, dtype,
                                              data, fillvalue, **kwargs)
        keys = name.split('/')
        group = self
        for k in keys[:-1]:
            group = group._require_child_group(k)
        return group._create_child_variable(keys[-1], dimensions, dtype, data,
                                            fillvalue, **kwargs)

    def _get_child(self, key):
        try:
            return self.variables[key]
        except KeyError:
            return self.groups[key]

    def __getitem__(self, key):
        if key.startswith('/'):
            return self._root[key[1:]]
        keys = key.split('/')
        item = self
        for k in keys:
            item = item._get_child(k)
        return item

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

    _cls_name = 'h5netcdf.Group'

    def __repr__(self):
        if self._root._closed:
            return '<Closed %s>' % self._cls_name
        header = ('<%s %r (%s members)>'
                  % (self._cls_name, self.name, len(self)))
        return '\n'.join([header] +
                         ['Groups:'] + ['    %s' % g for g in self.groups] +
                         ['Variables:'] +
                         ['    %s: %r %s' % (k, v.dimensions, v.dtype)
                          for k, v in self.variables.items()] +
                         ['Attributes:'] +
                         ['    %s: %r' % (k, v)
                          for k, v in self.attrs.items()])


class File(Group):
    def __init__(self, path, mode='a', **kwargs):
        self._file = h5py.File(path, mode, **kwargs)
        self._dim_sizes = {}
        self._dim_order = {}
        self._mode = mode
        self._root = self
        self._closed = False
        super(File, self).__init__(self, self._file)

    @property
    def mode(self):
        return self._h5group.mode

    @property
    def filename(self):
        return self._h5group.filename

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

    _cls_name = 'h5netcdf.File'

    def __repr__(self):
        if self._closed:
            return '<Closed %s>' % self._cls_name
        header = '<%s %r (mode %s)>' % (self._cls_name,
                                        self.filename.split('/')[-1],
                                        self.mode)
        return '\n'.join([header] +
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
