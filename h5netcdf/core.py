# TODO:
# avoid circular references to ensure __del__ works?
# handle unlimited dimensions?
#
# For details on how netCDF4 builds on HDF5:
# https://www.unidata.ucar.edu/software/netcdf/docs/netcdf/NetCDF_002d4-Format.html
import h5py
import numpy as np

from .compat import OrderedDict
from .attrs import HasAttributesMixin, Attributes
from .utils import Frozen


NOT_A_VARIABLE = b'This is a netCDF dimension but not a netCDF variable.'


class Group(HasAttributesMixin):
    def __init__(self, parent, h5group):
        self._parent = parent
        self._root = parent._root
        self._h5group = h5group

        self._variables = OrderedDict()
        self._groups = OrderedDict()
        for k, v in h5group.items():
            if isinstance(v, h5py.Group):
                self._groups[k] = Group(self, v)
            else:
                if v.attrs.get('CLASS') == b'DIMENSION_SCALE':
                    dim_id = v.attrs['_Netcdf4Dimid']
                    if '_Netcdf4Coordinates' in v.attrs:
                        coord_ids = v.attrs['_Netcdf4Coordinates']
                        size = v.shape[list(coord_ids).index(dim_id)]
                    else:
                        size = v.size
                    self._root._dim_sizes[k] = size
                    self._root._dim_order[k] = dim_id
                if NOT_A_VARIABLE not in v.attrs.get('NAME', b''):
                    name = k
                    if k.startswith('_nc4_non_coord_'):
                        name = k[len('_nc4_non_coord_'):]
                    self._variables[name] = Variable(self._root, v, k)

    def createGroup(self, name):
        if name in self._groups:
            raise IOError('group %r already exists' % name)
        h5group = self._h5group.create_group(name)
        group = Group(self, h5group)
        self._groups[name] = group
        return group

    def createVariable(self, name, dtype, dimensions, **kwargs):
        if name in self._variables:
            raise IOError('variable %r already exists' % name)

        shape = tuple(self._root.dimensions[d] for d in dimensions)
        if name in self._root.dimensions and name not in dimensions:
            h5name = '_nc4_non_coord_' + name
        else:
            h5name = name

        h5ds = self._h5group.create_dataset(h5name, shape, dtype, **kwargs)
        variable = Variable(self._root, h5ds, h5name, dimensions)
        self._variables[name] = variable
        return variable

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

    def createDimension(self, name, size=None):
        if name in self._dim_sizes:
            raise IOError('dimension %r already exists' % name)
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
                         ['Dimensions: %r' % self.dimensions] +
                         ['Variables:'] +
                         ['    %s: %s' % (k, v)
                          for k, v in self.variables.items()]
                         ['Attributes:'] +
                         ['    %s: %s' % (k, v)
                          for k, v in self.attrs.items()])


def reverse_dict(dict_):
    return dict(zip(dict_.values(), dict_.keys()))


class Variable(HasAttributesMixin):
    def __init__(self, root, h5ds, name, dimensions=None):
        self._root = root
        self._h5ds = h5ds
        self._name = name
        self._dimensions = dimensions

    def _lookup_dimensions(self):
        attrs = self._h5ds.attrs
        if '_Netcdf4Coordinates' in attrs:
            order_dim = reverse_dict(self._root._dim_order)
            return tuple(order_dim[coord_id]
                         for coord_id in attrs['_Netcdf4Coordinates'])
        elif self._name in self._root.dimensions:
            return (self._name,)
        else:
            return tuple(k[0].name[1:] for k in self._h5ds.dims)

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
        return ('<%s dimensions=%s dtype=%s>'
                % (type(self).__name__, self.dimensions, self.dtype))
