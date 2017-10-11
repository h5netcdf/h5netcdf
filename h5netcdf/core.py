# For details on how netCDF4 builds on HDF5:
# http://www.unidata.ucar.edu/software/netcdf/docs/file_format_specifications.html#netcdf_4_spec
from collections import Mapping
import os.path
import warnings

import h5py
import numpy as np

from .compat import ChainMap, OrderedDict, unicode
from .attrs import Attributes
from .dimensions import Dimensions
from .utils import Frozen


__version__ = '0.4.3'


_NC_PROPERTIES = (u'version=1|h5netcdfversion=%s|hdf5libversion=%s'
                  % (__version__, h5py.version.hdf5_version))

NOT_A_VARIABLE = b'This is a netCDF dimension but not a netCDF variable.'


def _reverse_dict(dict_):
    return dict(zip(dict_.values(), dict_.keys()))


def _join_h5paths(parent_path, child_path):
    return '/'.join([parent_path.rstrip('/'), child_path.lstrip('/')])


class CompatibilityError(Exception):
    """Raised when using features that are not part of the NetCDF4 API."""


def _invalid_netcdf_feature(feature, allow, file, stacklevel=0):
    if allow is None:
        msg = ('{} are supported by h5py, but not part of the NetCDF API. '
               'You are writing an HDF5 file that is not a valid NetCDF file! '
               'In the future, this will be an error, unless you set '
               'invalid_netcdf=True.'.format(feature))
        warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
        file._write_ncproperties = False
    elif not allow:
        msg = ('{} are not a supported NetCDF feature, and are not allowed by '
               'h5netcdf unless invalid_netcdf=True.'.format(feature))
        raise CompatibilityError(msg)


class BaseVariable(object):

    def __init__(self, parent, name, dimensions=None):
        self._parent = parent
        self._root = parent._root
        self._h5path = _join_h5paths(parent.name, name)
        self._dimensions = dimensions
        self._initialized = True

    @property
    def _h5ds(self):
        # Always refer to the root file and store not h5py object
        # subclasses:
        return self._root._h5file[self._h5path]

    @property
    def name(self):
        return self._h5ds.name

    def _lookup_dimensions(self):
        attrs = self._h5ds.attrs
        if '_Netcdf4Coordinates' in attrs:
            order_dim = _reverse_dict(self._parent._dim_order)
            return tuple(order_dim[coord_id]
                         for coord_id in attrs['_Netcdf4Coordinates'])

        child_name = self.name.split('/')[-1]
        if child_name in self._parent.dimensions:
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
            name = dim[0].name.split('/')[-1]
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
        return Attributes(self._h5ds.attrs,
                          self._root._check_valid_netcdf_dtype)

    _cls_name = 'h5netcdf.Variable'

    def __repr__(self):
        if self._parent._root._closed:
            return '<Closed %s>' % self._cls_name
        header = ('<%s %r: dimensions %s, shape %s, dtype %s>' %
                  (self._cls_name, self.name, self.dimensions, self.shape,
                   self.dtype))
        return '\n'.join([header] +
                         ['Attributes:'] +
                         ['    %s: %r' % (k, v)
                          for k, v in self.attrs.items()])


class Variable(BaseVariable):

    @property
    def chunks(self):
        return self._h5ds.chunks

    @property
    def compression(self):
        return self._h5ds.compression

    @property
    def compression_opts(self):
        return self._h5ds.compression_opts

    @property
    def fletcher32(self):
        return self._h5ds.fletcher32

    @property
    def shuffle(self):
        return self._h5ds.shuffle


class _LazyObjectLookup(Mapping):
    def __init__(self, parent, object_cls):
        self._parent = parent
        self._object_cls = object_cls
        self._objects = OrderedDict()

    def __setitem__(self, name, obj):
        self._objects[name] = obj

    def add(self, name):
        self._objects[name] = None

    def __iter__(self):
        for name in self._objects:
            yield name

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, key):
        if self._objects[key] is not None:
            return self._objects[key]
        else:
            self._objects[key] = self._object_cls(self._parent, key)
            return self._objects[key]


class Group(Mapping):

    _variable_cls = Variable

    @property
    def _group_cls(self):
        return Group

    def __init__(self, parent, name):
        self._parent = parent
        self._root = parent._root
        self._h5path = _join_h5paths(parent.name, name)

        if parent is not self:
            self._dim_sizes = parent._dim_sizes.new_child()
            self._current_dim_sizes = parent._current_dim_sizes.new_child()
            self._dim_order = parent._dim_order.new_child()

        self._variables = _LazyObjectLookup(self, self._variable_cls)
        self._groups = _LazyObjectLookup(self, self._group_cls)

        for k, v in self._h5group.items():
            if isinstance(v, h5py.Group):
                self._groups.add(k)
            else:
                if v.attrs.get('CLASS') == b'DIMENSION_SCALE':
                    dim_id = v.attrs.get('_Netcdf4Dimid')
                    if '_Netcdf4Coordinates' in v.attrs:
                        assert dim_id is not None
                        coord_ids = v.attrs['_Netcdf4Coordinates']
                        size = v.shape[list(coord_ids).index(dim_id)]
                        current_size = size
                    else:
                        assert len(v.shape) == 1
                        # Unlimited dimensions are represented as None.
                        size = None if v.maxshape == (None,) else v.size
                        current_size = v.size

                    self._dim_sizes[k] = size
                    self._current_dim_sizes[k] = current_size
                    if dim_id is None:
                        dim_id = len(self._dim_order)
                    self._dim_order[k] = dim_id
                if NOT_A_VARIABLE not in v.attrs.get('NAME', b''):
                    var_name = k
                    if k.startswith('_nc4_non_coord_'):
                        var_name = k[len('_nc4_non_coord_'):]
                    self._variables.add(var_name)

        # One last pass to get the current sizes.
        #
        # I'm not entirely sure how the netcdf C api determines the current
        # size - should probably be looked into.
        def _get_all_variables(g):
            var = list(g.variables.values())
            for _g in g.groups.values():
                var.extend(_get_all_variables(_g))
            return var

        all_vars = _get_all_variables(self)

        for d in self.dimensions:
            if self.dimensions[d] is not None:
                continue
            max_size = self._current_dim_sizes[d]
            for var in all_vars:
                for i, name in enumerate(var.dimensions):
                    if name == d:
                        max_size = max(var.shape[i], max_size)
            self._current_dim_sizes[d] = max_size

        self._initialized = True

    @property
    def _h5group(self):
        # Always refer to the root file and store not h5py object
        # subclasses:
        return self._root._h5file[self._h5path]

    @property
    def name(self):
        return self._h5group.name

    def _create_dimension(self, name, size=None):
        if name in self._dim_sizes.maps[0]:
            raise ValueError('dimension %r already exists' % name)

        self._dim_sizes[name] = size
        self._current_dim_sizes[name] = 0 if size is None else size
        self._dim_order[name] = len(self._dim_order)

    @property
    def dimensions(self):
        return Dimensions(self)

    @dimensions.setter
    def dimensions(self, value):
        for k, v in self._dim_sizes.maps[0].items():
            if k in value:
                if v != value[k]:
                    raise ValueError('cannot modify existing dimension %r' % k)
            else:
                raise ValueError('new dimensions do not include existing '
                                 'dimension %r' % k)
        self.dimensions.update(value)

    def _create_child_group(self, name):
        if name in self:
            raise ValueError('unable to create group %r (name already exists)'
                             % name)
        self._h5group.create_group(name)
        self._groups[name] = self._group_cls(self, name)
        return self._groups[name]

    def _require_child_group(self, name):
        try:
            return self._groups[name]
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
        stacklevel = 4  # correct if name does not start with '/'

        if name in self:
            raise ValueError('unable to create variable %r '
                             '(name already exists)' % name)

        if data is not None:
            data = np.asarray(data)
            for d, s in zip(dimensions, data.shape):
                if d not in self.dimensions:
                    self.dimensions[d] = s

        if dtype is None:
            dtype = data.dtype

        if dtype == np.bool_:
            # never warn since h5netcdf has always errored here
            _invalid_netcdf_feature('boolean dtypes',
                                    allow=bool(self._root.invalid_netcdf),
                                    file=self._root,
                                    stacklevel=stacklevel)
        else:
            self._root._check_valid_netcdf_dtype(dtype, stacklevel=stacklevel)

        compression = kwargs.get('compression')
        if compression not in {None, 'gzip'}:
            _invalid_netcdf_feature('{} compression'.format(compression),
                                    allow=self._root.invalid_netcdf,
                                    file=self._root,
                                    stacklevel=stacklevel)

        if 'scaleoffset' in kwargs:
            _invalid_netcdf_feature('scale-offset filters',
                                    allow=self._root.invalid_netcdf,
                                    file=self._root,
                                    stacklevel=stacklevel)

        if name in self.dimensions and name not in dimensions:
            h5name = '_nc4_non_coord_' + name
        else:
            h5name = name

        shape = tuple(self._current_dim_sizes[d] for d in dimensions)
        maxshape = tuple(self._dim_sizes[d] for d in dimensions)

        # If it is passed directly it will change the default compression
        # settings.
        if shape != maxshape:
            kwargs["maxshape"] = maxshape

        self._h5group.create_dataset(h5name, shape, dtype=dtype,
                                     data=data, fillvalue=fillvalue,
                                     **kwargs)

        self._variables[h5name] = self._variable_cls(self, h5name, dimensions)
        variable = self._variables[h5name]

        if fillvalue is not None:
            value = variable.dtype.type(fillvalue)
            variable.attrs._h5attrs['_FillValue'] = value
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

    def _create_dim_scales(self):
        dim_order = self._dim_order.maps[0]
        for dim in sorted(dim_order, key=lambda d: dim_order[d]):
            if dim not in self._h5group:
                size = self._current_dim_sizes[dim]
                kwargs = {}
                if self._dim_sizes[dim] is None:
                    kwargs["maxshape"] = (None,)
                self._h5group.create_dataset(
                    name=dim, shape=(size,), dtype='S1', **kwargs)

            h5ds = self._h5group[dim]
            h5ds.attrs['_Netcdf4Dimid'] = dim_order[dim]

            if len(h5ds.shape) > 1:
                dims = self._variables[dim].dimensions
                coord_ids = np.array([dim_order[d] for d in dims], 'int32')
                h5ds.attrs['_Netcdf4Coordinates'] = coord_ids

            scale_name = dim if dim in self.variables else NOT_A_VARIABLE
            h5ds.dims.create_scale(h5ds, scale_name)

        for subgroup in self.groups.values():
            subgroup._create_dim_scales()

    def _attach_dim_scales(self, parent_h5groups=None):
        if parent_h5groups is None:
            all_h5groups = ChainMap(self._h5group)
        else:
            all_h5groups = parent_h5groups.new_child(self._h5group)

        for name, var in self.variables.items():
            if name not in self.dimensions:
                for n, dim in enumerate(var.dimensions):
                    var._h5ds.dims[n].attach_scale(all_h5groups[dim])

        for subgroup in self.groups.values():
            subgroup._attach_dim_scales(all_h5groups)

    @property
    def parent(self):
        return self._parent

    def flush(self):
        self._root.flush()
    sync = flush

    @property
    def groups(self):
        return Frozen(self._groups)

    @property
    def variables(self):
        return Frozen(self._variables)

    @property
    def attrs(self):
        return Attributes(self._h5group.attrs,
                          self._root._check_valid_netcdf_dtype)

    _cls_name = 'h5netcdf.Group'

    def _repr_body(self):
        return (
            ['Dimensions:'] +
            ['    %s: %s' % (
             k, ("Unlimited (current: %s)" % self._current_dim_sizes[k])
             if v is None else v) for k, v in self.dimensions.items()] +
            ['Groups:'] +
            ['    %s' % g for g in self.groups] +
            ['Variables:'] +
            ['    %s: %r %s' % (k, v.dimensions, v.dtype)
             for k, v in self.variables.items()] +
            ['Attributes:'] +
            ['    %s: %r' % (k, v) for k, v in self.attrs.items()])

    def __repr__(self):
        if self._root._closed:
            return '<Closed %s>' % self._cls_name
        header = ('<%s %r (%s members)>'
                  % (self._cls_name, self.name, len(self)))
        return '\n'.join([header] + self._repr_body())

    def resize_dimension(self, dimension, size):
        """
        Resize a dimension to a certain size.

        This will pad with zeros where necessary.
        """
        if self.dimensions[dimension] is not None:
            raise ValueError("Dimension '%s' is not unlimited and thus "
                             "cannot be resized." % dimension)

        # Resize the dimension.
        self._current_dim_sizes[dimension] = size

        for var in self.variables.values():
            new_shape = list(var.shape)
            for i, d in enumerate(var.dimensions):
                if d == dimension:
                    new_shape[i] = size
            new_shape = tuple(new_shape)
            if new_shape != var.shape:
                var._h5ds.resize(new_shape)

        # Recurse as dimensions are visible to this group and all child groups.
        for i in self.groups.values():
            i.resize_dimension(dimension, size)


class File(Group):

    def __init__(self, path, mode='a', invalid_netcdf=None, **kwargs):
        self._preexisting_file = os.path.exists(path)
        try:
            self._h5file = h5py.File(path, mode, **kwargs)
        except Exception:
            self._closed = True
            raise
        else:
            self._closed = False

        # Three maps to keep track of dimensions in terms of size (might be
        # unlimited), current size (identical to size for limited dimensions),
        # and their position.
        self._dim_sizes = ChainMap()
        self._current_dim_sizes = ChainMap()
        self._dim_order = ChainMap()

        self._mode = mode
        self._root = self
        self._h5path = '/'
        self.invalid_netcdf = invalid_netcdf
        # If invalid_netcdf is None, we'll disable writing _NCProperties only
        # if we actually use invalid NetCDF features.
        self._write_ncproperties = (invalid_netcdf is not True)
        super(File, self).__init__(self, self._h5path)

    def _check_valid_netcdf_dtype(self, dtype, stacklevel=3):
        dtype = np.dtype(dtype)

        if dtype == bool:
            description = 'boolean'
        elif dtype == complex:
            description = 'complex'
        elif h5py.check_dtype(enum=dtype) is not None:
            description = 'enum'
        elif h5py.check_dtype(ref=dtype) is not None:
            description = 'reference'
        elif h5py.check_dtype(vlen=dtype) not in {None, unicode, bytes}:
            description = 'non-string variable length'
        else:
            description = None

        if description is not None:
            _invalid_netcdf_feature('{} dtypes'.format(description),
                                    allow=self.invalid_netcdf,
                                    file=self,
                                    stacklevel=stacklevel+1)

    @property
    def mode(self):
        return self._h5file.mode

    @property
    def filename(self):
        return self._h5file.filename

    @property
    def parent(self):
        return None

    def flush(self):
        if 'r' not in self._mode:
            self._create_dim_scales()
            self._attach_dim_scales()
            if not self._preexisting_file and self._write_ncproperties:
                self.attrs._h5attrs['_NCProperties'] = _NC_PROPERTIES
    sync = flush

    def close(self):
        if not self._closed:
            self.flush()
            self._h5file.close()
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
        return '\n'.join([header] + self._repr_body())
