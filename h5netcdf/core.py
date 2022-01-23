# For details on how netCDF4 builds on HDF5:
# http://www.unidata.ucar.edu/software/netcdf/docs/file_format_specifications.html#netcdf_4_spec
import os.path
import warnings
import weakref
from collections import ChainMap, Counter, OrderedDict, defaultdict
from collections.abc import Mapping

import h5py
import numpy as np
from packaging import version

from .attrs import Attributes
from .dimensions import Dimension, Dimensions
from .utils import Frozen

try:
    import h5pyd
except ImportError:
    no_h5pyd = True
    h5_group_types = (h5py.Group,)
    h5_dataset_types = (h5py.Dataset,)
else:
    no_h5pyd = False
    h5_group_types = (h5py.Group, h5pyd.Group)
    h5_dataset_types = (h5py.Dataset, h5pyd.Dataset)

__version__ = "0.13.0"


_NC_PROPERTIES = "version=2,h5netcdf=%s,hdf5=%s,h5py=%s" % (
    __version__,
    h5py.version.hdf5_version,
    h5py.__version__,
)

NOT_A_VARIABLE = b"This is a netCDF dimension but not a netCDF variable."


def _join_h5paths(parent_path, child_path):
    return "/".join([parent_path.rstrip("/"), child_path.lstrip("/")])


def _name_from_dimension(dim):
    # First value in a dimension is the actual dimension scale
    # which we'll use to extract the name.
    return dim[0].name.split("/")[-1]


class CompatibilityError(Exception):
    """Raised when using features that are not part of the NetCDF4 API."""


def _invalid_netcdf_feature(feature, allow):
    if not allow:
        msg = (
            "{} are not a supported NetCDF feature, and are not allowed by "
            "h5netcdf unless invalid_netcdf=True.".format(feature)
        )
        raise CompatibilityError(msg)


def _transform_1d_boolean_indexers(key):
    """Find and transform 1D boolean indexers to int"""
    key = [
        np.asanyarray(k).nonzero()[0]
        if isinstance(k, (np.ndarray, list)) and type(k[0]) in (bool, np.bool_)
        else k
        for k in key
    ]
    return tuple(key)


def _expanded_indexer(key, ndim):
    """Expand indexing key to tuple with length equal the number of dimensions."""
    # ToDo: restructure this routine to gain more performance
    # short circuit, if we have only slice
    if key is tuple and all(isinstance(k, slice) for k in key):
        return key

    # always return tuple and force colons to slices
    key = np.index_exp[key]

    # dimensions
    len_key = len(key)

    # find Ellipsis
    ellipsis = [i for i, k in enumerate(key) if k is Ellipsis]
    if len(ellipsis) > 1:
        raise IndexError(
            f"an index can only have a single ellipsis ('...'), {len(ellipsis)} given"
        )
    else:
        # expand Ellipsis wherever it is
        len_key -= len(ellipsis)
        res_dim_cnt = ndim - len_key
        res_dims = res_dim_cnt * (slice(None),)
        ellipsis = ellipsis[0] if ellipsis else None

    # check for correct dimensionality
    if ndim and res_dim_cnt < 0:
        raise IndexError(
            f"too many indices for array: array is {ndim}-dimensional, but {len_key} were indexed"
        )

    # convert remaining integer indices to slices
    key = tuple([slice(k, k + 1) if isinstance(k, int) else k for k in key])

    # slices to build resulting key
    k1 = slice(ellipsis)
    k2 = slice(len_key, None) if ellipsis is None else slice(ellipsis + 1, None)
    return key[k1] + res_dims + key[k2]


class BaseVariable(object):
    def __init__(self, parent, name, dimensions=None):
        self._parent_ref = weakref.ref(parent)
        self._root_ref = weakref.ref(parent._root)
        self._h5path = _join_h5paths(parent.name, name)
        self._dimensions = dimensions
        self._initialized = True

    @property
    def _parent(self):
        return self._parent_ref()

    @property
    def _root(self):
        return self._root_ref()

    @property
    def _h5ds(self):
        # Always refer to the root file and store not h5py object
        # subclasses:
        return self._root._h5file[self._h5path]

    @property
    def name(self):
        # fix name if _nc4_non_coord_
        return self._h5ds.name.replace("_nc4_non_coord_", "")

    def _lookup_dimensions(self):
        attrs = self._h5ds.attrs
        # coordinate variable and dimension, eg. 1D ("time") or 2D string variable
        if (
            "_Netcdf4Coordinates" in attrs
            and attrs.get("CLASS", None) == b"DIMENSION_SCALE"
        ):
            order_dim = {
                value._dimid: key for key, value in self._parent._all_dimensions.items()
            }
            return tuple(
                order_dim[coord_id] for coord_id in attrs["_Netcdf4Coordinates"]
            )
        # normal variable carrying DIMENSION_LIST
        # extract hdf5 file references and get objects name
        if "DIMENSION_LIST" in attrs:
            return tuple(
                self._root._h5file[ref[0]].name.split("/")[-1]
                for ref in list(self._h5ds.attrs.get("DIMENSION_LIST", []))
            )

        # need to use the h5ds name here to distinguish from collision dimensions
        child_name = self._h5ds.name.split("/")[-1]
        if child_name in self._parent._all_dimensions:
            return (child_name,)

        dims = []
        phony_dims = defaultdict(int)
        for axis, dim in enumerate(self._h5ds.dims):
            if len(dim):
                name = _name_from_dimension(dim)
            else:
                # if unlabeled dimensions are found
                if self._root._phony_dims_mode is None:
                    raise ValueError(
                        "variable %r has no dimension scale "
                        "associated with axis %s. \n"
                        "Use phony_dims=%r for sorted naming or "
                        "phony_dims=%r for per access naming."
                        % (self.name, axis, "sort", "access")
                    )
                else:
                    # get current dimension
                    dimsize = self._h5ds.shape[axis]
                    # get dimension names
                    dim_names = [
                        d.name
                        # for phony dims we need to look only in the current group
                        for d in self._parent._all_dimensions.maps[0].values()
                        if d.size == dimsize
                    ]
                    # extract wanted dimension name
                    name = dim_names[phony_dims[dimsize]].split("/")[-1]
                    phony_dims[dimsize] += 1
            dims.append(name)
        return tuple(dims)

    def _attach_dim_scales(self):
        """Attach dimension scales"""
        for n, dim in enumerate(self.dimensions):
            # find and attach dimensions also in parent groups
            self._h5ds.dims[n].attach_scale(self._parent._all_dimensions[dim]._h5ds)

    def _attach_coords(self):
        dims = self.dimensions
        # find dimensions also in parent groups
        coord_ids = np.array(
            [self._parent._all_dimensions[d]._dimid for d in dims],
            "int32",
        )
        if len(coord_ids) > 1:
            self._h5ds.attrs["_Netcdf4Coordinates"] = coord_ids

    def _ensure_dim_id(self):
        """Set _Netcdf4Dimid"""
        # set _Netcdf4Dimid, use id of first dimension
        # netCDF4 does this when the first variable's data is written
        if self.dimensions and not self._h5ds.attrs.get("_Netcdf4Dimid", False):
            dim = self._parent._all_h5groups[self.dimensions[0]]
            if "_Netcdf4Dimid" in dim.attrs:
                self._h5ds.attrs["_Netcdf4Dimid"] = dim.attrs["_Netcdf4Dimid"]

    def _maybe_resize_dimensions(self, key, value):
        """Resize according to given (expanded) key with respect to variable dimensions"""
        new_shape = ()
        v = None
        for i, dim in enumerate(self.dimensions):
            # is unlimited dimensions (check in all dimensions)
            if self._parent._all_dimensions[dim].isunlimited():
                if key[i].stop is None:
                    # if stop is None, get dimensions from value,
                    # they must match with variable dimension
                    if v is None:
                        v = np.asarray(value)
                    if v.ndim == self.ndim:
                        new_max = max(v.shape[i], self._h5ds.shape[i])
                    elif v.ndim == 0:
                        # for scalars we take the current dimension size (check in all dimensions
                        new_max = self._parent._all_dimensions[dim].size
                    else:
                        raise IndexError("shape of data does not conform to slice")
                else:
                    new_max = max(key[i].stop, self._h5ds.shape[i])
                # resize unlimited dimension if needed but no other variables
                # this is in line with `netcdf4-python` which only resizes
                # the dimension and this variable
                if self._parent._all_dimensions[dim].size < new_max:
                    self._parent.resize_dimension(dim, new_max)
                new_shape += (new_max,)
            else:
                new_shape += (self._parent._all_dimensions[dim].size,)

        # increase variable size if shape is changing
        if self._h5ds.shape != new_shape:
            self._h5ds.resize(new_shape)

    @property
    def dimensions(self):
        if self._dimensions is None:
            self._dimensions = self._lookup_dimensions()
        return self._dimensions

    @property
    def shape(self):
        # return actual dimensions sizes, this is in line with netcdf4-python
        return tuple([self._parent._all_dimensions[d].size for d in self.dimensions])

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
        from .legacyapi import Dataset

        if isinstance(self._parent._root, Dataset):
            # this is only for legacyapi
            key = _expanded_indexer(key, self.ndim)
            key = _transform_1d_boolean_indexers(key)

        if getattr(self._root, "decode_vlen_strings", False):
            string_info = h5py.check_string_dtype(self._h5ds.dtype)
            if string_info and string_info.length is None:
                return self._h5ds.asstr()[key]

        # return array padded with fillvalue (both api)
        if self.dtype != str and self.dtype.kind in ["f", "i", "u"]:
            sdiff = [d0 - d1 for d0, d1 in zip(self.shape, self._h5ds.shape)]
            if sum(sdiff):
                fv = self.dtype.type(self._h5ds.fillvalue)
                padding = [(0, s) for s in sdiff]
                return np.pad(
                    self._h5ds,
                    pad_width=padding,
                    mode="constant",
                    constant_values=fv,
                )[key]

        return self._h5ds[key]

    def __setitem__(self, key, value):
        from .legacyapi import Dataset

        if isinstance(self._parent._root, Dataset):
            # resize on write only for legacyapi
            key = _expanded_indexer(key, self.ndim)
            key = _transform_1d_boolean_indexers(key)
            # resize on write only for legacy API
            self._maybe_resize_dimensions(key, value)
        self._h5ds[key] = value

    @property
    def attrs(self):
        return Attributes(self._h5ds.attrs, self._root._check_valid_netcdf_dtype)

    _cls_name = "h5netcdf.Variable"

    def __repr__(self):
        if self._parent._root._closed:
            return "<Closed %s>" % self._cls_name
        header = "<%s %r: dimensions %s, shape %s, dtype %s>" % (
            self._cls_name,
            self.name,
            self.dimensions,
            self.shape,
            self.dtype,
        )
        return "\n".join(
            [header]
            + ["Attributes:"]
            + ["    %s: %r" % (k, v) for k, v in self.attrs.items()]
        )


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
        self._parent_ref = weakref.ref(parent)
        self._object_cls = object_cls
        self._objects = OrderedDict()

    @property
    def _parent(self):
        return self._parent_ref()

    def __setitem__(self, name, obj):
        self._objects[name] = obj

    def add(self, name):
        self._objects[name] = None

    def __iter__(self):
        for name in self._objects:
            # fix variable name for variable which clashes with dim name
            yield name.replace("_nc4_non_coord_", "")

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, key):
        # check for _nc4_non_coord_ variable
        if key not in self._objects and "_nc4_non_coord_" + key in self._objects:
            key = "_nc4_non_coord_" + key
        if self._objects[key] is not None:
            return self._objects[key]
        else:
            self._objects[key] = self._object_cls(self._parent, key)
            return self._objects[key]


def _netcdf_dimension_but_not_variable(h5py_dataset):
    return NOT_A_VARIABLE in h5py_dataset.attrs.get("NAME", b"")


def _unlabeled_dimension_mix(h5py_dataset):
    dims = sum([len(j) for j in h5py_dataset.dims])
    if dims:
        if dims != h5py_dataset.ndim:
            name = h5py_dataset.name.split("/")[-1]
            raise ValueError(
                "malformed variable {0} has mixing of labeled and "
                "unlabeled dimensions.".format(name)
            )
    return dims


class Group(Mapping):

    _variable_cls = Variable
    _dimension_cls = Dimension

    @property
    def _group_cls(self):
        return Group

    def __init__(self, parent, name):
        self._parent_ref = weakref.ref(parent)
        self._root_ref = weakref.ref(parent._root)
        self._h5path = _join_h5paths(parent._h5path, name)

        self._dimensions = Dimensions(self)

        # this map keeps track of all dimensions
        if parent is self:
            self._all_dimensions = ChainMap(self._dimensions)
        else:
            self._all_dimensions = parent._all_dimensions.new_child(self._dimensions)
            self._all_h5groups = parent._all_h5groups.new_child(self._h5group)

        self._variables = _LazyObjectLookup(self, self._variable_cls)
        self._groups = _LazyObjectLookup(self, self._group_cls)

        # initialize phony dimension counter
        if self._root._phony_dims_mode is not None:
            phony_dims = Counter()

        for k, v in self._h5group.items():
            if isinstance(v, h5_group_types):
                # add to the groups collection if this is a h5py(d) Group
                # instance
                self._groups.add(k)
            else:
                if v.attrs.get("CLASS") == b"DIMENSION_SCALE":
                    # add dimension and retrieve size
                    self._dimensions.add(k)
                else:
                    if self._root._phony_dims_mode is not None:

                        # check if malformed variable
                        if not _unlabeled_dimension_mix(v):
                            # if unscaled variable, get phony dimensions
                            phony_dims |= Counter(v.shape)

                if not _netcdf_dimension_but_not_variable(v):
                    if isinstance(v, h5_dataset_types):
                        self._variables.add(k)

        # iterate over found phony dimensions and create them
        if self._root._phony_dims_mode is not None:
            # retrieve labeled dims count from already acquired dimensions
            labeled_dims = Counter(
                [d._maxsize for d in self._dimensions.values() if not d._phony]
            )
            for size, cnt in phony_dims.items():
                # only create missing dimensions
                for pcnt in range(labeled_dims[size], cnt):
                    name = self._root._phony_dim_count
                    # for sort mode, we need to add precalculated max_dim_id + 1
                    if self._root._phony_dims_mode == "sort":
                        name += self._root._max_dim_id + 1
                    name = "phony_dim_{}".format(name)
                    self._dimensions[name] = size

        self._initialized = True

    @property
    def _root(self):
        return self._root_ref()

    @property
    def _parent(self):
        return self._parent_ref()

    @property
    def _h5group(self):
        # Always refer to the root file and store not h5py object
        # subclasses:
        return self._root._h5file[self._h5path]

    @property
    def _track_order(self):
        # TODO: make a suggestion to upstream to create a property
        # for files to get if they track the order
        # As of version 3.6.0 this property did not exist
        from h5py.h5p import CRT_ORDER_INDEXED, CRT_ORDER_TRACKED

        gcpl = self._h5group.id.get_create_plist()
        attr_creation_order = gcpl.get_attr_creation_order()
        order_tracked = bool(attr_creation_order & CRT_ORDER_TRACKED)
        order_indexed = bool(attr_creation_order & CRT_ORDER_INDEXED)
        return order_tracked and order_indexed

    @property
    def name(self):
        from .legacyapi import Dataset

        name = self._h5group.name
        # get group name only instead of full path for legacyapi
        if isinstance(self._parent._root, Dataset) and len(name) > 1:
            name = name.split("/")[-1]
        return name

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        for k, v in self._all_dimensions.maps[0].items():
            if k in value:
                if v != value[k]:
                    raise ValueError("cannot modify existing dimension %r" % k)
            else:
                raise ValueError(
                    "new dimensions do not include existing dimension %r" % k
                )
        self._dimensions.update(value)

    def _create_child_group(self, name):
        if name in self:
            raise ValueError("unable to create group %r (name already exists)" % name)
        self._h5group.create_group(name, track_order=self._track_order)
        self._groups[name] = self._group_cls(self, name)
        return self._groups[name]

    def _require_child_group(self, name):
        try:
            return self._groups[name]
        except KeyError:
            return self._create_child_group(name)

    def create_group(self, name):
        if name.startswith("/"):
            return self._root.create_group(name[1:])
        keys = name.split("/")
        group = self
        for k in keys[:-1]:
            group = group._require_child_group(k)
        return group._create_child_group(keys[-1])

    def _create_child_variable(
        self, name, dimensions, dtype, data, fillvalue, chunks, **kwargs
    ):
        if name in self:
            raise ValueError(
                "unable to create variable %r " "(name already exists)" % name
            )

        if data is not None:
            data = np.asarray(data)

        if dtype is None:
            dtype = data.dtype

        if dtype == np.bool_:
            # never warn since h5netcdf has always errored here
            _invalid_netcdf_feature(
                "boolean dtypes",
                self._root.invalid_netcdf,
            )
        else:
            self._root._check_valid_netcdf_dtype(dtype)

        if "scaleoffset" in kwargs:
            _invalid_netcdf_feature(
                "scale-offset filters",
                self._root.invalid_netcdf,
            )

        # maybe create new dimensions depending on data
        if data is not None:
            for d, s in zip(dimensions, data.shape):
                # create new dimensions only ever if
                #  - they are not known via parent-groups
                #  - they are given in dimensions
                #  - it's not a coordinate variable, they will get special handling later
                if d not in self._all_dimensions and d in dimensions and d is not name:
                    # calls _create_dimension
                    self.dimensions[d] = s

        # coordinate variable
        need_dim_adding = False
        if dimensions:
            for dim in dimensions:
                if name not in self._all_dimensions and name == dim:
                    need_dim_adding = True

        # variable <-> dimension name clash
        if name in self._dimensions and (
            name not in dimensions or (len(dimensions) > 1 and dimensions[0] != name)
        ):
            h5name = "_nc4_non_coord_" + name
        else:
            h5name = name

        # get shape from all dimensions
        shape = tuple(self._all_dimensions[d].size for d in dimensions)
        maxshape = tuple(self._all_dimensions[d]._maxsize for d in dimensions if d)

        # If it is passed directly it will change the default compression
        # settings.
        if shape != maxshape:
            kwargs["maxshape"] = maxshape

        if isinstance(chunks, str):
            if chunks == "h5py":
                chunks = True
            elif chunks == "h5netcdf":
                chunks = _get_default_chunksizes(maxshape, dtype)
            else:
                raise ValueError(
                    "got unrecognized string value %s for chunks argument" % chunks
                )

        # Clear dummy HDF5 datasets with this name that were created for a
        # dimension scale without a corresponding variable.
        # Keep the references, to re-attach later
        refs = None
        if h5name in self._dimensions and h5name in self._h5group:
            refs = self._dimensions[name]._scale_refs
            self._dimensions[name]._detach_scale()
            del self._h5group[name]

        # create hdf5 variable
        h5ds = self._h5group.create_dataset(
            h5name,
            shape,
            dtype=dtype,
            data=data,
            chunks=chunks,
            fillvalue=fillvalue,
            track_order=self._track_order,
            **kwargs,
        )

        if chunks in {None, True} and (None in maxshape):
            h5netcdf_chunks = _get_default_chunksizes(maxshape, dtype)
            warnings.warn(
                "Using h5py's default chunking with unlimited dimensions can lead "
                "to increased file sizes and degraded performance (using chunks: %r). "
                'Consider passing ``chunks="h5netcdf"`` (would give chunks: %r; '
                "default in h5netcdf >= 1.0), or set chunk sizes explicitly. "
                'To silence this warning, pass ``chunks="h5py"``. '
                % (h5ds.chunks, h5netcdf_chunks),
                FutureWarning,
            )

        # create variable class instance
        variable = self._variable_cls(self, h5name, dimensions)
        self._variables[h5name] = variable

        # need to put coordinate variable into dimensions
        if need_dim_adding:
            self._dimensions.add(name)

        # Re-create dim-scale and re-attach references to coordinate variable.
        if name in self._all_dimensions and h5name in self._h5group:
            self._all_dimensions[name]._create_scale()
            if refs is not None:
                self._all_dimensions[name]._attach_scale(refs)

        # In case of data variables attach dim_scales and coords.
        if name in self.variables and h5name not in self._dimensions:
            variable._attach_dim_scales()
            variable._attach_coords()

        # This is a bit of a hack, netCDF4 attaches _Netcdf4Dimid to every variable
        # when a variable is first written to, after variable creation.
        # Here we just attach it to every variable on creation.
        # Todo: get this consistent with netcdf-c/netcdf4-python
        variable._ensure_dim_id()

        if fillvalue is not None:
            value = variable.dtype.type(fillvalue)
            variable.attrs._h5attrs["_FillValue"] = value
        return variable

    def create_variable(
        self,
        name,
        dimensions=(),
        dtype=None,
        data=None,
        fillvalue=None,
        chunks=None,
        **kwargs,
    ):
        # if root-variable
        if name.startswith("/"):
            return self._root.create_variable(
                name[1:], dimensions, dtype, data, fillvalue, **kwargs
            )
        # else split groups and iterate child groups
        keys = name.split("/")
        group = self
        for k in keys[:-1]:
            group = group._require_child_group(k)
        return group._create_child_variable(
            keys[-1], dimensions, dtype, data, fillvalue, chunks, **kwargs
        )

    def _get_child(self, key):
        try:
            return self.variables[key]
        except KeyError:
            return self.groups[key]

    def __getitem__(self, key):
        if key.startswith("/"):
            return self._root[key[1:]]
        keys = key.split("/")
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
    def dims(self):
        return Frozen(self._dimensions)

    @property
    def attrs(self):
        return Attributes(self._h5group.attrs, self._root._check_valid_netcdf_dtype)

    _cls_name = "h5netcdf.Group"

    def _repr_body(self):
        return (
            ["Dimensions:"]
            + [
                "    %s: %s"
                % (
                    k,
                    ("Unlimited (current: %s)" % self._dimensions[k].size)
                    if v is None
                    else v,
                )
                for k, v in self.dimensions.items()
            ]
            + ["Groups:"]
            + ["    %s" % g for g in self.groups]
            + ["Variables:"]
            + [
                "    %s: %r %s" % (k, v.dimensions, v.dtype)
                for k, v in self.variables.items()
            ]
            + ["Attributes:"]
            + ["    %s: %r" % (k, v) for k, v in self.attrs.items()]
        )

    def __repr__(self):
        if self._root._closed:
            return "<Closed %s>" % self._cls_name
        header = "<%s %r (%s members)>" % (self._cls_name, self.name, len(self))
        return "\n".join([header] + self._repr_body())

    def resize_dimension(self, dim, size):
        """Resize a dimension to a certain size.

        It will pad with the underlying HDF5 data sets' fill values (usually
        zero) where necessary.
        """
        self._dimensions[dim]._resize(size)


class File(Group):
    def __init__(
        self, path, mode=None, invalid_netcdf=False, phony_dims=None, **kwargs
    ):
        """NetCDF4 file constructor.

        Parameters
        ----------
        path: path-like
            Location of the netCDF4 file to be accessed.

        mode: "r", "r+", "a", "w"
            A valid file access mode. See

        invalid_netcdf: bool
            Allow writing netCDF4 with data types and attributes that would
            otherwise not generate netCDF4 files that can be read by other
            applications. See
            https://github.com/h5netcdf/h5netcdf#invalid-netcdf-files
            for more details.

        phony_dims: 'sort', 'access'
            See:
            https://github.com/h5netcdf/h5netcdf#datasets-with-missing-dimension-scales

        **kwargs:
            Additional keyword arguments to be passed to the ``h5py.File``
            constructor.

        Notes
        -----
        In h5netcdf version 0.12.0 and earlier, order tracking was disabled in
        HDF5 file. As this is a requirement for the current netCDF4 standard,
        it has been enabled without deprecation as of version 0.13.0 [1]_.

        Datasets created with h5netcdf version 0.12.0 that are opened with
        newer versions of h5netcdf will continue to disable order tracker.

        .. [1] https://github.com/h5netcdf/h5netcdf/issues/128

        """
        # 2022/01/09
        # netCDF4 wants the track_order parameter to be true
        # through this might be getting relaxed in a more recent version of the
        # standard
        # https://github.com/Unidata/netcdf-c/issues/2054
        # https://github.com/h5netcdf/h5netcdf/issues/128
        # 2022/01/20: hmaarrfk
        # However, it was found that this causes issues with attrs and h5py
        # https://github.com/h5netcdf/h5netcdf/issues/136
        # https://github.com/h5py/h5py/issues/1385
        track_order = kwargs.pop("track_order", False)

        # When the issues with track_order in h5py are resolved, we
        # can consider uncommenting the code below
        # if not track_order:
        #     self._closed = True
        #     raise ValueError(
        #         f"track_order, if specified must be set to to True (got {track_order})"
        #         "to conform to the netCDF4 file format. Please see "
        #         "https://github.com/h5netcdf/h5netcdf/issues/130 "
        #         "for more details."
        #     )

        # Deprecating mode='a' in favor of mode='r'
        # If mode is None default to 'a' and issue a warning
        if mode is None:
            msg = (
                "Falling back to mode='a'. "
                "In future versions, mode will default to read-only. "
                "It is recommended to explicitly set mode='r' to prevent any unintended "
                "changes to the file."
            )
            warnings.warn(msg, FutureWarning, stacklevel=2)
            mode = "a"

        if version.parse(h5py.__version__) >= version.parse("3.0.0"):
            self.decode_vlen_strings = kwargs.pop("decode_vlen_strings", None)
        try:
            if isinstance(path, str):
                if path.startswith(("http://", "https://", "hdf5://")):
                    if no_h5pyd:
                        raise ImportError(
                            "No module named 'h5pyd'. h5pyd is required for "
                            "opening urls: {}".format(path)
                        )
                    try:
                        with h5pyd.File(path, "r") as f:  # noqa
                            pass
                        self._preexisting_file = True
                    except IOError:
                        self._preexisting_file = False
                    self._h5file = h5pyd.File(
                        path, mode, track_order=track_order, **kwargs
                    )
                else:
                    self._preexisting_file = os.path.exists(path) and mode != "w"
                    self._h5file = h5py.File(
                        path, mode, track_order=track_order, **kwargs
                    )
            else:  # file-like object
                if version.parse(h5py.__version__) < version.parse("2.9.0"):
                    raise TypeError(
                        "h5py version ({}) must be greater than 2.9.0 to load "
                        "file-like objects.".format(h5py.__version__)
                    )
                else:
                    self._preexisting_file = mode in {"r", "r+", "a"}
                    self._h5file = h5py.File(
                        path, mode, track_order=track_order, **kwargs
                    )
        except Exception:
            self._closed = True
            raise
        else:
            self._closed = False

        self._mode = mode
        self._writable = mode != "r"
        self._root_ref = weakref.ref(self)
        self._h5path = "/"
        self.invalid_netcdf = invalid_netcdf

        # phony dimension handling
        self._phony_dims_mode = phony_dims
        if phony_dims is not None:
            self._phony_dim_count = 0
            if phony_dims not in ["sort", "access"]:
                raise ValueError(
                    "unknown value %r for phony_dims\n"
                    "Use phony_dims=%r for sorted naming, "
                    "phony_dims=%r for per access naming."
                    % (phony_dims, "sort", "access")
                )

        # string decoding
        if version.parse(h5py.__version__) >= version.parse("3.0.0"):
            if "legacy" in self._cls_name:
                if self.decode_vlen_strings is not None:
                    msg = (
                        "'decode_vlen_strings' keyword argument is not allowed in h5netcdf "
                        "legacy API."
                    )
                    raise TypeError(msg)
                self.decode_vlen_strings = True
            else:
                if self.decode_vlen_strings is None:
                    msg = (
                        "String decoding changed with h5py >= 3.0. "
                        "See https://docs.h5py.org/en/latest/strings.html and "
                        "https://github.com/h5netcdf/h5netcdf/issues/132 for more details. "
                        "Currently backwards compatibility with h5py < 3.0 is kept by "
                        "decoding vlen strings per default. This will change in future "
                        "versions for consistency with h5py >= 3.0. To silence this "
                        "warning set kwarg ``decode_vlen_strings=False`` which will "
                        "return Python bytes from variables containing vlen strings. Setting "
                        "``decode_vlen_strings=True`` forces vlen string decoding which returns "
                        "Python strings from variables containing vlen strings."
                    )
                    warnings.warn(msg, FutureWarning, stacklevel=2)
                    self.decode_vlen_strings = True

        self._max_dim_id = -1
        # This maps keeps track of all HDF5 datasets corresponding to this group.
        self._all_h5groups = ChainMap(self._h5group)
        super(File, self).__init__(self, self._h5path)
        # get maximum dimension id and count of labeled dimensions
        if self._writable:
            self._max_dim_id = self._get_maximum_dimension_id()
        # initialize all groups to detect/create phony dimensions
        # mimics netcdf-c style naming
        if phony_dims == "sort":
            self._determine_phony_dimensions()

    def _get_maximum_dimension_id(self):
        dimids = []

        def _dimids(name, obj):
            if obj.attrs.get("CLASS", None) == b"DIMENSION_SCALE":
                dimids.append(obj.attrs.get("_Netcdf4Dimid", -1))

        self._h5file.visititems(_dimids)

        return max(dimids) if dimids else -1

    def _determine_phony_dimensions(self):
        def create_phony_dimensions(grp):
            for name in grp.groups:
                create_phony_dimensions(grp[name])

        create_phony_dimensions(self)

    def _check_valid_netcdf_dtype(self, dtype):
        dtype = np.dtype(dtype)

        if dtype == bool:
            description = "boolean"
        elif dtype == complex:
            description = "complex"
        elif h5py.check_dtype(enum=dtype) is not None:
            description = "enum"
        elif h5py.check_dtype(ref=dtype) is not None:
            description = "reference"
        elif h5py.check_dtype(vlen=dtype) not in {None, str, bytes}:
            description = "non-string variable length"
        else:
            description = None

        if description is not None:
            _invalid_netcdf_feature(
                "{} dtypes".format(description),
                self.invalid_netcdf,
            )

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
        if self._writable:
            if not self._preexisting_file and not self.invalid_netcdf:
                self.attrs._h5attrs["_NCProperties"] = np.array(
                    _NC_PROPERTIES,
                    dtype=h5py.string_dtype(
                        encoding="ascii", length=len(_NC_PROPERTIES)
                    ),
                )

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

    _cls_name = "h5netcdf.File"

    def __repr__(self):
        if self._closed:
            return "<Closed %s>" % self._cls_name
        header = "<%s %r (mode %s)>" % (
            self._cls_name,
            self.filename.split("/")[-1],
            self.mode,
        )
        return "\n".join([header] + self._repr_body())


def _get_default_chunksizes(dimsizes, dtype):
    # This is a modified version of h5py's default chunking heuristic
    # https://github.com/h5py/h5py/blob/aa31f03bef99e5807d1d6381e36233325d944279/h5py/_hl/filters.py#L334-L389
    # (published under BSD-3-Clause, included at licenses/H5PY_LICENSE.txt)
    # See also https://github.com/h5py/h5py/issues/2029 for context.

    CHUNK_BASE = 16 * 1024  # Multiplier by which chunks are adjusted
    CHUNK_MIN = 8 * 1024  # Soft lower limit (8k)
    CHUNK_MAX = 1024 * 1024  # Hard upper limit (1M)

    type_size = np.dtype(dtype).itemsize

    is_unlimited = np.array([x is None for x in dimsizes])

    # For unlimited dimensions start with a guess of 1024
    chunks = np.array([x if x is not None else 1024 for x in dimsizes], dtype="=f8")

    ndims = len(dimsizes)
    if ndims == 0:
        raise ValueError("Chunks not allowed for scalar datasets.")

    if not np.all(np.isfinite(chunks)):
        raise ValueError("Illegal value in chunk tuple")

    # Determine the optimal chunk size in bytes using a PyTables expression.
    # This is kept as a float.
    dset_size = np.product(chunks[~is_unlimited]) * type_size
    target_size = CHUNK_BASE * (2 ** np.log10(dset_size / (1024 * 1024)))

    if target_size > CHUNK_MAX:
        target_size = CHUNK_MAX
    elif target_size < CHUNK_MIN:
        target_size = CHUNK_MIN

    i = 0
    while True:
        # Repeatedly loop over the axes, dividing them by 2.
        # Start by reducing unlimited axes first.
        # Stop when:
        # 1a. We're smaller than the target chunk size, OR
        # 1b. We're within 50% of the target chunk size, AND
        #  2. The chunk is smaller than the maximum chunk size

        idx = i % ndims

        chunk_bytes = np.product(chunks) * type_size

        done = (
            chunk_bytes < target_size
            or abs(chunk_bytes - target_size) / target_size < 0.5
        ) and chunk_bytes < CHUNK_MAX

        if done:
            break

        if np.product(chunks) == 1:
            break  # Element size larger than CHUNK_MAX

        nelem_unlim = np.product(chunks[is_unlimited])

        if nelem_unlim == 1 or is_unlimited[idx]:
            chunks[idx] = np.ceil(chunks[idx] / 2.0)

        i += 1

    return tuple(int(x) for x in chunks)
