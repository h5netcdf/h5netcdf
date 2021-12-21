# For details on how netCDF4 builds on HDF5:
# http://www.unidata.ucar.edu/software/netcdf/docs/file_format_specifications.html#netcdf_4_spec
import os.path
import warnings
import weakref
from collections import ChainMap, OrderedDict, defaultdict
from collections.abc import Mapping
from distutils.version import LooseVersion

import h5py
import numpy as np

from .attrs import Attributes
from .dimensions import Dimensions
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

__version__ = "0.12.0"


_NC_PROPERTIES = "version=2,h5netcdf=%s,hdf5=%s,h5py=%s" % (
    __version__,
    h5py.version.hdf5_version,
    h5py.__version__,
)

NOT_A_VARIABLE = b"This is a netCDF dimension but not a netCDF variable."


def _reverse_dict(dict_):
    return dict(zip(dict_.values(), dict_.keys()))


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
        if "_Netcdf4Coordinates" in attrs:
            order_dim = _reverse_dict(self._parent._dim_order)
            return tuple(
                order_dim[coord_id] for coord_id in attrs["_Netcdf4Coordinates"]
            )

        child_name = self.name.split("/")[-1]
        if child_name in self._parent.dimensions:
            return (child_name,)

        dims = []
        phony_dims = defaultdict(int)
        for axis, dim in enumerate(self._h5ds.dims):
            # get current dimension
            dimsize = self.shape[axis]
            phony_dims[dimsize] += 1
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
                    # get dimension name
                    name = self._parent._phony_dims[(dimsize, phony_dims[dimsize] - 1)]
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
        if getattr(self._root, "decode_vlen_strings", False):
            string_info = h5py.check_string_dtype(self._h5ds.dtype)
            if string_info and string_info.length is None:
                return self._h5ds.asstr()[key]
        return self._h5ds[key]

    def __setitem__(self, key, value):
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

    @property
    def _group_cls(self):
        return Group

    def __init__(self, parent, name):
        self._parent_ref = weakref.ref(parent)
        self._root_ref = weakref.ref(parent._root)
        self._h5path = _join_h5paths(parent.name, name)

        if parent is not self:
            self._dim_sizes = parent._dim_sizes.new_child()
            self._current_dim_sizes = parent._current_dim_sizes.new_child()
            self._dim_order = parent._dim_order.new_child()
            self._all_h5groups = parent._all_h5groups.new_child(self._h5group)

        self._variables = _LazyObjectLookup(self, self._variable_cls)
        self._groups = _LazyObjectLookup(self, self._group_cls)

        # # initialize phony dimension counter
        if self._root._phony_dims_mode is not None:
            self._phony_dims = {}
            phony_dims = defaultdict(int)
            labeled_dims = defaultdict(int)

        for k, v in self._h5group.items():
            if isinstance(v, h5_group_types):
                # add to the groups collection if this is a h5py(d) Group
                # instance
                self._groups.add(k)
            else:
                if v.attrs.get("CLASS") == b"DIMENSION_SCALE":
                    dim_id = v.attrs.get("_Netcdf4Dimid")
                    if "_Netcdf4Coordinates" in v.attrs:
                        assert dim_id is not None
                        coord_ids = v.attrs["_Netcdf4Coordinates"]
                        size = v.shape[list(coord_ids).index(dim_id)]
                        current_size = size
                    else:
                        assert len(v.shape) == 1
                        # Unlimited dimensions are represented as None.
                        size = None if v.maxshape == (None,) else v.size
                        current_size = v.size

                    self._dim_sizes[k] = size

                    # keep track of found labeled dimensions
                    if self._root._phony_dims_mode is not None:
                        labeled_dims[size] += 1
                        self._phony_dims[(size, labeled_dims[size] - 1)] = k

                    # Figure out the current size of a dimension, which for
                    # unlimited dimensions requires looking at the actual
                    # variables.
                    self._current_dim_sizes[k] = self._determine_current_dimension_size(
                        k, current_size
                    )

                    self._dim_order[k] = dim_id
                else:
                    if self._root._phony_dims_mode is not None:
                        # check if malformed variable
                        if not _unlabeled_dimension_mix(v):
                            # if unscaled variable, get phony dimensions
                            vdims = defaultdict(int)
                            for i in v.shape:
                                vdims[i] += 1
                            for dimsize, cnt in vdims.items():
                                phony_dims[dimsize] = max(phony_dims[dimsize], cnt)

                if not _netcdf_dimension_but_not_variable(v):
                    if isinstance(v, h5_dataset_types):
                        self._variables.add(k)

        # iterate over found phony dimensions and create them
        if self._root._phony_dims_mode is not None:
            grp_phony_count = 0
            for size, cnt in phony_dims.items():
                # only create missing dimensions
                for pcnt in range(labeled_dims[size], cnt):
                    name = grp_phony_count + self._root._phony_dim_count
                    grp_phony_count += 1
                    if self._root._phony_dims_mode == "access":
                        name = "phony_dim_{}".format(name)
                        self._create_dimension(name, size)
                    self._phony_dims[(size, pcnt)] = name
            # finally increase phony dim count at file level
            self._root._phony_dim_count += grp_phony_count

        self._initialized = True

    @property
    def _root(self):
        return self._root_ref()

    @property
    def _parent(self):
        return self._parent_ref()

    def _create_phony_dimensions(self):
        # this is for 'sort' naming
        for key, value in self._phony_dims.items():
            if isinstance(value, int):
                value += self._root._labeled_dim_count
                name = "phony_dim_{}".format(value)
                self._create_dimension(name, key[0])
                self._phony_dims[key] = name

    def _determine_current_dimension_size(self, dim_name, max_size):
        """
        Helper method to determine the current size of a dimension.
        """
        # Limited dimension.
        if self.dimensions[dim_name] is not None:
            return max_size

        dim_variable = _find_dim(self._h5group, dim_name)

        if "REFERENCE_LIST" not in dim_variable.attrs:
            return max_size

        root = self._h5group["/"]

        for ref, _ in dim_variable.attrs["REFERENCE_LIST"]:
            var = root[ref]

            for i, var_d in enumerate(var.dims):
                name = _name_from_dimension(var_d)
                if name == dim_name:
                    max_size = max(var.shape[i], max_size)
        return max_size

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
            raise ValueError("dimension %r already exists" % name)

        self._dim_sizes[name] = size
        self._current_dim_sizes[name] = 0 if size is None else size
        self._dim_order[name] = None

    @property
    def dimensions(self):
        return Dimensions(self)

    @dimensions.setter
    def dimensions(self, value):
        for k, v in self._dim_sizes.maps[0].items():
            if k in value:
                if v != value[k]:
                    raise ValueError("cannot modify existing dimension %r" % k)
            else:
                raise ValueError(
                    "new dimensions do not include existing " "dimension %r" % k
                )
        self.dimensions.update(value)

    def _create_child_group(self, name):
        if name in self:
            raise ValueError("unable to create group %r (name already exists)" % name)
        self._h5group.create_group(name)
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
        self, name, dimensions, dtype, data, fillvalue, **kwargs
    ):
        if name in self:
            raise ValueError(
                "unable to create variable %r " "(name already exists)" % name
            )

        if data is not None:
            data = np.asarray(data)
            for d, s in zip(dimensions, data.shape):
                if d not in self.dimensions:
                    self.dimensions[d] = s

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

        # variable <-> dimension name clash
        if name in self.dimensions and (
            name not in dimensions or (len(dimensions) > 1 and dimensions[0] != name)
        ):
            h5name = "_nc4_non_coord_" + name
        else:
            h5name = name

        shape = tuple(self._current_dim_sizes[d] for d in dimensions)
        maxshape = tuple(self._dim_sizes[d] for d in dimensions)

        # If it is passed directly it will change the default compression
        # settings.
        if shape != maxshape:
            kwargs["maxshape"] = maxshape

        # Clear dummy HDF5 datasets with this name that were created for a
        # dimension scale without a corresponding variable.
        if h5name in self.dimensions and h5name in self._h5group:
            h5ds = self._h5group[h5name]
            if _netcdf_dimension_but_not_variable(h5ds):
                self._detach_dim_scale(h5name)
                del self._h5group[h5name]

        self._h5group.create_dataset(
            h5name, shape, dtype=dtype, data=data, fillvalue=fillvalue, **kwargs
        )

        self._variables[h5name] = self._variable_cls(self, h5name, dimensions)
        variable = self._variables[h5name]

        if fillvalue is not None:
            value = variable.dtype.type(fillvalue)
            variable.attrs._h5attrs["_FillValue"] = value
        return variable

    def create_variable(
        self, name, dimensions=(), dtype=None, data=None, fillvalue=None, **kwargs
    ):
        if name.startswith("/"):
            return self._root.create_variable(
                name[1:], dimensions, dtype, data, fillvalue, **kwargs
            )
        keys = name.split("/")
        group = self
        for k in keys[:-1]:
            group = group._require_child_group(k)
        return group._create_child_variable(
            keys[-1], dimensions, dtype, data, fillvalue, **kwargs
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

    def _create_dim_scales(self):
        """Create all necessary HDF5 dimension scale."""
        dim_order = self._dim_order.maps[0]
        for dim in sorted(dim_order, key=lambda d: dim_order[d]):
            dimlen = bytes(f"{self._current_dim_sizes[dim]:10}", "ascii")
            scale_name = (
                dim
                if dim in self._variables and dim in self._h5group
                else NOT_A_VARIABLE + dimlen
            )
            if dim not in self._h5group:
                size = self._current_dim_sizes[dim]
                kwargs = {}
                if self._dim_sizes[dim] is None:
                    kwargs["maxshape"] = (None,)
                self._h5group.create_dataset(
                    name=dim, shape=(size,), dtype=">f4", **kwargs
                )

            h5ds = self._h5group[dim]
            h5ds.attrs["_Netcdf4Dimid"] = np.int32(dim_order[dim])

            if len(h5ds.shape) > 1:
                dims = self._variables[dim].dimensions
                coord_ids = np.array([dim_order[d] for d in dims], "int32")
                h5ds.attrs["_Netcdf4Coordinates"] = coord_ids

            if not h5py.h5ds.is_scale(h5ds.id):
                if h5py.__version__ < LooseVersion("2.10.0"):
                    h5ds.dims.create_scale(h5ds, scale_name)
                else:
                    h5ds.make_scale(scale_name)

        for subgroup in self.groups.values():
            subgroup._create_dim_scales()

    def _attach_dim_scales(self):
        """Attach dimension scales to all variables."""
        for name, var in self.variables.items():
            # also attach for _nc4_non_coord_ variables
            if name not in self.dimensions or "_nc4_non_coord_" in var._h5ds.name:
                for n, dim in enumerate(var.dimensions):
                    vards = var._h5ds
                    scale = self._all_h5groups[dim]
                    # attach only, if not already attached
                    if not h5py.h5ds.is_attached(vards.id, scale.id, n):
                        vards.dims[n].attach_scale(scale)

        for subgroup in self.groups.values():
            subgroup._attach_dim_scales()

    def _detach_dim_scale(self, name):
        """Detach the dimension scale corresponding to a dimension name."""
        for var in self.variables.values():
            for n, dim in enumerate(var.dimensions):
                if dim == name:
                    vards = var._h5ds
                    scale = self._all_h5groups[dim]
                    # only detach if attached
                    if h5py.h5ds.is_attached(vards.id, scale.id, n):
                        vards.dims[n].detach_scale(scale)

        for subgroup in self.groups.values():
            if dim not in subgroup._h5group:
                subgroup._detach_dim_scale(name)

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
        return Attributes(self._h5group.attrs, self._root._check_valid_netcdf_dtype)

    _cls_name = "h5netcdf.Group"

    def _repr_body(self):
        return (
            ["Dimensions:"]
            + [
                "    %s: %s"
                % (
                    k,
                    ("Unlimited (current: %s)" % self._current_dim_sizes[k])
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

    def resize_dimension(self, dimension, size):
        """
        Resize a dimension to a certain size.

        It will pad with the underlying HDF5 data sets' fill values (usually
        zero) where necessary.
        """
        if self.dimensions[dimension] is not None:
            raise ValueError(
                "Dimension '%s' is not unlimited and thus "
                "cannot be resized." % dimension
            )

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
    def __init__(
        self, path, mode=None, invalid_netcdf=False, phony_dims=None, **kwargs
    ):
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

        if h5py.__version__ >= LooseVersion("3.0.0"):
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
                    self._h5file = h5pyd.File(path, mode, **kwargs)
                else:
                    self._preexisting_file = os.path.exists(path) and mode != "w"
                    self._h5file = h5py.File(path, mode, **kwargs)
            else:  # file-like object
                if h5py.__version__ < LooseVersion("2.9.0"):
                    raise TypeError(
                        "h5py version ({}) must be greater than 2.9.0 to load "
                        "file-like objects.".format(h5py.__version__)
                    )
                else:
                    self._preexisting_file = mode in {"r", "r+", "a"}
                    self._h5file = h5py.File(path, mode, **kwargs)
        except Exception:
            self._closed = True
            raise
        else:
            self._closed = False

        self._mode = mode
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
        if h5py.__version__ >= LooseVersion("3.0.0"):
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
                        "See https://docs.h5py.org/en/latest/strings.html for more details. "
                        "Currently backwards compatibility with h5py < 3.0 is kept by "
                        "decoding vlen strings per default. This will change in future "
                        "versions for consistency with h5py >= 3.0. To silence this "
                        "warning set kwarg ``decode_vlen_strings=False``. Setting "
                        "``decode_vlen_strings=True`` forces vlen string decoding."
                    )
                    warnings.warn(msg, FutureWarning, stacklevel=2)
                    self.decode_vlen_strings = True

        # These maps keep track of dimensions in terms of size (might be
        # unlimited), current size (identical to size for limited dimensions),
        # their position, and look-up for HDF5 datasets corresponding to a
        # dimension.
        self._dim_sizes = ChainMap()
        self._current_dim_sizes = ChainMap()
        self._dim_order = ChainMap()
        self._all_h5groups = ChainMap(self._h5group)
        super(File, self).__init__(self, self._h5path)
        # initialize all groups to detect/create phony dimensions
        # mimics netcdf-c style naming
        if phony_dims == "sort":
            self._determine_phony_dimensions()

    def _determine_phony_dimensions(self):
        def get_labeled_dimension_count(grp):
            count = len(grp._dim_sizes.maps[0])
            for name in grp.groups:
                count += get_labeled_dimension_count(grp[name])
            return count

        def create_phony_dimensions(grp):
            grp._create_phony_dimensions()
            for name in grp.groups:
                create_phony_dimensions(grp[name])

        self._labeled_dim_count = get_labeled_dimension_count(self)
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

    def _set_unassigned_dimension_ids(self):
        max_dim_id = -1

        # collect the largest assigned dimension ID
        groups = [self]
        while groups:
            group = groups.pop()
            assigned_dim_ids = [
                dim_id for dim_id in group._dim_order.values() if dim_id is not None
            ]
            max_dim_id = max([max_dim_id] + assigned_dim_ids)
            groups.extend(group._groups.values())

        # set all dimension IDs to valid values
        next_dim_id = max_dim_id + 1
        groups = [self]
        while groups:
            group = groups.pop()
            for key in group._dim_order:
                if group._dim_order[key] is None:
                    group._dim_order[key] = next_dim_id
                    next_dim_id += 1
            groups.extend(group._groups.values())

    def flush(self):
        if self._mode != "r":
            self._set_unassigned_dimension_ids()
            self._create_dim_scales()
            self._attach_dim_scales()
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


def _find_dim(h5group, dim):
    if dim not in h5group:
        return _find_dim(h5group.parent, dim)
    return h5group[dim]


