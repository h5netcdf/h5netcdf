# For details on how netCDF4 builds on HDF5:
# https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#netcdf_4_spec
import os
import warnings
import weakref
from collections import ChainMap, Counter, OrderedDict, defaultdict
from collections.abc import Mapping

import numpy as np
from packaging import version
import logging

from . import __version__
from .attrs import Attributes
from .dimensions import Dimension, Dimensions
from .utils import Frozen

try:
    import h5py
except ImportError:
    no_h5py = True
else:
    no_h5py = False

try:
    import h5pyd
except ImportError:
    no_h5pyd = True
else:
    no_h5pyd = False

try:
    import pyfive
except ImportError:
    no_pyfive = True
else:
    no_pyfive = False


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
            f"{feature} are not a supported NetCDF feature, and are not allowed by "
            "h5netcdf unless invalid_netcdf=True."
        )
        raise CompatibilityError(msg)

def _transform_1d_boolean_indexers(key):
    """Find and transform 1D boolean indexers to int"""
    # return key, if not iterable
    try:
        key = [
            (
                np.asanyarray(k).nonzero()[0]
                if isinstance(k, (np.ndarray, list)) and type(k[0]) in (bool, np.bool_)
                else k
            )
            for k in key
        ]
    except TypeError:
        return key

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


def _parse_backend(backend, mode):
    """Parse the 'backend' keyword to File.__init__.

    Parameters
    ----------
    backend : str
        The backend parameter.

    mode : "r", "r+", "a", "w"
        A valid file access mode. Defaults to "r".

    Returns
    -------
    backend: str
        The backend end that is going to used. If the input backend is
        None, then a vlaue of the H5NETCDF_BACKEND environment
        variable is used.

    """
    if backend is None:
        backend = os.environ.get("H5NETCDF_BACKEND", "h5py")

    if backend not in ("pyfive", "h5py"):
        raise ValueError(
            f"Unknown backend {backend!r} - valid options are: "
            "None, 'pyfive', 'h5py'"
        )

    if backend == "pyfive" and no_pyfive:
        raise ImportError("No module named 'pyfive', backend not available")

    if backend == "h5py" and no_h5py:
        raise ImportError("No module named 'h5py', backend not available")

    return backend


class BaseObject:
    def __init__(self, parent, name):
        self._parent_ref = weakref.ref(parent)
        self._root_ref = weakref.ref(parent._root)
        self._h5path = _join_h5paths(parent.name, name)
        self._backend = getattr(parent, "_backend", None)

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
        """Return object name."""
        return self._h5ds.name

    @property
    def dtype(self):
        """Return NumPy dtype giving object’s dtype."""
        return self._h5ds.dtype


_h5type_mapping = {
    "H5T_INTEGER": 0,
    "H5T_FLOAT": 1,
    "H5T_STRING": 3,
    "H5T_COMPOUND": 6,
    "H5T_ENUM": 8,
    "H5T_VLEN": 9,
}


def _get_h5usertype_identifier(h5type):
    """Return H5 Type Identifier from given H5 Datatype."""
    try:
        # h5py first
        h5typeid = h5type.id.get_class()
    except AttributeError:
        # h5pyd second
        h5typeid = _h5type_mapping[h5type.id.type_json["class"]]
    return h5typeid


def _get_h5dstype_identifier(h5type):
    """Return H5 Type Identifier from given H5 Dataset."""
    try:
        # h5py first
        h5typeid = h5type.id.get_type().get_class()
    except AttributeError:
        # h5pyd second
        h5typeid = _h5type_mapping[h5type.id.type_json["class"]]
    return h5typeid


class UserType(BaseObject):
    _cls_name = "h5netcdf.UserType"

    @property
    def name(self):
        """Return user type name."""
        # strip hdf5 path
        return super().name.split("/")[-1]

    def __repr__(self):
        if self._parent._root._closed:
            return f"<Closed {self._cls_name!r}>"
        header = f"<class {self._cls_name!r}: name = {self.name!r}, numpy dtype = {self.dtype!r}"
        return header

    @property
    def _h5type_identifier(self):
        """Returns type identifier.

        See https://api.h5py.org/h5t.html#datatype-class-codes and
        https://docs.hdfgroup.org (enum H5T_class_t)

        """
        return _get_h5usertype_identifier(self._h5ds)

    @property
    def _h5datatype(self):
        """Returns comparable h5type.

        - DatatypeID for h5py
        - (dtype, dtype.metadata) for h5pyd
        """
        if self._root._h5py.__name__ == "h5py":
            return self._h5ds.id
        else:
            return self.dtype, self.dtype.metadata


class EnumType(UserType):
    _cls_name = "h5netcdf.EnumType"

    @property
    def enum_dict(self):
        """Dictionary containing the Enum field/value pairs."""
        return self.dtype.metadata["enum"]

    def __repr__(self):
        return super().__repr__() + f", fields / values = {self.enum_dict!r}"


class VLType(UserType):
    _cls_name = "h5netcdf.VLType"


def _string_to_char_array_dtype(dtype):
    """Converts fixed string to char array dtype."""
    if dtype.kind == "c":
        return None
    return np.dtype(
        {
            name: (
                np.dtype(("S1", fmt.itemsize)) if fmt.kind == "S" else fmt,
                offset,
            )
            for name, (fmt, offset) in dtype.fields.items()
        }
    )


def _char_array_to_string_dtype(dtype):
    """Converts char array to fixed string dtype."""
    if dtype.kind == "c":
        return None
    return np.dtype(
        {
            name: (
                np.dtype(f"S{fmt.shape[0]}") if fmt.base == "S1" else fmt,
                offset,
            )
            for name, (fmt, offset) in dtype.fields.items()
        }
    )


class CompoundType(UserType):
    _cls_name = "h5netcdf.CompoundType"

    @property
    def dtype_view(self):
        return _char_array_to_string_dtype(self.dtype)


class BaseVariable(BaseObject):
    def __init__(self, parent, name, dimensions=None):
        super().__init__(parent, name)
        self._dimensions = dimensions
        self._initialized = True

    @property
    def name(self):
        """Return variable name."""
        # fix name if _nc4_non_coord_
        return super().name.replace("_nc4_non_coord_", "")

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
            # check if malformed variable and raise
            if _unlabeled_dimension_mix(self._h5ds) == "labeled":
                # If a dimension has attached more than one scale for some reason, then
                # take the last one. This is in line with netcdf-c and netcdf4-python.
                return tuple(
                    self._root._h5file[ref[-1]].name.split("/")[-1]
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
                        f"variable {self.name!r} has no dimension scale "
                        f"associated with axis {axis}. \n"
                        f"Use phony_dims='sort' for sorted naming or "
                        f"phony_dims='access' for per access naming."
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

    def _add_fillvalue(self, fillvalue):
        """Add _FillValue attribute.

        This method takes care of adding fillvalue with the wanted
        variable dtype.
        """

        # trying to create correct type of fillvalue
        if self.dtype is str:
            value = fillvalue
        else:
            # todo: this always checks for dtype.metadata
            string_info = self._root._h5py.check_string_dtype(self.dtype)
            enum_info = self._root._h5py.check_enum_dtype(self.dtype)
            if (
                string_info
                and string_info.length is not None
                and string_info.length > 1
            ) or enum_info:
                value = fillvalue
            else:
                value = self.dtype.type(fillvalue)

        self.attrs["_FillValue"] = value

    @property
    def dimensions(self):
        """Return variable dimension names."""
        if self._dimensions is None:
            self._dimensions = self._lookup_dimensions()
        return self._dimensions

    @property
    def shape(self):
        """Return current sizes of all variable dimensions."""
        # return actual dimensions sizes, this is in line with netcdf4-python
        return tuple([self._parent._all_dimensions[d].size for d in self.dimensions])

    @property
    def ndim(self):
        """Return number of variable dimensions."""
        return len(self.shape)

    def __len__(self):
        return self.shape[0]

    @property
    def _h5type_identifier(self):
        """Returns type identifier.

        See https://api.h5py.org/h5t.html#datatype-class-codes and
        https://docs.hdfgroup.org (enum H5T_class_t)

        """
        return _get_h5dstype_identifier(self._h5ds)

    @property
    def _h5datatype(self):
        """Returns comparable h5type.

        This property can be used to compare two variables/datatypes or
        a variable and a datatype for equality of the underlying datatype.

        - DatatypeID for h5py
        - (dtype, dtype.metadata) for h5pyd
        """
        if self._root._h5py.__name__ == "h5py":
            return self._h5ds.id.get_type()
        else:
            return self.dtype, self.dtype.metadata

    @property
    def datatype(self):
        """Return datatype.

        Returns numpy dtype (for primitive types) or VLType/CompoundType/EnumType
        instance (for compound, vlen or enum data types).
        """
        # this is really painful as we have to iterate over all types
        # and check equality
        if self._backend not in (None, "pyfive"):
            usertype = self._parent._get_usertype_dict(self._h5type_identifier)
            if usertype is not None:
                for tid in usertype.values():
                    if self._h5datatype == tid._h5datatype:
                        return tid

        return self.dtype

    def _get_padding(self, key):
        """Return padding if needed, defaults to False."""
        padding = False
        if self.dtype != str and self.dtype.kind in ["f", "i", "u"]:
            key0 = _expanded_indexer(key, self.ndim)
            key0 = _transform_1d_boolean_indexers(key0)
            # extract max shape of key vs hdf5-shape
            h5ds_shape = self._h5ds.shape
            shape = self.shape

            # check for ndarray and list
            # see https://github.com/pydata/xarray/issues/7154
            # first get maximum index
            max_index = [
                max(k) + 1 if isinstance(k, (np.ndarray, list)) else k.stop
                for k in key0
            ]
            # second convert to max shape
            max_shape = tuple(
                [
                    shape[i] if k is None else max(h5ds_shape[i], k)
                    for i, k in enumerate(max_index)
                ]
            )

            # check if hdf5 dataset dimensions are smaller than
            # their respective netcdf dimensions
            sdiff = [d0 - d1 for d0, d1 in zip(max_shape, h5ds_shape)]
            # create padding only if hdf5 dataset is smaller than netcdf dimension
            if sum(sdiff):
                padding = [(0, s) for s in sdiff]
        return padding

    def __array__(self, *args, **kwargs):
        return self._h5ds.__array__(*args, **kwargs)

    def __getitem__(self, key):
        from .legacyapi import Dataset

        if isinstance(self._parent._root, Dataset):
            # this is only for legacyapi
            # fix boolean indexing for affected versions
            # https://github.com/h5py/h5py/pull/2079
            # https://github.com/h5netcdf/h5netcdf/pull/125/
            h5py_version = version.parse(h5py.__version__)
            if version.parse("3.0.0") <= h5py_version < version.parse("3.7.0"):
                key = _transform_1d_boolean_indexers(key)

        if getattr(self._root, "decode_vlen_strings", False):
            string_info = self._root._h5py.check_string_dtype(self._h5ds.dtype)
            if string_info and string_info.length is None:
                if self._backend == "pyfive":
                    # pyfive backend has already dealt with strings
                    return self._h5ds[key]
                else:
                    return self._h5ds.asstr()[key]

        # get padding
        padding = self._get_padding(key)

        # apply padding with fillvalue (both api)
        if padding:
            fv = self.dtype.type(self._h5ds.fillvalue)
            h5ds = np.pad(
                self._h5ds,
                pad_width=padding,
                mode="constant",
                constant_values=fv,
            )
        else:
            h5ds = self._h5ds

        if (
            isinstance(self.datatype, CompoundType)
            and (view := self.datatype.dtype_view) is not None
        ):
            return h5ds[key].view(view)
        else:
            return h5ds[key]

    def __setitem__(self, key, value):
        from .legacyapi import Dataset

        # check if provided values match enumtype values
        if enum_dict := self._root._h5py.check_enum_dtype(self.dtype):
            mask = np.isin(value, list(enum_dict.values()))
            wrong = set(np.asanyarray(value)[~mask])
            if not mask.all():
                raise ValueError(
                    f"Trying to assign illegal value(s) {wrong!r} to Enum variable {self.name!r}."
                    f" Valid values are {dict(enum_dict)!r}."
                )

        if isinstance(self._parent._root, Dataset):
            # resize on write only for legacyapi
            key = _expanded_indexer(key, self.ndim)
            key = _transform_1d_boolean_indexers(key)
            # resize on write only for legacy API
            self._maybe_resize_dimensions(key, value)

        if (
            isinstance(self.datatype, CompoundType)
            and (view := _string_to_char_array_dtype(self.datatype.dtype)) is not None
        ):
            self._h5ds[key] = value.view(view)
        else:
            self._h5ds[key] = value

    @property
    def attrs(self):
        """Return variable attributes."""
        return Attributes(
            self._h5ds.attrs, self._root._check_valid_netcdf_dtype, self._root._h5py
        )

    _cls_name = "h5netcdf.Variable"

    def __repr__(self):
        if self._parent._root._closed:
            return f"<Closed {self._cls_name}>"
        header = f"<{self._cls_name} {self.name!r}: dimensions {self.dimensions}, shape {self.shape}, dtype {self.dtype}>"
        return "\n".join(
            [header]
            + ["Attributes:"]
            + [f"    {k}: {v!r}" for k, v in self.attrs.items()]
        )


class Variable(BaseVariable):
    @property
    def chunks(self):
        if self.shape == ():
            # In HSDS, the layout can be chunked even for scalar datasets, but with only a single chunk.
            # Return None for scalar datasets since they shall be handled as non-chunked.
            assert self._h5ds.chunks in (None, (), (1,))
            return None
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
    # check if dataset has dims and get it
    dimlist = getattr(h5py_dataset, "dims", [])
    if not dimlist:
        status = "nodim"
    else:
        dimset = {len(j) for j in dimlist}
        # either all dimensions have exactly one scale
        # or all dimensions have no scale
        if dimset ^ {0} == set():
            status = "unlabeled"
        elif dimset & {0}:
            name = h5py_dataset.name.split("/")[-1]
            raise ValueError(
                f"malformed variable {name} has mixing of labeled and "
                "unlabeled dimensions."
            )
        else:
            status = "labeled"

    return status


def _check_dtype(group, dtype):
    """Check and handle dtypes when adding variable to given group.

    Raises errors and issues warnings according to given dtype.
    """

    if dtype == np.bool_:
        # never warn since h5netcdf has always errored here
        _invalid_netcdf_feature(
            "boolean dtypes",
            group._root.invalid_netcdf,
        )
    else:
        group._root._check_valid_netcdf_dtype(dtype)

    # we only allow h5netcdf user types, not named h5py.Datatype
    if isinstance(dtype, group._root._h5py.Datatype):
        raise TypeError(
            f"Argument dtype {dtype!r} is not allowed. "
            f"Please provide h5netcdf user type or numpy compatible type."
        )

    # is user type is given extract underlying h5py object
    # we just use the h5py user type here
    if isinstance(dtype, (EnumType, VLType, CompoundType)):
        h5type = dtype._h5ds
        if dtype._root._h5file.filename != group._root._h5file.filename:
            raise TypeError(
                f"Given dtype {dtype} is not committed into current file"
                f" {group._root._h5file.filename}. Instead it's committed into"
                f" file {dtype._root._h5file.filename}"
            )
        # check if committed type can be accessed in current group hierarchy
        user_type = group._get_usertype(h5type)
        if user_type is None:
            msg = (
                f"Given dtype {dtype.name!r} is not accessible in current group"
                f" {group._h5group.name!r} or any parent group. Instead it's defined at"
                f" {h5type.name!r}. Please create it in the current or any parent group."
            )
            raise TypeError(msg)
        # this checks for committed types which are overridden by re-definitions
        elif (actual := user_type._h5ds.name) != h5type.name:
            msg = (
                f"Given dtype {dtype.name!r} is defined at {h5type.name!r}."
                f" Another dtype with same name is defined at {actual!r} and"
                f" would override it."
            )
            raise TypeError(msg)
    elif np.dtype(dtype).kind == "c":
        itemsize = np.dtype(dtype).itemsize
        try:
            width = {8: "FLOAT", 16: "DOUBLE"}[itemsize]
        except KeyError as e:
            raise TypeError(
                "Currently only 'complex64' and 'complex128' dtypes are allowed."
            ) from e
        dname = f"_PFNC_{width}_COMPLEX_TYPE"
        # todo check compound type for existing complex types
        #  which may be used here
        # if dname is not available in current group-path
        # create and commit type in current group
        if dname not in group._all_cmptypes:
            dtype = group.create_cmptype(dtype, dname).dtype

    return dtype


def _check_fillvalue(group, fillvalue, dtype):
    """Handles fillvalues when adding variable to given group.

    Raises errors and issues warnings according to
    given fillvalue and dtype.
    """

    # handling default fillvalues for legacyapi
    # see https://github.com/h5netcdf/h5netcdf/issues/182
    from .legacyapi import Dataset, _get_default_fillvalue

    stacklevel = 5 if isinstance(group._root, Dataset) else 4

    h5fillvalue = fillvalue

    # if no fillvalue is provided take netcdf4 default values for legacyapi
    if fillvalue is None:
        if isinstance(group._root, Dataset):
            h5fillvalue = _get_default_fillvalue(dtype)

    # handling for EnumType
    if dtype is not None and isinstance(dtype, EnumType):
        if fillvalue is None:
            # 1. we need to warn the user that writing enums with default values
            # which are defined in the enum dict will mask those values
            if (h5fillvalue or 0) in dtype.enum_dict.values():
                reverse = {v: k for k, v in dtype.enum_dict.items()}
                msg = (
                    f"Creating variable with default fill_value {h5fillvalue or 0!r}"
                    f" which IS defined in enum type {dtype!r}."
                    f" This will mask entry {{{reverse[h5fillvalue or 0]!r}: {h5fillvalue or 0!r}}}."
                )
                warnings.warn(msg, stacklevel=stacklevel)
            else:
                # 2. we need to raise if the default fillvalue is not within the enum dict
                if (
                    h5fillvalue is not None
                    and h5fillvalue not in dtype.enum_dict.values()
                ):
                    msg = (
                        f"Creating variable with default fill_value {h5fillvalue!r}"
                        f" which IS NOT defined in enum type {dtype!r}."
                        f" Please provide a fitting fill_value or enum type."
                    )
                    raise ValueError(msg)
                if h5fillvalue is None and 0 not in dtype.enum_dict.values():
                    # 3. we should inform the user that a fillvalue of '0'
                    # will be interpreted as _UNDEFINED in netcdf-c
                    # if it is not defined in the enum dict
                    msg = (
                        f"Creating variable with default fill_value {0!r}"
                        f" which IS NOT defined in enum type {dtype!r}."
                        f" Value {0!r} will be interpreted as '_UNDEFINED' by netcdf-c."
                    )
                    warnings.warn(msg, stacklevel=stacklevel)
        else:
            if h5fillvalue not in dtype.enum_dict.values():
                # 4. we should inform the user that a fillvalue of '0'
                # will be interpreted as _UNDEFINED in netcdf-c
                # if it is not defined in the enum dict
                if h5fillvalue == 0:
                    msg = (
                        f"Creating variable with specified fill_value {h5fillvalue!r}"
                        f" which IS NOT defined in enum type {dtype!r}."
                        f" Value {0!r} will be interpreted as '_UNDEFINED' by netcdf-c."
                    )
                    warnings.warn(msg, stacklevel=stacklevel)
                # 5. we need to raise if the fillvalue is not within the enum_dict
                else:
                    msg = (
                        f"Creating variable with specified fill_value {h5fillvalue!r}"
                        f" which IS NOT defined in enum type {dtype!r}."
                        f" Please provide a matching fill_value or enum type."
                    )
                    raise ValueError(msg)

    if fillvalue is not None:
        # cast to wanted type
        fillvalue = np.array(h5fillvalue).astype(dtype)
        h5fillvalue = fillvalue

    return fillvalue, h5fillvalue


class Group(Mapping):
    _variable_cls = Variable
    _dimension_cls = Dimension
    _enumtype_cls = EnumType
    _vltype_cls = VLType
    _cmptype_cls = CompoundType

    @property
    def _group_cls(self):
        return Group

    def __init__(self, parent, name):
        """Create netCDF4 group.

        Groups are containers by which the netCDF4 (HDF5) files are organized.
        Each group is like a Dataset itself.
        """
        self._parent_ref = weakref.ref(parent)
        self._root_ref = weakref.ref(parent._root)
        self._h5path = _join_h5paths(parent._h5path, name)

        self._dimensions = Dimensions(self)
        self._enumtypes = _LazyObjectLookup(self, self._enumtype_cls)
        self._vltypes = _LazyObjectLookup(self, self._vltype_cls)
        self._cmptypes = _LazyObjectLookup(self, self._cmptype_cls)

        # this map keeps track of all dimensions
        if parent is self:
            self._all_dimensions = ChainMap(self._dimensions)
            self._all_enumtypes = ChainMap(self._enumtypes)
            self._all_vltypes = ChainMap(self._vltypes)
            self._all_cmptypes = ChainMap(self._cmptypes)

        else:
            self._all_dimensions = parent._all_dimensions.new_child(self._dimensions)
            self._all_h5groups = parent._all_h5groups.new_child(self._h5group)
            self._all_enumtypes = parent._all_enumtypes.new_child(self._enumtypes)
            self._all_vltypes = parent._all_vltypes.new_child(self._vltypes)
            self._all_cmptypes = parent._all_cmptypes.new_child(self._cmptypes)

        self._variables = _LazyObjectLookup(self, self._variable_cls)
        self._groups = _LazyObjectLookup(self, self._group_cls)

        # initialize phony dimension counter
        if self._root._phony_dims_mode is not None:
            phony_dims = Counter()

        skip_unsupported_hdf5_features = getattr(
            parent, "skip_unsupported_hdf5_features", False
        )
        backend = getattr(parent, "backend", None)

        for k in self._h5group:
            if backend == "pyfive":
                # Some backends might have unsupported HDF5
                # features. Either skip over them, or fail.
                try:
                    v = self._h5group[k]
                except Exception as e:
                    if skip_unsupported_hdf5_features:
                        warnings.warn(
                            f"{backend} backend: Skipping unsupported type "
                            "of HDF5 variable or dimension {k!r}"
                        )
                        continue

                    e.add_note(
                        f"{backend} backend: Found unsupported type of HDF5 "
                        f"variable or dimension {k!r}. Consider setting "
                        "skip_unsupported_hdf5_features=True"
                    )
                    raise e
            else:
                v = self._h5group[k]

            if isinstance(v, self._root._h5py.Group):
                # add to the groups collection if this is a h5py(d) Group
                # instance
                self._groups.add(k)
            elif isinstance(v, self._root._h5py.Datatype):
                # add usertypes (enum, vlen, compound)
                self._add_usertype(v)
            else:
                if v.attrs.get("CLASS") == b"DIMENSION_SCALE":
                    # add dimension and retrieve size
                    self._dimensions.add(k)
                else:
                    if self._root._phony_dims_mode is not None:
                        # check if malformed variable and raise
                        if _unlabeled_dimension_mix(v) == "unlabeled":
                            # if unscaled variable, get phony dimensions
                            phony_dims |= Counter(v.shape)

                if not _netcdf_dimension_but_not_variable(v):
                    if isinstance(v, self._root._h5py.Dataset):
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
                    name = f"phony_dim_{name}"
                    self._dimensions.add_phony(name, size)

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
        if self._root._h5py.__name__ == "h5pyd":
            return False
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
                    raise ValueError(f"cannot modify existing dimension {k!r}")
            else:
                raise ValueError(
                    f"new dimensions do not include existing dimension {k!r}"
                )
        self._dimensions.update(value)

    def _create_child_group(self, name):
        if name in self:
            raise ValueError(f"unable to create group {name!r} (name already exists)")
        kwargs = {}
        kwargs.update(track_order=self._track_order)

        self._h5group.create_group(name, **kwargs)
        self._groups[name] = self._group_cls(self, name)
        return self._groups[name]

    def _require_child_group(self, name):
        try:
            return self._groups[name]
        except KeyError:
            return self._create_child_group(name)

    def create_group(self, name):
        """Create NetCDF4 group.

        Parameters
        ----------
        name : str
            Name of new group.
        """

        if name.startswith("/"):
            return self._root.create_group(name[1:])
        keys = name.split("/")
        group = self
        for k in keys[:-1]:
            group = group._require_child_group(k)
        return group._create_child_group(keys[-1])

    def _create_child_variable(
        self,
        name,
        dimensions,
        dtype,
        data,
        fillvalue,
        chunks,
        chunking_heuristic,
        **kwargs,
    ):
        if name in self:
            raise ValueError(
                f"unable to create variable {name!r} (name already exists)"
            )
        if data is not None:
            data = np.asarray(data)

        if dtype is None:
            dtype = data.dtype

        # check and handle dtypes
        dtype = _check_dtype(self, dtype)

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

        has_unsized_dims = 0 in shape
        if has_unsized_dims and chunks in {None, True}:
            if chunking_heuristic in [None, "h5netcdf"]:
                chunks = _get_default_chunksizes(shape, dtype)
            elif chunking_heuristic == "h5py":
                # do nothing -> h5py will handle chunks internally
                pass
            else:
                raise ValueError(
                    f"got unrecognized value {chunking_heuristic} for chunking_heuristic argument "
                    '(has to be "h5py" or "h5netcdf")'
                )

        # Clear dummy HDF5 datasets with this name that were created for a
        # dimension scale without a corresponding variable.
        # Keep the references, to re-attach later
        refs = None
        if h5name in self._dimensions and h5name in self._h5group:
            refs = self._dimensions[name]._scale_refs
            self._dimensions[name]._detach_scale()
            del self._h5group[name]

        kwargs.update(dict(track_order=self._parent._track_order))

        # fill value handling
        fillvalue, h5fillvalue = _check_fillvalue(self, fillvalue, dtype)

        # create hdf5 variable
        self._h5group.create_dataset(
            h5name,
            shape,
            dtype=dtype,
            data=data,
            chunks=chunks,
            fillvalue=h5fillvalue,
            **kwargs,
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

        # add fillvalue attribute to variable
        if fillvalue is not None:
            variable._add_fillvalue(fillvalue)

        return variable

    def create_variable(
        self,
        name,
        dimensions=(),
        dtype=None,
        data=None,
        fillvalue=None,
        chunks=None,
        chunking_heuristic=None,
        **kwargs,
    ):
        """Creates a new variable.

        Parameters
        ----------
        name : str
            Name of the new variable. If given as a path, intermediate groups will be created,
            if not existent.
        dimensions : tuple
            Tuple containing dimension name strings. Defaults to empty tuple, effectively
            creating a scalar variable.
        dtype : numpy.dtype, str, UserType (Enum, VL, Compound), optional
            Datatype of the new variable. Defaults to None.
        fillvalue : scalar, optional
            Specify fillvalue for uninitialized parts of the variable. Defaults to ``None``.
        chunks : tuple, optional
            Tuple of integers specifying the chunksizes of each variable dimension.
        chunking_heuristic : str, optional
            Specify auto-chunking approach. Can be either of ``h5py`` or ``h5netcdf``. Defaults to
            ``h5netcdf``. Discussion on ``h5netcdf`` chunking can be found in (:issue:`52`)
            and (:pull:`127`).
        compression : str, optional
            Compression filter to apply, defaults to ``gzip``. ``zlib`` is an alias for ``gzip``.
        compression_opts : int
            Parameter for compression filter. For ``compression="gzip"``/``compression="zlib"`` Integer from 1 to 9 specifying
            the compression level. Defaults to 4.
        fletcher32 : bool
            If ``True``, HDF5 Fletcher32 checksum algorithm is applied. Defaults to ``False``.
        shuffle : bool, optional
            If ``True``, HDF5 shuffle filter will be applied. Defaults to ``True``.

        Note
        ----
        Please refer to ``h5py`` `documentation`_ for further parameters via keyword arguments.
        Any parameterizations which do not adhere to netCDF4 standard will only work on files
        created with ``invalid_netcdf=True``,

        .. _documentation: https://docs.h5py.org/en/stable/high/dataset.html#creating-datasets


        Returns
        -------
        var : h5netcdf.Variable
            Variable class instance
        """

        # if root-variable
        if name.startswith("/"):
            # handling default fillvalues for legacyapi
            # see https://github.com/h5netcdf/h5netcdf/issues/182
            from .legacyapi import Dataset, _get_default_fillvalue

            if fillvalue is None and isinstance(self._parent._root, Dataset):
                fillvalue = _get_default_fillvalue(dtype)
            return self._root.create_variable(
                name[1:],
                dimensions,
                dtype,
                data,
                fillvalue,
                chunks,
                chunking_heuristic,
                **kwargs,
            )
        # else split groups and iterate child groups
        keys = name.split("/")
        if not keys[-1]:
            raise ValueError("name parameter cannot be an empty string")
        group = self
        for k in keys[:-1]:
            group = group._require_child_group(k)

        # Allow zlib to be an alias for gzip
        # but use getters and setters so as not to change the behavior
        # of the default h5py functions
        if kwargs.get("compression", None) == "zlib":
            kwargs["compression"] = "gzip"

        return group._create_child_variable(
            keys[-1],
            dimensions,
            dtype,
            data,
            fillvalue,
            chunks,
            chunking_heuristic,
            **kwargs,
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
        yield from self.groups
        yield from self.variables

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

    def _add_usertype(self, h5type):
        """Add usertype to related usertype dict.

        The type is added by name to the dict attached to current group.
        """
        name = h5type.name.split("/")[-1]
        h5typeid = _get_h5usertype_identifier(h5type)
        # add usertype to corresponding dict
        self._get_usertype_dict(h5typeid).maps[0].add(name)

    def _get_usertype(self, h5type):
        """Get usertype from related usertype dict."""
        h5typeid = _get_h5usertype_identifier(h5type)
        return self._get_usertype_dict(h5typeid).get(h5type.name.split("/")[-1])

    def _get_usertype_dict(self, h5typeid):
        """Return usertype-dict related to given h5 type identifier.

        See https://api.h5py.org/h5t.html#datatype-class-codes and
        https://docs.hdfgroup.org (enum H5T_class_t)
        """
        return {
            6: self._all_cmptypes,
            8: self._all_enumtypes,
            9: self._all_vltypes,
        }.get(h5typeid)

    @property
    def enumtypes(self):
        """Return group defined enum types."""
        return Frozen(self._enumtypes)

    @property
    def vltypes(self):
        """Return group defined vlen types."""
        return Frozen(self._vltypes)

    @property
    def cmptypes(self):
        """Return group defined compound types."""
        return Frozen(self._cmptypes)

    @property
    def dims(self):
        return Frozen(self._dimensions)

    @property
    def attrs(self):
        return Attributes(
            self._h5group.attrs, self._root._check_valid_netcdf_dtype, self._root._h5py
        )

    _cls_name = "h5netcdf.Group"

    def _repr_body(self):
        return (
            ["Dimensions:"]
            + [
                "    {}: {}".format(
                    k,
                    (
                        f"Unlimited (current: {self._dimensions[k].size})"
                        if v is None
                        else v
                    ),
                )
                for k, v in self.dimensions.items()
            ]
            + ["Groups:"]
            + [f"    {g}" for g in self.groups]
            + ["Variables:"]
            + [
                f"    {k}: {v.dimensions!r} {v.dtype}"
                for k, v in self.variables.items()
            ]
            + ["Attributes:"]
            + [f"    {k}: {v!r}" for k, v in self.attrs.items()]
        )

    def __repr__(self):
        if self._root._closed:
            return f"<Closed {self._cls_name}>"
        header = f"<{self._cls_name} {self.name!r} ({len(self)} members)>"
        return "\n".join([header] + self._repr_body())

    def resize_dimension(self, dim, size):
        """Resize a dimension to a certain size.

        It will pad with the underlying HDF5 data sets' fill values (usually
        zero) where necessary.
        """
        self._dimensions[dim]._resize(size)

    def create_enumtype(self, datatype, datatype_name, enum_dict):
        """Create EnumType.

        datatype: np.dtype
            A numpy integer dtype object describing the base type for the Enum.
        datatype_name: string
            A Python string containing a description of the Enum data type.
        enum_dict: dict
            A Python dictionary containing the Enum field/value pairs.
        """
        et = self._root._h5py.enum_dtype(enum_dict, basetype=datatype)
        self._h5group[datatype_name] = et
        # create enumtype class instance
        enumtype = self._enumtype_cls(self, datatype_name)
        self._enumtypes[datatype_name] = enumtype
        return enumtype

    def create_vltype(self, datatype, datatype_name):
        """Create VLType.

        datatype: np.dtype
            A numpy dtype object describing the base type.
        datatype_name: string
            A Python string containing a description of the VL data type.
        """
        # wrap in numpy dtype first
        datatype = np.dtype(datatype)
        et = self._root._h5py.vlen_dtype(datatype)
        self._h5group[datatype_name] = et
        # create vltype class instance
        vltype = self._vltype_cls(self, datatype_name)
        self._vltypes[datatype_name] = vltype
        return vltype

    def create_cmptype(self, datatype, datatype_name):
        """Create CompoundType.

        datatype: np.dtype
            A numpy dtype object describing the structured type.
        datatype_name: string
            A Python string containing a description of the compound data type.
        """
        # wrap in numpy dtype first
        datatype = np.dtype(datatype)
        if (new_dtype := _string_to_char_array_dtype(datatype)) is not None:
            # "SN" -> ("S1", (N,))
            datatype = new_dtype
        self._h5group[datatype_name] = datatype
        # create compound class instance
        cmptype = self._cmptype_cls(self, datatype_name)
        self._cmptypes[datatype_name] = cmptype
        return cmptype


class File(Group):
    def __init__(self, path, mode="r", invalid_netcdf=False, phony_dims=None, backend=None, skip_unsupported_hdf5_features=False, **kwargs):
        """NetCDF4 file constructor.

        Parameters
        ----------
        path: path-like
            Location of the netCDF4 file to be accessed, or an h5py File object,
            or a Python file-like object (which should read/write bytes).

        mode: "r", "r+", "a", "w"
            A valid file access mode. Defaults to "r".

        invalid_netcdf: bool
            Allow writing netCDF4 with data types and attributes that would
            otherwise not generate netCDF4 files that can be read by other
            applications. See :ref:`invalid netcdf` for more details.

        phony_dims: 'sort', 'access'
            See :ref:`phony dims` for more details.

        backend: 'pyfive','h5py' or None
            The default backend is h5py (backend=None, or backend=h5py), but
            for reading data, the pure python pyfive backend is available.

        skip_unsupported_hdf5_features: bool
            If True, then skip over types of HDF5 variables or
            dimensions that are not supported by the backend. If False
            (the default) then an exception is raised when such a
            feature is found in the file.

        track_order: bool
            Corresponds to the h5py.File `track_order` parameter. Unless
            specified, the library will choose a default that enhances
            compatibility with netCDF4-c. If h5py version 3.7.0 or greater is
            installed, this parameter will be set to True by default.
            track_order is required to be true to for netCDF4-c libraries to
            append to a file. If an older version of h5py is detected, this
            parameter will be set to False by default to work around a bug in
            h5py limiting the number of attributes for a given variable.
            Ignored when for the 'pyfive' backend.

        **kwargs:
            Additional keyword arguments to be passed to the backend
            file constructor, which is ``h5py.File`` for the 'h5py'
            backend (the default), or ``pyfive.File`` for the 'pyfive'
            backend.

        Notes
        -----
        In h5netcdf version 0.12.0 and earlier, order tracking was disabled in
        HDF5 file. As this is a requirement for the current netCDF4 standard,
        it has been enabled without deprecation as of version 0.13.0 (:issue:`128`).

        Datasets created with h5netcdf version 0.12.0 that are opened with
        newer versions of h5netcdf will continue to disable order tracker.

        If an h5py File object is passed in, closing the h5netcdf wrapper will
        not close the h5py File. In other cases, closing the h5netcdf File object
        does close the underlying file.

        """
        # 2022/01/09
        # netCDF4 wants the track_order parameter to be true
        # through this might be getting relaxed in a more recent version of the
        # standard
        # https://github.com/Unidata/netcdf-c/issues/2054
        # https://github.com/h5netcdf/h5netcdf/issues/128
        # h5py versions less than 3.7.0 had a bug that limited the number of
        # attributes when track_order was set to true by default.
        # However, setting track_order to True helps with compatibility
        # with netcdf4-c and generally, keeping track of how things were added
        # to the dataset.
        # https://github.com/h5netcdf/h5netcdf/issues/136#issuecomment-1017457067
        track_order_default = version.parse(h5py.__version__) >= version.parse("3.7.0")
        track_order = kwargs.pop("track_order", track_order_default)

        self.decode_vlen_strings = kwargs.pop("decode_vlen_strings", None)
        self._close_h5file = True
        self._skip_unsupported_hdf5_features = bool(skip_unsupported_hdf5_features)

        backend = _parse_backend(backend, mode)
        self._backend = backend

        if backend == "pyfive":
            self._h5py = pyfive
            try:
                # We can ignore track_order because pyfive read-only
                self.__h5file = self._h5py.File(path, mode, **kwargs)
                self._preexisting_file = True
            except OSError:
                # pyfive is readonly, we need to raise this error
                self._closed = True
                raise
            except Exception:
                self._closed = True
                raise
            else:
                self._closed = False

        else:
            try:
                if isinstance(path, str):
                    if kwargs.get("driver") == "h5pyd" or (
                        path.startswith(("http://", "https://", "hdf5://"))
                        and "driver" not in kwargs
                    ):
                        if no_h5pyd:
                            raise ImportError(
                                "No module named 'h5pyd'. h5pyd is required for "
                                f"opening urls: {path}"
                            )
                        self._preexisting_file = mode in {"r", "r+", "a"}
                        # remap "a" -> "r+" to check file existence
                        # fallback to "w" if not
                        _mode = mode
                        if mode == "a":
                            mode = "r+"
                        self._h5py = h5pyd
                        try:
                            self.__h5file = self._h5py.File(
                                path, mode, track_order=track_order, **kwargs
                            )
                            self._preexisting_file = mode != "w"
                        except OSError:
                            # if file does not exist, create it
                            if _mode == "a":
                                mode = "w"
                                self.__h5file = self._h5py.File(
                                    path, mode, track_order=track_order, **kwargs
                                )
                                self._preexisting_file = False
                                msg = (
                                    "Append mode for h5pyd now probes with 'r+' first and "
                                    "only falls back to 'w' if the file is missing.\n"
                                    "To silence this warning use 'r+' (open-existing) or 'w' "
                                    "(create-new) directly."
                                )
                                warnings.warn(msg, UserWarning, stacklevel=2)
                            else:
                                raise
                    else:
                        self._preexisting_file = os.path.exists(path) and mode != "w"
                        self._h5py = h5py
                        self.__h5file = self._h5py.File(
                            path, mode, track_order=track_order, **kwargs
                        )
                elif isinstance(path, h5py.File):
                    self._preexisting_file = mode in {"r", "r+", "a"}
                    self._h5py = h5py
                    self.__h5file = path
                    # h5py File passed in: let the caller decide when to close it
                    self._close_h5file = False
                else:  # file-like object
                    self._preexisting_file = mode in {"r", "r+", "a"}
                    self._h5py = h5py
                    self.__h5file = self._h5py.File(
                        path, mode, track_order=track_order, **kwargs
                    )
            except Exception:
                self._closed = True
                raise
            else:
                self._closed = False

        self._filename = self._h5file.filename
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
                    f"unknown value {phony_dims!r} for phony_dims\n"
                    "Use phony_dims='sort' for sorted naming, "
                    "phony_dims='access' for per access naming."
                )

        # string decoding
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
                self.decode_vlen_strings = False

        self._max_dim_id = -1
        # This maps keeps track of all HDF5 datasets corresponding to this group.
        self._all_h5groups = ChainMap(self._h5group)
        super().__init__(self, self._h5path)
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

        if dtype == bool:  # noqa
            description = "boolean"
        elif self._h5py.check_dtype(ref=dtype) is not None:
            description = "reference"
        else:
            description = None

        if description is not None:
            _invalid_netcdf_feature(
                f"{description} dtypes",
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

    @property
    def _root(self):
        return self

    @property
    def backend(self) -> str:
        """The HDF5 backend.

        Returns either "h5py" (the backend is h5py, built on the HDF5
        C library) or "pyfive" (the backend is pyfive, which is pure
        Python).

        """
        return self._backend

    @property
    def skip_unsupported_hdf5_features(self) -> bool:
        """Whether to skip unsupported HDF5 variable or dimension types.

        If True, then types of HDF5 variables or dimensions that are
        not supported by the backend are skipped over. If False (the
        default) then an exception is raised when such a feature is
        found in the file.

        """
        return self._skip_unsupported_hdf5_features

    def flush(self):
        if self._writable:
            # only write `_NCProperties` in newly created files
            if not self._preexisting_file and not self.invalid_netcdf:
                _NC_PROPERTIES = (
                    f"version=2,h5netcdf={__version__},"
                    f"hdf5={self._h5py.version.hdf5_version},"
                    f"{self._h5py.__name__}={self._h5py.__version__}"
                )
                self.attrs._h5attrs["_NCProperties"] = np.array(
                    _NC_PROPERTIES,
                    dtype=self._h5py.string_dtype(
                        encoding="ascii", length=len(_NC_PROPERTIES)
                    ),
                )
            if self.invalid_netcdf:
                # see https://github.com/h5netcdf/h5netcdf/issues/165
                # warn user if .nc file extension is used for invalid netcdf features
                if os.path.splitext(self.filename)[1] == ".nc":
                    msg = (
                        f"You are writing invalid netcdf features to file "
                        f"`{self.filename}`. The file will thus be not conforming "
                        f"to NetCDF-4 standard and might not be readable by other "
                        f"netcdf tools. Consider using a different extension."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
                # remove _NCProperties if invalid_netcdf if exists
                if "_NCProperties" in self.attrs._h5attrs:
                    del self.attrs._h5attrs["_NCProperties"]

    sync = flush

    @property
    def _h5file(self):
        if self._closed:
            raise ValueError(f"I/O operation on {self}: {self._filename!r}")
        return self.__h5file

    def close(self):
        if not self._closed:
            self.flush()
            if self._close_h5file:
                self._h5file.close()
            self.__h5file = None
            self._closed = True

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    _cls_name = "h5netcdf.File"

    def __repr__(self):
        if self._closed:
            return f"<Closed {self._cls_name}>"
        header = (
            f"<{self._cls_name} "
            f"{os.path.basename(self.filename)!r} "
            f"(mode {self.mode}, backend {self.backend})>"
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

    is_unlimited = np.array([x == 0 for x in dimsizes])

    # For unlimited dimensions start with a guess of 1024
    chunks = np.array([x if x != 0 else 1024 for x in dimsizes], dtype="=f8")

    ndims = len(dimsizes)
    if ndims == 0:
        raise ValueError("Chunks not allowed for scalar datasets.")

    if not np.all(np.isfinite(chunks)):
        raise ValueError("Illegal value in chunk tuple")

    # Determine the optimal chunk size in bytes using a PyTables expression.
    # This is kept as a float.
    dset_size = np.prod(chunks[~is_unlimited]) * type_size
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

        chunk_bytes = np.prod(chunks) * type_size

        done = (
            chunk_bytes < target_size
            or abs(chunk_bytes - target_size) / target_size < 0.5
        ) and chunk_bytes < CHUNK_MAX

        if done:
            break

        if np.prod(chunks) == 1:
            break  # Element size larger than CHUNK_MAX

        nelem_unlim = np.prod(chunks[is_unlimited])

        if nelem_unlim == 1 or is_unlimited[idx]:
            chunks[idx] = np.ceil(chunks[idx] / 2.0)

        i += 1

    return tuple(int(x) for x in chunks)
