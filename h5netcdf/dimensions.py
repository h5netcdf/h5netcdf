import weakref
from collections import OrderedDict
from collections.abc import MutableMapping

import h5py
import numpy as np
from packaging.version import Version


class Dimensions(MutableMapping):
    def __init__(self, group):
        self._group_ref = weakref.ref(group)
        self._objects = OrderedDict()

    @property
    def _group(self):
        return self._group_ref()

    def __getitem__(self, name):
        return self._objects[name]

    def __setitem__(self, name, size):
        # creating new dimensions
        phony = "phony_dim" in name
        if not self._group._root._writable and not phony:
            raise RuntimeError("H5NetCDF: Write to read only")
        if name in self._objects:
            raise ValueError("dimension %r already exists" % name)

        self._objects[name] = Dimension(self._group, name, size, create_h5ds=True)

    def add(self, name):
        # adding dimensions which are already created in the file
        self._objects[name] = Dimension(self._group, name)

    def __delitem__(self, key):
        raise NotImplementedError("cannot yet delete dimensions")

    def __iter__(self):
        for key in self._objects:
            yield key

    def __len__(self):
        return len(self._objects)

    def __repr__(self):
        if self._group._root._closed:
            return "<Closed h5netcdf.Dimensions>"
        return "<h5netcdf.Dimensions: %s>" % ", ".join(
            "%s=%r" % (k, v) for k, v in self._objects.items()
        )


def _join_h5paths(parent_path, child_path):
    return "/".join([parent_path.rstrip("/"), child_path.lstrip("/")])


class Dimension(object):
    def __init__(self, parent, name, size=None, create_h5ds=False):
        self._parent_ref = weakref.ref(parent)
        self._phony = "phony_dim" in name
        self._root_ref = weakref.ref(parent._root)
        self._h5path = _join_h5paths(parent.name, name)
        self._name = name
        self._size = 0 if size is None else size
        if self._phony:
            self._root._phony_dim_count += 1
        else:
            self._root._max_dim_id += 1
        self._dimensionid = self._root._max_dim_id
        if parent._root._writable and create_h5ds and not self._phony:
            self._create_scale()
        self._initialized = True

    @property
    def _root(self):
        return self._root_ref()

    @property
    def _parent(self):
        return self._parent_ref()

    @property
    def name(self):
        if self._phony:
            return self._name
        return self._h5ds.name.split("/")[-1]

    @property
    def size(self):
        size = len(self)
        if self.isunlimited():
            # return actual dimensions sizes, this is in line with netcdf4-python
            # get sizes from all connected variables and calculate max
            # because netcdf unlimited dimensions can be any length
            # but connected variables dimensions can have a certain larger length.
            reflist = self._h5ds.attrs.get("REFERENCE_LIST", None)
            if reflist is not None:
                for ref, axis in reflist:
                    var = self._parent._h5group["/"][ref]
                    size = max(var.shape[axis], size)
        return size

    def group(self):
        return self._parent

    def isunlimited(self):
        if self._phony:
            return False
        return self._h5ds.maxshape == (None,)

    @property
    def _h5ds(self):
        if self._phony:
            return None
        return self._root._h5file[self._h5path]

    @property
    def _isscale(self):
        return h5py.h5ds.is_scale(self._h5ds.id)

    @property
    def _dimid(self):
        if self._phony:
            return False
        return self._h5ds.attrs.get("_Netcdf4Dimid", self._dimensionid)

    def _resize(self, size):
        from .legacyapi import Dataset

        if not self.isunlimited():
            raise ValueError(
                "Dimension '%s' is not unlimited and thus cannot be resized."
                % self.name
            )
        self._h5ds.resize((size,))

        # resize all referenced datasets for new API
        if not isinstance(self._root, Dataset):
            refs = self._scale_refs
            if refs:
                for var, dim in refs:
                    self._parent._all_h5groups[var].resize(size, dim)

    @property
    def _scale_refs(self):
        """Return dimension scale references"""
        return list(self._h5ds.attrs.get("REFERENCE_LIST", []))

    def _create_scale(self):
        """Create dimension scale for this dimension"""
        if self._name not in self._parent._h5group:
            kwargs = {}
            if self._size is None or self._size == 0:
                kwargs["maxshape"] = (None,)
            self._parent._h5group.create_dataset(
                name=self._name,
                shape=(self._size,),
                dtype=">f4",
                track_order=self._parent._track_order,
                **kwargs,
            )
        self._h5ds.attrs["_Netcdf4Dimid"] = np.array(self._dimid, dtype=np.int32)

        if len(self._h5ds.shape) > 1:
            dims = self._parent._variables[self._name].dimensions
            coord_ids = np.array(
                [self._parent._dimensions[d]._dimid for d in dims], "int32"
            )
            self._h5ds.attrs["_Netcdf4Coordinates"] = coord_ids

        # need special handling for size in case of scalar and tuple
        size = self._size
        if not size:
            size = 1
        if isinstance(size, tuple):
            size = size[0]
        dimlen = bytes(f"{size:10}", "ascii")

        NOT_A_VARIABLE = b"This is a netCDF dimension but not a netCDF variable."
        scale_name = (
            self.name
            if self.name in self._parent._variables
            else NOT_A_VARIABLE + dimlen
        )
        # don't re-create scales if they already exist.
        if not h5py.h5ds.is_scale(self._h5ds.id):
            if Version(h5py.__version__) < Version("2.10.0"):
                self._h5ds.dims.create_scale(self._h5ds, scale_name)
            else:
                self._h5ds.make_scale(scale_name)

    def _attach_scale(self, refs):
        """Attach dimension scale to references"""
        for var, dim in refs:
            self._parent._all_h5groups[var].dims[dim].attach_scale(self._h5ds)

    def _detach_scale(self):
        """Detach dimension scale from all references"""
        refs = self._scale_refs
        if refs:
            for var, dim in refs:
                self._parent._all_h5groups[var].dims[dim].detach_scale(self._h5ds)

    @property
    def _maxsize(self):
        return None if self.isunlimited() else self.size

    def __len__(self):
        if self._phony:
            return self._size
        return self._h5ds.shape[0]

    _cls_name = "h5netcdf.Dimension"

    def __repr__(self):
        if not self._phony and self._parent._root._closed:
            return "<Closed %s>" % self._cls_name
        special = ""
        if self._phony:
            special += " (phony_dim)"
        if self.isunlimited():
            special += " (unlimited)"
        header = "<%s %r: size %s%s>" % (self._cls_name, self.name, self.size, special)
        return "\n".join([header])
