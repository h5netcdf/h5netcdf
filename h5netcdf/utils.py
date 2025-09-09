from collections.abc import Mapping
import numpy as np
import h5py

class CompatibilityError(Exception):
    """Raised when using features that are not part of the NetCDF4 API."""


class Frozen(Mapping):
    """Wrapper around an object implementing the mapping interface to make it
    immutable. If you really want to modify the mapping, the mutable version is
    saved under the `_mapping` attribute.
    """

    def __init__(self, mapping):
        self._mapping = mapping

    def __getitem__(self, key):
        return self._mapping[key]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def __contains__(self, key):
        return key in self._mapping

    def __repr__(self):
        return f"{type(self).__name__}({self._mapping!r})"


def write_classic_string_attr(gid, name, value):
    """Write a string attribute to an HDF5 object with control over the strpad."""
    # Convert to bytes
    if isinstance(value, str):
        value = value.encode("utf-8")

    tid = h5py.h5t.C_S1.copy()
    tid.set_size(len(value))
    tid.set_strpad(h5py.h5t.STR_NULLTERM)
    sid = h5py.h5s.create(h5py.h5s.SCALAR)
    value = np.array(np.bytes_(value))
    if h5py.h5a.exists(gid, name.encode()):
        h5py.h5a.delete(gid, name.encode())
    aid = h5py.h5a.create(gid, name.encode(), tid, sid)
    aid.write(value, mtype=aid.get_type())
