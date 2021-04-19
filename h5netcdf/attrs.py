from collections.abc import MutableMapping

import numpy as np

_HIDDEN_ATTRS = frozenset(
    [
        "REFERENCE_LIST",
        "CLASS",
        "DIMENSION_LIST",
        "NAME",
        "_Netcdf4Dimid",
        "_Netcdf4Coordinates",
        "_nc3_strict",
        "_NCProperties",
    ]
)


class Attributes(MutableMapping):
    def __init__(self, h5attrs, check_dtype):
        self._h5attrs = h5attrs
        self._check_dtype = check_dtype

    def __getitem__(self, key):
        import h5py

        if key in _HIDDEN_ATTRS:
            raise KeyError(key)
        # see https://github.com/h5netcdf/h5netcdf/issues/94 for details
        if isinstance(self._h5attrs[key], h5py.Empty):
            string_info = h5py.check_string_dtype(self._h5attrs[key].dtype)
            if string_info and string_info.length == 1:
                return b""
        return self._h5attrs[key]

    def __setitem__(self, key, value):
        if key in _HIDDEN_ATTRS:
            raise AttributeError("cannot write attribute with reserved name %r" % key)
        if hasattr(value, "dtype"):
            dtype = value.dtype
        else:
            dtype = np.asarray(value).dtype
        self._check_dtype(dtype)
        self._h5attrs[key] = value

    def __delitem__(self, key):
        del self._h5attrs[key]

    def __iter__(self):
        for key in self._h5attrs:
            if key not in _HIDDEN_ATTRS:
                yield key

    def __len__(self):
        hidden_count = sum(1 if attr in self._h5attrs else 0 for attr in _HIDDEN_ATTRS)
        return len(self._h5attrs) - hidden_count

    def __repr__(self):
        return "\n".join(
            ["%r" % type(self)] + ["%s: %r" % (k, v) for k, v in self.items()]
        )
