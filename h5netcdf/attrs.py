from collections.abc import MutableMapping
from collections import namedtuple

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

# Local version of h5py.h5t.string_info
_string_info = namedtuple('string_info', ['encoding', 'length'])


class Attributes(MutableMapping):
    def __init__(self, h5attrs, check_dtype, h5py_pckg):
        self._h5attrs = h5attrs
        self._check_dtype = check_dtype
        self._h5py = h5py_pckg

    def __getitem__(self, key):
        if key in _HIDDEN_ATTRS:
            raise KeyError(key)

        # get original attribute via h5py low level api
        # see https://github.com/h5py/h5py/issues/2045
        if self._h5py.__name__ == "h5py":
            attr = self._h5attrs.get_id(key)
        else:
            # pyfive backend
            attr = self._h5attrs[key]

        # handle Empty types
        if isinstance(self._h5attrs[key], self._h5py.Empty):
            # see https://github.com/h5netcdf/h5netcdf/issues/94 for details
            string_info = self._h5py.check_string_dtype(self._h5attrs[key].dtype)
            if string_info and string_info.length == 1:
                return b""
            # see https://github.com/h5netcdf/h5netcdf/issues/154 for details
            else:
                return np.array([], dtype=attr.dtype)

        output = self._h5attrs[key]

        # string decoding subtleties
        # vlen strings are already decoded -> only decode fixed length strings
        # see https://github.com/h5netcdf/h5netcdf/issues/116
        # netcdf4-python returns string arrays as lists, we do as well
        if hasattr(attr, 'dtype'):
            string_info = self._h5py.check_string_dtype(attr.dtype)
        else:
            # A pyfive attribute could be a str or bytes object, which
            # does not have a dtype, so we can't use the
            # 'check_string_dtype' method in this case.
            if isinstance(attr, str):
                string_info = _string_info('utf-8', None)
            elif isinstance(attr, bytes):
                print (999999)
                string_info = _string_info('ascii', None)
            else:
                string_info = None

        if string_info is not None:
            # do not decode "S1"-type char arrays, as they are actually wanted as bytes
            # see https://github.com/Unidata/netcdf4-python/issues/271
            if string_info.length is not None and string_info.length > 1:
                encoding = string_info.encoding
                if np.isscalar(output):
                    output = output.decode(encoding, "surrogateescape")
                else:
                    output = [
                        b.decode(encoding, "surrogateescape") for b in output.flat
                    ]
            else:
                # transform string array to list
                if not np.isscalar(output):
                    output = output.tolist()

        # return item if single element list/array see
        # https://github.com/h5netcdf/h5netcdf/issues/116
        if not np.isscalar(output) and len(output) == 1:
            return output[0]

        return output

    def __setitem__(self, key, value):
        if key in _HIDDEN_ATTRS:
            raise AttributeError(f"cannot write attribute with reserved name {key!r}")
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
        return "\n".join([f"{type(self)!r}"] + [f"{k}: {v!r}" for k, v in self.items()])
