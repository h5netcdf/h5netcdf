from collections.abc import Mapping

import numpy as np


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


def _create_enum_dataset(group, name, shape, enum_type, fillvalue=None):
    """
    Create a dataset with a transient enum dtype.

    group    : h5netcdf.Group
    name     : str, dataset name
    shape    : tuple, dataset shape
    enum_type: h5netcdf EnumType
    fillvalue : optional scalar fill value
    """
    _h5py = group._root._h5py

    # copy from existing committed type
    enum_tid = enum_type._h5ds.id.copy()
    space = _h5py.h5s.create_simple(shape)

    dcpl = _h5py.h5p.create(_h5py.h5p.DATASET_CREATE)
    if fillvalue is not None:
        dcpl.set_fill_value(np.array(fillvalue, dtype=enum_type.dtype))

    _h5py.h5d.create(
        group._h5group.id, name.encode("ascii"), enum_tid, space, dcpl=dcpl
    )
    enum_tid.close()


def _create_enum_dataset_attribute(ds, name, value, enum_type):
    """
    Create an enum attribute at the dataset.

    ds    : h5netcdf.Variable
    name     : str, dataset name
    enum_type: h5netcdf EnumType
    """
    _h5py = ds._root._h5py

    tid = enum_type._h5ds.id.copy()
    space = _h5py.h5s.create_simple((1,))

    aid = _h5py.h5a.create(ds._h5ds.id, name.encode("ascii"), tid, space)
    aid.write(value, mtype=aid.get_type())


def _make_enum_tid(obj, enum_dict, basetype):
    """
    group    : h5netcdf object
    enum_dict: dict, with Enum field/value pairs
    basetype: np.dtype, basetype of the enum
    """
    items = sorted(enum_dict.items(), key=lambda kv: kv[1])  # sort by value
    base_tid = obj._root._h5py.h5t.py_create(np.dtype(basetype))
    tid = obj._root._h5py.h5t.enum_create(base_tid)
    for name, val in items:
        tid.enum_insert(name.encode("utf-8"), int(val))
    return tid


def _commit_enum_type(group, name, enum_dict, basetype):
    """
    Commit an enum type to the given group.

    group    : h5netcdf.Group
    name     : str, dataset name
    enum_dict: dict, with Enum field/value pairs
    basetype: np.dtype, basetype of the enum
    """
    tid = _make_enum_tid(group, enum_dict, basetype)
    tid.commit(group._h5group.id, name.encode("ascii"))
    tid.close()


def h5dump(fn: str, dataset=None, strict=False):
    """Call h5dump on an h5netcdf file."""
    import re
    import subprocess

    arglist = ["h5dump", "-A"]
    if dataset is not None:
        arglist.append(f"-d {dataset}")
    arglist.append(fn)

    out = subprocess.run(arglist, check=False, capture_output=True).stdout.decode()

    # Strip non-deterministic components
    out = re.sub(r"DATASET [0-9]+ ", "DATASET XXXX ", out)

    # Strip the _NCProperties header, which includes software versions which won't match.
    pattern = (
        r'ATTRIBUTE "_NCProperties"'  # match the attribute start
        r"\s*{"  # opening brace
        r"(?:[^{}]*{[^{}]*}[^{}]*)*"  # match multiple inner braces
        r"}"  # closing brace
    )
    out = re.sub(
        pattern,
        'ATTRIBUTE "_NCProperties" { ... }',
        out,
        flags=re.DOTALL,
    )

    if not strict:
        out = re.sub(r"STRPAD H5T_STR_NULL(?:TERM|PAD);", "STRPAD { ... };", out)
        out = re.sub(r"CSET H5T_CSET_(?:UTF8|ASCII);", "CSET { ... };", out)

    return out
