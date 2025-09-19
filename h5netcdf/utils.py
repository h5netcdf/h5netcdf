from collections.abc import Mapping

import h5py
import numpy as np


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


def _create_classic_string_dataset(gid, name, value, shape, chunks):
    """Write a string dataset to an HDF5 object with control over the strpad.

    Parameters
    ----------
    gid : h5py.h5g.GroupID
        Group ID where to write the dataset.
    name : str
        Dataset name.
    value : str
        Dataset contents to be written.
    shape : tuple
        Dataset shape.
    chunks : tuple or None
        Chunk shape.
    """
    # Todo: This function need to be re-checked!
    # Convert to bytes
    if isinstance(value, str):
        value = value.encode("utf-8")

    tid = h5py.h5t.C_S1.copy()
    tid.set_size(1)
    tid.set_strpad(h5py.h5t.STR_NULLTERM)
    kwargs = {}
    if len(shape) == 0:
        sid = h5py.h5s.create(h5py.h5s.SCALAR)
    else:
        if chunks is not None:
            # for resizing, we need to provide maxshape
            # with unlimited as the first dimension
            maxshape = (h5py.h5s.UNLIMITED,) + shape[1:]
            # and we also need to create a chunked dataset
            dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            # try automatic chunking
            dcpl.set_chunk(chunks)
            kwargs["dcpl"] = dcpl
        else:
            maxshape = shape
        sid = h5py.h5s.create_simple(shape, maxshape)
    did = h5py.h5d.create(gid, name.encode(), tid, sid, **kwargs)
    if value is not None:
        value = np.array(np.bytes_(value))
        did.write(h5py.h5s.ALL, h5py.h5s.ALL, value, mtype=did.get_type())


def _create_enum_dataset(group, name, shape, enum_type, fillvalue=None):
    """Create a dataset with a transient enum dtype.

    Parameters
    ----------
    group : h5netcdf.Group
    name : str
        dataset name
    shape : tuple
        dataset shape
    enum_type : h5netcdf.EnumType

    Keyword arguments
    -----------------
    fillvalue : optional scalar fill value
    """
    # copy from existing committed type
    enum_tid = enum_type._h5ds.id.copy()
    space = h5py.h5s.create_simple(shape)

    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    if fillvalue is not None:
        dcpl.set_fill_value(np.array(fillvalue, dtype=enum_type.dtype))

    h5py.h5d.create(group._h5group.id, name.encode("ascii"), enum_tid, space, dcpl=dcpl)
    enum_tid.close()


def _create_enum_dataset_attribute(ds, name, value, enum_type):
    """Create an enum attribute at the dataset.

    Parameters
    ----------
    ds : h5netcdf.Variable
    name : str
        dataset name
    value : array of ints
        enum values
    enum_type : h5netcdf.EnumType
    """
    tid = enum_type._h5ds.id.copy()
    space = h5py.h5s.create_simple((1,))

    aid = h5py.h5a.create(ds._h5ds.id, name.encode("ascii"), tid, space)
    aid.write(value, mtype=aid.get_type())


def _make_enum_tid(enum_dict, basetype):
    """Make enum tid

    Parameters
    ----------
    enum_dict : dict
        dictionary with Enum field/value pairs
    basetype : np.dtype
        basetype of the enum
    """
    items = sorted(enum_dict.items(), key=lambda kv: kv[1])  # sort by value
    base_tid = h5py.h5t.py_create(np.dtype(basetype))
    tid = h5py.h5t.enum_create(base_tid)
    for name, val in items:
        tid.enum_insert(name.encode("utf-8"), int(val))
    return tid


def _commit_enum_type(group, name, enum_dict, basetype):
    """Commit an enum type to the given group.

    Parameters
    ----------
    group : h5netcdf.Group
    name : str
        dataset name
    enum_dict : dict
        dictionary with Enum field/value pairs
    basetype : np.dtype
        basetype of the enum
    """
    tid = _make_enum_tid(enum_dict, basetype)
    tid.commit(group._h5group.id, name.encode("ascii"))
    tid.close()


def _create_string_attribute(gid, name, value):
    """Create a string attribute to an HDF5 object with control over the strpad.

    Parameters
    ----------
    gid : h5py.h5g.GroupID
      Group ID where to write the attribute.
    name : str
        Attribute name.
    value : str
        Attributes contents to be written.
    """
    # handle charset and encoding
    charset = h5py.h5t.CSET_ASCII
    if isinstance(value, str):
        if not value.isascii():
            value = value.encode("utf-8")
            charset = h5py.h5t.CSET_UTF8
        else:
            value = value.encode("ascii")
            charset = h5py.h5t.CSET_ASCII

    # create string type
    tid = h5py.h5t.C_S1.copy()
    tid.set_size(len(value))
    tid.set_strpad(h5py.h5t.STR_NULLTERM)
    tid.set_cset(charset)

    sid = h5py.h5s.create(h5py.h5s.SCALAR)

    if h5py.h5a.exists(gid, name.encode()):
        h5py.h5a.delete(gid, name.encode())

    aid = h5py.h5a.create(gid, name.encode(), tid, sid)
    value = np.array(value)
    aid.write(value, mtype=aid.get_type())


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
