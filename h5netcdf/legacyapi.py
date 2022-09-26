import sys

import h5py
import numpy as np

from . import core


#: default netcdf fillvalues
default_fillvals = {
    "S1": "\x00",
    "i1": -127,
    "u1": 255,
    "i2": -32767,
    "u2": 65535,
    "i4": -2147483647,
    "u4": 4294967295,
    "i8": -9223372036854775806,
    "u8": 18446744073709551614,
    "f4": 9.969209968386869e36,
    "f8": 9.969209968386869e36,
}


def _get_default_fillvalue(dtype):
    kind = np.dtype(dtype).kind
    fillvalue = None
    if kind in ["u", "i", "f"]:
        size = np.dtype(dtype).itemsize
        fillvalue = default_fillvals[f"{kind}{size}"]
    return fillvalue


def _check_return_dtype_endianess(endian="native"):
    little_endian = sys.byteorder == "little"
    endianess = "="
    if endian == "little":
        endianess = little_endian and endianess or "<"
    elif endian == "big":
        endianess = not little_endian and endianess or ">"
    elif endian == "native":
        pass
    else:
        raise ValueError(
            "'endian' keyword argument must be 'little','big' or 'native', got '%s'"
            % endian
        )
    return endianess


class HasAttributesMixin(object):
    _initialized = False

    def getncattr(self, name):
        """Retrieve a netCDF4 attribute."""
        return self.attrs[name]

    def setncattr(self, name, value):
        """Set a netCDF4 attribute."""
        self.attrs[name] = value

    def ncattrs(self):
        """Return netCDF4 attribute names."""
        return list(self.attrs)

    def __getattr__(self, name):
        try:
            return self.attrs[name]
        except KeyError:
            raise AttributeError(
                "NetCDF: attribute {0}:{1} not found".format(type(self).__name__, name)
            )

    def __setattr__(self, name, value):
        if self._initialized and name not in self.__dict__:
            self.attrs[name] = value
        else:
            object.__setattr__(self, name, value)


class Variable(core.BaseVariable, HasAttributesMixin):
    _cls_name = "h5netcdf.legacyapi.Variable"

    def chunking(self):
        """Return variable chunking information.

        The chunksize is returned as a sequence with the size for each dimension.
        If the dataset is defined to be contiguous (no chunking) the word 'contiguous'
        is returned.
        """
        chunks = self._h5ds.chunks
        if chunks is None:
            return "contiguous"
        else:
            return chunks

    def filters(self):
        """Return HDF5 filter parameters dictionary."""
        complevel = self._h5ds.compression_opts
        return {
            "complevel": 0 if complevel is None else complevel,
            "fletcher32": self._h5ds.fletcher32,
            "shuffle": self._h5ds.shuffle,
            "zlib": self._h5ds.compression == "gzip",
        }

    @property
    def dtype(self):
        """Return netCDF4.Variable datatype."""
        dt = self._h5ds.dtype
        if h5py.check_dtype(vlen=dt) is str:
            return str
        return dt


class Group(core.Group, HasAttributesMixin):
    _cls_name = "h5netcdf.legacyapi.Group"
    _variable_cls = Variable

    @property
    def _group_cls(self):
        return Group

    createGroup = core.Group.create_group

    def createDimension(self, name, size):
        """Creates a new dimension with given name and size.

        Parameters
        ----------
        name : str
            Dimension name
        size : int, None
            size must be a positive integer or None (unlimited).
            Specifying size=0 results in an unlimited dimension too.

        Returns
        -------
        dim : h5netcdf.legacyapi.Dimension
            Dimension class instance.
        """
        self._dimensions[name] = size
        return self._dimensions[name]

    def createVariable(
        self,
        varname,
        datatype,
        dimensions=(),
        zlib=False,
        complevel=4,
        shuffle=True,
        fletcher32=False,
        chunksizes=None,
        fill_value=None,
        endian="native",
    ):
        """Creates a new variable.

        Parameters
        ----------
        varname : str
            Name of the new variable. If given as a path, intermediate groups will be created,
            if not existent.
        datatype : numpy.dtype, str
            Dataype of the new variable
        dimensions : tuple
            Tuple containing dimension name strings. Defaults to empty tuple, effectively
            creating a scalar variable.
        zlib : bool, optional
            If ``True``, variable data will be gzip compressed.
        complevel : int, optional
            Integer between 1 and 9 defining compression level. Defaults to 4.
            Ignored if ``zlib=False``.
        shuffle : bool, optional
            If ``True``, HDF5 shuffle filter will be applied. Defaults to ``True``.
            Ignored if ``zlib=False``.
        fletcher32 : bool, optional
            If ``True``, HDF5 Fletcher32 checksum algorithm is applied. Defaults to ``False``.
        chunksizes : tuple, optional
            Tuple of integers specifying the chunksizes of each variable dimension.
            Discussion on ``h5netcdf`` chunksizes can be found in (:issue:`52`) and (:pull:`127`).
        fill_value : scalar, optional
            Specify ``_FillValue`` for uninitialized parts of the variable. Defaults to ``None``.
        endian : str, optional
            Control on-disk storage format.
            Can be any of ``little``, ``big`` or ``native`` (default).

        Returns
        -------
        var : h5netcdf.legacyapi.Variable
            Variable class instance
        """
        if len(dimensions) == 0:  # it's a scalar
            # rip off chunk and filter options for consistency with netCDF4-python

            chunksizes = None
            zlib = False
            fletcher32 = False
            shuffle = False

        if datatype is str:
            datatype = h5py.special_dtype(vlen=str)

        kwds = {}
        if zlib:
            # only add compression related keyword arguments if relevant (h5py
            # chokes otherwise)
            kwds["compression"] = "gzip"
            kwds["compression_opts"] = complevel
            kwds["shuffle"] = shuffle

        # control endian-ess
        endianess = _check_return_dtype_endianess(endian)
        # needs swapping?
        if endianess != "=":
            # transform to numpy dtype and swap endianess
            dtype = np.dtype(datatype)
            if dtype.byteorder != "|":
                datatype = dtype.newbyteorder("S")

        # closer to netCDF4 chunking behavior
        kwds["chunking_heuristic"] = "h5netcdf"

        return super(Group, self).create_variable(
            varname,
            dimensions,
            dtype=datatype,
            fletcher32=fletcher32,
            chunks=chunksizes,
            fillvalue=fill_value,
            **kwds
        )


class Dimension(core.Dimension):
    _cls_name = "h5netcdf.legacyapi.Dimensions"


class Dataset(core.File, Group, HasAttributesMixin):
    _cls_name = "h5netcdf.legacyapi.Dataset"
