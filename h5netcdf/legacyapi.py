import sys

import h5py
import numpy as np

from . import core


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
        return self.attrs[name]

    def setncattr(self, name, value):
        self.attrs[name] = value

    def ncattrs(self):
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
        chunks = self._h5ds.chunks
        if chunks is None:
            return "contiguous"
        else:
            return chunks

    def filters(self):
        complevel = self._h5ds.compression_opts
        return {
            "complevel": 0 if complevel is None else complevel,
            "fletcher32": self._h5ds.fletcher32,
            "shuffle": self._h5ds.shuffle,
            "zlib": self._h5ds.compression == "gzip",
        }

    @property
    def dtype(self):
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
    createDimension = core.Group._create_dimension

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

        return super(Group, self).create_variable(
            varname,
            dimensions,
            dtype=datatype,
            fletcher32=fletcher32,
            chunks=chunksizes,
            fillvalue=fill_value,
            **kwds
        )


class Dataset(core.File, Group, HasAttributesMixin):
    _cls_name = "h5netcdf.legacyapi.Dataset"
