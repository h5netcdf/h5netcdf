from . import core


class HasAttributesMixin(object):
    _initialized = False

    def getncattr(self, name):
        return self.attrs[name]

    def setncattr(self, name, value):
        self.attrs[name] = value

    def ncattrs(self):
        return list(self.attrs)

    def __getattr__(self, name):
        return self.attrs[name]

    def __setattr__(self, name, value):
        if self._initialized and name not in self.__dict__:
            self.attrs[name] = value
        else:
            object.__setattr__(self, name, value)


class Variable(core.Variable, HasAttributesMixin):
    pass


class Group(core.Group, HasAttributesMixin):
    _variable_cls = Variable

    @property
    def _group_cls(self):
        return Group

    createGroup = core.Group.create_group

    def createVariable(self, varname, datatype, dimensions=(), zlib=False,
                       complevel=4, shuffle=True, fletcher32=False,
                       chunksizes=None, fill_value=None):
        kwds = {}
        if zlib:
            # only add compression related keyword arguments if relevant (h5py
            # chokes otherwise)
            kwds['compression'] = 'gzip'
            kwds['compression_ops'] = complevel
            kwds['shuffle'] = shuffle

        return super(Group, self).create_variable(
            varname, dimensions, dtype=datatype, fletcher32=fletcher32,
            chunks=chunksizes, fillvalue=fill_value, **kwds)


class Dataset(core.Dataset, Group, HasAttributesMixin):
    createDimension = core.Dataset.create_dimension
