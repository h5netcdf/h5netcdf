from collections.abc import MutableMapping


class Dimensions(MutableMapping):
    def __init__(self, group):
        self._group = group

    def __getitem__(self, key):
        return self._group._dim_sizes[key]

    def __setitem__(self, key, value):
        self._group._create_dimension(key, value)

    def __delitem__(self, key):
        raise NotImplementedError("cannot yet delete dimensions")

    def __iter__(self):
        for key in self._group._dim_sizes:
            yield key

    def __len__(self):
        return len(self._group._dim_sizes)

    def __repr__(self):
        if self._group._root._closed:
            return "<Closed h5netcdf.Dimensions>"
        return "<h5netcdf.Dimensions: %s>" % ", ".join(
            "%s=%r" % (k, v) for k, v in self.items()
        )
