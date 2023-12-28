from collections.abc import Mapping


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


def _get_cached_properties(cls):
    return [attr for attr in cls.__dir__() if attr.startswith("_cached")]


def _clear_class_caches(cls):
    attrs = _get_cached_properties(cls)
    for attr in attrs:
        try:
            delattr(cls, attr)
        except AttributeError:
            pass
