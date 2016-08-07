import sys

PY2 = sys.version_info[0] < 3

if PY2:
    unicode = unicode
else:
    unicode = str


try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

if sys.version_info < (3, 4):
    # we need the optional argument to ChainMap.new_child
    from ._chainmap import ChainMap
else:
    from collections import ChainMap

