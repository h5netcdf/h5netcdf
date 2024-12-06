from importlib.util import find_spec

import pytest

try:
    import requests

    has_h5pyd = find_spec("h5pyd") is not None
    has_hsds = find_spec("hsds") is not None
    with_reqd_pkgs = has_h5pyd and has_hsds
except ImportError:
    with_reqd_pkgs = False


@pytest.fixture(scope="session")
def hsds_up():
    """Check if hsds is running."""
    if with_reqd_pkgs:
        page = requests.get("http://127.0.0.1:5101/about")
        yield page.status_code == 200
    else:
        yield False
