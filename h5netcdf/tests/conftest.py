import pytest
import requests


@pytest.fixture(scope="session")
def hsds_up():
    """Check if hsds is running."""
    page = requests.get("http://127.0.0.1:5101/about")
    yield page.status_code == 200
