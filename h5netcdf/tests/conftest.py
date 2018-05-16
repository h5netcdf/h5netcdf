def pytest_addoption(parser):
    parser.addoption('--restapi', action='store_true', dest="restapi",
                     default=False, help="Enable HDF5 REST API tests")
