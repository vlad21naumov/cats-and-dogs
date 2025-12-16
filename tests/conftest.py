def pytest_addoption(parser):
    parser.addoption(
        "--model-path",
        action="store",
        default=None,
        help="Model path for testing",
    )
    parser.addoption(
        "--dummy",
        action="store",
        default=None,
        help="Dummy",
    )
