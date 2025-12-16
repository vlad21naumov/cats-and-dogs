import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports(request):
    dummy = request.config.getoption("--dummy")
    if dummy:
        pytest.skip("Dummy")

    from cats_and_dogs.pl_modules.classifiers import ConvClassifier, SimpleClassifier
    from cats_and_dogs.pl_modules.data import MyDataModule
    from cats_and_dogs.pl_modules.model import ImageClassifier

    assert True, "Problems with imports!"
