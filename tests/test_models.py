import pytest
from flaretorch import (
    ResNetMCD,
    ResNetQR,
    LaplaceWrapper,
    CQRWrapper,
    CPWrapper,
    __version__,
)


def test_version():
    assert __version__ == "0.1.0"


def test_import_models():
    assert ResNetMCD is not None
    assert ResNetQR is not None


def test_import_wrappers():
    assert LaplaceWrapper is not None
    assert CQRWrapper is not None
    assert CPWrapper is not None


def test_resnet_mcd_instantiation():
    # Basic check if class can be referenced and has expected methods
    assert hasattr(ResNetMCD, "forward")
    assert hasattr(ResNetMCD, "training_step")


def test_resnet_qr_instantiation():
    # Basic check if class can be referenced and has expected methods
    assert hasattr(ResNetQR, "forward")
    assert hasattr(ResNetQR, "training_step")
