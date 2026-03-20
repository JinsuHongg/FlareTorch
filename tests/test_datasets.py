import pytest
from flaretorch import FlareHelioviewerRegDataset, FlareSuryaClsDataset


def test_import_datasets():
    assert FlareHelioviewerRegDataset is not None
    assert FlareSuryaClsDataset is not None


def test_flare_helioviewer_reg_dataset_methods():
    # Basic check if class can be referenced and has expected methods
    assert hasattr(FlareHelioviewerRegDataset, "__len__")
    assert hasattr(FlareHelioviewerRegDataset, "__getitem__")


def test_flare_surya_cls_dataset_methods():
    # Basic check if class can be referenced and has expected methods
    assert hasattr(FlareSuryaClsDataset, "__len__")
    assert hasattr(FlareSuryaClsDataset, "__getitem__")
