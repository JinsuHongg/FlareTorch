import pytest
from flaretorch import FlareHelioviewerRegDataset, FlareSuryaClsDataset


def test_import_datasets():
    assert FlareHelioviewerRegDataset is not None
    assert FlareSuryaClsDataset is not None


def test_flare_helioviewer_reg_dataset_methods():
    # Basic check if class can be referenced and has expected methods
    assert hasattr(FlareHelioviewerRegDataset, "__len__")
    assert hasattr(FlareHelioviewerRegDataset, "__getitem__")


def test_flare_surya_bench_dataset_target_transform():
    from flaretorch.datasets.flare_cls_datasets import FlareSuryaBenchDataset
    from unittest.mock import MagicMock
    import pandas as pd
    import numpy as np

    # Mock the dataset as much as possible to just test the method
    # Need to pass enough to __init__ to avoid failure, or just mock everything.
    # Actually, simpler to just instantiate an object if I can mock the dependencies.
    # But just calling the method should be enough if I mock the attributes it accesses.
    
    # Actually, just mocking `transform_target` doesn't test the new logic. 
    # I want to test the new logic.
    
    # Create a dummy object to mimic the dataset structure
    class MockDataset:
        def __init__(self, target_norm_type):
            self.target_norm_type = target_norm_type
            self.label_type = "dummy"

    # Instantiate the method by binding it
    obj = MockDataset(target_norm_type="multi_class")
    obj.transform_target = FlareSuryaBenchDataset.transform_target.__get__(obj, MockDataset)

    # Test mapping
    assert obj.transform_target("A1.0") == 0
    assert obj.transform_target("B2.3") == 1
    assert obj.transform_target("C5.0") == 2
    assert obj.transform_target("M1.0") == 3
    assert obj.transform_target("X10.0") == 4
    assert obj.transform_target("FQ") == 0
    assert obj.transform_target(np.nan) == 0
    
    # Test binary/log
    obj.target_norm_type = "binary"
    assert obj.transform_target(1) == 1
    
    obj.target_norm_type = "log"
    # Need to mock label_type for the error message
    obj.label_type = "test"
    np.testing.assert_almost_equal(obj.transform_target(10), 10.0)

