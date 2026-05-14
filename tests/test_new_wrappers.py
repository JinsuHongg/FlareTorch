import torch
import pytest
from flaretorch import ClsCPWrapper, APSWrapper, OrdinalAPSWrapper

def test_wrappers_instantiation():
    # Mock model (simple linear layer for 5 classes)
    model = torch.nn.Linear(10, 5)
    
    # Instantiate wrappers
    cls_cp = ClsCPWrapper(model)
    aps = APSWrapper(model)
    ord_aps = OrdinalAPSWrapper(model)
    
    assert cls_cp is not None
    assert aps is not None
    assert ord_aps is not None
    
    # Simple check for forward pass
    x = torch.randn(2, 10)
    assert cls_cp(x).shape == (2, 5)
    assert aps(x).shape == (2, 5)
    assert ord_aps(x).shape == (2, 5)

    print("Instantiation and forward pass check passed.")

if __name__ == "__main__":
    test_wrappers_instantiation()
