import torch
from flaretorch.metrics import ClassificationUQMetrics

def test_classification_uq_metrics():
    num_classes = 5
    metrics = ClassificationUQMetrics(num_classes=num_classes)
    
    # Mock data: 2 samples, 5 classes
    # Sample 0: {0, 1, 2} (Size 3, Span 3, SFS 0, MDJ 0) -> Contiguous
    # Sample 1: {0, 4} (Size 2, Span 5, SFS 3, MDJ 3) -> Disjoint
    
    prediction_sets = torch.tensor([
        [True, True, True, False, False],
        [True, False, False, False, True]
    ])
    target = torch.tensor([1, 0])
    
    metrics.update(prediction_sets, target)
    results = metrics.compute()
    
    print(results)
    
    assert results["marginal_coverage"] == 1.0 # 0 is in sample 1, 1 is in sample 0
    assert results["avg_set_size"] == 2.5 # (3+2)/2
    assert results["avg_sfs"] == 1.5 # (0+3)/2
    assert results["avg_mdj"] == 1.5 # (0+3)/2
    assert results["ccr"] == 0.5 # 1 contiguous, 1 not

    print("UQ metrics test passed.")

if __name__ == "__main__":
    test_classification_uq_metrics()
