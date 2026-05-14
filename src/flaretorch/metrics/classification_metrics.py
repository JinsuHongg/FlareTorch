import torch
import torch.nn as nn
from torchmetrics import Metric


class MultiClassClassificationMetrics(Metric):
    """Multi-class classification metrics including Skill Scores.

    Args:
        num_classes: Number of classes.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.add_state(
            "conf_matrix",
            default=torch.zeros(num_classes, num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update confusion matrix."""
        preds = torch.argmax(preds, dim=1)
        # Assuming target is not one-hot encoded
        cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=preds.device)
        for p, t in zip(preds, target):
            cm[t, p] += 1
        self.conf_matrix += cm

    def compute(self):
        """Compute all metrics."""
        cm = self.conf_matrix
        tp = cm.diag()
        row_sum = cm.sum(dim=1)
        col_sum = cm.sum(dim=0)
        n = cm.sum()

        # Standard metrics
        accuracy = tp.sum() / n
        
        # Balanced accuracy
        balanced_accuracy = (tp / row_sum).mean()

        # Macro metrics
        precision = (tp / (col_sum + 1e-12)).mean()
        recall = (tp / (row_sum + 1e-12)).mean()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

        # Skill Scores
        # TSS = (sum(tp) - sum(row_i * col_i)/n) / (n - sum(row_i^2)/n)
        # HSS = (n * sum(tp) - sum(row_i * col_i)) / (n^2 - sum(row_i * col_i))
        
        sum_tp = tp.sum()
        sum_product_marginals = (row_sum * col_sum).sum()
        sum_row_sq = (row_sum**2).sum()
        
        hss = (n * sum_tp - sum_product_marginals) / (n**2 - sum_product_marginals + 1e-12)
        tss = (n * sum_tp - sum_product_marginals) / (n**2 - sum_row_sq + 1e-12)

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "tss": tss,
            "hss": hss,
        }

