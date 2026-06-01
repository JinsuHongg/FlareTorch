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


class ClassificationUQMetrics(Metric):
    """Uncertainty Quantification metrics for multi-class classification.

    Args:
        num_classes: Number of classes.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("coverage", default=[], dist_reduce_fx="cat")
        self.add_state("set_sizes", default=[], dist_reduce_fx="cat")
        self.add_state("sfs", default=[], dist_reduce_fx="cat")
        self.add_state("mdj", default=[], dist_reduce_fx="cat")

    def update(self, prediction_sets: torch.Tensor, target: torch.Tensor):
        """Update metrics state.

        Args:
            prediction_sets: Boolean mask of shape [Batch, Num_Classes].
            target: Ground truth labels of shape [Batch].
        """
        batch_size = prediction_sets.shape[0]
        device = prediction_sets.device

        # Marginal Coverage
        # Check if true class is in prediction set
        coverage = torch.gather(prediction_sets, 1, target.unsqueeze(1)).squeeze(1)
        self.coverage.append(coverage.float())

        # Set Size
        set_sizes = prediction_sets.sum(dim=1)
        self.set_sizes.append(set_sizes.float())

        # SFS and MDJ
        batch_sfs = torch.zeros(batch_size, device=device)
        batch_mdj = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            set_indices = torch.nonzero(prediction_sets[i]).squeeze()
            if set_indices.numel() == 0:
                continue
                
            if set_indices.dim() == 0: # Handle single element
                set_indices = set_indices.unsqueeze(0)

            # SFS = Span - Size
            span = set_indices.max() - set_indices.min() + 1
            batch_sfs[i] = span - set_indices.numel()

            # MDJ
            if set_indices.numel() > 1:
                diffs = torch.diff(set_indices)
                batch_mdj[i] = torch.max(diffs) - 1
            else:
                batch_mdj[i] = 0

        self.sfs.append(batch_sfs)
        self.mdj.append(batch_mdj)

    def compute(self):
        """Compute all UQ metrics."""
        coverage = torch.cat(self.coverage)
        set_sizes = torch.cat(self.set_sizes)
        sfs = torch.cat(self.sfs)
        mdj = torch.cat(self.mdj)

        return {
            "marginal_coverage": coverage.mean(),
            "avg_set_size": set_sizes.mean(),
            "avg_sfs": sfs.mean(),
            "avg_mdj": mdj.mean(),
            "ccr": (sfs == 0).float().mean(),
        }

