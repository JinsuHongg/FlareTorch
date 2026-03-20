import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """Pinball loss for quantile regression.

    This loss function is used to train models to predict specific quantiles
     of the target distribution.

    Args:
        quantiles: List of quantiles to estimate (e.g., [0.05, 0.5, 0.95]).

    Attributes:
        quantiles: List of quantiles to estimate.
    """

    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """Calculates the pinball loss.

        Args:
            preds: Predicted quantiles of shape (Batch, Num_Quantiles).
            target: Ground truth values of shape (Batch).

        Returns:
            The calculated mean pinball loss.
        """
        # Ensure target shape matches preds for broadcasting
        # Target: [Batch] -> [Batch, 1]
        target = target.view(-1, 1)

        # Define errors: (Batch, Num_Quantiles)
        errors = target - preds

        losses = []
        for i, q in enumerate(self.quantiles):
            # Extract error for this specific quantile column
            e = errors[:, i]

            # Basic Pinball Loss Formula: max(q * e, (q - 1) * e)
            loss = torch.max(q * e, (q - 1) * e)
            losses.append(loss)

        # Stack losses and average over batch and quantiles
        total_loss = torch.stack(losses, dim=1).mean()
        return total_loss
