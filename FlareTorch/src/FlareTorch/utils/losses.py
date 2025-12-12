import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    def __init__(self, quantiles: list[float]):
        """
        Args:
            quantiles: List of quantiles to estimate (e.g., [0.05, 0.5, 0.95])
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        Args:
            preds: (Batch, Num_Quantiles) - The model outputs
            target: (Batch) - The true values
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