import torch
import torch.nn as nn
from torchmetrics.regression import R2Score
from .backbone import (
    ResNet18Regressor,
    ResNet34Regressor,
    ResNet50Regressor,
    AlexNetRegressor,
    MobileNetRegressor,
)
from .base import BaseModule
from ..utils.losses import PinballLoss


class ResNetMCD(BaseModule):
    """ResNet with Monte Carlo Dropout for uncertainty estimation.

    This model performs multiple forward passes with dropout enabled during
    inference to estimate predictive uncertainty.

    Args:
        model_type: Type of ResNet backbone (e.g., 'resnet18', 'resnet34').
        module_dict: Configuration for the MCDropout module.
        base_model_dict: Configuration for the base regressor.
        loss_type: Type of loss function to use (e.g., 'mse').
        optimizer_dict: Configuration for the optimizer.
        scheduler_dict: Configuration for the learning rate scheduler.

    Attributes:
        num_forwards: Number of MC forward passes during prediction.
        base_model: The underlying ResNet regressor.
        loss_fn: The loss function used for training.
        train_r2: R2 score metric for training.
        val_r2: R2 score metric for validation.
    """

    def __init__(
        self,
        model_type,
        module_dict,
        base_model_dict,
        loss_type,
        optimizer_dict,
        scheduler_dict,
    ):
        super().__init__(optimizer_dict=optimizer_dict, scheduler_dict=scheduler_dict)
        self.save_hyperparameters()
        self.num_forwards = module_dict.get("num_forwards", 100)

        match model_type:
            case "resnet34":
                self.base_model = ResNet34Regressor(
                    in_channels=base_model_dict.in_channels,
                    time_steps=base_model_dict.time_steps,
                    num_classes=1,
                    dropout=base_model_dict.p_drop,
                )

            case "resnet18":
                self.base_model = ResNet18Regressor(
                    in_channels=base_model_dict.in_channels,
                    time_steps=base_model_dict.time_steps,
                    num_classes=1,
                    dropout=base_model_dict.p_drop,
                )

        match loss_type:
            case "mse":
                self.loss_fn = nn.MSELoss()

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Model output.
        """
        # Standard forward pass
        return self.base_model(x)

    def predict_step(self, batch, batch_idx):
        """Custom prediction step for MC Dropout.

        This runs multiple forward passes with dropout enabled to calculate
        mean and standard deviation of predictions.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            A dictionary containing 'mean' and 'std' of predictions.
        """
        x, _, timestamps = batch

        # Enable Dropout manually
        self.base_model.train()

        # Freeze BatchNorm layers to keep stats stable
        for module in self.base_model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()

        # Run N forward passes
        mc_predictions = []
        for _ in range(self.num_forwards):
            with torch.no_grad():
                pred = self.base_model(x)
                mc_predictions.append(pred)

        # Shape: [Num_Forwards, Batch, 1]
        mc_predictions = torch.stack(mc_predictions)

        # Calculate Statistics
        mean_pred = mc_predictions.mean(dim=0)  # [Batch, 1]
        std_pred = mc_predictions.std(dim=0)  # [Batch, 1]

        # Return dict for easy analysis later
        return {
            "mean": mean_pred,
            "std": std_pred,
            # "raw_samples": mc_predictions
        }

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            The calculated loss.
        """
        # Standard training loop
        x, y, timestamps = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        self.train_r2(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.
        """
        x, y, timestamps = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        self.val_r2(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)


class ResNetQR(BaseModule):
    """ResNet with Quantile Regression for uncertainty estimation.

    This model predicts multiple quantiles of the target distribution to
    provide prediction intervals.

    Args:
        model_type: Type of ResNet backbone (e.g., 'resnet18', 'resnet34').
        base_model_dict: Configuration for the base regressor.
        optimizer_dict: Configuration for the optimizer.
        scheduler_dict: Configuration for the learning rate scheduler.
        module_dict: Configuration for the Quantile Regression module.

    Attributes:
        quantiles: List of quantiles to predict.
        loss_fn: Pinball loss function.
        median_idx: Index of the 0.5 quantile.
        train_r2: R2 score metric for training.
        val_r2: R2 score metric for validation.
        base_model: The underlying ResNet regressor.
    """

    def __init__(
        self,
        model_type,
        base_model_dict,
        optimizer_dict,
        scheduler_dict,
        module_dict,
    ):
        super().__init__(optimizer_dict=optimizer_dict, scheduler_dict=scheduler_dict)
        self.save_hyperparameters()
        self.quantiles = module_dict.get("quantiles", [0.025, 0.5, 0.975])

        # Initialize Loss
        self.loss_fn = PinballLoss(quantiles=self.quantiles)

        # find median index
        try:
            self.median_idx = self.quantiles.index(0.5)
        except ValueError:
            # Fallback: if 0.5 isn't in list, use the middle column
            self.median_idx = len(self.quantiles) // 2
            print(
                "Warning: 0.5 quantile not found. Using index",
                self.median_idx,
                "for R2.",
            )

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()

        match model_type:
            case "resnet34":
                self.base_model = ResNet34Regressor(
                    in_channels=base_model_dict.in_channels,
                    time_steps=base_model_dict.time_steps,
                    num_classes=len(self.quantiles),
                    dropout=base_model_dict.p_drop,
                )
            case "resnet18":
                self.base_model = ResNet18Regressor(
                    in_channels=base_model_dict.in_channels,
                    time_steps=base_model_dict.time_steps,
                    num_classes=len(self.quantiles),
                    dropout=base_model_dict.p_drop,
                )

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Model output containing predicted quantiles.
        """
        return self.base_model(x)

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            The calculated loss.
        """
        x, y, _ = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.train_r2(preds[:, self.median_idx], y)
        self.log("train_r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            The calculated loss.
        """
        # Lightning sets .eval() automatically here
        x, y, _ = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.val_r2(preds[:, self.median_idx], y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            A dictionary mapping quantile strings to predicted values.
        """
        x, _, _ = batch
        preds = self(x)

        # Dynamic return based on your config
        return {str(q): preds[:, i] for i, q in enumerate(self.quantiles)}
