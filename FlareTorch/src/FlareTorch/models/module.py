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
        # Standard forward pass
        return self.base_model(x)

    def predict_step(self, batch, batch_idx):
        """
        Custom prediction step for MC Dropout.
        This runs automatically when you call trainer.predict()
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
        # Standard training loop
        x, y, timestamps = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        self.train_r2(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, timestamps = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        self.val_r2(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)


class ResNetQR(BaseModule):
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
        return self.base_model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.train_r2(preds[:, self.median_idx], y)
        self.log("train_r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Lightning sets .eval() automatically here
        x, y, _ = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.val_r2(preds[:, self.median_idx], y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        preds = self(x)

        # Dynamic return based on your config
        return {str(q): preds[:, i] for i, q in enumerate(self.quantiles)}
