import torch
import torch.nn as nn
from terratorch_surya.downstream_examples.solar_flare_forecasting.models import ResNet34Classifier

from .base import BaseModule


class ResNet34MCP(BaseModule):
    def __init__(
            self,
            model_type,
            num_forwards,
            p_drop,
            base_model_dict,
            loss_type,
            optimizer_dict,
            scheduler_dict
            ):
        super().__init__(
            optimizer_dict=optimizer_dict,
            scheduler_dict=scheduler_dict
        )
        self.save_hyperparameters()
        self.num_forwards = num_forwards
        
        match model_type:
            case "resnet34":
                self.base_model = ResNet34Classifier(
                    in_channels=base_model_dict.in_channels,
                    time_steps=base_model_dict.time_steps,
                    num_classes=1,
                    dropout=p_drop,
                )

        match loss_type:
            case "mse":
                self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # Standard forward pass
        return self.base_model(x)

    def predict_step(self, batch, batch_idx):
        """
        Custom prediction step for MC Dropout.
        This runs automatically when you call trainer.predict()
        """
        x, _ = batch
        
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
        mean_pred = mc_predictions.mean(dim=0) # [Batch, 1]
        std_pred = mc_predictions.std(dim=0)   # [Batch, 1]
        
        # Return dict for easy analysis later
        return {
            "mean": mean_pred,
            "std": std_pred,
            "raw_samples": mc_predictions
        }

    def training_step(self, batch, batch_idx):
        # Standard training loop
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Standard validation
        self.base_model.eval() 
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        self.log('val_loss', loss)