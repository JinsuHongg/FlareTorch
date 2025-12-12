import torch
import lightning as L
import numpy as np
from loguru import logger as lgr_logger


class CPWrapper(L.LightningModule):
    def __init__(
            self, 
            trained_model: L.LightningModule,
            score_type: str = "l1",
            alpha: float = 0.05):
        """
        Args:
            alpha: Error rate (e.g., 0.05 for 95% coverage).
        """
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha
        
        match score_type:
            case "l1":
                self.score_metric = lambda y, y_hat: torch.abs(y - y_hat)
            case "mse":
                self.score_metric = lambda y, y_hat: torch.square(y - y_hat)
                
        # Register buffer to save 'q_hat' in the checkpoint
        self.register_buffer("q_hat", torch.tensor(float('inf')))

    def forward(self, x):
        return self.base_model(x)

    def calibrate(self, calibration_dataloader):
        """
        Runs calibration to find the scalar 'q_hat'.
        Call this ONCE after loading the trained model.
        """
        lgr_logger.info("Starting Calibration...")
        self.base_model.eval()
        
        scores = []
        device = self.device

        # Collect Non-Conformity Scores
        with torch.no_grad():
            for batch in calibration_dataloader:
                x, y, _ = batch
                
                # Move to correct device
                x = x.to(device)
                y = y.to(device)

                # Get prediction
                preds = self.base_model(x)
                preds = preds.squeeze()
                y = y.squeeze()

                score = self.score_metric(y, preds)
                scores.append(score)

        scores = torch.cat(scores)
        
        # Compute Quantile
        # (1 - alpha) * (n+1)/n quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level)) 
        
        q_val = torch.quantile(scores, q_level)
        
        # Store in the buffer
        self.q_hat = q_val
        lgr_logger.info(f"Calibration Complete. Q_hat = {self.q_hat.item():.4f}")
        lgr_logger.info(f"Intervals will be: Y_hat ± {self.q_hat.item():.4f}")

    def predict_step(self, batch, batch_idx):
        """
        Returns the Conformal Prediction Interval.
        """
        x, _, _ = batch 
        
        preds = self.base_model(x).squeeze()
        
        return {
            "y_hat": preds,
            "lower": preds - self.q_hat,
            "upper": preds + self.q_hat,
        }