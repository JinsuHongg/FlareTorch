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
    

class CQRWrapper(L.LightningModule):
    def __init__(
            self, 
            trained_model: L.LightningModule,
            alpha: float = 0.1,
            lower_idx: int = 0,   # Index of the lower quantile (e.g., 0.05) in model output
            upper_idx: int = -1   # Index of the upper quantile (e.g., 0.95) in model output
            ):
        """
        Conformalized Quantile Regression (CQR) Wrapper.
        
        Args:
            trained_model: Pre-trained Quantile Regression model.
            alpha: Desired error rate (e.g., 0.1 for 90% coverage).
            lower_idx: Column index of the lower bound output (default 0).
            upper_idx: Column index of the upper bound output (default -1 for last).
        """
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha
        self.lower_idx = lower_idx
        self.upper_idx = upper_idx
        
        # Register buffer for the correction factor
        self.register_buffer("q_hat", torch.tensor(0.0))

    def forward(self, x):
        return self.base_model(x)

    def calibrate(self, calibration_dataloader):
        """
        Runs calibration to find the scalar 'q_hat' correction factor.
        """
        lgr_logger.info("Starting CQR Calibration...")
        self.base_model.eval()
        
        scores = []
        device = self.device

        with torch.no_grad():
            for batch in calibration_dataloader:
                x, y, _ = batch
                x = x.to(device)
                y = y.to(device)

                # Get Quantile Predictions [Batch, Num_Quantiles]
                preds = self.base_model(x)
                
                # Extract Lower and Upper Bounds
                y = y.squeeze()
                pred_lo = preds[:, self.lower_idx]
                pred_hi = preds[:, self.upper_idx]

                # Calculate CQR Non-Conformity Score
                # Score = max( lower - y,  y - upper )
                # Meaning: "How far is the point outside the interval?"
                #   - If point is inside, score is negative (distance to boundary)
                #   - If point is outside, score is positive
                score = torch.max(pred_lo - y, y - pred_hi)
                scores.append(score)

        scores = torch.cat(scores)
        
        # Compute Quantile for Correction
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level)) 

        q_val = torch.quantile(scores, q_level)
        
        # Store q_hat
        self.q_hat = q_val
        lgr_logger.info(f"CQR Calibration Complete.")
        lgr_logger.info(f"Correction Factor (Q_hat) = {self.q_hat.item():.4f}")
        lgr_logger.info("Logic: Final_Lower = Pred_Lower - Q_hat, Final_Upper = Pred_Upper + Q_hat")

    def predict_step(self, batch, batch_idx):
        """
        Returns the Conformalized Quantile Interval.
        """
        x, _, _ = batch 
        
        # Get raw quantiles from ResNet34QR
        preds = self.base_model(x)
        
        pred_lo = preds[:, self.lower_idx]
        pred_hi = preds[:, self.upper_idx]
        pred_median = preds[:, 1] if preds.shape[1] > 2 else (pred_lo + pred_hi) / 2
        
        # Apply CQR Correction
        corrected_lo = pred_lo - self.q_hat
        corrected_hi = pred_hi + self.q_hat
        
        return {
            "median": pred_median,
            "lower": corrected_lo,
            "upper": corrected_hi,
            "raw_lower": pred_lo,
            "raw_upper": pred_hi
        }
    
    
class LaplaceWrapper(L.LightningModule):
    def __init__(
            self, 
            trained_model: L.LightningModule, 
            subset_size: int = 2000,
            alpha: float = 0.05):
        """
        Args:
            trained_model: Your trained ResNet34.
            subset_size: Number of training samples to use for fitting the Hessian.
                         (Using full dataset is usually too slow/memory intensive).
        """
        super().__init__()
        self.base_model = trained_model
        self.subset_size = subset_size
        self.alpha = alpha
        self.la = None

    def forward(self, x):
        # Standard forward (uses the mean weights, i.e., the original trained model)
        return self.base_model(x)

    def fit_laplace(self, train_dataloader):
        """
        Fits the Laplace Approximation (Post-hoc).
        Call this ONCE before testing.
        """
        lgr_logger.info("Fitting Laplace Approximation...")
        
        # Extract the underlying PyTorch model from your LightningModule
        if hasattr(self.base_model, "base_model"):
            model = self.base_model.base_model # ResNet34Classifier
        else:
            model = self.base_model

        # Initialize Laplace
        # 'subset_of_weights="last_layer"' is highly recommended for ResNets (LLLA).
        self.la = Laplace(
            model, 
            likelihood="regression", 
            subset_of_weights="last_layer", 
            hessian_structure="kron" # Kronecker factorization is efficient
        )

        # Fit on a subset of data
        # We manually iterate to get a subset of X and Y
        X_collect = []
        Y_collect = []
        count = 0
        
        device = self.device
        model.eval()

        for batch in train_dataloader:
            x, y, _ = batch
            X_collect.append(x.to(device))
            Y_collect.append(y.to(device))
            count += x.shape[0]
            if count >= self.subset_size:
                break
        
        X = torch.cat(X_collect)[:self.subset_size]
        Y = torch.cat(Y_collect)[:self.subset_size]

        # Fit the model
        self.la.fit(torch.utils.data.TensorDataset(X, Y))
        
        # Optimize the prior precision (regularization) automatically
        self.la.optimize_prior_precision(method="marglik")
        
        lgr_logger.info("Laplace Fitting Complete.")

    def predict_step(self, batch, batch_idx):
        """
        Returns prediction with uncertainty.
        """
        if self.la is None:
            raise RuntimeError("You must call .fit_laplace() before predicting!")

        x, _, _ = batch
        
        # The library handles the sampling internally
        # f_mean: Prediction
        # f_var: Epistemic Uncertainty (Model uncertainty)
        f_mean, f_var = self.la(x)
        
        # Total Uncertainty = Model Uncertainty (f_var) + Aleatoric Noise
        # For regression, we usually add the observational noise (sigma^2)
        # estimated during prior optimization.
        sigma_noise = self.la.sigma_noise.item() # learned noise
        total_std = torch.sqrt(f_var + sigma_noise**2).squeeze()
        f_mean = f_mean.squeeze()

        normal_dist = torch.distributions.Normal(0, 1)
        z_score = normal_dist.icdf(torch.tensor(1 - self.alpha / 2, device=x.device))
        return {
            "mean": f_mean,
            "std": total_std,
            "lower": f_mean - z_score * total_std,
            "upper": f_mean + z_score * total_std,
            "zscore": z_score,
        }