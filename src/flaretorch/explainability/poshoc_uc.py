import torch
import torch.nn as nn
from laplace import Laplace
import lightning as L
import numpy as np
from loguru import logger as lgr_logger
from ..metrics.classification_metrics import ClassificationUQMetrics


class CPWrapper(L.LightningModule):
    """Conformal Prediction wrapper for uncertainty quantification.

    This wrapper applies split conformal prediction to a pre-trained model
    to provide valid prediction intervals with a specified coverage.

    Args:
        trained_model: The pre-trained LightningModule.
        score_type: Type of non-conformity score ('l1' or 'mse').
        alpha: Error rate (e.g., 0.05 for 95% coverage).

    Attributes:
        base_model: The underlying pre-trained model.
        alpha: The specified error rate.
        score_metric: Function to calculate non-conformity scores.
        q_hat: The calculated quantile for prediction intervals.
    """

    def __init__(
        self,
        trained_model: L.LightningModule,
        score_type: str = "l1",
        alpha: float = 0.05,
    ):
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha

        match score_type:
            case "l1":
                self.score_metric = lambda y, y_hat: torch.abs(y - y_hat)
            case "mse":
                self.score_metric = lambda y, y_hat: torch.square(y - y_hat)

        # Register buffer to save 'q_hat' in the checkpoint
        self.register_buffer("q_hat", torch.tensor(float("inf")))

    def forward(self, x):
        """Forward pass using the base model.

        Args:
            x: Input tensor.

        Returns:
            Model output.
        """
        return self.base_model(x)

    def calibrate(self, calibration_dataloader):
        """Runs calibration to find the scalar 'q_hat'.

        This method should be called once after loading the trained model
        using a calibration dataset.

        Args:
            calibration_dataloader: DataLoader for the calibration set.
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
        """Returns the Conformal Prediction Interval.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            A dictionary containing 'y_hat', 'lower', and 'upper' bounds.
        """
        x, _, _ = batch

        preds = self.base_model(x).squeeze()

        return {
            "y_hat": preds,
            "lower": preds - self.q_hat,
            "upper": preds + self.q_hat,
        }


class CQRWrapper(L.LightningModule):
    """Conformalized Quantile Regression (CQR) wrapper.

    This wrapper applies conformal prediction to a pre-trained quantile
    regression model to provide valid prediction intervals.

    Args:
        trained_model: Pre-trained Quantile Regression model.
        alpha: Desired error rate (e.g., 0.1 for 90% coverage).
        lower_idx: Column index of the lower bound output.
        upper_idx: Column index of the upper bound output.

    Attributes:
        base_model: The underlying pre-trained model.
        alpha: The specified error rate.
        lower_idx: Index of the lower quantile.
        upper_idx: Index of the upper quantile.
        q_hat: The calculated correction factor for prediction intervals.
    """

    def __init__(
        self,
        trained_model: L.LightningModule,
        alpha: float = 0.1,
        lower_idx: int = 0,  # Index of the lower quantile (e.g., 0.05) in model output
        upper_idx: int = -1,  # Index of the upper quantile (e.g., 0.95) in model output
    ):
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha
        self.lower_idx = lower_idx
        self.upper_idx = upper_idx

        # Register buffer for the correction factor
        self.register_buffer("q_hat", torch.tensor(0.0))

    def forward(self, x):
        """Forward pass using the base model.

        Args:
            x: Input tensor.

        Returns:
            Model output.
        """
        return self.base_model(x)

    def calibrate(self, calibration_dataloader):
        """Runs calibration to find the scalar 'q_hat' correction factor.

        Args:
            calibration_dataloader: DataLoader for the calibration set.
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
        lgr_logger.info(
            "Logic: Final_Lower = Pred_Lower - Q_hat, Final_Upper = Pred_Upper + Q_hat"
        )

    def predict_step(self, batch, batch_idx):
        """Returns the Conformalized Quantile Interval.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            A dictionary containing corrected and raw quantile predictions.
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
            "raw_upper": pred_hi,
        }


class OrdinalCQRWrapper(L.LightningModule):
    """Ordinal Conformalized Quantile Regression (OrdinalCQR) wrapper.

    Provides class-conditional interval coverage for ordinal flare classes.
    """

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int,
        class_mapping: dict,
        thresholds: list,
        alpha: float = 0.1,
        lower_idx: int = 0,
        upper_idx: int = -1,
    ):
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha
        self.lower_idx = lower_idx
        self.upper_idx = upper_idx
        self.class_names = list(class_mapping.keys())
        self.class_mapping = class_mapping
        self.num_classes = num_classes
        self.thresholds = thresholds  # Use provided thresholds

        # Register buffer for class-specific correction factors
        # 5 classes: A, B, C, M, X
        self.register_buffer("q_hats", torch.ones(num_classes) * 0.0)

    def _get_class_idx_from_value(self, value):
        v = value.item()
        if v < self.thresholds[0]:
            return 0
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i] <= v < self.thresholds[i + 1]:
                return i + 1
        return len(self.thresholds)

    def get_prediction_set(self, L, U, target_classes):
        """Determines the prediction set (boolean mask) for a given interval [L, U] and target classes."""
        batch_size = len(target_classes)
        num_classes = len(self.class_mapping)
        prediction_set = torch.zeros(
            batch_size, num_classes, dtype=torch.bool, device=self.device
        )

        # Define class intervals based on thresholds
        intervals = []
        # Class A: (-inf, thresholds[0])
        intervals.append((-float("inf"), self.thresholds[0]))
        # Intermediate classes: [thresholds[i], thresholds[i+1])
        for i in range(len(self.thresholds) - 1):
            intervals.append((self.thresholds[i], self.thresholds[i + 1]))
        # Class X: [thresholds[-1], inf)
        intervals.append((self.thresholds[-1], float("inf")))

        for i in range(batch_size):
            for cls_idx, (t_start, t_end) in enumerate(intervals):
                # Check for overlap between [L, U] and [t_start, t_end)
                if L[i] < t_end and U[i] > t_start:
                    prediction_set[i, cls_idx] = True
        return prediction_set

    def calibrate(self, calibration_dataloader):
        """Runs class-conditional calibration."""
        lgr_logger.info("Starting OrdinalCQR Calibration...")
        self.base_model.eval()

        # Group scores by true class
        class_scores = [[] for _ in range(self.num_classes)]
        device = self.device

        with torch.no_grad():
            for batch in calibration_dataloader:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)

                preds = self.base_model(x)
                y = y.squeeze()
                pred_lo = preds[:, self.lower_idx]
                pred_hi = preds[:, self.upper_idx]

                # CQR Score = max( lower - y, y - upper )
                scores = torch.max(pred_lo - y, y - pred_hi)

                for i in range(len(y)):
                    cls_idx = self._get_class_idx_from_value(y[i])
                    class_scores[cls_idx].append(scores[i].item())

        # Compute Quantile for each class
        for i in range(5):
            if len(class_scores[i]) > 0:
                scores_tensor = torch.tensor(class_scores[i])
                n = len(scores_tensor)
                q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
                q_level = min(1.0, max(0.0, q_level))
                self.q_hats[i] = torch.quantile(scores_tensor, q_level)
            else:
                # Fallback: global quantile if class is empty
                lgr_logger.warning(
                    f"Class {self.class_names[i]} has no samples. Using 0.0 correction."
                )
                self.q_hats[i] = 0.0

        lgr_logger.info(f"OrdinalCQR Calibration Complete. Q_hats: {self.q_hats}")

    def get_class_set(self, L, U):
        """Returns set of flare classes covered by interval [L, U]"""
        set_classes = []
        # Class intervals:
        # A: (-inf, 2), B: [2, 3), C: [3, 4), M: [4, 5), X: [5, inf)
        intervals = []
        intervals.append((-float("inf"), self.thresholds[0]))
        for i in range(len(self.thresholds) - 1):
            intervals.append((self.thresholds[i], self.thresholds[i + 1]))
        intervals.append((self.thresholds[-1], float("inf")))

        for i, (t_start, t_end) in enumerate(intervals):
            # Overlap if L < t_end and U > t_start
            if L < t_end and U > t_start:
                set_classes.append(self.class_names[i])
        return ",".join(set_classes)

    def predict_step(self, batch, batch_idx):
        """Returns corrected interval and covered class set."""
        x, y, _ = batch
        preds = self.base_model(x)

        pred_lo = preds[:, self.lower_idx]
        pred_hi = preds[:, self.upper_idx]
        # For ordinal regression, we often predict the midpoint or a value related to it.
        # Here, we'll use the mean of the lower and upper bounds as a proxy for the prediction that informs class mapping.
        pred_mid_proxy = (pred_lo + pred_hi) / 2

        lowers = []
        uppers = []
        targets = []

        # Map true labels to integer indices
        for true_val in y.squeeze():
            targets.append(self._get_class_idx_from_value(true_val))

        targets_tensor = torch.tensor(targets, device=pred_lo.device)

        targets_tensor = torch.tensor(targets, device=pred_lo.device)

        # Generate prediction sets using the new helper method
        prediction_sets = self.get_prediction_set(pred_lo, pred_hi, targets_tensor)

        for i in range(len(pred_lo)):
            lowers.append(
                (pred_lo[i]).item()
            )  # Keep original pred_lo for potential future use
            uppers.append(
                (pred_hi[i]).item()
            )  # Keep original pred_hi for potential future use

        return {
            "lower": lowers,
            "upper": uppers,
            "prediction_set": prediction_sets,
            "target": targets_tensor,  # Return mapped integer targets
        }

    def forward(self, x):
        """Forward pass with shape adjustments.

        Args:
            x: Input tensor.

        Returns:
            Adjusted output tensor.
        """
        if x.ndim == 4:
            x = x.unsqueeze(2)

        out = self.base_model(x)

        # Fix Output Shape (1D -> 2D)
        if out.ndim == 1:
            out = out.unsqueeze(1)

        return out


class LaplaceWrapper(L.LightningModule):
    """Laplace Approximation wrapper for uncertainty quantification.

    This wrapper applies Laplace approximation to a pre-trained model
    to estimate epistemic uncertainty.

    Args:
        trained_model: The pre-trained LightningModule.
        subset_size: Number of training samples to use for fitting the Hessian.
        alpha: Error rate for prediction intervals.

    Attributes:
        base_model: The underlying pre-trained model.
        subset_size: Number of samples for Hessian fitting.
        alpha: The specified error rate.
        la: The Laplace approximation object.
    """

    def __init__(
        self,
        trained_model: L.LightningModule,
        subset_size: int = 2000,
        alpha: float = 0.05,
    ):
        super().__init__()
        self.base_model = trained_model
        self.subset_size = subset_size
        self.alpha = alpha
        self.la = None

    def forward(self, x):
        """Forward pass using the base model.

        Args:
            x: Input tensor.

        Returns:
            Model output.
        """
        # Standard forward (uses the mean weights, i.e., the original trained model)
        return self.base_model(x)

    def fit_laplace(self, train_dataloader):
        """Fits the Laplace Approximation (Post-hoc).

        This method should be called once before testing to fit the Hessian
        on a subset of the training data.

        Args:
            train_dataloader: DataLoader for the training set.
        """
        lgr_logger.info("Fitting Laplace Approximation...")

        # Extract the underlying PyTorch model from your LightningModule
        if hasattr(self.base_model, "base_model"):
            raw_model = self.base_model.base_model  # ResNet34Classifier
        else:
            raw_model = self.base_model

        safe_model = SafeLaplaceModel(raw_model)

        # Initialize Laplace
        # 'subset_of_weights="last_layer"' is highly recommended for ResNets (LLLA).
        self.la = Laplace(
            safe_model,
            likelihood="regression",
            subset_of_weights="last_layer",
            hessian_structure="kron",
        )

        # Fit on a subset of data
        # We manually iterate to get a subset of X and Y
        X_collect = []
        Y_collect = []
        count = 0

        device = self.device
        safe_model.eval()

        for batch in train_dataloader:
            x, y, _ = batch
            X_collect.append(x.to(device))
            Y_collect.append(y.to(device))
            count += x.shape[0]
            if count >= self.subset_size:
                break

        X = torch.cat(X_collect)[: self.subset_size]
        Y = torch.cat(Y_collect)[: self.subset_size]

        if Y.ndim == 1:
            Y = Y.unsqueeze(1)

        # Fit the model
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        self.la.fit(loader)

        # Optimize the prior precision (regularization) automatically
        self.la.optimize_prior_precision(method="marglik")

        lgr_logger.info("Laplace Fitting Complete.")

    def predict_step(self, batch, batch_idx):
        """Returns prediction with uncertainty.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            A dictionary containing 'mean' and 'std' of predictions.
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
        sigma_noise = self.la.sigma_noise.item()  # learned noise
        total_std = torch.sqrt(f_var + sigma_noise**2).squeeze()
        f_mean = f_mean.squeeze()

        # normal_dist = torch.distributions.Normal(0, 1)
        # z_score = normal_dist.icdf(torch.tensor(1 - self.alpha / 2, device=x.device))
        return {
            "mean": f_mean,
            "std": total_std,
            # "lower": f_mean - z_score * total_std,
            # "upper": f_mean + z_score * total_std,
            # "zscore": z_score,
        }


class ClsCPWrapper(L.LightningModule):
    """Conformal Prediction wrapper for Classification (LAC).

    This wrapper applies the Least Ambiguous Coverage (LAC) method to provide
    valid prediction sets with specified coverage.

    Args:
        trained_model: The pre-trained LightningModule for classification.
        alpha: Error rate (e.g., 0.05 for 95% coverage).

    Attributes:
        base_model: The underlying pre-trained model.
        alpha: The specified error rate.
        q_hat: The calculated quantile for prediction sets.
    """

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int = 5,
        alpha: float = 0.05,
    ):
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha
        self.register_buffer("q_hat", torch.tensor(1.0))
        self.test_uq_metrics = ClassificationUQMetrics(num_classes=num_classes)

    def forward(self, x):
        """Forward pass using the base model.

        Args:
            x: Input tensor.

        Returns:
            Model logits.
        """
        return self.base_model(x)

    def calibrate(self, dataloader):
        """Runs calibration to find the scalar 'q_hat'.

        Args:
            dataloader: DataLoader for the calibration set.
        """
        lgr_logger.info("Starting Classification CP (LAC) Calibration...")
        self.base_model.eval()
        scores = []
        device = self.device
        with torch.no_grad():
            for batch in dataloader:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                logits = self.base_model(x)
                probs = torch.softmax(logits, dim=1)
                # LAC score: 1 - prob of true class
                true_probs = probs[torch.arange(len(y)), y]
                score = 1.0 - true_probs
                scores.append(score)
        scores = torch.cat(scores)
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))
        self.q_hat = torch.quantile(scores, q_level)
        lgr_logger.info(
            f"Classification CP (LAC) Calibration Complete. Q_hat = {self.q_hat.item():.4f}"
        )

    def predict_step(self, batch, batch_idx):
        """Returns the Conformal Prediction Set.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            A dictionary containing 'probs', 'prediction_set', 'y_hat', and 'target'.
        """
        x, y, _ = batch
        logits = self.base_model(x)
        probs = torch.softmax(logits, dim=1)
        # Prediction set: {y : probs[y] >= 1 - q_hat}
        prediction_sets = probs >= (1.0 - self.q_hat)
        return {
            "probs": probs,
            "prediction_set": prediction_sets,
            "y_hat": torch.argmax(probs, dim=1),
            "target": y,
        }

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        out = self.predict_step(batch, batch_idx)
        self.test_uq_metrics.update(out["prediction_set"], y)
        return out

    def on_test_epoch_end(self):
        results = self.test_uq_metrics.compute()
        self.log_dict({f"test_{k}": v for k, v in results.items()}, prog_bar=True)
        self.test_uq_metrics.reset()


class APSWrapper(L.LightningModule):
    """Adaptive Prediction Sets (APS) for Classification.

    Implements the standard APS method (Romano et al. 2020) which produces
    adaptive prediction sets by sorting class probabilities.

    Args:
        trained_model: The pre-trained LightningModule for classification.
        alpha: Error rate (e.g., 0.05 for 95% coverage).

    Attributes:
        base_model: The underlying pre-trained model.
        alpha: The specified error rate.
        q_hat: The calculated quantile for prediction sets.
    """

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int = 5,
        alpha: float = 0.05,
    ):
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha
        self.register_buffer("q_hat", torch.tensor(1.0))
        self.test_uq_metrics = ClassificationUQMetrics(num_classes=num_classes)

    def forward(self, x):
        """Forward pass using the base model.

        Args:
            x: Input tensor.

        Returns:
            Model logits.
        """
        return self.base_model(x)

    def _compute_scores(self, probs, y=None):
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)
        # Cumulative sums
        cum_probs = torch.cumsum(sorted_probs, dim=1)

        if y is not None:
            # Find rank of true class y
            # We want the index j such that sorted_indices[i, j] == y[i]
            ranks = (sorted_indices == y.unsqueeze(1)).nonzero()[:, 1]
            # Score is the cumulative sum up to the true class
            scores = cum_probs[torch.arange(len(y)), ranks]
            return scores
        else:
            return cum_probs, sorted_indices

    def calibrate(self, dataloader):
        """Runs calibration to find the scalar 'q_hat'.

        Args:
            dataloader: DataLoader for the calibration set.
        """
        lgr_logger.info("Starting APS Calibration...")
        self.base_model.eval()
        scores = []
        device = self.device
        with torch.no_grad():
            for batch in dataloader:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                logits = self.base_model(x)
                probs = torch.softmax(logits, dim=1)
                score = self._compute_scores(probs, y)
                scores.append(score)
        scores = torch.cat(scores)
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))
        self.q_hat = torch.quantile(scores, q_level)
        lgr_logger.info(f"APS Calibration Complete. Q_hat = {self.q_hat.item():.4f}")

    def predict_step(self, batch, batch_idx):
        """Returns the Adaptive Prediction Set.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            A dictionary containing 'probs', 'prediction_set', 'y_hat', and 'target'.
        """
        x, y, _ = batch
        logits = self.base_model(x)
        probs = torch.softmax(logits, dim=1)
        cum_probs, sorted_indices = self._compute_scores(probs)

        batch_size, K = probs.shape
        prediction_sets = torch.zeros(
            batch_size, K, dtype=torch.bool, device=probs.device
        )

        for i in range(batch_size):
            mask = cum_probs[i] <= self.q_hat
            classes = sorted_indices[i, mask]
            # Ensure at least the top class is included if q_hat is very small
            if len(classes) == 0:
                classes = sorted_indices[i, :1]
            prediction_sets[i, classes] = True

        return {
            "probs": probs,
            "prediction_set": prediction_sets,
            "y_hat": torch.argmax(probs, dim=1),
            "target": y,
        }

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        out = self.predict_step(batch, batch_idx)
        self.test_uq_metrics.update(out["prediction_set"], y)
        return out

    def on_test_epoch_end(self):
        results = self.test_uq_metrics.compute()
        self.log_dict({f"test_{k}": v for k, v in results.items()}, prog_bar=True)
        self.test_uq_metrics.reset()


class OrdinalAPSWrapper(L.LightningModule):
    """Ordinal Adaptive Prediction Sets (OAPS) for Ordinal Classification.

    Implements the OAPS method for labels with a natural ordering (Tiberi et al. 2023).
    This ensures that prediction sets are intervals in the ordinal space.

    Args:
        trained_model: The pre-trained LightningModule for classification.
        alpha: Error rate (e.g., 0.05 for 95% coverage).

    Attributes:
        base_model: The underlying pre-trained model.
        alpha: The specified error rate.
        q_hat: The calculated quantile for prediction sets.
    """

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int = 5,
        alpha: float = 0.05,
    ):
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha
        self.register_buffer("q_hat", torch.tensor(1.0))
        self.test_uq_metrics = ClassificationUQMetrics(num_classes=num_classes)

    def forward(self, x):
        """Forward pass using the base model.

        Args:
            x: Input tensor.

        Returns:
            Model logits.
        """
        return self.base_model(x)

    def _get_all_intervals_max_probs(self, probs):
        B, K = probs.shape
        cumprobs = torch.cumsum(probs, dim=1)
        cumprobs = torch.cat([torch.zeros(B, 1, device=probs.device), cumprobs], dim=1)

        P = torch.zeros(B, K, device=probs.device)
        I_start = torch.zeros(B, K, dtype=torch.long, device=probs.device)

        for s in range(1, K + 1):
            s_probs = cumprobs[:, s:] - cumprobs[:, :-s]
            max_p, start_idx = torch.max(s_probs, dim=1)
            P[:, s - 1] = max_p
            I_start[:, s - 1] = start_idx
        return P, I_start

    def _compute_score(self, probs, y):
        P, I_start = self._get_all_intervals_max_probs(probs)
        B, K = probs.shape
        scores = torch.full((B,), 1.5, device=probs.device)
        for s in range(1, K + 1):
            start = I_start[:, s - 1]
            end = start + s - 1
            mask = (y >= start) & (y <= end)
            scores = torch.where(mask & (P[:, s - 1] < scores), P[:, s - 1], scores)
        return scores

    def calibrate(self, dataloader):
        """Runs calibration to find the scalar 'q_hat'.

        Args:
            dataloader: DataLoader for the calibration set.
        """
        lgr_logger.info("Starting Ordinal APS Calibration...")
        self.base_model.eval()
        all_scores = []
        device = self.device
        with torch.no_grad():
            for batch in dataloader:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                logits = self.base_model(x)
                probs = torch.softmax(logits, dim=1)
                scores = self._compute_score(probs, y)
                all_scores.append(scores)
        all_scores = torch.cat(all_scores)
        n = len(all_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))
        self.q_hat = torch.quantile(all_scores, q_level)
        lgr_logger.info(
            f"Ordinal APS Calibration Complete. Q_hat = {self.q_hat.item():.4f}"
        )

    def predict_step(self, batch, batch_idx):
        """Returns the Ordinal Adaptive Prediction Set.

        Args:
            batch: The input batch.
            batch_idx: Index of the batch.

        Returns:
            A dictionary containing 'probs', 'prediction_set', 'y_hat', and 'target'.
        """
        x, y, _ = batch
        logits = self.base_model(x)
        probs = torch.softmax(logits, dim=1)
        P, I_start = self._get_all_intervals_max_probs(probs)

        B, K = probs.shape
        prediction_sets = torch.zeros(B, K, dtype=torch.bool, device=probs.device)
        for s in range(1, K + 1):
            mask = P[:, s - 1] <= self.q_hat
            for i in range(B):
                if mask[i]:
                    start = I_start[i, s - 1]
                    prediction_sets[i, start : start + s] = True

        return {
            "probs": probs,
            "prediction_set": prediction_sets,
            "y_hat": torch.argmax(probs, dim=1),
            "target": y,
        }

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        out = self.predict_step(batch, batch_idx)
        self.test_uq_metrics.update(out["prediction_set"], y)
        return out

    def on_test_epoch_end(self):
        results = self.test_uq_metrics.compute()
        self.log_dict({f"test_{k}": v for k, v in results.items()}, prog_bar=True)
        self.test_uq_metrics.reset()
