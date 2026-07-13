import torch.utils.data
from typing_extensions import override
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from laplace import Laplace
from loguru import logger as lgr_logger
from torch.utils.data import DataLoader

from ..metrics.classification_metrics import ClassificationUQMetrics


class SafeLaplaceModel(nn.Module):
    """A lightweight adapter to decouple standard PyTorch nn.Module architectures
    from PyTorch Lightning structural hooks.

    The laplace-torch library inspects model parameter structures and forward signatures.
    Passing a raw LightningModule container directly can trigger attribute routing
    conflicts during Kronecker-factored Hessian matrix computation.
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.backbone = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class LaplaceWrapper(L.LightningModule):
    """Laplace Approximation wrapper for epistemic uncertainty quantification.

    Estimates a post-hoc Gaussian posterior over network weights by fitting
    the curvature (Hessian) on a calibration subset.

    Attributes:
        base_model (L.LightningModule): The underlying pre-trained model.
        subset_size (int): Max number of samples for Hessian matrix construction.
        alpha (float): Target miscoverage rate.
        la (Optional[Laplace]): Fitted Laplace approximation instance.
    """

    base_model: L.LightningModule
    alpha: float
    num_classes: int
    class_mapping: dict[str, int]
    thresholds: list[float]
    q_hat: torch.Tensor
    q_hats: torch.Tensor
    lambda_hat: torch.Tensor
    test_uq_metrics: Any

    def __init__(
        self,
        trained_model: L.LightningModule,
        subset_size: int = 2000,
        alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_model = trained_model
        self.subset_size = subset_size
        self.alpha = alpha
        self.la: Any = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard deterministic forward pass using MAP point estimates."""
        return self.base_model(x)

    def fit_laplace(self, train_dataloader: torch.utils.data.DataLoader[Any]) -> None:
        """Fits the Laplace Approximation post-hoc on a data subset."""
        lgr_logger.info("Initializing post-hoc Laplace Approximation fitting...")

        # Extract the raw nn.Module from the Lightning container
        raw_model = getattr(self.base_model, "base_model", self.base_model)
        safe_model = SafeLaplaceModel(raw_model).to(self.device)
        safe_model.eval()

        # Initialize Last-Layer Laplace Approximation (LLLA)
        self.la = Laplace(
            safe_model,
            likelihood="regression",
            subset_of_weights="last_layer",
            hessian_structure="kron",
        )

        X_collect: list[torch.Tensor] = []
        Y_collect: list[torch.Tensor] = []
        count = 0

        with torch.no_grad():
            for batch in train_dataloader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                X_collect.append(x)
                Y_collect.append(y)
                count += x.size(0)
                if count >= self.subset_size:
                    break

        X = torch.cat(X_collect, dim=0)[: self.subset_size]
        Y = torch.cat(Y_collect, dim=0)[: self.subset_size]

        # Ensure target tensor is 2D for regression likelihood format: [N, 1]
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        self.la.fit(loader)
        self.la.optimize_prior_precision(method="marglik")
        lgr_logger.info("Laplace Approximation Hessian fitting completed successfully.")

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Returns predictive mean and epistemic + aleatoric standard deviation."""
        if self.la is None:
            raise RuntimeError(
                "Fatal Error: You must execute .fit_laplace() prior to inference!"
            )

        x = batch[0]
        f_mean, f_var = self.la(x)

        # Incorporate learned observational noise sigma^2 into total variance
        sigma_noise = self.la.sigma_noise.item()
        total_std = torch.sqrt(f_var + (sigma_noise**2)).squeeze(-1)
        f_mean = f_mean.squeeze(-1)

        return {
            "mean": f_mean,
            "std": total_std,
        }


class OrdinalCQRWrapper(L.LightningModule):
    """Ordinal Conformalized Quantile Regression (OrdinalCQR) wrapper.

    Bridges continuous regression intervals [L, U] to discrete ordinal classes
    by evaluating interval intersection against continuous class thresholds.
    """

    base_model: L.LightningModule
    alpha: float
    num_classes: int
    class_mapping: dict[str, int]
    thresholds: list[float]
    q_hat: torch.Tensor
    q_hats: torch.Tensor
    lambda_hat: torch.Tensor
    test_uq_metrics: Any

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int,
        class_mapping: dict[str, int],
        thresholds: list[float],
        alpha: float = 0.1,
        lower_idx: int = 0,
        upper_idx: int = -1,
    ) -> None:
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha
        self.lower_idx = lower_idx
        self.upper_idx = upper_idx
        self.class_mapping = class_mapping
        self.class_names = list(class_mapping.keys())
        self.num_classes = num_classes
        self.thresholds = thresholds

        self.register_buffer("q_hats", torch.zeros(num_classes, dtype=torch.float32))
        self.test_uq_metrics = ClassificationUQMetrics(num_classes=num_classes)

    def _get_class_idx_from_value(self, value: torch.Tensor) -> int:
        """Maps a continuous scalar to its corresponding ordinal integer label."""
        v = value.item()
        if v < self.thresholds[0]:
            return 0
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i] <= v < self.thresholds[i + 1]:
                return i + 1
        return len(self.thresholds)

    def get_prediction_set(
        self, L: torch.Tensor, U: torch.Tensor, target_classes: torch.Tensor
    ) -> torch.Tensor:
        """Constructs boolean prediction sets by checking interval overlaps."""
        batch_size = target_classes.size(0)
        num_classes = len(self.class_mapping)
        prediction_set = torch.zeros(
            (batch_size, num_classes), dtype=torch.bool, device=self.device
        )

        # Build contiguous boundary bins
        intervals = [(-float("inf"), self.thresholds[0])]
        for i in range(len(self.thresholds) - 1):
            intervals.append((self.thresholds[i], self.thresholds[i + 1]))
        intervals.append((self.thresholds[-1], float("inf")))

        for i in range(batch_size):
            for cls_idx, (t_start, t_end) in enumerate(intervals):
                if L[i] < t_end and U[i] > t_start:
                    prediction_set[i, cls_idx] = True

        return prediction_set

    def calibrate(
        self, calibration_dataloader: torch.utils.data.DataLoader[Any]
    ) -> None:
        """Executes class-conditional (Mondrian) CQR calibration."""
        lgr_logger.info("Initializing OrdinalCQR Mondrian Calibration...")
        self.base_model.eval()

        class_scores: list[list[float]] = [[] for _ in range(self.num_classes)]

        with torch.no_grad():
            for batch in calibration_dataloader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                preds = self.base_model(x)

                # Enforce 1D tensor view to guarantee stability across varying batch sizes
                y_flat = y.view(-1)
                pred_lo = preds[:, self.lower_idx]
                pred_hi = preds[:, self.upper_idx]

                # CQR non-conformity score: max(lower - y, y - upper)
                scores = torch.max(pred_lo - y_flat, y_flat - pred_hi)

                for i in range(y_flat.size(0)):
                    cls_idx = self._get_class_idx_from_value(y_flat[i])
                    class_scores[cls_idx].append(scores[i].item())

        for c in range(self.num_classes):
            if len(class_scores[c]) > 0:
                scores_tensor = torch.tensor(class_scores[c], dtype=torch.float32)
                n = scores_tensor.numel()
                q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
                q_level = min(max(q_level, 0.0), 1.0)
                self.q_hats[c] = torch.quantile(scores_tensor, q_level)
            else:
                lgr_logger.warning(
                    f"Class {self.class_names[c]} has 0 calibration samples. Defaulting q_hat to 0.0."
                )
                self.q_hats[c] = 0.0

        lgr_logger.info(
            f"OrdinalCQR Calibration successfully completed. Q_hats: {self.q_hats.cpu().tolist()}"
        )

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, Any]:
        """Returns CQR interval bounds and mapped ordinal prediction sets."""
        x, y = batch[0], batch[1]
        preds = self.base_model(x)

        pred_lo = preds[:, self.lower_idx]
        pred_hi = preds[:, self.upper_idx]

        y_flat = y.view(-1)
        targets = [self._get_class_idx_from_value(val) for val in y_flat]
        targets_tensor = torch.tensor(targets, device=pred_lo.device, dtype=torch.long)

        prediction_sets = self.get_prediction_set(pred_lo, pred_hi, targets_tensor)

        return {
            "lower": pred_lo.detach().cpu().tolist(),
            "upper": pred_hi.detach().cpu().tolist(),
            "prediction_set": prediction_sets,
            "target": targets_tensor,
        }

    def test_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, Any]:
        out = self.predict_step(batch, batch_idx)
        self.test_uq_metrics.update(out["prediction_set"], out["target"])
        return out

    def on_test_epoch_end(self) -> None:
        results = self.test_uq_metrics.compute()
        self.log_dict({f"test_{k}": v for k, v in results.items()}, prog_bar=True)
        self.test_uq_metrics.reset()


class CPWrapper(L.LightningModule):
    """Split Conformal Prediction wrapper for standard regression models."""

    base_model: L.LightningModule
    alpha: float
    num_classes: int
    class_mapping: dict[str, int]
    thresholds: list[float]
    q_hat: torch.Tensor
    q_hats: torch.Tensor
    lambda_hat: torch.Tensor
    test_uq_metrics: Any

    def __init__(
        self,
        trained_model: L.LightningModule,
        score_type: str = "l1",
        alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha
        self.score_metric = (
            (lambda y, y_hat: torch.abs(y - y_hat))
            if score_type == "l1"
            else (lambda y, y_hat: torch.square(y - y_hat))
        )
        self.register_buffer("q_hat", torch.tensor(float("inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

    def calibrate(
        self, calibration_dataloader: torch.utils.data.DataLoader[Any]
    ) -> None:
        lgr_logger.info("Initializing Regression Split CP Calibration...")
        self.base_model.eval()
        scores: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in calibration_dataloader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                preds = self.base_model(x)

                # Enforce 1D tensor view to preserve structural stability during inference
                preds_flat = preds.view(-1)
                y_flat = y.view(-1)
                scores.append(self.score_metric(y_flat, preds_flat))

        all_scores = torch.cat(scores, dim=0)
        n = all_scores.numel()
        q_level = min(max(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0), 1.0)
        self.q_hat = torch.quantile(all_scores, q_level)
        lgr_logger.info(f"Calibration completed. Q_hat = {self.q_hat.item():.4f}")

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        x = batch[0]
        preds = self.base_model(x).view(-1)
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

    base_model: L.LightningModule
    alpha: float
    num_classes: int
    class_mapping: dict[str, int]
    thresholds: list[float]
    q_hat: torch.Tensor
    q_hats: torch.Tensor
    lambda_hat: torch.Tensor
    test_uq_metrics: Any

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

    def calibrate(
        self, calibration_dataloader: torch.utils.data.DataLoader[Any]
    ) -> None:
        """Runs calibration to find the scalar 'q_hat' correction factor.

        Args:
            calibration_dataloader: torch.utils.data.DataLoader[Any] for the calibration set.
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
        lgr_logger.info("CQR Calibration Complete.")
        lgr_logger.info(f"Correction Factor (Q_hat) = {self.q_hat.item():.4f}")
        lgr_logger.info(
            "Logic: Final_Lower = Pred_Lower - Q_hat, Final_Upper = Pred_Upper + Q_hat"
        )

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
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

    base_model: L.LightningModule
    alpha: float
    num_classes: int
    class_mapping: dict[str, int]
    thresholds: list[float]
    q_hat: torch.Tensor
    q_hats: torch.Tensor
    lambda_hat: torch.Tensor
    test_uq_metrics: Any

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int = 5,
        alpha: float = 0.05,
        class_wise: bool = False,
        class_mapping: dict[str, int] | None = None,
        thresholds: list[float] | None = None,
    ):
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha
        self.class_wise = class_wise
        self.num_classes = num_classes
        self.class_mapping = class_mapping or {"FQ": 0, "B": 1, "C": 2, "M": 3, "X": 4}
        self.thresholds = thresholds or [2, 3, 4, 5]

        if self.class_wise:
            self.register_buffer("q_hats", torch.ones(num_classes) * 1.0)
        else:
            self.register_buffer("q_hat", torch.tensor(1.0))
        self.test_uq_metrics = ClassificationUQMetrics(num_classes=num_classes)

    def _get_class_idx_from_value(self, value):
        v = value.item()
        if v < self.thresholds[0]:
            return 0
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i] <= v < self.thresholds[i + 1]:
                return i + 1
        return len(self.thresholds)

    def forward(self, x):
        """Forward pass using the base model.

        Args:
            x: Input tensor.

        Returns:
            Model logits.
        """
        return self.base_model(x)

    def calibrate(self, dataloader: torch.utils.data.DataLoader[Any]) -> None:
        """Runs calibration to find the scalar 'q_hat'."""
        lgr_logger.info("Starting Classification CP (LAC) Calibration...")
        self.base_model.eval()
        if self.class_wise:
            class_scores = [[] for _ in range(self.num_classes)]
        else:
            scores = []
        device = self.device
        with torch.no_grad():
            for batch in dataloader:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                logits = self.base_model(x)
                probs = torch.softmax(logits, dim=1)
                true_probs = probs[torch.arange(len(y)), y]
                score = 1.0 - true_probs
                if self.class_wise:
                    for i in range(len(y)):
                        cls_idx = self._get_class_idx_from_value(y[i])
                        class_scores[cls_idx].append(score[i].item())
                else:
                    scores.append(score)

        if self.class_wise:
            for i in range(self.num_classes):
                if len(class_scores[i]) > 0:
                    scores_tensor = torch.tensor(class_scores[i], device=device)
                    n = len(scores_tensor)
                    q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
                    q_level = min(1.0, max(0.0, q_level))
                    self.q_hats[i] = torch.quantile(scores_tensor, q_level)
                else:
                    lgr_logger.warning(f"Class {i} has no samples. Using 1.0.")
                    self.q_hats[i] = 1.0
            lgr_logger.info(f"Calibration Complete. Q_hats = {self.q_hats}")
        else:
            scores = torch.cat(scores)
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(1.0, max(0.0, q_level))
            self.q_hat = torch.quantile(scores, q_level)
            lgr_logger.info(f"Calibration Complete. Q_hat = {self.q_hat.item():.4f}")

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Returns the Conformal Prediction Set."""
        x, y, _ = batch
        logits = self.base_model(x)
        probs = torch.softmax(logits, dim=1)

        if self.class_wise:
            prediction_sets = torch.zeros(
                probs.shape, dtype=torch.bool, device=probs.device
            )
            for k in range(self.num_classes):
                prediction_sets[:, k] = probs[:, k] >= (1.0 - self.q_hats[k])
        else:
            prediction_sets = probs >= (1.0 - self.q_hat)

        return {
            "probs": probs,
            "prediction_set": prediction_sets,
            "y_hat": torch.argmax(probs, dim=1),
            "target": y,
        }

    def test_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        x, y, _ = batch
        out = self.predict_step(batch, batch_idx)
        self.test_uq_metrics.update(out["prediction_set"], y)
        return out

    def on_test_epoch_end(self) -> None:
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

    base_model: L.LightningModule
    alpha: float
    num_classes: int
    class_mapping: dict[str, int]
    thresholds: list[float]
    q_hat: torch.Tensor
    q_hats: torch.Tensor
    lambda_hat: torch.Tensor
    test_uq_metrics: Any

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int = 5,
        alpha: float = 0.05,
        class_wise: bool = False,
        class_mapping: dict[str, int] | None = None,
        thresholds: list[float] | None = None,
    ):
        super().__init__()
        self.base_model = trained_model
        self.alpha = alpha
        self.class_wise = class_wise
        self.num_classes = num_classes
        self.class_mapping = class_mapping or {"FQ": 0, "B": 1, "C": 2, "M": 3, "X": 4}
        self.thresholds = thresholds or [2, 3, 4, 5]

        if self.class_wise:
            self.register_buffer("q_hats", torch.ones(num_classes) * 1.0)
        else:
            self.register_buffer("q_hat", torch.tensor(1.0))
        self.test_uq_metrics = ClassificationUQMetrics(num_classes=num_classes)

    def _get_class_idx_from_value(self, value):
        v = value.item()
        if v < self.thresholds[0]:
            return 0
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i] <= v < self.thresholds[i + 1]:
                return i + 1
        return len(self.thresholds)

    def _compute_class_aps_scores(self, probs):
        # probs: (Batch, K)
        batch_size, K = probs.shape
        sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=1)

        ranks = torch.zeros(batch_size, K, dtype=torch.long, device=probs.device)
        for i in range(batch_size):
            ranks[i, sorted_indices[i]] = torch.arange(K, device=probs.device)

        # aps_scores[i, k] = APS score for class k in sample i
        aps_scores = cum_probs[torch.arange(batch_size).unsqueeze(1), ranks]
        return aps_scores

    def calibrate(self, dataloader: torch.utils.data.DataLoader[Any]) -> None:
        """Runs calibration to find the scalar 'q_hat'."""
        lgr_logger.info("Starting APS Calibration...")
        self.base_model.eval()
        if self.class_wise:
            class_scores = [[] for _ in range(self.num_classes)]
        else:
            scores = []
        device = self.device
        with torch.no_grad():
            for batch in dataloader:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                logits = self.base_model(x)
                probs = torch.softmax(logits, dim=1)

                if self.class_wise:
                    aps_scores = self._compute_class_aps_scores(probs)
                    for i in range(len(y)):
                        cls_idx = self._get_class_idx_from_value(y[i])
                        # The score for the true class y[i]
                        class_scores[cls_idx].append(aps_scores[i, y[i]].item())
                else:
                    # Original logic
                    sorted_probs, sorted_indices = torch.sort(
                        probs, dim=1, descending=True
                    )
                    cum_probs = torch.cumsum(sorted_probs, dim=1)
                    ranks = (sorted_indices == y.unsqueeze(1)).nonzero()[:, 1]
                    score = cum_probs[torch.arange(len(y)), ranks]
                    scores.append(score)

        if self.class_wise:
            for i in range(self.num_classes):
                if len(class_scores[i]) > 0:
                    scores_tensor = torch.tensor(class_scores[i], device=device)
                    n = len(scores_tensor)
                    q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
                    q_level = min(1.0, max(0.0, q_level))
                    self.q_hats[i] = torch.quantile(scores_tensor, q_level)
                else:
                    lgr_logger.warning(f"Class {i} has no samples. Using 1.0.")
                    self.q_hats[i] = 1.0
            lgr_logger.info(f"Calibration Complete. Q_hats = {self.q_hats}")
        else:
            scores = torch.cat(scores)
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(1.0, max(0.0, q_level))
            self.q_hat = torch.quantile(scores, q_level)
            lgr_logger.info(f"Calibration Complete. Q_hat = {self.q_hat.item():.4f}")

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Returns the Adaptive Prediction Set."""
        x, y, _ = batch
        logits = self.base_model(x)
        probs = torch.softmax(logits, dim=1)

        batch_size, K = probs.shape
        prediction_sets = torch.zeros(
            batch_size, K, dtype=torch.bool, device=probs.device
        )

        if self.class_wise:
            aps_scores = self._compute_class_aps_scores(probs)
            for i in range(batch_size):
                for k in range(K):
                    if aps_scores[i, k] <= self.q_hats[k]:
                        prediction_sets[i, k] = True
                # Ensure at least one class included if none
                if prediction_sets[i].sum() == 0:
                    _, top_class = torch.max(probs[i], 0)
                    prediction_sets[i, top_class] = True
        else:
            # Original logic
            sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=1)

            for i in range(batch_size):
                mask = cum_probs[i] <= self.q_hat
                classes = sorted_indices[i, mask]
                if len(classes) == 0:
                    classes = sorted_indices[i, :1]
                prediction_sets[i, classes] = True

        return {
            "probs": probs,
            "prediction_set": prediction_sets,
            "y_hat": torch.argmax(probs, dim=1),
            "target": y,
        }

    def test_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        x, y, _ = batch
        out = self.predict_step(batch, batch_idx)
        self.test_uq_metrics.update(out["prediction_set"], y)
        return out

    def on_test_epoch_end(self) -> None:
        results = self.test_uq_metrics.compute()
        self.log_dict({f"test_{k}": v for k, v in results.items()}, prog_bar=True)
        self.test_uq_metrics.reset()


class OrdinalAPSWrapper(L.LightningModule):
    """Ordinal Adaptive Prediction Sets (OAPS) for Ordinal Classification.

    Enforces contiguity over ordered label spaces by evaluating cumulative
    probability distribution functions (CDFs) outward from the predicted mode
    or across natural sequential thresholds. This implementation supports both
    standard marginal calibration and Mondrian (class-conditional) calibration.

    Attributes:
        base_model (L.LightningModule): Backbone neural network providing classification logits.
        num_classes (int): Cardinality of the ordinal label space (K).
        alpha (float): Target miscoverage rate (e.g., 0.05 for 95% marginal/class-wise coverage).
        class_wise (bool): If True, applies Mondrian calibration generating K class-specific quantiles.
        class_mapping (Dict[str, int]): Mapping from domain-specific string labels to ordinal integers.
    """

    base_model: L.LightningModule
    alpha: float
    num_classes: int
    class_mapping: dict[str, int]
    thresholds: list[float]
    q_hat: torch.Tensor
    q_hats: torch.Tensor
    lambda_hat: torch.Tensor
    test_uq_metrics: Any

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int = 5,
        alpha: float = 0.05,
        class_wise: bool = False,
        class_mapping: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.base_model = trained_model
        self.num_classes = num_classes
        self.alpha = alpha
        self.class_wise = class_wise
        self.class_mapping = class_mapping or {"FQ": 0, "B": 1, "C": 2, "M": 3, "X": 4}

        # Register non-conformity quantiles as persistent buffers to ensure proper device serialization
        if self.class_wise:
            self.register_buffer("q_hats", torch.ones(num_classes, dtype=torch.float32))
        else:
            self.register_buffer("q_hat", torch.tensor(1.0, dtype=torch.float32))

        # Placeholder for user-defined metrics (uncomment when integrating metric tracking)
        # self.test_uq_metrics = ClassificationUQMetrics(num_classes=num_classes)

    def _compute_nonconformity_scores(
        self, probs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Computes vectorized OAPS non-conformity scores based on cumulative ordinal mass.

        Args:
            probs (torch.Tensor): Softmax probability matrix of shape (B, K).
            targets (torch.Tensor): Ground truth ordinal class indices of shape (B,).

        Returns:
            torch.Tensor: Non-conformity scores of shape (B,).
        """
        B, K = probs.shape
        # Compute ordinal cumulative probability distribution (CDF) along the class dimension
        cum_probs = probs.cumsum(dim=-1)

        # Gather the cumulative probability mass assigned up to the true ordinal target class
        # This acts as the standard APS non-conformity score: lower mass = higher non-conformity
        target_indices = targets.view(-1, 1)
        scores = cum_probs.gather(dim=1, index=target_indices).squeeze(dim=1)

        return scores

    def calibrate(self, dataloader: torch.utils.data.DataLoader[Any]) -> None:
        """Executes finite-sample conformal calibration to establish optimal q_hat thresholds.

        Applies order-statistic sorting to guarantee empirical coverage without relying on
        parametric distributional assumptions.
        """
        lgr_logger.info("Initializing Ordinal APS calibration sequence...")
        self.base_model.eval()

        score_list: list[torch.Tensor] = []
        target_list: list[torch.Tensor] = []

        # Execute forward pass without gradient tracking to conserve GPU VRAM
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.base_model(x)
                probs = torch.softmax(logits, dim=-1)

                scores = self._compute_nonconformity_scores(probs, y)
                score_list.append(scores)
                target_list.append(y)

        # Concatenate memory blocks to perform vectorized GPU operations
        all_scores = torch.cat(score_list, dim=0)
        all_targets = torch.cat(target_list, dim=0)

        if self.class_wise:
            # Mondrian (Class-Conditional) Calibration
            for c in range(self.num_classes):
                class_mask = all_targets == c
                if class_mask.sum() > 0:
                    c_scores = all_scores[class_mask]
                    n = c_scores.numel()

                    # Apply standard conformal index formula: ceil((n + 1) * (1 - alpha))
                    q_idx = int(np.ceil((n + 1) * (1.0 - self.alpha)))
                    # Clamp index to [1, n] to prevent index-out-of-bounds in extreme imbalance scenarios
                    q_idx = min(max(q_idx, 1), n) - 1

                    sorted_scores, _ = torch.sort(c_scores)
                    self.q_hats[c] = sorted_scores[q_idx]
                else:
                    lgr_logger.warning(
                        f"Class {c} encountered 0 samples during calibration. Defaulting q_hat to 1.0."
                    )
                    self.q_hats[c] = 1.0
            lgr_logger.info(
                f"Mondrian calibration completed successfully. Calibrated q_hats: {self.q_hats.cpu().tolist()}"
            )
        else:
            # Standard Marginal Calibration
            n = all_scores.numel()
            q_idx = int(np.ceil((n + 1) * (1.0 - self.alpha)))
            q_idx = min(max(q_idx, 1), n) - 1

            sorted_scores, _ = torch.sort(all_scores)
            self.q_hat = sorted_scores[q_idx]
            lgr_logger.info(
                f"Marginal calibration completed successfully. Calibrated q_hat: {self.q_hat.item():.4f}"
            )

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Generates contiguous ordinal prediction sets during inference.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing raw probabilities, boolean prediction sets,
                                     and point predictions.
        """
        x = batch[0]
        logits = self.base_model(x)
        probs = torch.softmax(logits, dim=-1)
        B, K = probs.shape

        cum_probs = probs.cumsum(dim=-1)
        prediction_sets = torch.zeros((B, K), dtype=torch.bool, device=probs.device)

        # Dynamic threshold assignment based on calibration regime
        if self.class_wise:
            point_preds = torch.argmax(probs, dim=-1)
            thresholds = self.q_hats[point_preds].unsqueeze(1)
        else:
            thresholds = self.q_hat

        # Step 1: Include all classes where cumulative probability satisfies the conformal threshold
        included_mask = cum_probs <= thresholds

        for i in range(B):
            if not included_mask[i].any():
                # Guarantee non-empty sets: fallback to mode if threshold is ultra-conservative
                mode_idx = torch.argmax(probs[i]).item()
                prediction_sets[i, mode_idx] = True
            else:
                # Step 2: Enforce strict contiguity by filling the interval from index 0 to the boundary
                last_included_idx = torch.where(included_mask[i])[0][-1].item()
                # Include the boundary class that caused cumulative probability to exceed threshold
                boundary_idx = min(last_included_idx + 1, K - 1)
                prediction_sets[i, : boundary_idx + 1] = True

        return {
            "probs": probs,
            "prediction_set": prediction_sets,
            "y_hat": torch.argmax(probs, dim=-1),
            "target": batch[1],
        }

    def test_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self.predict_step(batch, batch_idx)
        # Integrate custom metrics updating here:
        # self.test_uq_metrics.update(out["prediction_set"], batch[1])
        return out


class MinCPSWrapper(L.LightningModule):
    """Minimum-Length Conformal Prediction Sets (min-CPS) for Ordinal Classification.

    Frames ordinal prediction as a constrained combinatorial optimization problem.
    It utilizes a sliding-window search across discrete softmax distributions to
    identify the shortest contiguous interval that satisfies empirical coverage guarantees.

    Attributes:
        base_model (L.LightningModule): Backbone neural network.
        num_classes (int): Total number of ordinal classes.
        alpha (float): Target error rate.
    """

    base_model: L.LightningModule
    alpha: float
    num_classes: int
    class_mapping: dict[str, int]
    thresholds: list[float]
    q_hat: torch.Tensor
    q_hats: torch.Tensor
    lambda_hat: torch.Tensor
    test_uq_metrics: Any

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int = 5,
        alpha: float = 0.05,
        class_mapping: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.base_model = trained_model
        self.num_classes = num_classes
        self.alpha = alpha
        self.class_mapping = class_mapping or {"FQ": 0, "B": 1, "C": 2, "M": 3, "X": 4}

        self.register_buffer("q_hat", torch.tensor(1.0, dtype=torch.float32))

    def _compute_mincps_scores(
        self, probs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculates optimal min-CPS non-conformity scores via interval probability evaluation.

        To guarantee minimum-length properties, the non-conformity score is defined as
        (1.0 - max_prob) of the SHORTEST valid interval containing the ground truth target y.
        """
        B, K = probs.shape
        scores = torch.ones(B, device=probs.device)

        # Evaluate intervals dynamically to identify the maximum probability mass
        # assigned to the tightest contiguous bounds surrounding target y
        for i in range(B):
            y_true = targets[i].item()
            best_interval_prob = 0.0
            min_len = K + 1

            # Search exhaustive contiguous sub-intervals [start, end] containing y_true
            for start in range(y_true + 1):
                for end in range(y_true, K):
                    interval_len = end - start + 1
                    interval_prob = probs[i, start : end + 1].sum().item()

                    # Optimize for shorter length first; tie-break with higher probability mass
                    if interval_len < min_len or (
                        interval_len == min_len and interval_prob > best_interval_prob
                    ):
                        min_len = interval_len
                        best_interval_prob = interval_prob

            # Non-conformity score inversely proportional to the mass of the tightest covering interval
            scores[i] = 1.0 - best_interval_prob

        return scores

    def calibrate(self, dataloader: torch.utils.data.DataLoader[Any]) -> None:
        """Executes marginal calibration for the min-CPS optimization objective."""
        lgr_logger.info("Initializing min-CPS calibration sequence...")
        self.base_model.eval()

        score_list: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.base_model(x)
                probs = torch.softmax(logits, dim=-1)

                scores = self._compute_mincps_scores(probs, y)
                score_list.append(scores)

        all_scores = torch.cat(score_list, dim=0)
        n = all_scores.numel()

        # Compute empirical quantile index with boundary safety constraints
        q_idx = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        q_idx = min(max(q_idx, 1), n) - 1

        sorted_scores, _ = torch.sort(all_scores)
        self.q_hat = sorted_scores[q_idx]
        lgr_logger.info(
            f"min-CPS calibration completed successfully. Calibrated q_hat: {self.q_hat.item():.4f}"
        )

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Resolves the minimum-length contiguous interval satisfying the calibrated threshold."""
        x = batch[0]
        logits = self.base_model(x)
        probs = torch.softmax(logits, dim=-1)
        B, K = probs.shape

        prediction_sets = torch.zeros((B, K), dtype=torch.bool, device=probs.device)
        threshold_prob = 1.0 - self.q_hat.item()

        for i in range(B):
            valid_intervals = []

            # Exhaustive linear search over discrete probability space
            for start in range(K):
                for end in range(start, K):
                    interval_prob = probs[i, start : end + 1].sum().item()

                    # Interval is valid if its probability mass satisfies the conformal constraint
                    if interval_prob >= threshold_prob:
                        valid_intervals.append(
                            (start, end, end - start + 1, interval_prob)
                        )

            if valid_intervals:
                # Sort primarily by interval length (ascending), secondarily by probability (descending)
                valid_intervals.sort(key=lambda x: (x[2], -x[3]))
                best_start, best_end, _, _ = valid_intervals[0]
                prediction_sets[i, best_start : best_end + 1] = True
            else:
                # Fallback to point prediction mode if no valid interval satisfies conservative threshold
                mode_idx = torch.argmax(probs[i]).item()
                prediction_sets[i, mode_idx] = True

        return {
            "probs": probs,
            "prediction_set": prediction_sets,
            "y_hat": torch.argmax(probs, dim=-1),
            "target": batch[1],
        }

    def test_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self.predict_step(batch, batch_idx)
        self.test_uq_metrics.update(out["prediction_set"], out["target"])
        return out


class MinRCPSWrapper(MinCPSWrapper):
    """Minimum-Length Regularized Conformal Prediction Sets (min-RCPS).

    Extends min-CPS by incorporating an explicit length regularization penalty
    into the non-conformity scoring objective. This penalizes wider intervals
    during calibration, incentivizing tighter prediction set sizes while
    strictly preserving marginal coverage guarantees.

    Attributes:
        reg_weight (float): Regularization hyperparameter controlling the penalty
                            applied to interval cardinality (length).
    """

    base_model: L.LightningModule
    alpha: float
    num_classes: int
    class_mapping: dict[str, int]
    thresholds: list[float]
    q_hat: torch.Tensor
    q_hats: torch.Tensor
    lambda_hat: torch.Tensor
    test_uq_metrics: Any

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int = 5,
        alpha: float = 0.05,
        reg_weight: float = 0.01,
        class_mapping: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            trained_model=trained_model,
            num_classes=num_classes,
            alpha=alpha,
            class_mapping=class_mapping,
        )
        self.reg_weight = reg_weight

    def _compute_mincps_scores(
        self, probs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Overrides the base min-CPS scoring to integrate length regularization.

        Score formula: S(x, y) = (1.0 - P(Interval)) + (reg_weight * Interval_Length)
        Evaluates the optimal regularized covering interval for target y.
        """
        B, K = probs.shape
        scores = torch.zeros(B, device=probs.device, dtype=torch.float32)

        for i in range(B):
            y_true = targets[i].item()
            best_reg_score = float("inf")

            # Exhaustive evaluation of valid contiguous bounds containing ground truth y_true
            for start in range(y_true + 1):
                for end in range(y_true, K):
                    interval_len = end - start + 1
                    interval_prob = probs[i, start : end + 1].sum().item()

                    # Regularized objective: balance coverage probability against set cardinality
                    reg_score = (1.0 - interval_prob) + (self.reg_weight * interval_len)

                    if reg_score < best_reg_score:
                        best_reg_score = reg_score

            scores[i] = best_reg_score

        return scores

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Resolves the shortest regularized interval satisfying the calibrated quantile."""
        x = batch[0]
        logits = self.base_model(x)
        probs = torch.softmax(logits, dim=-1)
        B, K = probs.shape

        prediction_sets = torch.zeros((B, K), dtype=torch.bool, device=probs.device)
        threshold_val = self.q_hat.item()

        for i in range(B):
            valid_intervals: list[tuple[int, int, int, float]] = []

            for start in range(K):
                for end in range(start, K):
                    interval_len = end - start + 1
                    interval_prob = probs[i, start : end + 1].sum().item()
                    reg_score = (1.0 - interval_prob) + (self.reg_weight * interval_len)

                    # Interval is valid if its regularized score falls below the conformal threshold
                    if reg_score <= threshold_val:
                        valid_intervals.append(
                            (start, end, interval_len, interval_prob)
                        )

            if valid_intervals:
                # Sort primarily by interval length (ascending), secondarily by probability mass (descending)
                valid_intervals.sort(key=lambda x: (x[2], -x[3]))
                best_start, best_end, _, _ = valid_intervals[0]
                prediction_sets[i, best_start : best_end + 1] = True
            else:
                # Safe fallback: select point prediction mode to guarantee non-empty sets
                mode_idx = torch.argmax(probs[i]).item()
                prediction_sets[i, mode_idx] = True

        return {
            "probs": probs,
            "prediction_set": prediction_sets,
            "y_hat": torch.argmax(probs, dim=-1),
            "target": batch[1],
        }


class UnimodalityViolationError(RuntimeError):
    """Custom exception raised when a backbone model violates COPOC unimodality constraints."""

    pass


class COPOCWrapper(L.LightningModule):
    """Conformal Ordinal Prediction under Order Constraints (COPOC).

    Implements the COPOC calibration framework (Dey et al., 2023). COPOC strictly
    requires the underlying backbone model to emit unimodal probability distributions
    over the ordinal class space.

    When unimodality holds, standard Least Ambiguous Classifier (LAC) thresholding
    mathematically guarantees strictly contiguous (zero-disjoint) prediction sets
    without requiring any post-hoc heuristic gap-filling.

    Attributes:
        base_model (L.LightningModule): Backbone neural network with constrained unimodal heads
                                        (e.g., Poisson ordinal regression or binomial layers).
        num_classes (int): Cardinality of label space.
        alpha (float): Target miscoverage rate.
        tol (float): Numerical floating-point tolerance for monotonicity verification.
    """

    base_model: L.LightningModule
    alpha: float
    num_classes: int
    class_mapping: dict[str, int]
    thresholds: list[float]
    q_hat: torch.Tensor
    q_hats: torch.Tensor
    lambda_hat: torch.Tensor
    test_uq_metrics: Any

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int = 5,
        alpha: float = 0.05,
        class_mapping: dict[str, int] | None = None,
        numerical_tolerance: float = 1e-5,
    ) -> None:
        super().__init__()
        self.base_model = trained_model
        self.num_classes = num_classes
        self.alpha = alpha
        self.class_mapping = class_mapping or {"FQ": 0, "B": 1, "C": 2, "M": 3, "X": 4}
        self.tol = numerical_tolerance

        self.register_buffer("q_hat", torch.tensor(1.0, dtype=torch.float32))

    def _verify_unimodality(
        self, probs: torch.Tensor, context: str = "Execution"
    ) -> None:
        """Vectorized GPU verification of discrete unimodal distribution constraints.

        A discrete probability distribution is unimodal if there exists a mode index m
        such that probabilities are non-decreasing for k <= m and non-increasing for k >= m.

        Raises:
            UnimodalityViolationError: If any sample in the batch violates monotonicity.
        """
        B, K = probs.shape
        if K <= 2:
            return  # Distributions with 1 or 2 classes are trivially unimodal

        # Find the mode (peak probability index) for each sample in the batch: shape (B, 1)
        modes = torch.argmax(probs, dim=-1, keepdim=True)

        # Compute adjacent class differences: diffs[i, j] = probs[i, j+1] - probs[i, j]
        # Shape: (B, K-1)
        diffs = probs[:, 1:] - probs[:, :-1]

        # Create positional index matrix for difference boundaries: shape (1, K-1)
        indices = torch.arange(K, device=probs.device).unsqueeze(0)

        # Mask 1: Check boundaries strictly BEFORE the mode (indices[:, 1:] <= modes)
        # In this region, probabilities must be monotonically non-decreasing (diffs >= -tol)
        before_mode_mask = indices[:, 1:] <= modes
        violations_before = before_mode_mask & (diffs < -self.tol)

        # Mask 2: Check boundaries strictly AFTER the mode (indices[:, :-1] >= modes)
        # In this region, probabilities must be monotonically non-increasing (diffs <= tol)
        after_mode_mask = indices[:, :-1] >= modes
        violations_after = after_mode_mask & (diffs > self.tol)

        # Aggregate violations across the batch
        if violations_before.any() or violations_after.any():
            invalid_sample_idx = torch.where(
                violations_before.any(dim=-1) | violations_after.any(dim=-1)
            )[0][0].item()
            faulty_dist = probs[invalid_sample_idx].detach().cpu().numpy().round(4)
            mode_idx = modes[invalid_sample_idx].item()

            error_msg = (
                f"\n[COPOC Architectural Fatal Error during {context}]\n"
                f"The backbone model output an invalid non-unimodal probability distribution at sample index {invalid_sample_idx}.\n"
                f"Observed Softmax Probabilities: {faulty_dist} (Mode at Class {mode_idx})\n\n"
                f"Reason: COPOC mathematically requires unimodal probability outputs. Standard unconstrained "
                f"Softmax layers (e.g., vanilla ResNet) violate this assumption. You must replace the classification head "
                f"with an order-constrained unimodal architecture (e.g., Binomial/Poisson ordinal regression layer)."
            )
            lgr_logger.error(error_msg)
            raise UnimodalityViolationError(error_msg)

    def calibrate(self, dataloader: torch.utils.data.DataLoader[Any]) -> None:
        """Executes marginal calibration using standard LAC non-conformity scores."""
        lgr_logger.info(
            "Initializing COPOC calibration sequence with strict unimodality assertions..."
        )
        self.base_model.eval()

        score_list: list[torch.Tensor] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.base_model(x)
                probs = torch.softmax(logits, dim=-1)

                # STRICT ASSERTION: Verify unimodality before computing calibration scores
                self._verify_unimodality(
                    probs, context=f"Calibration (Batch {batch_idx})"
                )

                # Vectorized LAC score: 1.0 - P(Y_true)
                batch_indices = torch.arange(y.size(0), device=self.device)
                true_probs = probs[batch_indices, y]
                scores = 1.0 - true_probs

                score_list.append(scores)

        all_scores = torch.cat(score_list, dim=0)
        n = all_scores.numel()

        # Finite-sample order statistic indexing with out-of-bounds protection
        q_idx = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        q_idx = min(max(q_idx, 1), n) - 1

        sorted_scores, _ = torch.sort(all_scores)
        self.q_hat = sorted_scores[q_idx]
        lgr_logger.info(
            f"COPOC calibration completed successfully. Calibrated q_hat: {self.q_hat.item():.4f}"
        )

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Generates contiguous prediction sets via pure level-set LAC thresholding."""
        x = batch[0]
        logits = self.base_model(x)
        probs = torch.softmax(logits, dim=-1)

        # STRICT ASSERTION: Verify unimodality during inference
        self._verify_unimodality(probs, context=f"Inference (Batch {batch_idx})")

        # Because unimodality is mathematically guaranteed by the assertion above,
        # level-set thresholding inherently generates strictly contiguous intervals!
        prediction_sets = probs >= (1.0 - self.q_hat)

        # Safety fallback: If threshold is overly conservative and yields an empty set,
        # default to mode prediction to guarantee non-empty sets.
        empty_mask = ~prediction_sets.any(dim=-1)
        if empty_mask.any():
            modes = torch.argmax(probs, dim=-1)
            prediction_sets[empty_mask, modes[empty_mask]] = True

        return {
            "probs": probs,
            "prediction_set": prediction_sets,
            "y_hat": torch.argmax(probs, dim=-1),
            "target": batch[1],
        }

    def test_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self.predict_step(batch, batch_idx)
        self.test_uq_metrics.update(out["prediction_set"], out["target"])
        return out


class RiskControlWrapper(L.LightningModule):
    """Conformal Risk Control (CRC) for Ordinal Classification.

    Implements the CRC framework (Angelopoulos et al., 2024; Xu et al., 2023) to bound
    the expected ordinal misclassification severity (e.g., Mean Absolute Error distance)
    rather than standard marginal miscoverage.

    Attributes:
        base_model (L.LightningModule): Backbone neural network.
        num_classes (int): Cardinality of the ordinal label space (K).
        alpha (float): Target expected risk bound (e.g., MAE <= 0.5 class distance).
        max_penalty (float): Maximum possible ordinal distance loss (K - 1).
        delta (float): Confidence level for finite-sample Hoeffding bound correction.
    """

    base_model: L.LightningModule
    alpha: float
    num_classes: int
    class_mapping: dict[str, int]
    thresholds: list[float]
    q_hat: torch.Tensor
    q_hats: torch.Tensor
    lambda_hat: torch.Tensor
    test_uq_metrics: Any

    def __init__(
        self,
        trained_model: L.LightningModule,
        num_classes: int = 5,
        alpha: float = 0.5,
        class_mapping: dict[str, int] | None = None,
        delta: float = 0.1,  # 90% confidence that expected risk is strictly bounded by alpha
    ) -> None:
        super().__init__()
        self.base_model = trained_model
        self.num_classes = num_classes
        self.alpha = alpha
        self.class_mapping = class_mapping or {"FQ": 0, "B": 1, "C": 2, "M": 3, "X": 4}

        # In ordinal distance loss, the maximum possible distance between classes is K - 1
        self.max_penalty = float(num_classes - 1)
        self.delta = delta

        self.register_buffer("lambda_hat", torch.tensor(0.0, dtype=torch.float32))

    def _compute_vectorized_risk(
        self, pred_sets: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized computation of ordinal distance loss across an entire batch.

        Loss definition:
            - If set is non-empty: L(C, y) = min_{c in C} |c - y|
            - If set is empty:     L(C, y) = max_penalty (K - 1)
        """
        B, K = pred_sets.shape

        # Create positional class matrix: shape (1, K) -> [[0, 1, 2, ..., K-1]]
        class_indices = torch.arange(
            K, device=pred_sets.device, dtype=torch.float32
        ).unsqueeze(0)

        # Expand targets to match class matrix: shape (B, 1)
        target_indices = targets.unsqueeze(1).to(torch.float32)

        # Compute absolute ordinal distance from true target to ALL classes: shape (B, K)
        abs_distances = torch.abs(class_indices - target_indices)

        # Mask out classes that are NOT included in the prediction set by setting their distance to infinity
        masked_distances = torch.where(
            pred_sets, abs_distances, torch.full_like(abs_distances, float("inf"))
        )

        # Find the minimum distance to any class currently inside the prediction set: shape (B,)
        min_distances, _ = torch.min(masked_distances, dim=-1)

        # If a set was empty, min_distance will be inf. Replace with max_penalty.
        empty_set_mask = torch.isinf(min_distances)
        min_distances[empty_set_mask] = self.max_penalty

        return min_distances

    def calibrate(self, dataloader: torch.utils.data.DataLoader[Any]) -> None:
        """Executes risk-controlling calibration via vectorized grid search and finite-sample bounds."""
        lgr_logger.info(
            f"Initializing Risk Control Calibration (Target Expected Risk <= {self.alpha})..."
        )
        self.base_model.eval()

        prob_list: list[torch.Tensor] = []
        target_list: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.base_model(x)
                probs = torch.softmax(logits, dim=-1)

                prob_list.append(probs)
                target_list.append(y)

        all_probs = torch.cat(prob_list, dim=0)
        all_targets = torch.cat(target_list, dim=0)
        n = all_probs.size(0)

        # Generate 1000 search steps for high-precision threshold resolution
        lambdas = torch.linspace(0.0, 1.0, steps=1000, device=self.device)
        empirical_risks = torch.zeros_like(lambdas)

        # Finite-sample Hoeffding upper-bound correction factor for risk control
        # Guarantees that true out-of-distribution risk <= alpha with probability >= (1 - delta)
        hoeffding_bound_slack = self.max_penalty * np.sqrt(
            np.log(1.0 / self.delta) / (2.0 * n)
        )
        adjusted_target_alpha = max(0.0, self.alpha - hoeffding_bound_slack)

        lgr_logger.info(
            f"Calibration Sample Size: {n} | Hoeffding Slack: {hoeffding_bound_slack:.4f} | Adjusted Target Alpha: {adjusted_target_alpha:.4f}"
        )

        # Vectorized Grid Search across all candidate lambda thresholds
        for i, lbd in enumerate(lambdas):
            # 1. Thresholding
            raw_sets = all_probs >= lbd

            # 2. Vectorized Post-Hoc Contiguity Enforcement (simulating inference behavior)
            # Find leftmost and rightmost active boundaries per sample without loops
            active_exists = raw_sets.any(dim=-1)

            # Use cumsum tricks to identify span boundaries on GPU
            cumsum_fwd = raw_sets.int().cumsum(dim=-1)
            cumsum_bwd = raw_sets.int().flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

            # A class is inside the contiguous bridge if it is between the first and last active class
            contiguous_sets = (cumsum_fwd > 0) & (cumsum_bwd > 0)

            # Fallback for empty sets: activate argmax mode
            if (~active_exists).any():
                modes = torch.argmax(all_probs, dim=-1)
                empty_indices = torch.where(~active_exists)[0]
                contiguous_sets[empty_indices, modes[empty_indices]] = True

            # 3. Compute empirical risk using fully vectorized matrix operations
            distances = self._compute_vectorized_risk(contiguous_sets, all_targets)
            empirical_risks[i] = distances.mean()

        # Select the HIGHEST lambda (smallest prediction sets) that strictly satisfies the adjusted risk bound
        valid_mask = empirical_risks <= adjusted_target_alpha

        if valid_mask.any():
            valid_indices = torch.where(valid_mask)[0]
            self.lambda_hat = lambdas[valid_indices[-1]]
            lgr_logger.info(
                f"Calibration completed successfully. Lambda_hat = {self.lambda_hat.item():.4f} (Empirical Risk: {empirical_risks[valid_indices[-1]].item():.4f})"
            )
        else:
            # Fallback to conservative lower bound if target risk cannot be satisfied
            min_risk_idx = torch.argmin(empirical_risks)
            self.lambda_hat = lambdas[min_risk_idx]
            lgr_logger.warning(
                f"Could not find lambda to satisfy target risk {self.alpha} (Minimum achievable empirical risk: {empirical_risks[min_risk_idx].item():.4f}). "
                f"Defaulting to lambda_hat = {self.lambda_hat.item():.4f}."
            )

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Generates regularized risk-controlled prediction sets with vectorized gap-filling."""
        x = batch[0]
        logits = self.base_model(x)
        probs = torch.softmax(logits, dim=-1)
        B, K = probs.shape

        raw_sets = probs >= self.lambda_hat

        # Vectorized contiguity enforcement (identical to calibration logic)
        cumsum_fwd = raw_sets.int().cumsum(dim=-1)
        cumsum_bwd = raw_sets.int().flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        prediction_sets = (cumsum_fwd > 0) & (cumsum_bwd > 0)

        # Fallback for empty sets
        empty_mask = ~prediction_sets.any(dim=-1)
        if empty_mask.any():
            modes = torch.argmax(probs, dim=-1)
            prediction_sets[empty_mask, modes[empty_mask]] = True

        return {
            "probs": probs,
            "prediction_set": prediction_sets,
            "y_hat": torch.argmax(probs, dim=-1),
            "target": batch[1],
        }

    def test_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self.predict_step(batch, batch_idx)
        self.test_uq_metrics.update(out["prediction_set"], out["target"])
        return out
