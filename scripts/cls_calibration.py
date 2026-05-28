import os
import csv
import hydra
from loguru import logger as lgr_logger
import torch
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from flaretorch.datamodules import FlareSuryaBenchDataModule
from flaretorch.explainability import ClsCPWrapper, APSWrapper, OrdinalAPSWrapper
from flaretorch.models import ResNetCls


class UQCSVWriter(BasePredictionWriter):
    def __init__(self, output_dir, method_name, write_interval="batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.file_path = os.path.join(output_dir, f"{method_name}_predictions.csv")
        self.header_written = False

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        probs = prediction["probs"].cpu().numpy()
        prediction_sets = prediction["prediction_set"].cpu().numpy()
        y_hat = prediction["y_hat"].cpu().numpy()
        target = prediction["target"].cpu().numpy()
        B, K = probs.shape

        rows = []
        for i in range(B):
            row = {
                "target": target[i],
                "y_hat": y_hat[i],
            }
            for k in range(K):
                row[f"prob_{k}"] = probs[i, k]
                row[f"in_set_{k}"] = int(prediction_sets[i, k])
            rows.append(row)

        keys = rows[0].keys()
        with open(self.file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerows(rows)


@hydra.main(
    config_path="../configs/",
    config_name="CLS_resnet18_calibration_surya_bench.yaml",
    version_base=None,
)
def run_uc_cal(cfg):
    datamodule = FlareSuryaBenchDataModule(cfg=cfg)
    datamodule.setup(stage="calibrate")
    datamodule.setup(stage="test")

    calibration_loader = datamodule.cal_dataloader()
    test_loader = datamodule.test_dataloader()

    # Load Model
    cls_pretrained_path = os.path.join(
        cfg.check_point.base, cfg.check_point.resnet18_cls
    )
    model = ResNetCls.load_from_checkpoint(
        cls_pretrained_path, strict=False, weights_only=False
    )

    # Move model to device once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert method to a list if it's not already
    methods = [cfg.uc.method] if isinstance(cfg.uc.method, str) else cfg.uc.method

    for method_name in methods:
        lgr_logger.info(f"Starting run for UQ method: {method_name}")

        # Initialize Wrapper
        alpha = cfg.uc.significance_level
        num_classes = cfg.uc.num_classes
        wrapper = None

        match method_name:
            case "lac":
                wrapper = ClsCPWrapper(
                    trained_model=model, num_classes=num_classes, alpha=alpha
                )
            case "aps":
                wrapper = APSWrapper(
                    trained_model=model, num_classes=num_classes, alpha=alpha
                )
            case "oaps":
                wrapper = OrdinalAPSWrapper(
                    trained_model=model, num_classes=num_classes, alpha=alpha
                )
            case _:
                raise ValueError(f"Unknown method: {method_name}")

        # Calibration
        wrapper.to(device)
        lgr_logger.info(f"Running Calibration for {method_name}...")
        wrapper.calibrate(calibration_loader)

        # Loggers
        method_output_dir = os.path.join(cfg.uc.csv_path, method_name)
        os.makedirs(method_output_dir, exist_ok=True)

        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{method_name}_alpha{alpha}",
            save_dir=cfg.wandb.save_dir,
        )
        csv_logger = CSVLogger(save_dir=method_output_dir, name="summary")

        # Trainer for Testing
        trainer = L.Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            logger=[wandb_logger, csv_logger],
            callbacks=[
                UQCSVWriter(output_dir=method_output_dir, method_name=method_name)
            ],
        )

        lgr_logger.info(f"Running Evaluation on Test Set for {method_name}...")
        trainer.test(wrapper, test_loader)

        lgr_logger.info(f"Running Prediction to Save CSV for {method_name}...")
        trainer.predict(wrapper, test_loader)

        lgr_logger.info(f"Finished run for UQ method: {method_name}")

    lgr_logger.info("All UQ methods processed.")


if __name__ == "__main__":
    run_uc_cal()
