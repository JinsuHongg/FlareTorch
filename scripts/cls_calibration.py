import os
import csv
import hydra
from loguru import logger as lgr_logger
import torch
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.loggers import WandbLogger, CSVLogger


def save_batch_to_csv(file_path, batch_dict, header_written=False):
    """
    Helper to save a batch of dictionary results to CSV.
    Handles both vectors (per-sample) and scalars (constants).
    """
    keys = list(batch_dict.keys())

    # Determine Batch Size from the first VECTOR found
    batch_size = 1
    for k in keys:
        val = batch_dict[k]
        if hasattr(val, "ndim") and val.ndim > 0:
            batch_size = len(val)
            break
        elif isinstance(val, list):
            batch_size = len(val)
            break

    rows = []
    for idx in range(batch_size):
        row = {}
        for k in keys:
            val = batch_dict[k]

            if hasattr(val, "ndim") and val.ndim == 0:
                item = val
            elif not hasattr(val, "__getitem__") or isinstance(val, (int, float)):
                item = val
            else:
                if len(val) > idx:
                    item = val[idx]
                else:
                    item = None

            if isinstance(item, torch.Tensor):
                item = item.item()

            row[k] = item
        rows.append(row)

    # Write to CSV
    mode = "a" if header_written else "w"
    with open(file_path, mode=mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not header_written:
            writer.writeheader()
        writer.writerows(rows)



from flaretorch.datamodules import FlareSuryaBenchDataModule
from flaretorch.explainability import ClsCPWrapper, APSWrapper, OrdinalAPSWrapper
from flaretorch.models import ResNetCls


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
    # Prepare wrappers and calibrate them
    wrappers = {}
    for method_name in methods:
        lgr_logger.info(f"Initializing and calibrating UQ method: {method_name}")

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
        wrapper.calibrate(calibration_loader)
        wrappers[method_name] = wrapper

    # Initialize Loggers (once)
    loggers = []

    # Wandb Logger (using .get with defaults if wandb config is missing)
    if cfg.get("wandb"):
        wandb_logger = WandbLogger(
            project=cfg.wandb.get("project", "default_project"),
            entity=cfg.wandb.get("entity", "default_entity"),
            name=f"calibration_run_{cfg.experiment.task}",
            save_dir=cfg.wandb.get("save_dir", "./wandb_logs"),
        )
        loggers.append(wandb_logger)

    # CSV Logger
    # Assuming csv_logger path is general for all methods, or needs to be method-specific
    # If method-specific CSV, it should be inside the loop.
    # For now, using a general path as per calibration.py example.
    csv_logger = CSVLogger(save_dir=cfg.uc.csv_path, name="summary")
    loggers.append(csv_logger)

    # Initialize Trainer once
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=loggers,  # Pass the list of loggers
        # Callbacks removed as prediction saving is handled manually later
    )

    lgr_logger.info("Running Evaluation on Test Set...")
    results = {}

    for method_name, wrapper in wrappers.items():
        lgr_logger.info(f"Running Prediction for {method_name}...")
        results[method_name] = trainer.predict(wrapper, test_loader)

    # Save Results ---
    lgr_logger.info("Saving results to CSV...")

    for method, preds in results.items():
        # Construct the path using the method name and alpha value from config
        path = os.path.join(cfg.uc.csv_path, f"{method}_alpha{cfg.uc.significance_level}_result_testset.csv")
        for i, batch_res in enumerate(preds):
            save_batch_to_csv(path, batch_res, header_written=(i > 0))

    lgr_logger.info("All UQ methods processed.")



if __name__ == "__main__":
    run_uc_cal()
