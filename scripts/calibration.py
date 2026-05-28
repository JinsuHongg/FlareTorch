import os
import csv
import hydra
from loguru import logger as lgr_logger
import torch
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from flaretorch.datamodules import (
    FlareHelioviewerRegDataModule,
    FlareSuryaBenchDataModule,
)
from flaretorch.explainability import (
    LaplaceWrapper,
    CQRWrapper,
    CPWrapper,
    OrdinalCQRWrapper,
)
from flaretorch.models import ResNetMCD, ResNetQR


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


@hydra.main(
    config_path="../configs/",
    config_name="resnet34_calibration.yaml",
    version_base=None,
)
def run_uc_cal(cfg):
    methods = cfg.uc.get("methods", ["mcd", "cp", "cqr", "lp"])

    if "input_zarr_path" in cfg.data:
        datamodule = FlareSuryaBenchDataModule(cfg=cfg)
    else:
        datamodule = FlareHelioviewerRegDataModule(cfg=cfg)
    datamodule.setup(stage="calibrate")
    datamodule.setup(stage="test")

    if hasattr(datamodule, "cal_dataloader"):
        calibration_loader = datamodule.cal_dataloader()
    else:
        lgr_logger.warning("No cal_dataloader found, using val_dataloader.")
        calibration_loader = datamodule.val_dataloader()

    test_loader = datamodule.test_dataloader()

    # Load Models
    base_path = cfg.check_point.base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcd = None
    qr = None

    if any(m in methods for m in ["mcd", "cp", "lp"]):
        if cfg.check_point.mcd is None:
            lgr_logger.warning(
                "MCD checkpoint is null, skipping MCD-dependent methods."
            )
        else:
            mcd_pretrained_path = os.path.join(base_path, "mcd", cfg.check_point.mcd)
            match cfg.check_point.model_type:
                case "resnet":
                    mcd = ResNetMCD.load_from_checkpoint(
                        mcd_pretrained_path, strict=False, weights_only=False
                    )
                    mcd.to(device)
                case _:
                    raise ValueError(f"Wrong model type: {cfg.check_point.model_type}")

    if "cqr" in methods:
        if cfg.check_point.qr is None:
            lgr_logger.warning("QR checkpoint is null, skipping CQR.")
        else:
            qr_pretrained_path = os.path.join(base_path, "qr", cfg.check_point.qr)
            match cfg.check_point.model_type:
                case "resnet":
                    qr = ResNetQR.load_from_checkpoint(
                        qr_pretrained_path, strict=False, weights_only=False
                    )
                    qr.to(device)
                case _:
                    raise ValueError(f"Wrong model type: {cfg.check_point.model_type}")

    # Initialize Wrappers
    alpha = cfg.uc.significance_level

    wrappers = {}

    if "cp" in methods and mcd is not None:
        wrappers["cp"] = CPWrapper(
            trained_model=mcd, score_type=cfg.uc.cp.score_type, alpha=alpha
        ).to(device)

    if "cqr" in methods and qr is not None:
        wrappers["cqr"] = CQRWrapper(
            trained_model=qr,
            alpha=alpha,
            lower_idx=cfg.uc.cqr.lower_idx,
            upper_idx=cfg.uc.cqr.upper_idx,
        ).to(device)

    if "ordinal_cqr" in methods and qr is not None:
        wrappers["ordinal_cqr"] = OrdinalCQRWrapper(
            qr,
            num_classes=cfg.uc.num_classes,  # Assuming num_classes is in cfg.data
            class_mapping=cfg.uc.class_mapping,
            thresholds=cfg.uc.thresholds,  # Assuming thresholds are defined in config
            alpha=alpha,
            lower_idx=cfg.uc.cqr.lower_idx,
            upper_idx=cfg.uc.cqr.upper_idx,
        ).to(device)

    if "lp" in methods and mcd is not None:
        wrappers["lp"] = LaplaceWrapper(
            trained_model=mcd,
            alpha=alpha,
            subset_size=cfg.uc.lp.subset_size,
        ).to(device)

    # Calibration -------------------------------------------------------------
    lgr_logger.info(f"Running Calibration for methods: {methods}")

    if "cp" in wrappers:
        wrappers["cp"].calibrate(calibration_loader)
        lgr_logger.info(f"CP Q_hat: {wrappers['cp'].q_hat.item():.4f}")

    if "cqr" in wrappers:
        wrappers["cqr"].calibrate(calibration_loader)
        lgr_logger.info(f"CQR Q_hat: {wrappers['cqr'].q_hat.item():.4f}")

    if "ordinal_cqr" in wrappers:
        wrappers["ordinal_cqr"].calibrate(calibration_loader)
        lgr_logger.info(f"OrdinalCQR Calibrated.")

    if "lp" in wrappers:
        wrappers["lp"].fit_laplace(calibration_loader)

    # Prediction --------------------------------------------------------------
    lgr_logger.info("Running Prediction on Test Set...")

    # Initialize Loggers
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
    csv_logger = CSVLogger(save_dir=cfg.uc.csv_path, name="summary")
    loggers.append(csv_logger)

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=loggers,  # Pass the list of loggers
    )

    results = {}

    if "mcd" in methods and mcd is not None:
        results["mcd"] = trainer.predict(mcd, test_loader)

    if "cp" in wrappers:
        results["cp"] = trainer.predict(wrappers["cp"], test_loader)

    if "cqr" in wrappers:
        results["cqr"] = trainer.predict(wrappers["cqr"], test_loader)

    if "ordinal_cqr" in wrappers:
        results["ordinal_cqr"] = trainer.predict(wrappers["ordinal_cqr"], test_loader)

    if "lp" in wrappers:
        results["lp"] = trainer.predict(wrappers["lp"], test_loader)

    # Save Results ---
    lgr_logger.info("Saving results to CSV...")

    for method, preds in results.items():
        path = os.path.join(
            cfg.uc.csv_path, f"{method}_alpha{alpha}_result_testset.csv"
        )
        for i, batch_res in enumerate(preds):
            save_batch_to_csv(path, batch_res, header_written=(i > 0))

    lgr_logger.info("Done.")


if __name__ == "__main__":
    run_uc_cal()
