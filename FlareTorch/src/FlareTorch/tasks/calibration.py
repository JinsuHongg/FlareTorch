import os
import csv
import hydra
from loguru import logger as lgr_logger
# import numpy as np

import torch
# from torch.utils.data import DataLoader
import lightning as L

from FlareTorch.datamodules import FlareHelioviewerClsDataModule
from FlareTorch.explainability import LaplaceWrapper, CQRWrapper, CPWrapper
from FlareTorch.models import ResNetMCD, ResNetQR


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
    config_path="../../../configs/",
    config_name="resnet34_calibration.yaml",
    version_base=None,
)
def run_uc_cal(cfg):

    datamodule = FlareHelioviewerClsDataModule(cfg=cfg)
    datamodule.setup(stage="calibrate")

    if hasattr(datamodule, "cal_dataloader"):
        calibration_loader = datamodule.cal_dataloader()
    else:
        lgr_logger.warning("No cal_dataloader found, using val_dataloader.")
        calibration_loader = datamodule.val_dataloader()

    test_loader = datamodule.test_dataloader()

    # Load Models
    base_path = cfg.check_point.base
    mcd_pretrained_path = os.path.join(base_path, "mcd", cfg.check_point.mcd)
    qr_pretrained_path = os.path.join(base_path, "qr", cfg.check_point.qr)

    match cfg.check_point.model_type:
        case "resnet":
            mcd = ResNetMCD.load_from_checkpoint(mcd_pretrained_path)
            qr = ResNetQR.load_from_checkpoint(qr_pretrained_path)
        case _:
            raise ValueError(f"Wrong model type: {cfg.check_point.model_type}")

    # Initialize Wrappers
    # Common significance level
    alpha = cfg.uc.significance_level

    cp_model = CPWrapper(
        trained_model=mcd, score_type=cfg.uc.cp.score_type, alpha=alpha
    )

    cqr_model = CQRWrapper(
        trained_model=qr,
        alpha=alpha,
        lower_idx=cfg.uc.cqr.lower_idx,
        upper_idx=cfg.uc.cqr.upper_idx,
    )

    lp_model = LaplaceWrapper(
        trained_model=mcd,
        alpha=alpha,
        subset_size=cfg.uc.lp.subset_size,
    )

    # Calibration -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp_model.to(device)
    cqr_model.to(device)
    lp_model.to(device)

    lgr_logger.info("Running Calibration...")

    # CP Calibration
    cp_model.calibrate(calibration_loader)
    lgr_logger.info(f"CP Q_hat: {cp_model.q_hat.item():.4f}")

    # CQR Calibration
    cqr_model.calibrate(calibration_loader)
    lgr_logger.info(f"CQR Q_hat: {cqr_model.q_hat.item():.4f}")

    # Laplace Fitting
    lp_model.fit_laplace(calibration_loader)

    # Prediction --------------------------------------------------------------
    lgr_logger.info("Running Prediction on Test Set...")

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices, logger=False
    )

    # Get all batches of predictions
    preds_mcd = trainer.predict(mcd, test_loader)
    preds_cp = trainer.predict(cp_model, test_loader)
    preds_cqr = trainer.predict(cqr_model, test_loader)
    preds_lp = trainer.predict(lp_model, test_loader)

    # Save Results ---
    lgr_logger.info("Saving results to CSV...")

    # Define paths
    path_mcd = os.path.join(cfg.uc.csv_path, "mcd_result_testset.csv")
    path_cp = os.path.join(cfg.uc.csv_path, "cp_result_testset.csv")
    path_cqr = os.path.join(cfg.uc.csv_path, "cqr_result_testset.csv")
    path_lp = os.path.join(cfg.uc.csv_path, "lp_result_testset.csv")

    # Iterate through batches and save
    # Save MCD
    for i, batch_res in enumerate(preds_mcd):
        save_batch_to_csv(path_mcd, batch_res, header_written=(i > 0))
    # Save CP
    for i, batch_res in enumerate(preds_cp):
        save_batch_to_csv(path_cp, batch_res, header_written=(i > 0))

    # Save CQR
    for i, batch_res in enumerate(preds_cqr):
        save_batch_to_csv(path_cqr, batch_res, header_written=(i > 0))

    # Save Laplace
    for i, batch_res in enumerate(preds_lp):
        save_batch_to_csv(path_lp, batch_res, header_written=(i > 0))

    lgr_logger.info("Done.")


if __name__ == "__main__":
    run_uc_cal()
