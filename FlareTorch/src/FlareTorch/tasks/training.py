# import argparse
import os
import hydra
from loguru import logger as lgr_logger
from omegaconf import OmegaConf

import torch
from lightning.pytorch import Trainer

from FlareTorch.datamodules import FlareHelioviewerClsDataModule
from FlareTorch.models import ResNetMCD, ResNetQR
from FlareTorch.utils import build_wandb, build_callbacks

torch.set_float32_matmul_precision("medium")


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = OmegaConf.load(f)
    lgr_logger.info(f"Loaded config from {config_path}")
    return cfg


def build_model(cfg):
    module_type = cfg.model.module_type

    if module_type == "mcd":
        return ResNetMCD(
            model_type=cfg.model.type,
            module_dict=cfg.model.get(cfg.model.module_type),
            base_model_dict=cfg.model.get(cfg.model.type, "resnet"),
            loss_type=cfg.model.loss.type,
            optimizer_dict=cfg.optimizer,
            scheduler_dict=cfg.scheduler,
        )

    elif module_type == "qr":
        return ResNetQR(
            model_type=cfg.model.type,
            module_dict=cfg.model.get(cfg.model.module_type),
            base_model_dict=cfg.model.get(cfg.model.type, "resnet34"),
            optimizer_dict=cfg.optimizer,
            scheduler_dict=cfg.scheduler,
        )


@hydra.main(
    config_path="../../../configs",
    config_name="resnet_helioviewer_config.yaml",
    version_base=None,
)
def train(cfg):

    # Datamodule
    datamodule = FlareHelioviewerClsDataModule(cfg=cfg)

    # Load model
    model = build_model(cfg=cfg)

    # Create wandb obejct
    wandb_logger = build_wandb(cfg=cfg, model=model)

    # Trainer
    callbacks = build_callbacks(cfg=cfg, wandb_logger=wandb_logger)
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        strategy=cfg.trainer.strategy,
    )

    lgr_logger.info(f"Start training...")
    ckpt = (
        os.path.join(cfg.model.save_ckpt_path, cfg.model.ckpt)
        if cfg.model.ckpt
        else None
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)
    # trainer.test(dataloaders=datamodule)


if __name__ == "__main__":
    train()
