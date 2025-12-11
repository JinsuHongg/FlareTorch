import argparse
import hydra
from loguru import logger as lgr_logger
from omegaconf import OmegaConf

import torch
from lightning.pytorch import Trainer

from FlareTorch.datamodules import FlareHelioviewerDataModule
from FlareTorch.models import ResNet34MCP
from FlareTorch.utils import build_wandb, build_callbacks

torch.set_float32_matmul_precision('medium')


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = OmegaConf.load(f)
    lgr_logger.info(f"Loaded config from {config_path}")
    return cfg


def build_model(cfg):

    return ResNet34MCP(
        model_type=cfg.model.type,
        num_forwards=cfg.model.num_forwards,
        p_drop=cfg.model.p_drop,
        base_model_dict=cfg.model.get(cfg.model.type, "resnet34"),
        loss_type=cfg.model.loss.type,
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
    datamodule = FlareHelioviewerDataModule(
        cfg=cfg
    )

    # Load model
    if cfg.model.pretrained_weights_path:
        if cfg.model.type == "resnet34":
            model_obj = ResNet34MCP
        model = model_obj.load_from_checkpoint(cfg.model.pretrained_weights_path)
    else: 
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
    trainer.fit(model=model, datamodule=datamodule)
    # trainer.test(dataloaders=datamodule)


if __name__ == "__main__":
    train()