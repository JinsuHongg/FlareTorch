import argparse

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


def build_model(config):

    return ResNet34MCP(
        model_type=config.model.type,
        num_forwards=config.model.num_forwards,
        p_drop=config.model.p_drop,
        base_model_dict=config.get(config.model.type, "resnet34"),
        loss_type=config.model.loss.type,
        optimizer_dict=config.optimizer,
        scheduler_dict=config.scheduler,
    )


def train(config_path):

    # load config
    config = (config_path)

    # Datamodule
    datamodule = FlareHelioviewerDataModule(
        config_path=config_path
    )

    # Load model
    if config["pretrained_weights_path"]:
        if config.model.type == "resnet34":
            model_obj = ResNet34MCP
        model = model_obj.load_from_checkpoint(config["pretrained_downstream_model_path"])
    else: 
        model = build_model(config=config)
    
    # Create wandb obejct
    wandb_logger = build_wandb(cfg=config, model=model)

    # Trainer
    callbacks = build_callbacks(cfg=config, wandb_logger=wandb_logger)
    trainer = Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        num_nodes=config.trainer.num_nodes,
        max_epochs=config.trainer.max_epochs,
        precision=config.trainer.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=config.trainer.log_every_n_steps,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        strategy=config.trainer.strategy,
    )

    lgr_logger.info(f"Start training...")
    trainer.fit(model=model, datamodule=datamodule)
    # trainer.test(dataloaders=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Surya-Flare-finetuning',
                    description='Finetuning Surya with flare dataset')
    parser.add_argument('--config-path', default="../configs/first_experiement_model_comparison.yaml")  
    args = parser.parse_args()
    train(config_path=args.config_path)