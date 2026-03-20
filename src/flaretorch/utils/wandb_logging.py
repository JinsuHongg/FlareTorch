import datetime
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger


def build_wandb(cfg, model):
    name = f"{cfg.model.type}_lr{cfg.optimizer.lr}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb_logger = WandbLogger(
        entity=cfg["wandb"]["entity"],
        project=cfg["wandb"]["project"],
        save_dir=cfg["wandb"]["save_dir"],
        offline=cfg["wandb"]["offline"],
        log_model=cfg["wandb"]["log_model"],
        save_code=cfg["wandb"]["save_code"],
        notes=cfg["wandb"]["notes"],
        tags=cfg["wandb"]["tag"],
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    # # selected hparams for WandB
    # wandb_logger.log_hyperparams({
    #     # optimizer / training
    #     "lr": cfg["optimizer"]["lr"],
    #     "batch_size": cfg["data"]["batch_size"],
    # })

    # wandb_logger.watch(model, log="parameters", log_freq=2000)
    return wandb_logger