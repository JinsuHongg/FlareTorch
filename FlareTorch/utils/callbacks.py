from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
)


def build_callbacks(cfg, wandb_logger):
    ckpt_name = (
        f"{wandb_logger.experiment.id}_"
        f"{cfg['experimentt']['ckpt_file_name']}_"
        "{epoch}-{val_loss:.4f}"
    )

    return [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor=cfg["scheduler"]["monitor"],
            dirpath=cfg["model"]["pretrained_weights_path"],
            filename=ckpt_name,
            save_top_k=3,
            verbose=True,
            mode="min",
        ),
    ]