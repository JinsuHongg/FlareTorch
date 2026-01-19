import os
import hydra
import torch
import numpy as np
import pandas as pd
from FlareTorch.datamodules import FlareHelioviewerDataModule


@hydra.main(
    config_path="../../../configs/",
    config_name="MCD_resnet34_train.yaml",
)
def main(cfg):

    datamodule = FlareHelioviewerDataModule(cfg=cfg)
    datamodule.batch_size = 1
    datamodule.setup(stage="test")

    test_loader = datamodule.test_dataloader()
    limb_mask = datamodule.test_ds.limb_mask

    for x, target, time in test_loader:
        x_inversed = torch.abs(x)
        img_sum = (x_inversed * limb_mask).sum().item()

        time = pd.to_datetime(time[0])
        datamodule.test_ds.index.loc[time, "pixel_sum"] = img_sum

    datamodule.test_ds.index.to_csv(
        "../../assets/data/test_pixel_sum.csv", index=False
    )


if __name__ == "__main__":
    main()
