import os
import lightning as L
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from loguru import logger as lgr_logger
from ..datasets import FlareHelioviewerDataset


class FlareHelioviewerDataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: str
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.data.batch_size

    def get_dataset(self, phase, index_path):
        return FlareHelioviewerDataset(
            task=self.cfg.experiment.task,
            index_path=index_path,
            input_time_delta=self.cfg.data.input_time_delta,
            input_stat_path=self.cfg.data.input_stat_path,
            limb_mask_path=self.cfg.data.limb_mask_path,
            scaler_mul=self.cfg.data.scaler_mul,
            scaler_shift=self.cfg.data.scaler_shift,
            scaler_div=self.cfg.data.scaler_div,
            label_type=self.cfg.data.label_type,
            phase=phase,
        )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train_ds = self.get_dataset(
                "train",
                os.path.join(
                    self.cfg.data.index.path,
                    self.cfg.data.index.train)
                )

        # Assign validation dataset for use in dataloader(s)
        if stage in ("fit", "validate", None):
            self.val_ds = self.get_dataset(
                "validation",
                os.path.join(
                    self.cfg.data.index.path,
                    self.cfg.data.index.val)
                )
        
        # Assign test dataset for use in dataloader(s)
        if stage in (None, "test", "calibrate"):
            self.test_ds = self.get_dataset(
                "test",
                os.path.join(
                    self.cfg.data.index.path,
                    self.cfg.data.index.test)
                )

        if stage in (None, "predict", "calibrate"):
            self.pred_ds = self.get_dataset(
                "test",
                os.path.join(
                    self.cfg.data.index.path,
                    self.cfg.data.index.test)
                )
            
        if stage in (None, "calibrate"):
            self.cal_ds = self.get_dataset(
                "calibration",
                os.path.join(
                    self.cfg.data.index.path,
                    self.cfg.data.index.cal)
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            num_workers=self.cfg.data.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.cfg.data.pin_memory
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            num_workers=self.cfg.data.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.cfg.data.pin_memory
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            num_workers=self.cfg.data.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.cfg.data.pin_memory
            )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_ds,
            num_workers=self.cfg.data.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.cfg.data.pin_memory
            )
    
    def cal_dataloader(self):
        return DataLoader(
            self.cal_ds,
            num_workers=self.cfg.data.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.cfg.data.pin_memory
            )
