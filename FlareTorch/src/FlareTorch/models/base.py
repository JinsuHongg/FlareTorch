import torch
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup
from loguru import logger as lgr_logger


class BaseModule(L.LightningModule):
    def __init__(
            self, 
            optimizer_dict,
            scheduler_dict):
        super().__init__()
        self.optimizer_dict = optimizer_dict
        self.scheduler_dict = scheduler_dict

    def configure_optimizers(self):
        opt_type = self.optimizer_dict.get("type", "adamw")
        lr = self.optimizer_dict.get("lr", 1e-4)
        weight_decay = self.optimizer_dict.get("weight_decay", 0.01)
        
        params = filter(lambda p: p.requires_grad, self.parameters())

        match opt_type.lower():
            case "adam":
                optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
            case "adamw":
                optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            case _:
                raise ValueError(f"Unknown optimizer type: {opt_type}")

        sched_use = self.scheduler_dict.get("use", None)
        monitor = self.scheduler_dict.get("monitor", "val_loss")
        hyper_params = self.scheduler_dict.get(sched_use, {})

        if not sched_use:
            return optimizer

        if sched_use == "cosine_warmup":
            warmup_ratio = hyper_params.get("warmup_ratio", 0.1)
            
            # SAFEGUARD: Calculate steps
            total_steps = self.trainer.estimated_stepping_batches
            lgr_logger.debug(f"Scheduler initialized with TOTAL_STEPS = {total_steps}")
            # Check for edge cases where Lightning returns infinity or valid steps are unknown
            if isinstance(total_steps, (float, int)) and (total_steps == float('inf') or total_steps == 0):
                lgr_logger.warning("Warning: Could not calculate total steps automatically.")
                total_steps = self.optimizer_dict.total_steps
                
            num_warmup_steps = int(total_steps * warmup_ratio)

            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=total_steps
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        elif sched_use == "plateau":
            # Pass params directly using **kwargs unpacking
            scheduler = ReduceLROnPlateau(optimizer, **hyper_params)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return optimizer