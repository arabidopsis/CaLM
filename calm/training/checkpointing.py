import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int, dirpath: str | os.PathLike):
        super().__init__()
        self.every = every
        self.dirpath = Path(dirpath)

        if not  self.dirpath.exists():
            self.dirpath.mkdir(exist_ok=True)

    def on_before_zero_grad(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.global_step > 0 and pl_module.global_step % self.every == 0:
            assert self.dirpath is not None
            current = self.dirpath / f"latest-{pl_module.global_step}.ckpt"
            prev = self.dirpath / f"latest-{pl_module.global_step - self.every}.ckpt"
            print("saving", current)
            trainer.save_checkpoint(current)
            if prev.exists():
                prev.unlink()
