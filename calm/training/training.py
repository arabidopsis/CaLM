"""PyTorch Lightning module for standard training."""

import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from .data_module import CodonDataModule
from .checkpointing import PeriodicCheckpoint

from ..model import ProteinBertModel
from ..utils import ArgparseMixin, optional
from ..alphabet import Alphabet


@dataclass
class CodonModelCfg(ArgparseMixin):
    batch_size: int = 46
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    lr_scheduler: str = "warmup_cosine"
    learning_rate: float = 4e-4
    accumulate_gradients: int = 40
    num_steps: int = 121000
    random_seed: int = -1


class CodonModel(pl.LightningModule):
    """PyTorch Lightning module for standard training."""

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return CodonModelCfg.add_args(parser)

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.cfg = CodonModelCfg.create(args)

        self.model = ProteinBertModel(args)

        def init_weights(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.normal_(module.weight, std=0.02)

            if isinstance(module, (torch.nn.Linear)):
                module.bias.data.fill_(0)

        self.model.apply(init_weights)

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
        self.save_hyperparameters()

    @property
    def alphabet(self) -> Alphabet:
        return self.model.alphabet

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        if self.cfg.lr_scheduler == "none":
            return optimizer
        elif self.cfg.lr_scheduler == "warmup_sqrt":

            def schedule(global_step):
                if global_step < self.cfg.warmup_steps:
                    return (global_step + 1) / self.cfg.warmup_steps
                else:
                    return np.sqrt(self.cfg.warmup_steps / global_step)

        elif self.cfg.lr_scheduler == "warmup_cosine":

            def schedule(global_step):
                if global_step < self.cfg.warmup_steps:
                    return (global_step + 1) / self.cfg.warmup_steps
                else:
                    progress = (
                        global_step - self.cfg.warmup_steps
                    ) / self.cfg.num_steps
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        else:
            raise ValueError("Unrecognised learning rate scheduler")

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch["input"].to(), train_batch["labels"].to(
            dtype=torch.int64
        )

        output = self.model(data)
        likelihoods = output["logits"]
        loss = self.loss_fn(
            likelihoods.view(-1, len(self.model.alphabet.all_toks)), labels.view(-1)
        )

        if batch_idx % self.cfg.accumulate_gradients == 0:
            self.log("train_loss", loss, batch_size=self.cfg.batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch["input"].to(), val_batch["labels"].to(
            dtype=torch.int64
        )

        output = self.model(data)
        likelihoods = output["logits"]
        loss = self.loss_fn(
            likelihoods.view(-1, len(self.model.alphabet.all_toks)), labels.view(-1)
        )
        self.log("val_loss", loss, batch_size=self.cfg.batch_size)
        return loss


@dataclass
class TrainingCfg(ArgparseMixin):
    name: str = "training-run"
    ckpt_path: str | None = field(
        default=None,
        metadata=dict(type=optional(str), help="load weights from checkpoint"),
    )
    no_progress_bar: bool = False


def init_args() -> tuple[argparse.Namespace, list[Path]]:
    parser = argparse.ArgumentParser(prog="python -m calm.training")

    parser = ProteinBertModel.add_args(parser)
    parser = CodonDataModule.add_args(parser)
    parser = CodonModel.add_args(parser)
    parser = TrainingCfg.add_args(parser)

    parser.add_argument("fasta_files", nargs="*", help="Training Fasta files")

    args = parser.parse_args()
    training_data: list[str] = args.fasta_files

    if not training_data:
        raise ValueError("no training data")

    fasta_files = [Path(f).expanduser() for f in training_data]

    if any(not f.exists() for f in fasta_files):
        raise ValueError("some training files don't exist!")
    return args, fasta_files


def train() -> None:

    args, fasta_files = init_args()

    training_cfg = TrainingCfg.create(args)
    dm_cfg = CodonDataModule.create_cfg(args)

    # model
    # arguments and their types will be save to the ckpt files

    ckpt_path = (
        None
        if training_cfg.ckpt_path is None
        else Path(training_cfg.ckpt_path).expanduser()
    )
    if ckpt_path is not None:
        print("loading checkpoint", ckpt_path)
        model = CodonModel.load_from_checkpoint(ckpt_path, args=args)
    else:
        model = CodonModel(args)
    assert model.model.max_positions == dm_cfg.max_positions

    codon_cfg = model.cfg

    # data
    datamodule = CodonDataModule(
        dm_cfg,
        model.model.alphabet,
        fasta_files=fasta_files,
        batch_size=codon_cfg.batch_size,
        random_seed=codon_cfg.random_seed,
    )

    # training
    name = training_cfg.name
    # logger = WandbLogger(name=name, project='12layers', version='restart3')
    # logger = CSVLogger("logs", name=name)
    # pip install tensorboard
    logger = TensorBoardLogger(save_dir="logs", name=name)
    trainer = pl.Trainer(
        num_nodes=1,
        precision="bf16-mixed",
        max_steps=codon_cfg.num_steps,
        logger=logger,
        log_every_n_steps=1,
        # val_check_interval=100*codon_cfg.accumulate_gradients,
        # accumulate_grad_batches=codon_cfg.accumulate_gradients,
        # limit_val_batches=1.0,
        accelerator="auto",
        enable_progress_bar=not training_cfg.no_progress_bar,
        # max_time="00:00:01:00",
        callbacks=[
            PeriodicCheckpoint(1000, name),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(model, datamodule=datamodule)  # , ckpt_path=ckpt_path)


if __name__ == "__main__":
    train()
