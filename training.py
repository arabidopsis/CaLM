"""PyTorch Lightning module for standard training."""

import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from data_module import CodonDataModule
from checkpointing import PeriodicCheckpoint
from calm.alphabet import Alphabet
from calm.model import ProteinBertModel, ProteinBertModelCfg
from calm.utils import ArgparseMixin, optional


@dataclass
class CodonModelCfg(ArgparseMixin):
    batch_size: int = 46
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    lr_scheduler: str = "warmup_cosine"
    learning_rate: float = 4e-4
    accumulate_gradients: int = 40
    num_steps: int = 121000


class CodonModel(pl.LightningModule):
    """PyTorch Lightning module for standard training."""

    def __init__(
        self, cfg: CodonModelCfg, model_cfg: ProteinBertModelCfg, alphabet: Alphabet
    ):
        super().__init__()
        self.args = cfg
        self.alphabet = alphabet
        self.model = ProteinBertModel(model_cfg, alphabet)

        def init_weights(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.normal_(module.weight, std=0.02)

            if isinstance(module, (torch.nn.Linear)):
                module.bias.data.fill_(0)

        self.model.apply(init_weights)

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        if self.args.lr_scheduler == "none":
            return optimizer
        elif self.args.lr_scheduler == "warmup_sqrt":

            def schedule(global_step):
                if global_step < self.args.warmup_steps:
                    return (global_step + 1) / self.args.warmup_steps
                else:
                    return np.sqrt(self.args.warmup_steps / global_step)

        elif self.args.lr_scheduler == "warmup_cosine":

            def schedule(global_step):
                if global_step < self.args.warmup_steps:
                    return (global_step + 1) / self.args.warmup_steps
                else:
                    progress = (
                        global_step - self.args.warmup_steps
                    ) / self.args.num_steps
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
            likelihoods.view(-1, len(self.alphabet.all_toks)), labels.view(-1)
        )

        if batch_idx % self.args.accumulate_gradients == 0:
            self.log("train_loss", loss, batch_size=self.args.batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch["input"].to(), val_batch["labels"].to(
            dtype=torch.int64
        )

        output = self.model(data)
        likelihoods = output["logits"]
        loss = self.loss_fn(
            likelihoods.view(-1, len(self.alphabet.all_toks)), labels.view(-1)
        )
        self.log("val_loss", loss, batch_size=self.args.batch_size)
        return loss


@dataclass
class TrainingCfg(ArgparseMixin):
    ckpt_path: str | None = field(default=None, metadata=dict(type=optional(str)))
    data: str = "training_data.fasta"
    no_progress_bar: bool = field(default=False, metadata=dict(action="store_true"))


def train():
    # parsing
    parser = argparse.ArgumentParser()

    parser = ProteinBertModel.add_args(parser)
    parser = CodonDataModule.add_args(parser)
    parser = CodonModelCfg.add_args(parser)
    parser = TrainingCfg.add_args(parser)
    args = parser.parse_args()

    training_cfg = TrainingCfg.create(args)
    codon_cfg = CodonModelCfg.create(args)
    model_cfg = ProteinBertModelCfg.create(args)
    dm_cfg = CodonDataModule.create(args)

    # data
    alphabet = Alphabet.from_architecture("CodonModel")
    datamodule = CodonDataModule(
        dm_cfg,
        alphabet,
        str(Path(training_cfg.data).expanduser()),
        codon_cfg.batch_size,
    )

    # model
    model = CodonModel(codon_cfg, model_cfg, alphabet)

    # training
    name = "training-run"
    # logger = WandbLogger(name=name, project='12layers', version='restart3')
    logger = CSVLogger("logs", name=name)
    trainer = pl.Trainer(
        num_nodes=1,
        precision="bf16-mixed",
        max_steps=codon_cfg.num_steps,
        logger=logger,
        log_every_n_steps=1,
        # val_check_interval=100*args.accumulate_gradients,
        # accumulate_grad_batches=args.accumulate_gradients,
        limit_val_batches=0.25,
        accelerator="auto",
        enable_progress_bar=not training_cfg.no_progress_bar,
        callbacks=[
            PeriodicCheckpoint(1000, name),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    ckpt_path = (
        None
        if training_cfg.ckpt_path is None
        else str(Path(training_cfg.ckpt_path).expanduser())
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    train()
