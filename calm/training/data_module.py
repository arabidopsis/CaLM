"""Data modules for PyTorch Lightning."""

from dataclasses import dataclass
from argparse import Namespace, ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

# from sklearn.model_selection import train_test_split  # type: ignore

from ..alphabet import Alphabet
from ..dataset import MultiSequenceDataset
from ..pipeline2 import (
    standard_pipeline,
    PipelineCfg,
)
from ..utils import ArgparseMixin


@dataclass
class CondonDataModuleCfg(PipelineCfg, ArgparseMixin):
    num_data_workers: int = 2


class CodonDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for manipulating FASTA files
    containing sequences of codons."""

    @classmethod
    def add_args(cls, argparser: ArgumentParser) -> ArgumentParser:
        return CondonDataModuleCfg.add_args(argparser)

    @classmethod
    def create_cfg(cls, args: Namespace) -> CondonDataModuleCfg:
        return CondonDataModuleCfg.create(args)

    def __init__(
        self,
        cfg: CondonDataModuleCfg,
        alphabet: Alphabet,
        *,
        fasta_files: list[Path],
        batch_size: int,
        random_seed: int = -1,
        test_size: float = 0.01,
    ):
        super().__init__()
        self.fasta_files = fasta_files
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = cfg.num_data_workers

        self.pipeline = standard_pipeline(cfg, alphabet=alphabet)

        self.train_data: Dataset | None = None
        self.val_data: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        dataset = MultiSequenceDataset(self.fasta_files, codon_sequence=True)

        generator = torch.Generator()
        if self.random_seed != -1:
            generator = generator.manual_seed(self.random_seed)

        self.train_data, self.val_data = random_split(
            dataset,
            [1.0 - self.test_size, self.test_size],
            generator=generator,
        )
        # train_idx, val_idx = train_test_split(
        #     range(len(dataset)),
        #     test_size=self.test_size,
        #     shuffle=True,
        #     random_state=self.random_seed,
        # )

    def train_dataloader(self) -> DataLoader:
        assert self.train_data is not None
        return DataLoader(
            self.train_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.pipeline,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_data is not None
        return DataLoader(
            self.val_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=self.pipeline,
        )
