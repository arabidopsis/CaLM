"""Data modules for PyTorch Lightning."""

from dataclasses import dataclass
from argparse import Namespace, ArgumentParser
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split  # type: ignore

from ..alphabet import Alphabet
from ..dataset import MultiSequenceDataset, LazyDataset
from ..pipeline import (
    standard_pipeline,
    PipelineCfg,
)
from ..utils import ArgparseMixin


@dataclass
class CondonDataModuleCfg(PipelineCfg, ArgparseMixin):
    pass


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
        args: CondonDataModuleCfg,
        alphabet: Alphabet,
        *,
        fasta_files: list[Path],
        batch_size: int,
        random_seed: int = 42,
        test_size: float = 0.01,
    ):
        super().__init__()
        self.fasta_files = fasta_files
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        # self.pipeline = Pipeline(
        #     [
        #         DataCollator(
        #             args.mask_proportion,
        #             args.mask_percent,
        #             args.leave_percent,
        #             alphabet,
        #         ),
        #         DataTrimmer(max_positions, alphabet),
        #         DataPadder(max_positions, alphabet),
        #         DataPreprocessor(alphabet),
        #     ]
        # )

        self.pipeline = standard_pipeline(args, alphabet=alphabet)

        self.train_data: Dataset | None = None
        self.val_data: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        dataset = MultiSequenceDataset(self.fasta_files, codon_sequence=True)
        train_idx, val_idx = train_test_split(
            range(len(dataset)),
            test_size=self.test_size,
            shuffle=True,
            random_state=self.random_seed,
        )
        self.train_data = LazyDataset(dataset, train_idx)
        self.val_data = LazyDataset(dataset, val_idx)

    def train_dataloader(self) -> DataLoader:
        assert self.train_data is not None
        return DataLoader(
            self.train_data,
            num_workers=2,
            batch_size=self.batch_size,
            collate_fn=self.pipeline,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_data is not None
        return DataLoader(
            self.val_data,
            num_workers=2,
            batch_size=self.batch_size,
            collate_fn=self.pipeline,
        )
