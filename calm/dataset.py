"""Common class for sequence datasets."""

import os
from typing import Sequence as SequenceType
from typing import TypeVar, Generic
from pathlib import Path
from torch.utils.data import Dataset
from Bio import SeqIO
from .sequence import Sequence, CodonSequence, AminoAcidSequence
from .fasta import nnfastas

class SequenceDataset(Dataset):
    """Common class for sequence datasets."""

    def __init__(self, fasta_file: Path, codon_sequence: bool = True):
        self.fasta_file = fasta_file
        self.codon_sequence = codon_sequence
        # self._titles = []
        constructor = CodonSequence if self.codon_sequence else AminoAcidSequence
        self._sequences: list[Sequence] = list(
            constructor(record.seq) for record in SeqIO.parse(fasta_file, "fasta")
        )

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Sequence:
        return self._sequences[idx]


class MultiSequenceDataset(Dataset):
    def __init__(
        self, fasta_files: SequenceType[str | os.PathLike], codon_sequence: bool = True
    ):

        self._fs = nnfastas(fasta_files)
        self.constructor = CodonSequence if codon_sequence else AminoAcidSequence

    def __len__(self) -> int:
        return len(self._fs)

    def __getitem__(self, idx: int) -> Sequence:
        r = self._fs[idx]
        return self.constructor(r.seq)


T = TypeVar("T")


class LazyDataset(Dataset, Generic[T]):
    def __init__(self, dataset: Dataset[T], indexes: list[int]):
        """Use index to create a new dataset from another"""
        self._dataset = dataset
        self._indexes = indexes
        assert len(dataset) > max(self._indexes) and min(self._indexes) >= 0  # type: ignore

    def __len__(self) -> int:
        return len(self._indexes)

    def __getitem__(self, idx: int) -> T:
        idx = self._indexes[idx]
        return self._dataset[idx]
