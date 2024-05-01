"""Common class for sequence datasets."""

from pathlib import Path
import torch
from Bio import SeqIO

from .sequence import Sequence, CodonSequence, AminoAcidSequence


class SequenceDataset(torch.utils.data.Dataset):
    """Common class for sequence datasets."""

    def __init__(self, fasta_file: Path, codon_sequence: bool = True):
        self.fasta_file = fasta_file
        self.codon_sequence = codon_sequence
        # self._titles = []
        constructor = CodonSequence if self.codon_sequence else AminoAcidSequence
        self._sequences : list[Sequence] = list(
            constructor(record.seq) for record in SeqIO.parse(fasta_file, "fasta")
        )

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Sequence:
        return self._sequences[idx]
