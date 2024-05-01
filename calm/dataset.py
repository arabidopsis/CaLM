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
        self._sequences: list[Sequence] = []
        # self._titles = []

        for record in SeqIO.parse(fasta_file, "fasta"):
            # self._titles.append(record.id)
            if self.codon_sequence:
                self._sequences.append(CodonSequence(record.seq))
            else:
                self._sequences.append(AminoAcidSequence(record.seq))

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx:int) -> Sequence:
        return self._sequences[idx]
