"""Classes to deal with codon sequences."""

import abc


def _split_into_codons(seq: str):
    """Yield successive 3-letter chunks of a string/sequence."""
    for i in range(0, len(seq), 3):
        yield seq[i : i + 3]


class Sequence(abc.ABC):
    """Abstract base class for sequence data."""

    def __init__(self, seq_: str):
        seq = seq_.upper()
        _tokens = [
            "<cls>",
            *self._sanitize(self.split(seq)),
            "<eos>",
        ]
        # self._tokens = self._sanitize(_tokens)
        self._seq = " ".join(_tokens)

    @property
    def seq(self) -> str:
        return self._seq

    @property
    def tokens(self) -> list[str]:
        return self._seq.split()

    def _sanitize(self, tokens: list[str]):
        for x in tokens:
            x = x.strip()
            if x:
                yield x

    @abc.abstractmethod
    def split(self, seq: str) -> list[str]:
        raise NotImplementedError

    # @property
    # def seq(self) -> str:
    #     return " ".join(self._tokens)

    # @property
    # def tokens(self) -> list[str]:
    #     return self._tokens


class CodonSequence(Sequence):
    """Class containing a sequence of codons.

    >>> seq = CodonSequence('ATGGCGCTAAAGCGGATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']

    >>> seq = CodonSequence('ATG GCG CTA AAG CGG ATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']
    """

    def split(self, seq):
        return _split_into_codons(seq.replace("T", "U").replace(" ", ""))


class AminoAcidSequence(Sequence):

    def split(self, seq):
        return list(seq)
