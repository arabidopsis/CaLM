"""Classes to deal with codon sequences."""

import abc
import re

WHITE = re.compile(r"\W+", re.I | re.M)


def remove_white(s: str) -> str:
    return WHITE.sub("", s)


def _split_into_codons(seq: str):
    """Yield successive 3-letter chunks of a string/sequence."""
    for i in range(0, len(seq), 3):
        yield seq[i : i + 3]


class Sequence(abc.ABC):
    """Abstract base class for sequence data."""

    def __init__(self, seq_: str):
        _tokens = [
            "<cls>",
            *self._sanitize(self.split(remove_white(seq_.upper()))),
            "<eos>",
        ]
        # self._tokens = _tokens
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

    def split(self, seq: str) -> list[str]:
        return _split_into_codons(seq.replace("T", "U"))


class AminoAcidSequence(Sequence):

    def split(self, seq: str) -> list[str]:
        """split on letters"""
        return list(seq)
