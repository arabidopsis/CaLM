"""Implementation of the Alphabet and BatchConverter classes.

This code has been modified from the original implementation
by Facebook Research, describing its ESM-1b paper.

# see https://github.com/facebookresearch/esm/esm/data.py
"""

from typing import Sequence, TypedDict

import torch


class Tokens(TypedDict):
    toks: list[str]
    coding_toks: list[str]


proteinseq_toks = Tokens(
    toks=list("LAGVSERTIDPKQNFYMHWCXBUZO.-"),
    coding_toks=list("ARNDCQEGHILKMFPSTWYV"),
)
CODONS = [
    "AAA",
    "AAU",
    "AAC",
    "AAG",
    "AUA",
    "AUU",
    "AUC",
    "AUG",
    "ACA",
    "ACU",
    "ACC",
    "ACG",
    "AGA",
    "AGU",
    "AGC",
    "AGG",
    "UAA",
    "UAU",
    "UAC",
    "UAG",
    "UUA",
    "UUU",
    "UUC",
    "UUG",
    "UCA",
    "UCU",
    "UCC",
    "UCG",
    "UGA",
    "UGU",
    "UGC",
    "UGG",
    "CAA",
    "CAU",
    "CAC",
    "CAG",
    "CUA",
    "CUU",
    "CUC",
    "CUG",
    "CCA",
    "CCU",
    "CCC",
    "CCG",
    "CGA",
    "CGU",
    "CGC",
    "CGG",
    "GAA",
    "GAU",
    "GAC",
    "GAG",
    "GUA",
    "GUU",
    "GUC",
    "GUG",
    "GCA",
    "GCU",
    "GCC",
    "GCG",
    "GGA",
    "GGU",
    "GGC",
    "GGG",
]
codonseq_toks = Tokens(toks=CODONS, coding_toks=CODONS)


class Alphabet:
    def __init__(
        self,
        tokens: Tokens,
    ):
        standard_toks = tokens["toks"]
        self.coding_toks = tokens["coding_toks"]

        prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
        append_toks = ("<mask>",)
        self._all_toks = (*prepend_toks, *standard_toks, *append_toks)

        self._tok_to_idx = {tok: i for i, tok in enumerate(self._all_toks)}
        self._unique_no_split_tokens = set(self._all_toks)

        self.unk_idx = self._tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")

    def __len__(self):
        return len(self._all_toks)

    def get_idx(self, tok: str) -> int:
        return self._tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind: int) -> str:
        return self._all_toks[ind]

    def get_batch_converter(self):
        return BatchConverter(self)

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks
        elif name in ("CodonModel",):
            standard_toks = codonseq_toks

        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks)

    def tokens_to_id(self, tokens: list[str]) -> list[int]:
        return [self.get_idx(tok) for tok in tokens]

    def tokens_ok(self, tokens: list[str]) -> bool:
        return not bool(set(tokens) - self._unique_no_split_tokens)


class BatchConverter:
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet: Alphabet):
        self.alphabet = alphabet

    def from_tokens(self, tokens: Sequence[list[str]]) -> torch.Tensor:
        return self._tokens_to_tensor(
            [self.alphabet.tokens_to_id(seq_str) for seq_str in tokens]
        )

    def __call__(self, tokens: Sequence[list[str]]) -> torch.Tensor:
        return self.from_tokens(tokens)

    def _tokens_to_tensor(self, seq_encoded_list: Sequence[list[int]]) -> torch.Tensor:
        batch_size = len(seq_encoded_list)
        max_len = len(max(seq_encoded_list, key=len))
        tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)

        for i, seq_encoded in enumerate(seq_encoded_list):
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, 0 : len(seq_encoded)] = seq

        return tokens
