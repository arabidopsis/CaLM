"""Implementation of the Alphabet and BatchConverter classes.

This code has been modified from the original implementation
by Facebook Research, describing its ESM-1b paper."""

import itertools
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
        self.standard_toks = tokens["toks"]
        self.coding_toks = tokens["coding_toks"]

        self.prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
        self.append_toks = ("<mask>",)
        # self.all_special_tokens = ("<eos>", "<unk>", "<pad>", "<cls>", "<mask>")
        self.prepend_bos = False
        self.append_eos = False

        self.all_toks = (*self.prepend_toks, *self.standard_toks, *self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
        self.unique_no_split_tokens = set(self.all_toks)

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok: str) -> int:
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind: int) -> str:
        return self.all_toks[ind]

    # def to_dict(self):
    #     return self.tok_to_idx.copy()

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

    def _tokenize(self, text: str) -> list[str]:
        return text.split()

    def tokenize(self, text: str, **kwargs) -> list[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok: str, text: str) -> list[str]:
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list: set[str], text: str) -> list[str]:
            if not text.strip():
                return []

            tokenized_text: list[str] = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        (
                            self._tokenize(token)
                            if token not in self.unique_no_split_tokens
                            else [token]
                        )
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text: str) -> list[int]:
        return self.tokens2id(self.tokenize(text))

    def tokens2id(self, tokens: list[str]) -> list[int]:
        return [self.get_idx(tok) for tok in tokens]

    def tokens_ok(self, tokens: list[str]) -> bool:
        return not bool(set(tokens) - self.unique_no_split_tokens)


class BatchConverter:
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet: Alphabet):
        self.alphabet = alphabet

    def from_seq(self, seq: str) -> torch.Tensor:
        _, _, tokens = self([("", seq)])
        return tokens

    def from_seqs(self, seqs: Sequence[str]) -> torch.Tensor:
        _, _, tokens = self([("", seq) for seq in seqs])
        return tokens

    def __call__(
        self, raw_batch: Sequence[tuple[str, str]]
    ) -> tuple[list[str], list[str], torch.Tensor]:
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        max_len = len(max(seq_encoded_list, key=len))
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = (
                    self.alphabet.eos_idx
                )

        return labels, strs, tokens
