"""Utilities to preprocess data for training."""

from dataclasses import dataclass
import abc
from copy import deepcopy
from typing import NamedTuple, TypedDict, cast
from typing_extensions import NotRequired

import torch
from torch.utils.data import random_split, Dataset
import numpy as np


from .alphabet import Alphabet
from .sequence import Sequence


@dataclass
class PipelineCfg:
    mask_proportion: float = 0.25
    leave_percent: float = 0.1
    mask_percent: float = 0.8
    max_positions: int = 1024


# def _split_array(array: np.ndarray, chunks: list[int]) -> list[np.ndarray]:
#     """Split an array into N chunks of defined size."""
#     assert np.sum(chunks) == len(array)
#     acc = 0
#     arrays = []
#     for chunk in chunks:
#         arrays.append(array[acc : acc + chunk])
#         acc += chunk
#     return arrays


class PipelineInput(NamedTuple):
    sequences: list[Sequence]


class SeqInfo(NamedTuple):
    ground_truth: list[str]
    masked_seq: list[str]
    target_mask: np.ndarray


class PipelineOutput(TypedDict):
    input: torch.Tensor
    labels: torch.Tensor
    ground_truth: list[list[str]]
    seq_list: NotRequired[list[SeqInfo]]


class PipelineBlock(abc.ABC):
    """Base class for data preprocessing pipeline blocks."""

    @abc.abstractmethod
    def __call__(self, input_: list[SeqInfo]) -> list[SeqInfo]:
        """Apply the block to a sequence."""
        raise NotImplementedError


class PipelineEntrypoint(abc.ABC):
    """Starting point for a pipeline."""

    @abc.abstractmethod
    def __call__(self, input_: list[Sequence]) -> list[SeqInfo]:
        """Apply the block to a sequence."""
        raise NotImplementedError


class PipelineEndpoint(abc.ABC):
    """Final point for a pipeline."""

    @abc.abstractmethod
    def __call__(self, input_: list[SeqInfo]) -> PipelineOutput:
        """Apply the block to the data."""
        raise NotImplementedError


class Pipeline:
    """Class to preprocess data for training.

    This class is used to preprocess data for training. It is a pipeline of
    transformations that are applied to the data. The pipeline is defined by a
    list of callables that are applied in order.
    """

    def __init__(
        self, pipeline: list[PipelineEntrypoint | PipelineBlock | PipelineEndpoint]
    ):
        """Initialize the pipeline.

        Args:
            pipeline: List of callables that are applied in order.
        """

        if not issubclass(type(pipeline[0]), PipelineEntrypoint):
            raise ValueError("First block in a pipeline must be PipelineEntrypoint.")
        for block in pipeline[1:-1]:
            if issubclass(type(block), PipelineEntrypoint) or issubclass(
                type(block), PipelineEndpoint
            ):
                raise ValueError(
                    "Intermediate blocks cannot be PipelineEntrypoint or PipelineEndpoint."
                )
        self.pipeline = pipeline

    def __call__(self, data_: list[Sequence]) -> PipelineOutput:
        """Apply the pipeline to the data.

        Args:
            data: Data to apply the pipeline to.

        Returns:
            Data after the pipeline has been applied.
        """

        data: list[Sequence] | list[SeqInfo] | PipelineOutput = data_
        for transform in self.pipeline:
            data = transform(data)  # type: ignore
        return data  # type: ignore


class MaskAndChange(PipelineEntrypoint):
    """Class to process sequences and apply random masking. The output
    of a call to MaskAndChange are strings of tokens, separated by spaces,
    and arrays which are zero except where a token change has occurred."""

    def __init__(
        self,
        cfg: PipelineCfg,
        coding_toks: list[str],
    ):
        self.cfg = cfg
        self.coding_toks = coding_toks

    def __call__(self, input_: list[Sequence]) -> list[SeqInfo]:
        output = []

        for seq in input_:
            tokens = seq.tokens  # this splits seq.seq
            tokens, mask = self._mask_seq(tokens)
            output.append(SeqInfo(seq.tokens, tokens, mask))

        return output

    def _mask_seq(self, tokens_: list[str]) -> tuple[list[str], np.ndarray]:
        tokens = deepcopy(tokens_)
        num_tokens = len(tokens)
        num_changed_tokens = int(num_tokens * self.cfg.mask_proportion)
        num_to_mask = int(num_changed_tokens * self.cfg.mask_percent)
        num_to_leave = int(num_changed_tokens * self.cfg.leave_percent)
        num_to_change = num_changed_tokens - num_to_mask - num_to_leave

        # Apply masking
        idxs: np.ndarray[int] = np.random.choice(
            np.arange(1, num_tokens - 1),  # avoid <cls> and <eos>
            size=num_changed_tokens,
            replace=False,
        )
        # idxs_mask, _, idxs_change = _split_array(
        #     idxs, [num_to_mask, num_to_leave, num_to_change]
        # )

        idxs_mask, _, idxs_change = random_split(
            cast(Dataset[int], idxs), [num_to_mask, num_to_leave, num_to_change]
        )

        for idx_mask in iter(idxs_mask):
            tokens[idx_mask] = "<mask>"
        for idx_change in iter(idxs_change):
            tokens[idx_change] = np.random.choice(self.coding_toks)

        # Generate masks
        mask = np.zeros(num_tokens)
        mask[idxs] = 1.0

        return tokens, mask


class DataTrimmer(PipelineBlock):
    """Class to trim sequences. Returns sequences and masks that have
    been trimmed to the maximum number of positions of the model."""

    def __init__(self, max_positions: int):
        self.max_positions = max_positions

    def __call__(self, input_: list[SeqInfo]) -> list[SeqInfo]:

        return [self._trim_seq(seqinfo) for seqinfo in input_]

    def _trim_seq(
        self,
        sinfo: SeqInfo,
    ) -> SeqInfo:

        n_tokens = len(sinfo.ground_truth)
        if n_tokens <= self.max_positions:
            return sinfo
        else:

            start = np.random.randint(0, n_tokens - self.max_positions)
            end = start + self.max_positions
            new_original_seq = sinfo.ground_truth[start:end]
            new_masked_seq = sinfo.masked_seq[start:end]
            new_mask = sinfo.target_mask[start:end]
            return SeqInfo(new_original_seq, new_masked_seq, new_mask)


class DataPadder(PipelineBlock):
    """Class to pad sequences."""

    def __init__(self, max_positions: int) -> None:
        self.max_positions = max_positions

    def __call__(self, input_: list[SeqInfo]) -> list[SeqInfo]:

        return [self._pad_seq(seqinfo) for seqinfo in input_]

    def _pad_seq(
        self,
        sinfo: SeqInfo,
    ) -> SeqInfo:

        npadding = self.max_positions - len(sinfo.ground_truth)
        if npadding > 0:
            padding = ["<pad>"] * npadding
            ground_truth = sinfo.ground_truth + padding
            masked_seq = sinfo.masked_seq + padding
            mask = np.concatenate([sinfo.target_mask, np.zeros(npadding)])
            return SeqInfo(ground_truth, masked_seq, mask)
        else:
            return sinfo


class DataPreprocessor(PipelineEndpoint):
    """Class to transform tokens into PyTorch Tensors."""

    def __init__(self, alphabet: Alphabet):
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, seqs_list: list[SeqInfo]) -> PipelineOutput:

        ground_truth = [s.ground_truth for s in seqs_list]
        new_input = self.batch_converter.from_tokens([s.masked_seq for s in seqs_list])
        labels = self.batch_converter.from_tokens(ground_truth)
        mask = torch.tensor(np.stack([s.target_mask for s in seqs_list], axis=0))
        labels[~mask.bool()] = -100
        return PipelineOutput(
            input=new_input.to(dtype=torch.int32),
            labels=labels,
            ground_truth=ground_truth,
        )


def standard_pipeline(cfg: PipelineCfg, alphabet: Alphabet) -> Pipeline:
    return Pipeline(
        [
            MaskAndChange(
                cfg,
                alphabet.coding_toks,  # for replacement tokens
            ),
            DataTrimmer(cfg.max_positions),
            DataPadder(cfg.max_positions),
            DataPreprocessor(alphabet),  # for batch_coverter
        ]
    )


def test_pipeline(max_positions: int = 100) -> Pipeline:
    return standard_pipeline(
        PipelineCfg(max_positions=max_positions),
        Alphabet.from_architecture("CodonModel"),
    )
