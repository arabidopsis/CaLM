"""Utilities to preprocess data for training."""

from dataclasses import dataclass
import abc
from copy import deepcopy
from typing import NamedTuple, Iterator, TypedDict


import torch
import numpy as np


from .alphabet import Alphabet
from .sequence import Sequence


@dataclass
class PipelineCfg:
    mask_proportion: float = 0.25
    leave_percent: float = 0.1
    mask_percent: float = 0.8
    max_positions: int = 1024


def _split_array(array: np.ndarray, chunks: list[int]) -> list[np.ndarray]:
    """Split an array into N chunks of defined size."""
    assert np.sum(chunks) == len(array)
    acc = 0
    arrays = []
    for chunk in chunks:
        arrays.append(array[acc : acc + chunk])
        acc += chunk
    return arrays


# PipelineInput = namedtuple("PipelineInput", ["sequence"])
# PipelineOutput = namedtuple("PipelineOutput", ["input", "labels", "ground_truth"])
# _PipelineData = namedtuple("_PipelineData", ["ground_truth", "sequence", "target_mask"])


class PipelineInput(NamedTuple):
    sequence: list[Sequence]


class _PipelineData(NamedTuple):
    ground_truth: list[str]
    sequence: list[str]
    target_mask: list[np.ndarray]


class PipelineOutput(TypedDict):
    input: torch.Tensor
    labels: torch.Tensor
    ground_truth: list[str]


class PipelineData(_PipelineData):
    """Data structure for inner pipeline data."""

    @property
    def size(self):
        """Number of sequences in the data, equivalent
        to batch size."""
        assert len(self.ground_truth) == len(self.sequence)
        assert len(self.ground_truth) == len(self.target_mask)
        return len(self.ground_truth)

    def iterate(self) -> Iterator[tuple[str, str, np.ndarray]]:
        """Iterate over the data."""
        for i in range(self.size):
            yield self.ground_truth[i], self.sequence[i], self.target_mask[i]


class PipelineBlock(abc.ABC):
    """Base class for data preprocessing pipeline blocks."""

    @abc.abstractmethod
    def __call__(self, input_: PipelineData) -> PipelineData:
        """Apply the block to a sequence."""
        raise NotImplementedError


class PipelineEntrypoint(abc.ABC):
    """Starting point for a pipeline."""

    @abc.abstractmethod
    def __call__(self, input_: PipelineInput) -> PipelineData:
        """Apply the block to a sequence."""
        raise NotImplementedError


class PipelineEndpoint(abc.ABC):
    """Final point for a pipeline."""

    @abc.abstractmethod
    def __call__(self, input_: PipelineData) -> PipelineOutput:
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

        data: PipelineInput | PipelineData | PipelineOutput = PipelineInput(
            sequence=data_
        )
        for transform in self.pipeline:
            data = transform(data)  # type: ignore
        return data  # type: ignore


class DataCollator(PipelineEntrypoint):
    """Class to process sequences and apply random masking. The output
    of a call to DataCollator are strings of tokens, separated by spaces,
    and arrays which are zero except where a token change has occurred."""

    def __init__(
        self,
        cfg: PipelineCfg,
        alphabet: Alphabet,
    ):
        self.cfg = cfg
        self.alphabet = alphabet

    def __call__(self, input_: PipelineInput) -> PipelineData:
        output = PipelineData(ground_truth=[], sequence=[], target_mask=[])

        for seq in input_.sequence:
            tokens, mask = self._mask_seq(seq.tokens)
            output.target_mask.append(mask)
            output.sequence.append(" ".join(tokens))
            output.ground_truth.append(" ".join(seq.tokens))

        return output

    def _mask_seq(self, tokens_: list[str]) -> tuple[list[str], np.ndarray]:
        tokens = deepcopy(tokens_)
        num_tokens = len(tokens)
        num_changed_tokens = int(num_tokens * self.cfg.mask_proportion)
        num_to_mask = int(num_changed_tokens * self.cfg.mask_percent)
        num_to_leave = int(num_changed_tokens * self.cfg.leave_percent)
        num_to_change = num_changed_tokens - num_to_mask - num_to_leave

        # Apply masking
        idxs = np.random.choice(
            np.arange(1, num_tokens - 1),  # avoid <cls> and <eos>
            size=num_changed_tokens,
            replace=False,
        )
        idxs_mask, _, idxs_change = _split_array(
            idxs, [num_to_mask, num_to_leave, num_to_change]
        )
        for idx_mask in idxs_mask:
            tokens[idx_mask] = "<mask>"
        for idx_change in idxs_change:
            tokens[idx_change] = np.random.choice(self.alphabet.coding_toks)

        # Generate masks
        mask = np.zeros(num_tokens)
        mask[idxs] = 1.0

        return tokens, mask


class DataTrimmer(PipelineBlock):
    """Class to trim sequences. Returns sequences and masks that have
    been trimmed to the maximum number of positions of the model."""

    def __init__(self, max_positions: int, alphabet: Alphabet):
        self.max_positions = max_positions
        self.alphabet = alphabet

    def __call__(self, input_: PipelineData) -> PipelineData:
        output = PipelineData(ground_truth=[], sequence=[], target_mask=[])

        for ground_truth, sequence, target_mask in input_.iterate():
            ground_truth, sequence, target_mask = self._trim_seq(
                ground_truth, sequence, target_mask
            )
            output.ground_truth.append(ground_truth)
            output.sequence.append(sequence)
            output.target_mask.append(target_mask)

        return output

    def _trim_seq(
        self, original_seq: str, masked_seq: str, mask: np.ndarray
    ) -> tuple[str, str, np.ndarray]:

        original_tokens = original_seq.split()
        masked_tokens = masked_seq.split()
        n_tokens = len(original_tokens)
        if n_tokens <= self.max_positions:
            return original_seq, masked_seq, mask
        else:
            start = np.random.randint(0, n_tokens - self.max_positions)
            end = start + self.max_positions
            new_original_seq = " ".join(original_tokens[start:end])
            new_masked_seq = " ".join(masked_tokens[start:end])
            new_mask = mask[start:end]
            return new_original_seq, new_masked_seq, new_mask


class DataPadder(PipelineBlock):
    """Class to pad sequences."""

    def __init__(self, max_positions: int, alphabet: Alphabet) -> None:
        self.max_positions = max_positions
        self.alphabet = alphabet

    def __call__(self, input_: PipelineData) -> PipelineData:
        output = PipelineData(ground_truth=[], sequence=[], target_mask=[])

        # max_positions = max(len(seq.split()) for seq in input_.sequence)

        for ground_truth, sequence, target_mask in input_.iterate():
            ground_truth, sequence, target_mask = self._pad_seq(
                ground_truth, sequence, target_mask, max_positions=self.max_positions
            )
            output.ground_truth.append(ground_truth)
            output.sequence.append(sequence)
            output.target_mask.append(target_mask)

        return output

    def _pad_seq(
        self,
        original_seq: str,
        masked_seq: str,
        mask: np.ndarray,
        max_positions: int,
    ) -> tuple[str, str, np.ndarray]:
        n_tokens = len(original_seq.split())
        if len(masked_seq.split()) < max_positions:
            original_seq_ = " ".join(
                original_seq.split() + ["<pad>"] * (max_positions - n_tokens)
            )
            masked_seq_ = " ".join(
                masked_seq.split() + ["<pad>"] * (max_positions - n_tokens)
            )
            mask_ = np.concatenate([mask, np.zeros(max_positions - n_tokens)])
            return original_seq_, masked_seq_, mask_
        else:
            return original_seq, masked_seq, mask


class DataPreprocessor(PipelineEndpoint):
    """Class to transform tokens into PyTorch Tensors."""

    def __init__(self, alphabet: Alphabet):
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, input_: PipelineData) -> PipelineOutput:
        new_input = self._compute_input(input_.sequence)
        labels = self._compute_input(input_.ground_truth)
        mask = self._compute_mask(input_.target_mask)
        labels[~mask.bool()] = -100
        return PipelineOutput(
            input=new_input, labels=labels, ground_truth=input_.ground_truth
        )

    def _compute_input(self, seq_list: list[str]) -> torch.Tensor:
        _, _, input_ = self.batch_converter([("", seq) for seq in seq_list])
        return input_.to(dtype=torch.int32)

    def _compute_mask(self, mask_list: list[np.ndarray]) -> torch.Tensor:
        return torch.tensor(np.stack(mask_list, axis=0))


def standard_pipeline(cfg: PipelineCfg, alphabet: Alphabet) -> Pipeline:
    return Pipeline(
        [
            DataCollator(
                cfg,
                alphabet,
            ),
            DataTrimmer(cfg.max_positions, alphabet),
            DataPadder(cfg.max_positions, alphabet),
            DataPreprocessor(alphabet),
        ]
    )
