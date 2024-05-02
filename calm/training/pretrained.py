import torch

from ..sequence import CodonSequence, Sequence
from ..model import ProteinBertModel
from .training import CodonModel


class TrainedModel:
    Model: type[torch.nn.Module] = CodonModel

    def __init__(self, weights_file: str):
        data = torch.load(weights_file)
        self.hyper_parameters = data["hyper_parameters"]
        model = self.Model(**self.hyper_parameters)
        model.load_state_dict(data["state_dict"])
        self.model: ProteinBertModel = self.get_inner_model(model)
        self.model.eval()
        self.batch_converter = self.model.alphabet.get_batch_converter()

    def get_inner_model(self, model: torch.nn.Module) -> ProteinBertModel:
        return model.model

    def tokenize(self, seq: Sequence) -> torch.Tensor:
        return self.batch_converter.from_seq(seq.seq)


    def to_tensor(self, seq: str) -> torch.Tensor:
        model = self.model
        repr_layer = model.cfg.num_layers
        s = CodonSequence(seq)
        r = model(self.tokenize(s), repr_layers=[repr_layer])["representations"][
            repr_layer
        ]
        return r


class BareModel(TrainedModel):
    Model = ProteinBertModel

    def get_inner_model(self, model: torch.nn.Module) -> ProteinBertModel:
        return model  # type: ignore
