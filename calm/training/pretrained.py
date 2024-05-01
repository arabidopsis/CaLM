import torch

from calm.sequence import CodonSequence
from calm.model import ProteinBertModel
from .training import CodonModel


class TrainedModel:
    Model: type[torch.nn.Module] = CodonModel

    def __init__(self, weights_file: str):
        data = torch.load(weights_file)
        self.hyper_parameters = data["hyper_parameters"]
        model = self.Model(**self.hyper_parameters)
        model.load_state_dict(data["state_dict"])
        self.model = self.get_inner_model(model)
        self.bc = self.model.alphabet.get_batch_converter()

    def get_inner_model(self, model):
        return model.model

    def tokenize(self, seq: CodonSequence) -> torch.Tensor:
        _, _, tokens = self.bc([("", seq.seq)])
        return tokens

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

    def get_inner_model(self, model):
        return model
