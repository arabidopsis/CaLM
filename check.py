from calm.sequence import CodonSequence
from calm.model import ProteinBertModel
import torch
from training import CodonModel
import click
from Bio import SeqIO


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
    Model= ProteinBertModel

    def get_inner_model(self, model):
        return model

@click.group()
def calm():
    pass


@calm.command()
@click.option('--torch', 'is_torch', is_flag=True)
@click.argument("weights_file", type=click.Path(dir_okay=False))
@click.argument("fasta", type=click.Path(dir_okay=False))
def to_tensor(weights_file: str, fasta: str, is_torch:bool) -> None:
    """Convert cDNA fasta sequences into CaLM Tensors"""
    c: TrainedModel
    if is_torch:
        c = BareModel(weights_file)
    else:
        c = TrainedModel(weights_file)
    with open(fasta, "rt", encoding="ascii") as fp:
        for rec in SeqIO.parse(fp, "fasta"):
            r = c.to_tensor(rec.seq)
            t = r.mean(axis=1)
            print(rec.id, t)


@calm.command()
@click.argument("weights_file", type=click.Path(dir_okay=False))
@click.argument("out", type=click.Path(dir_okay=False))
def convert(weights_file: str, out: str) -> None:
    """Convert lighting Model to torch model data"""
    c = TrainedModel(weights_file)
    model = c.model
    data = {"hyper_parameters": c.hyper_parameters, "state_dict": model.state_dict()}
    torch.save(data, out)


if __name__ == "__main__":
    calm()  # pylint: disable=no-value-for-parameter
