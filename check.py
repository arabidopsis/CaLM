from calm.sequence import CodonSequence
import torch
from training import CodonModel
import click
from Bio import SeqIO


class Generate:

    def __init__(self, weights_file: str):
        data = torch.load(weights_file)
        model = CodonModel(**data["hyper_parameters"])
        model.load_state_dict(data["state_dict"])
        self.model = model
        self.bc = model.alphabet.get_batch_converter()

    def tokenize(self, seq: CodonSequence) -> torch.Tensor:
        _, _, tokens = self.bc([("", seq.seq)])
        return tokens

    def to_tensor(self, seq: str) -> torch.Tensor:
        model = self.model
        s = CodonSequence(seq)
        r = model.model(self.tokenize(s), repr_layers=[model.args.num_layers])[
            "representations"
        ][model.args.num_layers]
        return r


@click.command()
@click.argument("weights_file", type=click.Path(dir_okay=False))
@click.argument("fasta", type=click.Path(dir_okay=False))
def check(weights_file: str, fasta: str) -> None:

    data = torch.load(weights_file)
    model = CodonModel(**data["hyper_parameters"])
    model.load_state_dict(data["state_dict"])

    c = Generate(weights_file)
    with open(fasta, "rt", encoding="ascii") as fp:
        for rec in SeqIO.parse(fp, "fasta"):
            r = c.to_tensor(rec.seq)
            t = r.mean(axis=1)
            print(rec.id, t)


if __name__ == "__main__":
    check()
