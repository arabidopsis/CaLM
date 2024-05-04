from typing import IO
import click


@click.group()
def calm():
    pass


@calm.command()
@click.option("-t", "--torch", "is_torch", is_flag=True)
@click.argument("weights_file", type=click.Path(dir_okay=False))
@click.argument("fasta_file", type=click.Path(dir_okay=False))
def to_tensor(weights_file: str, fasta_file: str, is_torch: bool) -> None:
    """Convert cDNA fasta sequences into CaLM Tensors"""
    from ..fasta import RandomFasta
    from .pretrained import TrainedModel, BareModel

    c: TrainedModel
    if is_torch:
        c = BareModel(weights_file)
    else:
        c = TrainedModel(weights_file)
    for rec in RandomFasta(fasta_file):
        r = c.to_tensor(rec.seq)
        t = r.mean(axis=1)  # type: ignore
        print(rec.id, t)


@calm.command()
@click.argument("weights_file", type=click.Path(dir_okay=False))
@click.argument("out", type=click.Path(dir_okay=False))
def convert(weights_file: str, out: str) -> None:
    """Convert lighting Model to torch model data"""
    import torch
    from .pretrained import TrainedModel

    c = TrainedModel(weights_file)
    model = c.model
    data = {"hyper_parameters": c.hyper_parameters, "state_dict": model.state_dict()}
    torch.save(data, out)


@calm.command()
@click.option("-s", "--start", default=0, help="start")
@click.option(
    "-n", "--number", default=200, help="number of sequences", show_default=True
)
@click.argument("fasta_file", type=click.File(mode="rt"))
@click.argument("out", type=click.File(mode="wt"))
def fasta(fasta_file: IO[str], out: IO[str], number: int, start: int) -> None:
    """slice a few sequences out of a fasta file for testing data"""
    from Bio import SeqIO
    from itertools import islice

    SeqIO.write(islice(SeqIO.parse(fasta_file, "fasta"), start, number), out, "fasta")


@calm.command()
@click.option(
    "--max-depth",
    default=-1,
    help="see pytorch_lightning.utilities.model_summary.ModelSummary",
)
@click.argument("checkpoint", type=click.Path(dir_okay=False))
def summary(checkpoint: str, max_depth: int) -> None:
    """Summary of model from checkpoint"""
    from pytorch_lightning.utilities.model_summary import ModelSummary
    from .training import CodonModel

    model = CodonModel.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint
    )
    click.echo(ModelSummary(model, max_depth=max_depth))


@calm.command()
@click.argument("fasta_files", type=click.Path(dir_okay=False), nargs=-1)
def rna_ok(fasta_files: tuple[str, ...]) -> None:
    from ..sequence import rna_ok
    from ..fasta import nnfastas

    nbad = 0
    fastaf = nnfastas(fasta_files)
    for rec in fastaf:
        msg = rna_ok(rec.seq)
        if msg:
            nbad += 1
            click.secho(f"{rec.id}: {msg}", fg="red")
    click.secho(f"{nbad}/{len(fasta)} bad", fg="red" if nbad else "green")


if __name__ == "__main__":
    calm()  # pylint: disable=no-value-for-parameter
