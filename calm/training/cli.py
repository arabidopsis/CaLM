import click


@click.group()
def calm():
    pass


@calm.command()
@click.option("--torch", "is_torch", is_flag=True)
@click.argument("weights_file", type=click.Path(dir_okay=False))
@click.argument("fasta", type=click.Path(dir_okay=False))
def to_tensor(weights_file: str, fasta: str, is_torch: bool) -> None:
    """Convert cDNA fasta sequences into CaLM Tensors"""
    from Bio import SeqIO
    from calm.training.pretrained import TrainedModel, BareModel

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
    import torch
    from calm.training.pretrained import TrainedModel

    c = TrainedModel(weights_file)
    model = c.model
    data = {"hyper_parameters": c.hyper_parameters, "state_dict": model.state_dict()}
    torch.save(data, out)


if __name__ == "__main__":
    calm()  # pylint: disable=no-value-for-parameter
