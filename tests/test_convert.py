import click
from pathlib import Path


@click.command()
@click.option("--out")
@click.argument("fasta_file", type=click.Path(dir_okay=False))
def convert(fasta_file: str, out: str | None | Path):
    import torch

    from calm.fasta import nnfastas
    from calm import CaLM

    cm = CaLM()
    fasta = nnfastas(fasta_file)
    ret = []
    for rec in fasta:
        t = cm.embed_sequence(rec.seq)
        ret.append(t)

    full = torch.cat(ret, dim=0)
    p = Path(fasta_file)
    if out is None:
        out = p.parent / (p.stem + ".pt")

    click.secho(f"saving {out}")
    torch.save(full, out)


if __name__ == "__main__":
    convert()  # pylint: disable=no-value-for-parameter
