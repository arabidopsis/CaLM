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


@calm.command(name="cds-ok")
@click.argument("fasta_files", type=click.Path(dir_okay=False), nargs=-1)
def cds_ok_cmd(fasta_files: tuple[str, ...]) -> None:
    from ..sequence import cds_ok, CodonSequence
    from ..fasta import nnfastas
    from ..alphabet import Alphabet

    alphabet = Alphabet.from_architecture("CodonModel")
    bc = alphabet.get_batch_converter()
    nbad = 0
    nunknown = 0
    fastaf = nnfastas(fasta_files)
    with click.progressbar(fastaf, length=len(fastaf)) as bar:
        for idx, rec in enumerate(bar, start=1):
            msg = cds_ok(rec.seq)
            if msg:
                nbad += 1
                click.secho(f"{rec.id}: {msg}", fg="red")
            tensor = bc.from_tokens([CodonSequence(rec.seq).tokens])
            n = (tensor == alphabet.unk_idx).sum()
            if n:
                nunknown += 1
            #     print('unknown',n)
            # print(tensor)
            # click.secho(f"{nbad}/{idx}")
    click.secho(
        f"{nbad}/{len(fastaf)} bad, unknown tokens={nunknown}",
        fg="red" if nbad else "green",
    )


@calm.command()
@click.option(
    "--out", help="output CSV filename", default="result.csv", show_default=True
)
@click.option("-b", "--batch-size", help="batch size", default=10, show_default=True)
@click.option("-x", "--without-progress-bar", 'without_pb', help="no progress bar", is_flag=True)
@click.option(
    "-r",
    "--round",
    "dec",
    default=6,
    help="round tensor values down",
    show_default=True,
)
@click.argument("fasta_files", type=click.Path(dir_okay=False), nargs=-1)
def fasta_to_tensors(
    fasta_files: tuple[str, ...], out: str, batch_size: int, dec: int, without_pb: bool
) -> None:
    # import pandas as pd
    import gzip
    from typing import TextIO, Protocol, Iterable, Any
    import csv
    from ..utils import batched
    from ..fasta import nnfastas, Record
    from ..pretrained import CaLM
    class CSVWriter(Protocol):
        def writerow(self, row: Iterable[Any]) -> Any:
            ...
        def writerows(self, rows: Iterable[Iterable[Any]]) -> None:
            ...
    # from ..sequence import CodonSequence

    cm = CaLM()

    def dobatch(batch: tuple[Record, ...], tgt: CSVWriter):
        # t = cm.tokenize_batch([CodonSequence(s.seq) for s in batch])
        tensor = cm.embed_sequences([s.seq for s in batch], average=True)
        npt = tensor.cpu().detach().numpy()  # float32
        assert len(npt) == len(batch)
        for rec, trow in zip(batch, npt):
            vlist = trow.tolist()
            if dec > 0:
                vlist = [round(t, dec) for t in vlist]
            row = [rec.id] + vlist
            tgt.writerow(row)

    def outf() -> TextIO:
        if out.endswith(".gz"):
            return gzip.open(out, mode="wt", encoding="utf8")
        return open(out, "wt", encoding="utf8")

    fastaf = nnfastas(fasta_files)
    nmem = len(fastaf) * 786 * 4
    click.secho(f"translating {len(fastaf)} sequences mem={nmem}")
    # res = []
    with outf() as fp:
        tgt = csv.writer(fp)
        # tgt.writerow(["id"] + [f"col_{n}" for n in range(768)])
        if not without_pb:
            with click.progressbar(fastaf, length=len(fastaf)) as it:
                for batch in batched(it, batch_size):
                    dobatch(batch, tgt)
                    fp.flush()
        else:
            for ib,batch in enumerate(batched(fastaf, batch_size),1):
                dobatch(batch, tgt)
                fp.flush()
                click.secho(f"done: {ib * batch_size}/{len(fastaf)}")
            # res.extend([dict(id=s.id, embedding=t) for s, t in zip(batch, npt)])

    # df = pd.DataFrame(res)
    # df.to_parquet(out)


if __name__ == "__main__":
    calm()  # pylint: disable=no-value-for-parameter
