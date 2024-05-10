from itertools import islice

from typing import Iterator, Iterable, TypeVar
import click
from calm.pipeline2 import (
    PipelineCfg,
    standard_pipeline,
    MaskAndChange,
    DataTrimmer,
    DataPadder,
    Pipeline,
    SeqInfo,
    show,
)
from calm.utils import add_args, create_from_ns, batched


from calm.alphabet import Alphabet
from calm.sequence import CodonSequence, BioSequence

T = TypeVar("T")


@click.command()
@click.option("--verbose", is_flag=True)
@click.option("--width", type=int)
@click.option("-c", "--compact", is_flag=True, help="use compact layout for display")
@click.argument("fasta_file", type=click.Path(dir_okay=False))
@click.argument("configuration", nargs=-1)
def pipeline2(
    fasta_file: str,
    verbose: bool,
    configuration: tuple[str, ...],
    compact: bool,
    width: int | None,
) -> None:
    from calm.fasta import nnfastas
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = add_args(parser, PipelineCfg)
    ns = parser.parse_args(configuration + ("--with-ground-truths",))

    alphabet = Alphabet.from_architecture("CodonModel")
    cfg = create_from_ns(ns, PipelineCfg)
    pipeline = standard_pipeline(cfg, alphabet)
    mac = MaskAndChange(cfg, alphabet.coding_toks)
    p2 = Pipeline([mac, DataTrimmer(cfg.max_positions), DataPadder(cfg.max_positions)])
    if compact:
        join_str = ""
        if width is None:
            width = 40
    else:
        join_str = " "
        if width is None:
            width = 20
    fasta = nnfastas([fasta_file])
    if verbose:
        click.echo(str(cfg))
    lengths = set()
    for batch in batched(fasta, 20):
        iseqs: list[BioSequence] = [CodonSequence(s.seq) for s in batch]
        oseqs = pipeline(iseqs)
        seq_list = mac(iseqs)
        olist: list[SeqInfo] = p2(iseqs)  # type: ignore
        gt = oseqs["ground_truths"]
        ip = oseqs["input"]

        assert ip.size() == (len(batch), cfg.max_positions), ip.size()
        assert ip.size() == oseqs["labels"].size()
        assert len(iseqs) == len(gt)
        assert len(seq_list) == len(batch)
        assert len(olist) == len(iseqs)
        mxlen = 0
        for seqin, seqout, sinfo, sinfo2, rec in zip(iseqs, gt, seq_list, olist, batch):
            mxlen = max(len(seqin.tokens), mxlen)
            lengths.add(len(seqout))
            if len(seqin.tokens) <= len(seqout):
                if not seqin.seq in " ".join(seqout):
                    raise ValueError(
                        "input sequence not a subset of output ground_truth"
                    )
            nmask = sinfo.masked_seq.count("<mask>")
            mmask = int(
                len(sinfo.ground_truth) * cfg.mask_proportion * cfg.mask_percent
            )
            assert mmask in {nmask - 1, nmask, nmask + 1}, (
                nmask,
                mmask,
                sinfo.ground_truth,
            )
            padding = sinfo.masked_seq.count("<pad>")
            if padding:
                assert len(seqin.tokens) + padding == cfg.max_positions
            if verbose:
                print(f">{rec.description}")
                show(
                    sinfo2,
                    seqin,
                    join_str=join_str,
                    width=width,
                )
    # all sequences of same length
    assert len(lengths) == 1, lengths
    assert lengths.pop() == cfg.max_positions
    if verbose:
        click.echo(f"max token length={mxlen}/{cfg.max_positions}")
    click.secho("OK", fg="green", bold=True)


if __name__ == "__main__":
    pipeline2()  # pylint: disable=no-value-for-parameter
