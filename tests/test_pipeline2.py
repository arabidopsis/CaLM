import click
from itertools import islice
from calm.pipeline2 import PipelineCfg, standard_pipeline, Pipeline, MaskAndChange
from calm.alphabet import Alphabet
from calm.sequence import CodonSequence, Sequence


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            break
        yield batch


@click.command()
@click.argument("fasta_file", type=click.Path(dir_okay=False))
def pipeline2(fasta_file: str) -> None:
    from calm.fasta import nnfastas

    alphabet = Alphabet.from_architecture("CodonModel")
    cfg = PipelineCfg()
    pipeline = standard_pipeline(cfg, alphabet)
    mac = MaskAndChange(cfg, alphabet.coding_toks)

    fasta = nnfastas([fasta_file])
    lengths = set()
    for batch in batched(fasta, 20):
        iseqs: list[Sequence] = [CodonSequence(s.seq) for s in batch]
        oseqs = pipeline(iseqs)
        seq_list = mac(iseqs)
        gt = oseqs["ground_truth"]
        ip = oseqs["input"]

        assert ip.size() == (len(batch), cfg.max_positions), ip.size()
        assert ip.size() == oseqs["labels"].size()
        assert len(iseqs) == len(gt)
        assert len(seq_list) == len(batch)
        for seqin, seqout, sinfo in zip(iseqs, gt, seq_list):

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
    assert len(lengths) == 1, lengths
    assert lengths.pop() == cfg.max_positions


if __name__ == "__main__":
    pipeline2()  # pylint: disable=no-value-for-parameter
