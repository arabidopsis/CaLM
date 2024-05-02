import sys
import random
from calm.fasta import RandomFasta, CollectionFasta
from Bio import SeqIO


def ok(r, rec):
    assert r.id == rec.id
    assert r.seq == str(rec.seq)
    assert r.description == rec.description


def test():

    ffs = sys.argv[1:]

    fasta = CollectionFasta(ffs) if len(ffs) > 1 else RandomFasta(ffs[0])

    full = []
    for ff in ffs:
        with open(ff, encoding="utf8") as h:
            for rec in SeqIO.parse(h, "fasta"):
                full.append(rec)

    for i, rec in enumerate(full):
        r = fasta[i]
        ok(r, rec)
    assert len(full) == len(fasta)
    total = len(full)
    print(total)
    for s, e in [
        (s, random.randint(s + 1, total)) for s in random.sample(range(0, total), min(2000, total))
    ]:
        if s == e:
            continue
        recs = full[s:e]
        rs = fasta[s:e]
        for rec, r in zip(recs, rs):
            ok(r, rec)

    for rl in [
        random.sample(range(0, total), random.randint(10, 60)) for _ in range(40)
    ]:
        rs = fasta[rl]
        recs = [full[i] for i in rl]
        for rec, r in zip(recs, rs):
            ok(r, rec)
    print("OK")


test()
