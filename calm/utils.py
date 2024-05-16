import argparse
import os
from typing import TypeVar, Iterable, Iterator, TYPE_CHECKING, IO
from itertools import islice
from dataclasses import dataclass, fields, asdict, is_dataclass
from typing_extensions import Self
from pathlib import Path

if TYPE_CHECKING:
    import numpy as np

T = TypeVar("T")


def kebab(s: str) -> str:
    return s.replace("_", "-")


def optional(op):
    def ret(v):
        if v is None:
            return None
        return op(v)

    return ret


def add_args(argparser: argparse.ArgumentParser, cls: type) -> argparse.ArgumentParser:
    if not is_dataclass(cls):
        return argparser
    for f in fields(cls):

        if isinstance(f.default, bool):
            m = {k: v for k, v in f.metadata.items() if k not in {"type", "default"}}
            if "action" not in m:
                m["action"] = "store_true"
            argparser.add_argument(
                f"--{kebab(f.name)}",
                default=f.default,
                **m,
            )
        else:
            m = {k: v for k, v in f.metadata.items() if k not in {"default"}}
            if "type" not in m:
                m["type"] = f.type
            if "metavar" not in m:
                if isinstance(f.default, int):
                    m["metavar"] = "N"
                elif isinstance(f.default, float):
                    m["metavar"] = "F"
            if f.default is not None:
                if "help" in m:
                    h = m["help"]
                    m["help"] = f"={f.default} {h}"
                else:
                    m["help"] = f"={f.default}"
            argparser.add_argument(f"--{kebab(f.name)}", default=f.default, **m)
    return argparser


def create_from_ns(ns: argparse.Namespace, cls: type[T]) -> T:
    if not is_dataclass(cls):
        raise ValueError("not a dataclass")
    return cls(
        **{f.name: getattr(ns, f.name) for f in fields(cls) if hasattr(ns, f.name)}
    )  # type: ignore


@dataclass
class ArgparseMixin:

    @classmethod
    def add_args(cls, argparser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return add_args(argparser, cls)

    @classmethod
    def create(cls, ns: argparse.Namespace) -> Self:
        return create_from_ns(ns, cls)

    def to_namespace(self, **kwargs) -> argparse.Namespace:
        d = asdict(self)
        d.update(kwargs)
        return argparse.Namespace(**d)


def batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            break
        yield batch


def opengz(path: str | os.PathLike, mode="rt") -> IO[str]:
    path = Path(path)
    if path.name.endswith(".gz"):
        import gzip

        return gzip.open(path, mode=mode, encoding="utf8")
    return open(path, mode=mode, encoding="utf8")


def load_embed(path: str | os.PathLike) -> Iterator[tuple[str, "np.ndarray"]]:
    import numpy as np
    import csv

    with opengz(path) as fp:
        reader = csv.reader(fp)
        for row in reader:
            agi, embed = row[0], row[1:]
            yield agi, np.array(embed, dtype=np.float32)
