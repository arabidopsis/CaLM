import argparse
from typing import TypeVar
from dataclasses import dataclass, fields, asdict, is_dataclass
from typing_extensions import Self

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
