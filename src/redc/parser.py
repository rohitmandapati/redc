# Simple parser to produce AST

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from lark import Lark, Transformer, UnexpectedInput


class CompileError(Exception):
    pass


@dataclass(frozen=True)
class AST:
    kind: str
    children: tuple
    line: int = 0
    column: int = 0

    def fail(self, message):
        raise CompileError(f"line {self.line}:{self.column}: {message}")


class BuildAST(Transformer):
    def __default__(self, data, children, meta):
        return AST(str(data), tuple(children), getattr(meta, 'line', 0),
                   getattr(meta, 'column', 0))

    def __default_token__(self, token):
        return str(token)


@lru_cache(maxsize=1)
def parser():
    return Lark(files('redc').joinpath('redc.lark').read_text(), parser='lalr',
                propagate_positions=True, maybe_placeholders=False)


def parse(source: str) -> AST:
    try:
        return BuildAST().transform(parser().parse(source))
    except UnexpectedInput as exc:
        raise CompileError(f"line {exc.line}:{exc.column}: syntax error\n"
                           f"{exc.get_context(source)}") from None
