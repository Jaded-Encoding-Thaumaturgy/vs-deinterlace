from typing import Literal, TypeAlias

__all__ = [
    "Types"
]


class Types:
    Match: TypeAlias = Literal['p'] | Literal['c'] | Literal['n'] | Literal['b'] | Literal['u']
    """A type representing possible fieldmatches."""
