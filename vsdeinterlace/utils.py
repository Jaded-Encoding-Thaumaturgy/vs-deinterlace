from __future__ import annotations

from fractions import Fraction
from functools import partial
from math import gcd
from typing import SupportsFloat

from vskernels import BicubicDidee, Catrom
from vstools import (
    CustomError, CustomValueError, Dar, FieldBased, FieldBasedT, FuncExceptT, Region, core, depth, get_prop, get_w,
    mod2, mod4, vs
)

from .helpers import check_ivtc_pattern

__all__ = [
    'interlace_patterns',
]


def interlace_patterns(clipa: vs.VideoNode, clipb: vs.VideoNode, length: int = 5) -> list[vs.VideoNode]:
    a_select = [clipa.std.SelectEvery(length, i) for i in range(length)]
    b_select = [clipb.std.SelectEvery(length, i) for i in range(length)]

    return [
        core.std.Interleave([
            (b_select if i == j else a_select)[j] for j in range(length)
        ]) for i in range(length)
    ]
