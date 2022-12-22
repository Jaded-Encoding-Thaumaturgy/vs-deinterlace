from __future__ import annotations

from typing import Literal, cast

from vsexprtools import ExprVars, aka_expr_available, norm_expr
from vsrgtools import sbr
from vstools import (
    ConvMode, CustomEnum, FieldBased, FieldBasedT, FuncExceptT, PlanesT, core, get_neutral_value, normalize_planes,
    scale_8bit, vs
)

__all__ = [
    'fix_telecined_fades',

    'vinverse'
]


def fix_telecined_fades(
    clip: vs.VideoNode, tff: bool | FieldBasedT | None = None, fade_type: Literal[1, 2] = 1,
    planes: PlanesT = None, func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Give a mathematically perfect solution to fades made *after* telecining (which made perfect IVTC impossible).

    This is an improved version of the Fix-Telecined-Fades plugin
    that deals with overshoot/undershoot by adding a check.

    Make sure to run this *after* IVTC/deinterlacing!

    If the value surpases thr * original value, it will not affect any pixels in that frame
    to avoid it damaging frames it shouldn't need to. This helps a lot with orphan fields as well,
    which would otherwise create massive swings in values, sometimes messing up the fade fixing.

    .. warning::
        | If you pass your own float clip, you'll want to make sure to properly dither it down after.
        | If you don't do this, you'll run into some serious issues!

    :param clip:                            Clip to process.
    :param tff:                             Top-field-first. `False` sets it to Bottom-Field-First.
                                            If `None`, get the field order from the _FieldBased prop.
    :param cuda:                            Use cupy for certain calculations. `False` uses numpy instead.

    :return:                                Clip with fades (and only fades) accurately deinterlaced.

    :raises UndefinedFieldBasedError:       No automatic ``tff`` can be determined.
    """
    func = func or fix_telecined_fades

    if not aka_expr_available:
        raise ExprVars._get_akarin_err()(func=func)

    clip = FieldBased.ensure_presence(clip, tff, func)

    fields = clip.std.Limiter().std.SeparateFields()

    planes = normalize_planes(clip, planes)

    for i in planes:
        fields = fields.std.PlaneStats(None, i, f'PAvg{i}')

    props_clip = core.akarin.PropExpr(
        [clip, fields[::2], fields[1::2]], lambda: {  # type: ignore[misc]
            f'f{t}Avg{i}': f'{c}.PAvg{i}'  # type: ignore[has-type]
            for t, c in ['ty', 'bz']
            for i in planes  # type: ignore
        }
    )

    fix = 'x TAVG@ BF@ x.ftAvg{i} x.fbAvg{i} ? + 2 / TAVG@ / *'

    return norm_expr(
        props_clip,
        'Y 2 % BF! BF@ x.fbAvg{i} x.ftAvg{i} ? TAVG! '
        + (f'TAVG@ 0 = x {fix} ?' if fade_type == 2 else fix),
        planes, i=planes, force_akarin=func
    )


class Vinverse(CustomEnum):
    V1 = object()
    V2 = object()
    MASKED = object()
    MASKEDV1 = object()
    MASKEDV2 = object()

    def __call__(
        self, clip: vs.VideoNode, sstr: float = 2.7, amount: int = 255, scale: float = 0.25,
        mode: ConvMode = ConvMode.VERTICAL, planes: PlanesT = None, vinverse2: bool = False
    ) -> vs.VideoNode:
        if amount <= 0:
            return clip

        neutral = get_neutral_value(clip)

        expr = f'y z - {sstr} * D1! x y - D2! D1@ abs D1A! D2@ abs D2A! '
        expr += f'D1@ D2@ xor D1A@ D2A@ < D1@ D2@ ? {scale} * D1A@ D2A@ < D1@ D2@ ? ? y + '

        if self in {Vinverse.V1, Vinverse.MASKEDV1}:
            blur = clip.std.Convolution([50, 99, 50], mode=mode, planes=planes)
            blur2 = blur.std.Convolution([1, 4, 6, 4, 1], mode=mode, planes=planes)
        elif self in {Vinverse.V2, Vinverse.MASKEDV2}:
            blur = sbr(clip, mode=mode, planes=planes)
            blur2 = blur.std.Convolution([1, 2, 1], mode=mode, planes=planes)

        if self in {Vinverse.MASKED, Vinverse.MASKEDV1, Vinverse.MASKEDV2}:
            if self is Vinverse.MASKED:
                find_combs = norm_expr(clip, f'x x 2 * x[0,-1] x[0,1] + + 4 / - {neutral} +', planes)
                decomb = norm_expr(
                    [find_combs, clip],
                    'x x 2 * x[0,-1] x[0,1] + + 4 / - B! y B@ x {n} - * 0 '
                    '< {n} B@ abs x {n} - abs < B@ {n} + x ? ? - {n} +', n=neutral
                )
            else:
                decomb = norm_expr(
                    [clip, blur, blur2], 'x x 2 * y + 4 / - {n} + FC@ FC@ FC@ 2 * y z - {n} + + 4 / - B! '
                    'x B@ FC@ {n} - * 0 < {n} B@ abs FC@ {n} - abs < B@ {n} + FC@ ? ? - {n} +', n=neutral
                )

            return norm_expr(
                [clip, decomb], f'{scale_8bit(clip, amount)} a! y y y y 2 * y[0,-1] y[0,1] + + 4 / - {sstr} '
                '* + y - {n} + D1! x y - {n} + D2! D1@ {n} - D2@ {n} - * 0 < D1@ {n} - abs D2@ {n} - abs < D1@ '
                'D2@ ? {n} - {scale} * {n} + D1@ {n} - abs D2@ {n} - abs < D1@ D2@ ? ? {n} - + merge! '
                'x a@ + merge@ < x a@ + x a@ - merge@ > x a@ - merge@ ? ?', n=neutral
            )

        if amount < 255:
            amn = scale_8bit(clip, amount)
            expr += f'LIM! x {amn} + LIM@ < x {amn} + x {amn} - LIM@ > x {amn} - LIM@ ? ?'

        return norm_expr([clip, blur, blur2], expr, planes)


vinverse = cast(Vinverse, Vinverse.V1)
