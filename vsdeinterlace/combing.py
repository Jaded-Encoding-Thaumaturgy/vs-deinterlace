from __future__ import annotations
from typing import Literal

from vsexprtools import norm_expr, aka_expr_available, ExprVars
from vstools import (
    CustomOverflowError, FieldBased, FieldBasedT, FuncExceptT, PlanesT, check_variable_format, core, get_neutral_value,
    normalize_planes, scale_value, vs
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
        raise ExprVars._get_akarin_err('You need the akarin plugin to run this function!')(func=func)

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


def vinverse(clip: vs.VideoNode, sstr: float = 2.0, amount: int = 128, scale: float = 1.5) -> vs.VideoNode:
    """
    Clean up residual combing after a deinterlacing pass.

    This is Setsugen no ao's implementation, adopted into vsdeinterlace.

    :param clip:        Clip to process.
    :param sstr:        Contrasharpening strength. Increase this if you find
                        the decombing blurs the image a bit too much.
    :param amount:      Maximum difference allowed between the original pixels and adjusted pixels.
                        Scaled to input clip's depth. Set to 255 to effectively disable this.
    :param scale:       Scale amount for vertical sharp * vertical blur.

    :return:            Clip with residual combing largely removed.

    :raises ValueError: ``amount`` is set above 255.
    """

    assert check_variable_format(clip, "vinverse")

    if amount > 255:
        raise CustomOverflowError("'amount' may not be set higher than 255!", vinverse)

    neutral = get_neutral_value(clip)

    # Expression to find combing and separate it from the rest of the clip
    find_combs = clip.akarin.Expr(f'{neutral} n! x x 2 * x[0,-1] x[0,1] + + 4 / - n@ +')

    # Expression to decomb it (creates blending)
    decomb = core.akarin.Expr(
        [find_combs, clip],
        f'{neutral} n! x 2 * x[0,-1] x[0,1] + + 4 / blur! y x blur@ - x n@ - * 0 < n@ x blur@ '
        ' - abs x n@ - abs < x blur@ - n@ + x ? ? - n@ +'
    )

    # Final expression to properly merge it and avoid creating too much damage
    return core.akarin.Expr(
        [clip, decomb],
        f'{neutral} n! {scale_value(amount, 8, clip.format.bits_per_sample)} a! y y y y 2 * y[0,-1]'
        f' y[0,1] + + 4 / - {sstr} * + y - n@ + sdiff! x y - n@ + diff! sdiff@ n@ - diff@ n@ - '
        f'* 0 < sdiff@ n@ - abs diff@ n@ - abs < sdiff@ diff@ ? n@ - {scale} * n@ + sdiff@ n@ '
        '- abs diff@ n@ - abs < sdiff@ diff@ ? ? n@ - + merge! x a@ + merge@ < x a@ + x a@ - '
        'merge@ > x a@ - merge@ ? ?'
    )
