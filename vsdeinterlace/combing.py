from __future__ import annotations

from typing import cast, overload

from vsexprtools import ExprVars, complexpr_available, norm_expr
from vsrgtools import sbr
from vstools import (
    MISSING, ConvMode, CustomEnum, FieldBasedT, FuncExceptT, FunctionUtil, MissingT, PlanesT, core,
    depth, expect_bits, get_neutral_values, scale_8bit, vs
)
import warnings

__all__ = [
    'fix_telecined_fades', 'fix_interlaced_fades',
    'vinverse'
]


@overload
def fix_telecined_fades(
    clip: vs.VideoNode, tff: bool | FieldBasedT | None, colors: float | list[float] = 0.0,
    planes: PlanesT = None, func: FuncExceptT | None = None
) -> vs.VideoNode:
    ...


@overload
def fix_telecined_fades(
    clip: vs.VideoNode, colors: float | list[float] = 0.0,
    planes: PlanesT = None, func: FuncExceptT | None = None
) -> vs.VideoNode:
    ...


def fix_telecined_fades(  # type: ignore[misc]
    clip: vs.VideoNode, tff: bool | FieldBasedT | None | float | list[float] | MissingT = MISSING,
    colors: float | list[float] | PlanesT = 0.0,
    planes: PlanesT | FuncExceptT = None, func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    * Deprecated, use "fix_interlaced_fades" *

    Give a mathematically perfect solution to decombing fades made *after* telecining
    (which made perfect IVTC impossible) that start or end in a solid color.

    Steps between the frames are not adjusted, so they will remain uneven depending on the telecining pattern,
    but the decombing is blur-free, ensuring minimum information loss. However, this may cause small amounts
    of combing to remain due to error amplification, especially near the solid-color end of the fade.

    This is an improved version of the Fix-Telecined-Fades plugin.

    Make sure to run this *after* IVTC/deinterlacing!

    :param clip:                            Clip to process.
    :param tff:                             This parameter is deprecated and unused. It will be removed in the future.
    :param colors:                          Fade source/target color (floating-point plane averages).

    :return:                                Clip with fades to/from `colors` accurately deinterlaced.
                                            Frames that don't contain such fades may be damaged.
    """

    warnings.warn(
        'fix_telecined_fades: This function is deprecated and as such it will be removed in the future! '
        'Please use "fix_interlaced_fades".'
    )

    # Gracefully handle positional arguments that either include or
    # exclude tff, hopefully without interfering with keyword arguments.
    # Remove this block when tff is fully dropped from the parameter list.
    if isinstance(tff, (float, list)):
        if colors == 0.0:
            tff, colors = MISSING, tff
        elif planes is None:
            tff, colors, planes = MISSING, tff, colors
        else:
            tff, colors, planes, func = MISSING, tff, colors, planes

    func = func or fix_telecined_fades

    if not complexpr_available:
        raise ExprVars._get_akarin_err()(func=func)

    if tff is not MISSING:
        print(DeprecationWarning('fix_telecined_fades: The tff parameter is unnecessary and therefore deprecated!'))

    f = FunctionUtil(clip, func, planes, (vs.GRAY, vs.YUV), 32)

    fields = f.work_clip.std.Limiter().std.SeparateFields(tff=True)

    for i in f.norm_planes:
        fields = fields.std.PlaneStats(None, i, f'P{i}')

    props_clip = core.akarin.PropExpr(
        [f.work_clip, fields[::2], fields[1::2]], lambda: {  # type: ignore[misc]
            f'f{t}Avg{i}': f'{c}.P{i}Average {color} -'  # type: ignore[has-type]
            for t, c in ['ty', 'bz']
            for i, color in zip(f.norm_planes, f.norm_seq(colors))
        }
    )

    fix = norm_expr(
        props_clip, 'Y 2 % x.fbAvg{i} x.ftAvg{i} ? AVG! '
        'AVG@ 0 = x x {color} - x.ftAvg{i} x.fbAvg{i} + 2 / AVG@ / * ? {color} +',
        planes, i=f.norm_planes, color=colors, force_akarin=func,
    )

    return f.return_clip(fix)


class FixInterlacedFades(CustomEnum):
    Average = object()
    Darken = object()
    Brighten = object()

    def __call__(
        self, clip: vs.VideoNode, colors: float | list[float] | PlanesT = 0.0,
        planes: PlanesT | FuncExceptT = None, func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """
        Give a mathematically perfect solution to decombing fades made *after* telecine
        (which made perfect IVTC impossible) that start or end in a solid color.

        Steps between the frames are not adjusted, so they will remain uneven depending on the telecine pattern,
        but the decombing is blur-free, ensuring minimum information loss. However, this may cause small amounts
        of combing to remain due to error amplification, especially near the solid-color end of the fade.

        This is an improved version of the Fix-Telecined-Fades plugin.

        Make sure to run this *after* IVTC!

        :param clip:                            Clip to process.
        :param colors:                          Fade source/target color (floating-point plane averages).

        :return:                                Clip with fades to/from `colors` accurately deinterlaced.
                                                Frames that don't contain such fades may be damaged.
        """
        func = func or self.__class__

        if not complexpr_available:
            raise ExprVars._get_akarin_err()(func=func)

        f = FunctionUtil(clip, func, planes, None, 32)

        fields = f.work_clip.std.Limiter().std.SeparateFields(tff=True)

        for i in f.norm_planes:
            fields = fields.std.PlaneStats(None, i, f'P{i}')

        props_clip = core.akarin.PropExpr(
            [f.work_clip, fields[::2], fields[1::2]], lambda: {  # type: ignore[misc]
                f'f{t}Avg{i}': f'{c}.P{i}Average {color} -'  # type: ignore[has-type]
                for t, c in ['ty', 'bz']
                for i, color in zip(f.norm_planes, f.norm_seq(colors))
            }
        )

        expr_mode = '+ 2 /' if self == self.Average else ('min' if self == self.Darken else 'max')

        fix = norm_expr(
            props_clip, 'Y 2 % x.fbAvg{i} x.ftAvg{i} ? AVG! '
            'AVG@ 0 = x x {color} - x.ftAvg{i} x.fbAvg{i} '
            '{expr_mode} AVG@ / * ? {color} +',
            planes, i=f.norm_planes, expr_mode=expr_mode,
            color=colors, force_akarin=func,
        )

        return f.return_clip(fix)


class Vinverse(CustomEnum):
    V1 = object()
    V2 = object()
    MASKED = object()
    MASKEDV1 = object()
    MASKEDV2 = object()

    def __call__(
        self, clip: vs.VideoNode, sstr: float = 2.7, amount: int = 255, scale: float = 0.25,
        mode: ConvMode = ConvMode.VERTICAL, planes: PlanesT = None
    ) -> vs.VideoNode:
        if amount <= 0:
            return clip

        clip, bits = expect_bits(clip, 32)

        neutral = get_neutral_values(clip)

        expr = f'y z - {sstr} * D1! x y - D2! D1@ abs D1A! D2@ abs D2A! '
        expr += f'D1@ D2@ xor D1A@ D2A@ < D1@ D2@ ? {scale} * D1A@ D2A@ < D1@ D2@ ? ? y + '

        if self in {Vinverse.V1, Vinverse.MASKEDV1}:
            blur = clip.std.Convolution([50, 99, 50], mode=mode, planes=planes)
            blur2 = blur.std.Convolution([1, 4, 6, 4, 1], mode=mode, planes=planes)
        elif self in {Vinverse.V2, Vinverse.MASKEDV2}:
            blur = sbr(clip, mode=mode, planes=planes)
            blur2 = blur.std.Convolution([1, 2, 1], mode=mode, planes=planes)

        if self in {Vinverse.MASKED, Vinverse.MASKEDV1, Vinverse.MASKEDV2}:
            search_str = 'x[-1,0] x[1,0]' if mode == ConvMode.HORIZONTAL else 'x[0,-1] x[0,1]'
            mask_search_str = search_str.replace('x', 'y')

            if self is Vinverse.MASKED:
                find_combs = norm_expr(clip, f'x x 2 * {search_str} + + 4 / - {{n}} +', planes, n=neutral)
                decomb = norm_expr(
                    [find_combs, clip],
                    'x x 2 * {search_str} + + 4 / - B! y B@ x {n} - * 0 '
                    '< {n} B@ abs x {n} - abs < B@ {n} + x ? ? - {n} +', n=neutral, search_str=search_str
                )
            else:
                decomb = norm_expr(
                    [clip, blur, blur2], 'x x 2 * y + 4 / - {n} + FC! FC@ FC@ 2 * y z - {n} + + 4 / - B! '
                    'x B@ FC@ {n} - * 0 < {n} B@ abs FC@ {n} - abs < B@ {n} + FC@ ? ? - {n} +', n=neutral
                )

            return norm_expr(
                [clip, decomb], f'{scale_8bit(clip, amount)} a! y y y y 2 * {mask_search_str} + + 4 / - {sstr} '
                '* + y - {n} + D1! x y - {n} + D2! D1@ {n} - D2@ {n} - * 0 < D1@ {n} - abs D2@ {n} - abs < D1@ '
                'D2@ ? {n} - {scale} * {n} + D1@ {n} - abs D2@ {n} - abs < D1@ D2@ ? ? {n} - + merge! '
                'x a@ + merge@ < x a@ + x a@ - merge@ > x a@ - merge@ ? ?', n=neutral, scale=scale
            )

        if amount < 255:
            amn = scale_8bit(clip, amount)
            expr += f'LIM! x {amn} + LIM@ < x {amn} + x {amn} - LIM@ > x {amn} - LIM@ ? ?'

        return depth(norm_expr([clip, blur, blur2], expr, planes), bits)


fix_interlaced_fades = cast(FixInterlacedFades, FixInterlacedFades.Average)
vinverse = cast(Vinverse, Vinverse.V1)
