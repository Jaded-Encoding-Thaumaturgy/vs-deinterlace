from __future__ import annotations

import warnings
from typing import Any, Sequence, cast, overload

from vsexprtools import ExprVars, complexpr_available, norm_expr
from vsrgtools import BlurMatrix
from vstools import (
    MISSING, ConvMode, CustomEnum, FieldBasedT, FuncExceptT, FunctionUtil, GenericVSFunction, KwargsT, MissingT,
    PlanesT, core, scale_8bit, vs
)

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


def vinverse(
    clip: vs.VideoNode,
    comb_blur: GenericVSFunction | Sequence[int] = [1, 2, 1],
    contra_blur: GenericVSFunction | Sequence[int] = [1, 4, 6, 4, 1],
    contra_str: float = 2.7, amnt: int = 255, scl: float = 0.25, planes: PlanesT = None,
    **kwargs: Any
) -> vs.VideoNode:
    """
    A simple but effective plugin to remove residual combing. Based on an AviSynth script by Didée.

    :param clip:            Clip to process.
    :param comb_blur:       Filter used to remove combing.
    :param contra_blur:     Filter used to calculate contra sharpening.
    :param contra_str:      Strength of contra sharpening.
    :param amnt:            Change no pixel by more than this in 8bit (default is 255, unrestricted).
    :param scl:             Scale factor for vshrpD*vblurD < 0.
    """

    func = FunctionUtil(clip, vinverse, planes, vs.YUV, 32)

    def_k = KwargsT(mode=ConvMode.VERTICAL)

    kwrg_a, kwrg_b = not callable(comb_blur), not callable(contra_blur)

    if not callable(comb_blur):
        comb_blur = BlurMatrix(comb_blur)

    if not callable(contra_blur):
        contra_blur = BlurMatrix(contra_blur)

    blurred = comb_blur(func.work_clip, planes=planes, **((def_k | kwargs) if kwrg_a else kwargs))
    blurred2 = contra_blur(blurred, planes=planes, **((def_k | kwargs) if kwrg_b else kwargs))

    combed = norm_expr(
        [func.work_clip, blurred, blurred2],
        'y z - {sstr} * D1! x y - D2! D1@ abs D1A! D2@ abs D2A! '
        'D1@ D2@ xor D1A@ D2A@ < D1@ D2@ ? {scl} * D1A@ D2A@ < D1@ D2@ ? ? y + '
        'LIM! x {amnt} + LIM@ < x {amnt} + x {amnt} - LIM@ > x {amnt} - LIM@ ? ?',
        planes, sstr=contra_str, amnt=scale_8bit(func.work_clip, amnt), scl=scl,
    )

    return func.return_clip(combed)


fix_interlaced_fades = cast(FixInterlacedFades, FixInterlacedFades.Average)
