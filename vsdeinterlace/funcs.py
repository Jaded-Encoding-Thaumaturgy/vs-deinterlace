from __future__ import annotations

from typing import Any, Sequence, cast

from stgpytools import CustomIntEnum
from vsexprtools import ExprVars, complexpr_available, norm_expr
from vsrgtools import BlurMatrix
from vsdenoise import MVTools
from vstools import (
    ConvMode, CustomEnum, FormatsMismatchError, InvalidFramerateError,
    FunctionUtil, FuncExceptT, GenericVSFunction, KwargsT, PlanesT,
    core, vs, scale_8bit, check_variable
)

__all__ = [
    'telop_resample',
    'fix_interlaced_fades',
    'vinverse'
]


class telop_resample(CustomIntEnum):
    TXT60i_on_24telecined = 0
    TXT60i_on_24duped = 1
    TXT30p_on_24telecined = 2

    def __call__(self, bobbed_clip: vs.VideoNode, pattern: int, **mv_args: Any) -> vs.VideoNode:
        """
        Virtually oversamples the video to 120 fps with motion interpolation on credits only, and decimates to 24 fps.
        Requires manually specifying the 3:2 pulldown pattern (the clip must be split into parts if it changes).

        :param bobbed_clip:             Bobbed clip. Framerate must be 60000/1001.
        :param pattern:                 First frame in the pattern.
        :param mv_args:                 Arguments to pass on to MVTools, used for motion compensation.

        :return:                        Decimated clip with text resampled down to 24p.

        :raises InvalidFramerateError:  Bobbed clip does not have a framerate of 60000/1001 (59.94)
        """

        assert check_variable(bobbed_clip, telop_resample)

        InvalidFramerateError.check(telop_resample, bobbed_clip, (60000, 1001))

        invpos = (5 - pattern * 2 % 5) % 5

        offset = [0, 0, -1, 1, 1][pattern]
        pattern = [0, 1, 0, 0, 1][pattern]
        direction = [-1, -1, 1, 1, 1][pattern]

        ivtc_fps, ivtc_fps_div = (dict(fpsnum=x, fpsden=1001) for x in (24000, 12000))

        pos = []
        assumefps = 0

        interlaced = self in (telop_resample.TXT60i_on_24telecined, telop_resample.TXT60i_on_24duped)
        decimate = self is telop_resample.TXT60i_on_24duped

        cycle = 10 // (1 + interlaced)

        def bb(idx: int, cut: bool = False) -> vs.VideoNode:
            if cut:
                return bobbed_clip[cycle:].std.SelectEvery(cycle, [idx])
            return bobbed_clip.std.SelectEvery(cycle, [idx])

        def intl(clips: list[vs.VideoNode], toreverse: bool, halv: list[int]) -> vs.VideoNode:
            clips = [c[::2] if i in halv else c for i, c in enumerate(clips)]
            if not toreverse:
                clips = list(reversed(clips))
            return core.std.Interleave(clips)

        if interlaced:
            if decimate:
                cleanpos = 4
                pos = [1, 2]

                if invpos > 2:
                    pos = [6, 7]
                    assumefps = 2
                elif invpos > 1:
                    pos = [2, 6]
                    assumefps = 1
            else:
                cleanpos = 1
                pos = [3, 4]

                if invpos > 1:
                    cleanpos = 6
                    assumefps = 1

                if invpos > 3:
                    pos = [4, 8]
                    assumefps = 1

            clean = bobbed_clip.std.SelectEvery(cycle, [cleanpos - invpos])
            jitter = bobbed_clip.std.SelectEvery(cycle, [p - invpos for p in pos])

            if assumefps:
                jitter = core.std.AssumeFPS(
                    bobbed_clip[0] * assumefps + jitter, **(ivtc_fps_div if cleanpos == 6 else ivtc_fps)
                )

            comp = MVTools(jitter, **mv_args).flow_interpolate()

            out = intl([comp, clean], decimate, [0])
            offs = 3 if decimate else 2

            return out[invpos // offs:]

        if pattern == 0:
            c1pos = [0, 2, 7, 5]
            c2pos = [3, 4, 9, 8]

            if offset == -1:
                c1pos = [2, 7, 5, 10]

            if offset == 1:
                c2pos = []
                c2 = core.std.Interleave([bb(4), bb(5), bb(0, True), bb(9)])
        else:
            c1pos = [2, 4, 9, 7]
            c2pos = [0, 1, 6, 5]

            if offset == 1:
                c1pos = []
                c1 = core.std.Interleave([bb(3), bb(5), bb(0, True), bb(8)])

            if offset == -1:
                c2pos = [1, 6, 5, 10]

        if c1pos:
            c1 = bobbed_clip.std.SelectEvery(cycle, [c + offset for c in c1pos])

        if c2pos:
            c2 = bobbed_clip.std.SelectEvery(cycle, [c + offset for c in c2pos])

        if offset == -1:
            c1, c2 = (core.std.AssumeFPS(bobbed_clip[0] + c, **ivtc_fps) for c in (c1, c2))

        fix1 = MVTools(c1, **mv_args).flow_interpolate(time=50 + direction * 25)
        fix2 = MVTools(c2, **mv_args).flow_interpolate()

        return intl([fix1, fix2], pattern == 0, [0, 1])


class FixInterlacedFades(CustomEnum):
    Average: FixInterlacedFades = object()  # type: ignore
    Darken: FixInterlacedFades = object()  # type: ignore
    Brighten: FixInterlacedFades = object()  # type: ignore

    def __call__(
        self, clip: vs.VideoNode, colors: float | list[float] | PlanesT = 0.0,
        planes: PlanesT = None, func: FuncExceptT | None = None
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

        f = FunctionUtil(clip, func, planes, vs.YUV, 32)

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

        expr_header = 'Y 2 % x.fbAvg{i} x.ftAvg{i} ? AVG! AVG@ 0 = x x {color} - '
        expr_footer = ' AVG@ / * ? {color} +'

        expr_mode, expr_mode_chroma = (
            ('+ 2 /', '+ 2 /') if self == self.Average else (('min', '<') if self == self.Darken else ('max', '>'))
        )

        fix = norm_expr(
            props_clip, (
                # luma
                expr_header + 'x.ftAvg{i} x.fbAvg{i} {expr_mode}' + expr_footer,
                # chroma
                expr_header + 'x.ftAvg{i} abs x.fbAvg{i} abs {expr_mode} x.ftAvg{i} x.fbAvg{i} ?' + expr_footer
            ),
            planes, i=f.norm_planes, expr_mode=(expr_mode, expr_mode_chroma),
            color=colors, force_akarin=func,
        )

        return f.return_clip(fix)


def vinverse(
    clip: vs.VideoNode,
    comb_blur: GenericVSFunction | vs.VideoNode | Sequence[int] = [1, 2, 1],
    contra_blur: GenericVSFunction | vs.VideoNode | Sequence[int] = [1, 2, 1],
    contra_str: float = 2.0, amnt: int = 255, scl: float = 0.25,
    thr: int = 0, planes: PlanesT = None,
    **kwargs: Any
) -> vs.VideoNode:
    """
    A simple but effective script to remove residual combing. Based on an AviSynth script by Didée.

    :param clip:            Clip to process.
    :param comb_blur:       Filter used to remove combing.
    :param contra_blur:     Filter used to calculate contra sharpening.
    :param contra_str:      Strength of contra sharpening.
    :param amnt:            Change no pixel by more than this in 8bit.
    :param thr:             Skip processing if abs(clip - comb_blur(clip)) < thr
    :param scl:             Scale factor for vshrpD * vblurD < 0.
    """

    func = FunctionUtil(clip, vinverse, planes)

    def_k = KwargsT(mode=ConvMode.VERTICAL)

    kwrg_a, kwrg_b = not callable(comb_blur), not callable(contra_blur)

    if isinstance(comb_blur, vs.VideoNode):
        blurred = comb_blur
    else:
        if not callable(comb_blur):
            comb_blur = BlurMatrix(comb_blur)  # type:ignore

        blurred = comb_blur(func.work_clip, planes=planes, **((def_k | kwargs) if kwrg_a else kwargs))

    if isinstance(contra_blur, vs.VideoNode):
        blurred2 = contra_blur
    else:
        if not callable(contra_blur):
            contra_blur = BlurMatrix(contra_blur)  # type:ignore

        blurred2 = contra_blur(blurred, planes=planes, **((def_k | kwargs) if kwrg_b else kwargs))

    FormatsMismatchError.check(func.func, func.work_clip, blurred, blurred2)

    combed = norm_expr(
        [func.work_clip, blurred, blurred2],  # type:ignore
        'x y - D1! D1@ abs D1A! D1A@ {thr} < x y z - {sstr} * D2! D1A@ D2@ abs < D1@ D2@ ? D3! '
        'D1@ D2@ xor D3@ {scl} * D3@ ? y + x {amnt} - x {amnt} + clip ?',
        planes, sstr=contra_str, amnt=scale_8bit(func.work_clip, amnt),
        scl=scl, thr=scale_8bit(func.work_clip, thr),
    )

    return func.return_clip(combed)


fix_interlaced_fades = cast(FixInterlacedFades, FixInterlacedFades.Average)
