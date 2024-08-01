from __future__ import annotations

import warnings

from stgpytools import KwargsT
from vsdenoise import MVTools
from vstools import FieldBased, FieldBasedT, InvalidFramerateError, check_variable, core, vs

__all__ = [
    'pulldown_credits'
]


def pulldown_credits(
    clip: vs.VideoNode, bob_clip: vs.VideoNode, frame_ref: int, tff: bool | FieldBasedT | None = None,
    interlaced: bool = True, decimate: bool | None = None, mv_args: KwargsT | None = None
) -> vs.VideoNode:
    """
    Deinterlacing function for interlaced credits (60i/30p) on top of telecined video (24p).

    This is a combination of havsfunc's dec_txt60mc, ivtc_txt30mc, and ivtc_txt60mc functions.
    The credits are interpolated and decimated to match the output clip.

    The function assumes you're passing a telecined clip (that's native 24p).
    If your clip is already fieldmatched, decimation will automatically be enabled unless set it to False.
    Likewise, if your credits are 30p (as opposed to 60i), you should set `interlaced` to False.

    The recommended way to use this filter is to trim out the area with interlaced credits,
    apply this function, and `vstools.insert_clip` the clip back into a properly IVTC'd clip.
    Alternatively, use `muvsfunc.VFRSplice` to splice the clip back in if you're dealing with a VFR clip.

    :param clip:                    Clip to process. Framerate must be 30000/1001.
    :param bob_clip:                Custom bobbed clip. Framerate must be 60000/1001.
    :param frame_ref:               First frame in the pattern. Expected pattern is ABBCD,
                                    except for when ``decimate`` is enabled, in which case it's AABCD.
    :param tff:                     Top-field-first. `False` sets it to Bottom-Field-First.
    :param interlaced:              60i credits. Set to false for 30p credits.
    :param decimate:                Decimate input clip as opposed to IVTC.
                                    Automatically enabled if certain fieldmatching props are found.
                                    Can be forcibly disabled by setting it to `False`.
    :param mv_args:                 Arguments to pass on to MVTools, used for motion compensation.

    :return:                        IVTC'd/decimated clip with credits pulled down to 24p.

    :raises ModuleNotFoundError:    Dependencies are missing.
    :raises ValueError:             Clip does not have a framerate of 30000/1001 (29.97).
    :raises TopFieldFirstError:     No automatic ``tff`` can be determined.
    :raises InvalidFramerateError:  Bobbed clip does not have a framerate of 60000/1001 (59.94)
    """

    assert check_variable(clip, pulldown_credits)

    InvalidFramerateError.check(pulldown_credits, clip, (30000, 1001))
    InvalidFramerateError.check(pulldown_credits, bob_clip, (60000, 1001))

    if decimate is not False:  # Automatically enable decimate unless set to False
        decimate = any(x in clip.get_frame(0).props for x in {"VFMMatch", "TFMMatch"})

        if decimate:
            warnings.warn(
                "pulldown_credits: Fieldmatched clip passed to function! decimate is now automatically set to `True`. "
                "If you want to disable this, set `decimate=False`!"
            )

    tff = FieldBased.from_param_or_video(tff, clip, True, pulldown_credits)
    clip = FieldBased.ensure_presence(clip, tff)

    field_ref = frame_ref * 2
    frame_ref %= 5
    invpos = (5 - field_ref) % 5

    offset = [0, 0, -1, 1, 1][frame_ref]
    pattern = [0, 1, 0, 0, 1][frame_ref]
    direction = [-1, -1, 1, 1, 1][frame_ref]

    mv_args = KwargsT(range_conversion=1.0, pel=2, refine=0) | (mv_args or KwargsT())

    ivtc_fps, ivtc_fps_div = (dict(fpsnum=x, fpsden=1001) for x in (24000, 12000))

    pos = []
    assumefps = 0

    cycle = 10 // (1 + interlaced)

    # 60i credits. Start of ABBCD
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

        clean = bob_clip.std.SelectEvery(cycle, [cleanpos - invpos])
        jitter = bob_clip.std.SelectEvery(cycle, [p - invpos for p in pos])

        if assumefps:
            jitter = core.std.AssumeFPS(bob_clip[0] * assumefps + jitter, **(ivtc_fps_div if cleanpos == 6 else ivtc_fps))

        comp = MVTools(jitter, **mv_args).flow_interpolate()

        out = core.std.Interleave([comp[::2], clean] if decimate else [clean, comp[::2]])
        offs = 3 if decimate else 2

        return out[invpos // offs:]

    def bb(idx: int, cut: bool = False) -> vs.VideoNode:
        if cut:
            return bob_clip[cycle:].std.SelectEvery(cycle, [idx])
        return bob_clip.std.SelectEvery(cycle, [idx])

    # 30i credits
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
        c1 = bob_clip.std.SelectEvery(cycle, [c + offset for c in c1pos])

    if c2pos:
        c2 = bob_clip.std.SelectEvery(cycle, [c + offset for c in c2pos])

    if offset == -1:
        c1, c2 = (core.std.AssumeFPS(bob_clip[0] + c, **ivtc_fps) for c in (c1, c2))

    fix1 = MVTools(c1, **mv_args).flow_interpolate(time=50 + direction * 25)
    fix2 = MVTools(c2, **mv_args).flow_interpolate()

    return core.std.Interleave([fix1[::2], fix2[::2]] if pattern == 0 else [fix2[::2], fix1[::2]])
