from __future__ import annotations

import warnings
from functools import partial

from vsrgtools import repair
from vstools import core, vs

__all__ = [
    'deblend'
]


def deblend(clip: vs.VideoNode, start: int = 0, rep: int | None = None, decimate: bool = True) -> vs.VideoNode:
    """
    Deblending function for blended AABBA patterns.

    .. warning:
        This function's base functionality and settings will be updated in a future version!

    Assuming there's a constant pattern of frames (labeled A, B, C, CD, and DA in this function),
    blending can be fixed by calculating the D frame by getting halves of CD and DA, and using that
    to fix up CD. DA is then dropped because it's a duplicated frame.

    Doing this will result in some of the artifacting being added to the deblended frame,
    but we can mitigate that by repairing the frame with the non-blended frame before it.

    For more information, please refer to `this blogpost by torchlight
    <https://mechaweaponsvidya.wordpress.com/2012/09/13/adventures-in-deblending/>`_.

    :param clip:        Clip to process.
    :param start:       First frame of the pattern (Default: 0).
    :param rep:         Repair mode for the deblended frames, no repair if None (Default: None).
    :param decimate:    Decimate the video after deblending (Default: True).

    :return:            Deblended clip.
    """
    warnings.warn("deblend: 'This function's base functionality and settings "
                  "will be updated in a future version!'", DeprecationWarning)

    blends_a = range(start + 2, clip.num_frames - 1, 5)
    blends_b = range(start + 3, clip.num_frames - 1, 5)
    expr_cd = ["z a 2 / - y x 2 / - +"]

    # Thanks Myaa, motbob and kageru!
    def deblend(n: int, clip: vs.VideoNode, rep: int | None) -> vs.VideoNode:
        if n % 5 in [0, 1, 4]:
            return clip
        else:
            if n in blends_a:
                c, cd, da, a = clip[n - 1], clip[n], clip[n + 1], clip[n + 2]
                debl = core.akarin.Expr([c, cd, da, a], expr_cd)
                return repair(debl, c, rep) if rep else debl
            return clip

    debl = core.std.FrameEval(clip, partial(deblend, clip=clip, rep=rep))
    return core.std.DeleteFrames(debl, blends_b).std.AssumeFPS(fpsnum=24000, fpsden=1001) if decimate else debl
