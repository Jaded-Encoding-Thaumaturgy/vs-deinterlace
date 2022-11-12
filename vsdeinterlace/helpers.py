from __future__ import annotations

from lvsfunc import clip_async_render
from vstools import Dar, FramePropError, core, get_prop, vs

__all__ = [
    'check_ivtc_pattern',
    'calculate_dar_from_props'
]


def check_ivtc_pattern(clip: vs.VideoNode, pattern: int = 0) -> bool:
    """:py:func:`vsdeinterlace.utils.check_patterns` rendering behaviour."""

    from .funcs import sivtc

    clip = sivtc(clip, pattern)
    clip = core.tdm.IsCombed(clip)

    frames: list[int] = []

    def _cb(n: int, f: vs.VideoFrame) -> None:
        if get_prop(f, '_Combed', int):
            frames.append(n)

    # TODO: Tried being clever and just exiting if any combing was found, but async_render had other plans :)
    clip_async_render(clip[::4], progress=f"Checking pattern {pattern}...", callback=_cb)

    if len(frames) > 0:
        print(f"check_patterns: 'Combing found with pattern {pattern}!'")
        return False

    print(f"check_patterns: 'Clean clip found with pattern {pattern}!'")
    return True


def calculate_dar_from_props(clip: vs.VideoNode) -> Dar:
    """Determine what DAR the clip is by checking default SAR props."""
    frame = clip.get_frame(0)

    try:
        sar = get_prop(frame, "_SARDen", int), get_prop(frame, "_SARNum", int)
    except FramePropError as e:
        raise FramePropError(
            "PARser", "", f"SAR props not found! Make sure your video indexing plugin sets them!\n\t{e}"
        )

    match sar:
        case (11, 10) | (9, 8): return Dar.FULLSCREEN
        case (33, 40) | (27, 32): return Dar.WIDESCREEN
        case _: raise ValueError("Could not calculate DAR. Please set the DAR manually.")
