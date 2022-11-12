from __future__ import annotations

from vstools import Dar, FramePropError, get_prop, get_render_progress, vs

__all__ = [
    'check_ivtc_pattern',
    'calculate_dar_from_props'
]


def check_ivtc_pattern(clip: vs.VideoNode, pattern: int = 0) -> bool:
    """:py:func:`vsdeinterlace.utils.check_patterns` rendering behaviour."""

    from .ivtc import sivtc

    clip = sivtc(clip, pattern).tdm.IsCombed()

    p = get_render_progress()
    task = p.add_task(f"Checking pattern {pattern}...", total=clip.num_frames)

    for f in clip[::4].frames(close=True):
        if get_prop(f, '_Combed', int):
            print(f"check_patterns: 'Combing found with pattern {pattern}!'")
            return False

        p.update(task, advance=1)

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
