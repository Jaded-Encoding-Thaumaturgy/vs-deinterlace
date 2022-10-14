from __future__ import annotations

from vstools import Dar, vs, get_prop, FramePropError

__all__ = [
    '_calculate_dar_from_props'
]


def _calculate_dar_from_props(clip: vs.VideoNode) -> Dar:
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
