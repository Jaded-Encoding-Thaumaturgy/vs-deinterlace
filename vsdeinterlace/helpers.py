from __future__ import annotations

from vstools import get_prop, get_render_progress, vs

__all__ = [
    'check_ivtc_pattern'
]


def check_ivtc_pattern(clip: vs.VideoNode, pattern: int = 0) -> bool:
    """:py:func:`vsdeinterlace.utils.check_patterns` rendering behaviour."""

    from .ivtc import sivtc

    clip = sivtc(clip, pattern).tdm.IsCombed()

    with get_render_progress() as p:
        task = p.add_task(f"Checking pattern {pattern}...", total=clip.num_frames)

        for f in clip[::4].frames(close=True):
            if get_prop(f, '_Combed', int):
                print(f"check_patterns: 'Combing found with pattern {pattern}!'")
                return False

            p.update(task, advance=1)

    print(f"check_patterns: 'Clean clip found with pattern {pattern}!'")

    return True
