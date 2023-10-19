from __future__ import annotations

from typing import Any

from vstools import CustomEnum, FieldBased, FieldBasedT, InvalidFramerateError, VSFunctionKwArgs, core, join, vs

from .blending import deblend

__all__ = [
    'IVTCycles',
    'sivtc', 'jivtc',
]


class IVTCycles(list[int], CustomEnum):
    cycle_10 = [[0, 3, 6, 8], [0, 2, 5, 8], [0, 2, 4, 7], [2, 4, 6, 9], [1, 4, 6, 8]]
    cycle_08 = [[0, 3, 4, 6], [0, 2, 5, 6], [0, 2, 4, 7], [0, 2, 4, 7], [1, 2, 4, 6]]
    cycle_05 = [[0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 4]]

    @property
    def pattern_length(self) -> int:
        return int(self._name_[6:])

    @property
    def length(self) -> int:
        return len(self.value)

    def decimate(self, clip: vs.VideoNode, pattern: int = 0) -> vs.VideoNode:
        assert 0 <= pattern < self.length
        return clip.std.SelectEvery(self.pattern_length, self.value[pattern])


def sivtc(
    clip: vs.VideoNode, pattern: int = 0, tff: bool | FieldBasedT = True, ivtc_cycle: IVTCycles = IVTCycles.cycle_10
) -> vs.VideoNode:
    """
    Simplest form of a fieldmatching function.

    This is essentially a stripped-down JIVTC offering JUST the basic fieldmatching and decimation part.
    As such, you may need to combine multiple instances if patterns change throughout the clip.

    :param clip:        Clip to process.
    :param pattern:     First frame of any clean-combed-combed-clean-clean sequence.
    :param tff:         Top-Field-First.

    :return:            IVTC'd clip.
    """

    tff = FieldBased.from_param(tff).field

    ivtc = clip.std.SeparateFields(tff=tff).std.DoubleWeave()
    ivtc = ivtc_cycle.decimate(ivtc, pattern)

    return FieldBased.PROGRESSIVE.apply(ivtc)


def jivtc(
    src: vs.VideoNode, pattern: int, tff: bool = True, chroma_only: bool = True,
    postprocess: VSFunctionKwArgs = deblend, postdecimate: IVTCycles | None = IVTCycles.cycle_05,
    ivtc_cycle: IVTCycles = IVTCycles.cycle_10, final_ivtc_cycle: IVTCycles = IVTCycles.cycle_08,
    **kwargs: Any
) -> vs.VideoNode:
    """
    This function should only be used when a normal ivtc or ivtc + bobber leaves chroma blend to a every fourth frame.
    You can disable chroma_only to use in luma as well, but it is not recommended.

    :param src:             Source clip. Has to be 60i.
    :param pattern:         First frame of any clean-combed-combed-clean-clean sequence.
    :param tff:             Set top field first (True) or bottom field first (False).
    :param chroma_only:     Decide whether luma too will be processed.
    :param postprocess:     Function to run after second decimation. Should be either a bobber or a deblender.
    :param postdecimate:    If the postprocess function doesn't decimate itself, put True.

    :return:                Inverse Telecined clip.
    """

    InvalidFramerateError.check(jivtc, src, (30000, 1001))

    ivtced = core.std.SeparateFields(src, tff=tff).std.DoubleWeave()
    ivtced = ivtc_cycle.decimate(ivtced, pattern)

    pprocess = postprocess(src if postdecimate else ivtced, **kwargs)

    if postdecimate:
        pprocess = postdecimate.decimate(pprocess, pattern)

    inter = core.std.Interleave([ivtced, pprocess])
    final = final_ivtc_cycle.decimate(inter, pattern)

    final = join(ivtced, final) if chroma_only else final

    return FieldBased.ensure_presence(final, FieldBased.PROGRESSIVE)
