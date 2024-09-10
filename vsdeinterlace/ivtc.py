from __future__ import annotations

from typing import Any

from vstools import (
    CustomEnum, FieldBased, FieldBasedT, InvalidFramerateError,
    VSFunctionKwArgs, core, join, vs, find_prop_rfs, FunctionUtil
)

from .blending import deblend

__all__ = [
    'IVTCycles',
    'sivtc', 'jivtc',
    'vfm', 'vdecimate'
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


def vfm(
    clip: vs.VideoNode, tff: bool | None = None, field: int = 2, mode: int = 1, mchroma: bool = True,
    cthresh: int = 9, mi: int = 80, chroma: bool = True, block: tuple | int = 16, y: tuple | bool = 16,
    scthresh: float | int = 12, micmatch: int = 1, micout: bool = False, postprocess: vs.VideoNode | None = None
) -> vs.VideoNode:
    
    func = FunctionUtil(clip, vfm, None, vs.YUV, 8)

    if isinstance(block, int):
        block = (block, block)
    if isinstance(y, int):
        y = (y, y)

    if func.work_clip != clip:
        clip2 = clip
    else:
        clip2 = None

    fieldmatch = func.work_clip.vivtc.VFM(
        order=tff, field=field, mode=mode, mchroma=mchroma, cthresh=cthresh, mi=mi, chroma=chroma, blockx=block[0], blocky=block[1],
        y0=y[0], y1=y[1], scthresh=scthresh, micmatch=micmatch, micout=micout, clip2=clip2
    )

    if not postprocess:
        fieldmatch = find_prop_rfs(fieldmatch, postprocess, prop="_Combed")
    
    return fieldmatch


def vdecimate(
    clip: vs.VideoNode, cycle: int = 5, chroma: bool = True, dupthresh: float | int = 1.1, scthresh: float | int = 15,
    block: tuple | int = 32, ovr: str | None = None, dryrun: bool = False, weight: float = 0.0
) -> vs.VideoNode:
    
    func = FunctionUtil(clip, vdecimate, None, vs.YUV, (8, 16))

    if isinstance(block, int):
        block = (block, block)

    if dryrun:
        weight = 0.0
    if weight:
        dryrun = True

    if func.work_clip != clip:
        clip2 = clip
    else:
        clip2 = None

    decimate_args = dict(
        cycle=cycle, chroma=chroma, dupthresh=dupthresh, scthresh=scthresh,
        blockx=block[0], blocky=block[1], ovr=ovr, dryrun=dryrun, clip2=clip2
    )

    if dryrun:
        stats = func.work_clip.vivtc.VDecimate(**decimate_args)
        if not weight:
            return stats
        else:
            decimate_args['dryrun'] = False
            avg = clip.std.AverageFrames(weights=[0, 1 - weight, weight])
            func.work_clip = find_prop_rfs(clip, avg, ref=stats, prop="VDecimateDrop")

    decimate = func.work_clip.vivtc.VDecimate(**decimate_args)

    return func.return_clip(decimate)
