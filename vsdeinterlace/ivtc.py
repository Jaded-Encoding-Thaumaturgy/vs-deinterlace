from __future__ import annotations

from typing import Any

from vstools import (
    CustomEnum, CustomIntEnum, FieldBased, FieldBasedT,
    FunctionUtil, InvalidFramerateError, VSFunctionKwArgs,
    VSFunctionNoArgs, core, find_prop_rfs, join, vs
)

from .blending import deblend

__all__ = [
    'IVTCycles',
    'sivtc', 'jivtc',
    'vfm', 'VFMMode',
    'vdecimate'
]


class VFMMode(CustomIntEnum):
    """
    Enum representing different matching modes for VFM.

    The mode determines the strategy used for matching fields and frames.
    Higher modes generally offer better matching in complex scenarios but
    may introduce more risk of jerkiness or duplicate frames.
    """

    TWO_WAY_MATCH = 0
    """2-way match (p/c). Safest option, but may output combed frames in cases of bad edits or blended fields."""

    TWO_WAY_MATCH_THIRD_COMBED = 1
    """2-way match + 3rd match on combed (p/c + n). Default mode."""

    TWO_WAY_MATCH_THIRD_SAME_ORDER = 2
    """2-way match + 3rd match (same order) on combed (p/c + u)."""

    TWO_WAY_MATCH_THIRD_FOURTH_FIFTH = 3
    """2-way match + 3rd match on combed + 4th/5th matches if still combed (p/c + n + u/b)."""

    THREE_WAY_MATCH = 4
    """3-way match (p/c/n)."""

    THREE_WAY_MATCH_FOURTH_FIFTH = 5
    """
    3-way match + 4th/5th matches on combed (p/c/n + u/b).
    Highest risk of jerkiness but best at finding good matches.
    """


class IVTCycles(list[int], CustomEnum):  # type: ignore[misc]
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
    clip: vs.VideoNode, tff: FieldBasedT | None = None,
    mode: VFMMode = VFMMode.TWO_WAY_MATCH_THIRD_COMBED,
    postprocess: vs.VideoNode | VSFunctionNoArgs | None = None,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Perform field matching using VFM.

    This function uses VIVTC's VFM plugin to detect and match pairs of fields in telecined content.

    You can pass a post-processing clip or function that will act on leftover combed frames.
    If you pass a clip, it will replace combed frames with that clip. If you pass a function,
    it will run that function on your input clip and replace combed frames with it.

    Example usage:

    .. code-block:: python

        # Run vsaa.Nnedi3 on combed frames
        >>> vfm(clip, postprocess=lambda x: Nnedi3().interpolate(x, double_y=False))

    :param clip:            Input clip to field matching telecine on.
    :param tff:             Field order of the input clip.
                            If None, it will be automatically detected.
    :param mode:            VFM matching mode. For more information, see :py:class:`VFMMode`.
                            Default: VFMMode.TWO_WAY_MATCH_THIRD_COMBED.
    :param postprocess:     Optional function or clip to process combed frames.
                            If a function is passed, it should take a clip as input and return a clip as output.
                            If a clip is passed, it will be used as the postprocessed clip.
    :param kwargs:          Additional keyword arguments to pass to VFM.
                            For a list of parameters, see the VIVTC documentation.

    :return:                Field matched clip with progressive frames.
    """

    func = FunctionUtil(clip, vfm, None, (vs.YUV, vs.GRAY), 8)

    tff = FieldBased.from_param_or_video(tff, clip, False, func.func)

    vfm_kwargs = dict[str, Any](
        order=tff.is_tff, mode=mode
    )

    if block := kwargs.pop('block', None):
        if isinstance(block, int):
            vfm_kwargs |= dict(blockx=block, blocky=block)
        else:
            vfm_kwargs |= dict(blockx=block[0], blocky=block[1])

    if (y := kwargs.pop('y', None)) and not isinstance(y, int):
        vfm_kwargs |= dict(y0=y[0], y1=y[1])

    if not kwargs.get('clip2', None) and func.work_clip.format is not clip.format:
        vfm_kwargs |= dict(clip2=clip)

    fieldmatch = func.work_clip.vivtc.VFM(**(vfm_kwargs | kwargs))

    if postprocess:
        if callable(postprocess):
            postprocess = postprocess(kwargs.get('clip2', clip))

        fieldmatch = find_prop_rfs(fieldmatch, postprocess, "_Combed", "==", 1)

    return func.return_clip(fieldmatch)


def vdecimate(clip: vs.VideoNode, weight: float = 0.0, **kwargs: Any) -> vs.VideoNode:
    """
    Perform frame decimation using VDecimate.

    This function uses VIVTC's VDecimate plugin to remove duplicate frames from telecined content.
    It's recommended to use the vfm function before running this.

    :param clip:            Input clip to decimate.
    :param weight:          Weight for frame blending. If > 0, blends duplicate frames before dropping one.
                            Default: 0.0 (frames are dropped, not blended).
    :param kwargs:          Additional keyword arguments to pass to VDecimate.
                            For a list of parameters, see the VIVTC documentation.

    :return:                Decimated clip with duplicate frames removed or blended.
    """

    func = FunctionUtil(clip, vdecimate, None, (vs.YUV, vs.GRAY), (8, 16))

    vdecimate_kwargs = dict[str, Any]()

    if block := kwargs.pop('block', None):
        if isinstance(block, int):
            vdecimate_kwargs |= dict(blockx=block, blocky=block)
        else:
            vdecimate_kwargs |= dict(blockx=block[0], blocky=block[1])

    if not kwargs.get('clip2', None) and func.work_clip.format is not clip.format:
        vdecimate_kwargs |= dict(clip2=clip)

    dryrun = kwargs.pop('dryrun', False)

    if dryrun or weight:
        stats = func.work_clip.vivtc.VDecimate(dryrun=True, **(vdecimate_kwargs | kwargs))

        if dryrun:
            return func.return_clip(stats)

        clip = kwargs.pop('clip2', clip)

        avg = clip.std.AverageFrames(weights=[0, 1 - weight, weight])
        splice = find_prop_rfs(clip, avg, "VDecimateDrop", "==", 1, stats)
        vdecimate_kwargs |= dict(clip2=splice)

    decimate = func.work_clip.vivtc.VDecimate(**(vdecimate_kwargs | kwargs))

    return func.return_clip(decimate)
