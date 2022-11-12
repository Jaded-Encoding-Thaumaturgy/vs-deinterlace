from __future__ import annotations

from fractions import Fraction
from functools import partial
from math import gcd
from typing import SupportsFloat

from vskernels import BicubicDidee, Catrom
from vstools import (
    CustomError, CustomValueError, Dar, FieldBased, FieldBasedT, FuncExceptT, Region, core, depth, fallback, get_prop,
    get_w, mod2, mod4, vs
)

from .helpers import check_ivtc_pattern

__all__ = [
    'seek_cycle',

    'check_patterns',

    'PARser'
]


def seek_cycle(clip: vs.VideoNode, write_props: bool = True, scale: int = -1) -> vs.VideoNode:
    """
    Purely visual tool to view telecining cycles.

    .. warning::
        | This is purely a visual tool and has no matching parameters!
        | Just use `Wobbly <https://github.com/dubhater/Wobbly>`_ instead if you need that.

    Displays the current frame, two previous and future frames,
    and whether they are combed or not.

    ``P`` indicates a progressive frame,
    and ``C`` a combed frame.

    Dependencies:

    * `VapourSynth-TDeintMod <https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TDeintMod>`_

    :param clip:            Clip to process.
    :param write_props:     Write props on frames. Disabling this will also speed up the function.
    :param scale:           Integer scaling of all clips. Must be to the power of 2.

    :return:                Viewing UI for standard telecining cycles.

    :raises ValueError:     `scale` is a value that is not to the power of 2.
    """

    if (scale & (scale - 1) != 0) and scale != 0 and scale != -1:
        raise CustomValueError("'scale' must be a power of 2!", seek_cycle)

    # TODO: 60i checks and flags somehow? false positives gonna be a pain though
    def check_combed(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.text.Text("C" if get_prop(f, "_Combed", int) else "P", 7)

    # Scaling of the main clip
    scale = 1 if scale == -1 else 2 ** scale

    height = clip.height * scale
    width = get_w(height, clip.width / clip.height)

    clip = clip.tdm.IsCombed()
    clip = Catrom().scale(clip, width, height)

    # Downscaling for the cycle clips
    clip_down = BicubicDidee().scale(clip, mod2(width / 4), mod2(height / 4))
    if write_props:
        clip_down = core.std.FrameEval(clip_down, partial(check_combed, clip=clip_down), clip_down).text.FrameNum(2)
    blank_frame = clip_down.std.BlankClip(length=1, color=[0] * 3)

    pad_c = clip_down.text.Text("Current", 8) if write_props else clip_down
    pad_a, pad_b = blank_frame * 2 + clip_down[:-2], blank_frame + clip_down[:-1]
    pad_d, pad_e = clip_down[1:] + blank_frame, clip_down[2:] + blank_frame * 2

    # Cycling
    cycle_clips = [pad_a, pad_b, pad_c, pad_d, pad_e]
    pad_x = [pad_a.std.BlankClip(mod4(pad_a.width / 15))] * 4
    cycle = cycle_clips + pad_x  # no shot this can't be done way cleaner
    cycle[::2], cycle[1::2] = cycle_clips, pad_x

    # Final stacking
    stack_abcde = core.std.StackHorizontal(cycle)

    vert_pad = stack_abcde.std.BlankClip(height=mod2(stack_abcde.height / 5))
    horz_pad = clip.std.BlankClip(mod2((stack_abcde.width - clip.width) / 2))

    stack = core.std.StackHorizontal([horz_pad, clip, horz_pad])
    return core.std.StackVertical([vert_pad, stack, vert_pad, stack_abcde])


def check_patterns(clip: vs.VideoNode, tff: bool | FieldBasedT | None = None) -> int:
    """
    Naïve function that iterates over a given clip and tries out every simple 3:2 IVTC pattern.

    This function will return the best pattern value that didn't result in any combing.
    If all of them resulted in combing, it will raise an error.

    Note that the clip length may seem off because I grab every fourth frame of a clip.
    This should make processing faster, and it will still find combed frames.

    This function should only be used for rudimentary testing.
    If I see it in any proper scripts, heads will roll.

    Dependencies:

    * `VapourSynth-TDeintMod <https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TDeintMod>`_

    :param clip:                    Clip to process.
    :param tff:                     Top-field-first. `False` sets it to Bottom-Field-First.
                                    If None, get the field order from the _FieldBased prop.

    :return:                        Integer representing the best pattern.

    :raises TopFieldFirstError:     No automatic ``tff`` can be determined.
    :raises StopIteration:          No pattern resulted in a clean match.
    """

    clip = FieldBased.ensure_presence(clip, tff, check_patterns)

    clip = depth(clip, 8)

    pattern = -1

    for n in [int(n) for n in range(0, 4)]:
        check = check_ivtc_pattern(clip, n)

        if check:
            pattern = n
            break

    if pattern == -1:
        raise CustomError[StopIteration](
            'None of the patterns resulted in a clip without combing. '
            'Please try performing proper IVTC on the clip.', check_patterns
        )

    return pattern


def PARser(
    clip: vs.VideoNode, active_area: int,
    dar: Dar | str | Fraction | None = None, height: int | None = None,
    region: Region | str = Region.NTSC,
    return_result: bool = False, func: FuncExceptT | None = None
) -> vs.VideoNode | dict[str, SupportsFloat | tuple[int, int] | str]:
    """
    Calculate SAR (sample aspect ratio) and attach result as frameprops.

    The active area is calculated by subtracting the dirty edges from the left and right side of the clip.
    Only take the darker pixels into account, and make sure you gather them from bright scenes!
    The SAR and anamorphic width/height will be added to the clip as props, but can also be printed.

    List of common valid active areas (note that ``active_area`` only sets the width!):

        * 704x480 (NTSC, MPEG-4)
        * 708x480 (NTSC, SMPTE Rp87-1995)
        * 710.85x486 (NTSC, ITU-R REC.601)
        * 711x480 (NTSC, MPEG-4)

    Make sure you absolutely DO NOT ACTUALLY CROP in the direction you're stretching!
    If you do, the end result will be off, and the aspect ratio will be wrong! Just leave them alone.
    It's okay to crop the top/bottom when going widescreen, and left/right when going fullscreen.

    If you absolutely *must* crop the pixels away, consider letting the video decoder handle that during playback.
    See: `display-window (x265) <https://x265.readthedocs.io/en/master/cli.html#cmdoption-display-window>`_

    It's not recommended to stretch to the final resolution after calculating the SAR.
    Instead, set the SAR in your encoder's settings and encode your video as anamorphic.
    This will result in the most accurate final image without introducing compounding resampling artefacting
    (don't worry, plenty programs still support anamorphic video).

    Core idea originated from a `private gist <https://gist.github.com/wiwaz/40883bae396bef5eb9fc99d4de2377ec>`_
    and was heavily modified by LightArrowsEXE.

    For more information, I highly recommend reading
    `this blogpost <https://web.archive.org/web/20140218044518/http://lipas.uwasa.fi/~f76998/video/conversion/>`_.

    :param clip:                Input clip.
    :param active_area:         Width you would end up with post-cropping.
                                Only take into account darker messed up edges!
    :param dar:                 Display Aspect Ratio. Refers to the analog television aspect ratio.
                                Must be a :py:attr:`vstools.Dar` enum, a string representing a Dar value,
                                or a Fraction object containing a user-defined DAR.
                                If None, automatically guesses DAR based on the SAR props.
                                Default: assume based on current SAR properties.
    :param height:              Height override. If None, auto-select based on region.
                                This is not particularly useful unless you want to set it to 486p
                                (to use with for example a 1920x1080 -> 864x486 -> 864x480 downscaled + cropped DVD
                                where the studio did not properly account for anamorphic resolutions)
                                or need to deal with ITU-R REC.601 video.
                                Default: input clip's height.
    :param region:              Analog television region. Must be either NTSC or PAL.
                                Must be a :py:attr:`vstools.Region` enum  or string representing a Region value.
                                Default: :py:attr:`vstools.Region.NTSC`.
    :param return_result:       Return the results as a dict. Default: False.

    :return:                    Clip with corrected SAR props and anamorphic width/height prop,
                                or a dictionary with all the results.

    :raises FramePropError:     DAR is None and no SAR props are set on the input clip.
    :raises ValueError:         Invalid :py:attr:`vstools.Dar` is passed.
    :raises ValueError:         Invalid :py:attr:`vstools.Region` is passed.
    """

    func = fallback(func, PARser)

    match dar:
        case Fraction(): new_dar = dar.numerator, dar.denominator
        case Dar.WIDE: new_dar = 16, 9
        case Dar.FULL: new_dar = 4, 3
        case Dar.SQUARE: return clip.std.SetFrameProps(_SARDen=1, _SARNum=1)
        case None: return PARser(clip, active_area, Dar.from_video(clip), height, region, return_result, func)
        case _: raise CustomValueError('Invalid DAR passed! Must be in {values} or None!', func, values=iter(Dar))

    props = dict[str, SupportsFloat | tuple[int, int] | str](dar=new_dar)

    if height is None:
        match region:
            case Region.NTSC: height = 480
            case Region.PAL: height = 576
            case _: raise CustomValueError('Invalid Region passed! Must be in {values}!', func, values=iter(Region))

    sar = new_dar[0] * height, new_dar[1] * active_area
    sargcd = gcd(sar[0], sar[1])

    sarden = sar[0] // sargcd
    sarnum = sar[1] // sargcd

    props.update(_SARDen=sarden, _SARNum=sarnum)

    match dar:
        case Dar.WIDE: props.update(amorph_width=clip.width * (sarden / sarnum))
        case Dar.FULL: props.update(amorph_height=clip.height * (sarnum / sarden))
        # TODO: autoguess which to return based on the sarnum maybe?
        case _: props.update(
            amorph__note="Use your best judgment to pick one!",
            amorph_width=clip.width * (sarden / sarnum),
            amorph_height=clip.height * (sarnum / sarden)
        )

    if return_result:
        return props

    return clip.std.SetFrameProps(**props)
