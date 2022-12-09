from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any

from vstools import CustomTypeError, FieldBased, FieldBasedT, core, get_render_progress, vs

__all__ = [
    'sivtc',
    'tivtc_2pass', 'tivtc_vfr'
]


def sivtc(clip: vs.VideoNode, pattern: int = 0, tff: bool | FieldBasedT = True, decimate: bool = True) -> vs.VideoNode:
    """
    Simplest form of a fieldmatching function.

    This is essentially a stripped-down JIVTC offering JUST the basic fieldmatching and decimation part.
    As such, you may need to combine multiple instances if patterns change throughout the clip.

    :param clip:        Clip to process.
    :param pattern:     First frame of any clean-combed-combed-clean-clean sequence.
    :param tff:         Top-Field-First.
    :param decimate:    Drop a frame every 5 frames to get down to 24000/1001.

    :return:            IVTC'd clip.
    """

    pattern = pattern % 5

    defivtc = core.std.SeparateFields(clip, tff=FieldBased.from_param(tff).field).std.DoubleWeave()
    selectlist = [[0, 3, 6, 8], [0, 2, 5, 8], [0, 2, 4, 7], [2, 4, 6, 9], [1, 4, 6, 8]]
    dec = core.std.SelectEvery(defivtc, 10, selectlist[pattern]) if decimate else defivtc
    return dec.std.SetFieldBased(0).std.SetFrameProp(prop='SIVTC_pattern', intval=pattern)


main_file = os.path.realpath(sys.argv[0]) if sys.argv[0] else None
main_file = os.path.splitext(os.path.basename(str(main_file)))[0]
main_file = "{yourScriptName}_" if main_file in ("__main___", "setup_") else main_file


def tivtc_2pass(
    clip: vs.VideoNode,
    decimate_mode: int | tuple[int, int], decimate: int | bool = True,
    tfm_in: Path | str = f".ivtc/{main_file}_matches.txt",
    tdec_in: Path | str = f".ivtc/{main_file}_metrics.txt",
    timecodes_out: Path | str = f".ivtc/{main_file}_timecodes.txt",
    tfm_args: dict[str, Any] | None = None, tdecimate_args: dict[str, Any] | None = None,
    tfm_pass_args: tuple[dict[str, Any] | None, dict[str, Any] | None] | None = None,
    tdecimate_pass_args: tuple[dict[str, Any] | None, dict[str, Any] | None] | None = None,
) -> vs.VideoNode:
    """
    Perform TFM and TDecimate on a clip.

    Includes automatic generation of a metrics/matches/timecodes txt file.

    Dependencies:

    * `TIVTC <https://github.com/dubhater/vapoursynth-tivtc>`_


    :param clip:                Clip to process.
    :param decimate_mode:       Decimate mode for both passes of TDecimate.
    :param decimate:            Whether to perform TDecimate on the clip or returns TFM'd clip only.
                                Set to -1 to use TDecimate without TFM.
    :param tfm_in:              Location for TFM's matches analysis.
    :param tdec_in:             Location for TDecimate's metrics analysis.
    :param timecodes_out:       Location for TDecimate's timecodes analysis.
    :param tfm_args:            Additional arguments to pass to TFM.
    :param tdecimate_args:      Additional arguments to pass to TDecimate.
    :param tfm_pass_args:       Arguments that will overwrite ``tfm_args`` in the first and second pass.
    :param tdecimate_pass_args: Arguments that will overwrite ``tdecimate_args`` in the first and second pass.

    :return:                    IVTC'd clip with external timecode/matches/metrics txt files.

    :raises TypeError:          Invalid ``decimate`` argument is passed.
    """

    decimate = int(decimate)

    if decimate not in {-1, 0, 1}:
        raise CustomTypeError(
            "Invalid 'decimate' argument. Must be True/False, their integer values, or -1!", tivtc_vfr
        )

    tfm_args, tdecimate_args = tfm_args or {}, tdecimate_args or {}

    if tfm_pass_args:
        tfm_pass1, tfm_pass2 = tuple((tfm_args | (x or {})) for x in tfm_pass_args)
    else:
        tfm_pass1, tfm_pass2 = tfm_args.copy(), tfm_args.copy()

    if tdecimate_pass_args:
        tdecimate_pass1, tdecimate_pass2 = tuple((tdecimate_args | (x or {})) for x in tdecimate_pass_args)
    else:
        tdecimate_pass1, tdecimate_pass2 = tdecimate_args.copy(), tdecimate_args.copy()

    if isinstance(decimate_mode, tuple):
        tdec_mode1, tdec_mode2 = decimate_mode
    else:
        tdec_mode1 = tdec_mode2 = decimate_mode

    tdecimate_pass1, tdecimate_pass2 = tdecimate_pass1 | dict(mode=tdec_mode1), tdecimate_pass2 | dict(mode=tdec_mode2)

    tfm_f = tdec_f = timecodes_f = Path()

    def _set_paths() -> None:
        nonlocal tfm_f, tdec_f, timecodes_f
        tfm_f = Path(tfm_in).resolve().absolute()
        tdec_f = Path(tdec_in).resolve().absolute()
        timecodes_f = Path(timecodes_out).resolve().absolute()

    _set_paths()

    for p in (tfm_f, tdec_f, timecodes_f):
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.stat().st_size == 0:
            p.unlink(True)

    if not (tfm_f.exists() and tdec_f.exists()):
        ivtc_clip = clip.tivtc.TFM(output=str(tfm_f), **tfm_pass1)
        ivtc_clip = ivtc_clip.tivtc.TDecimate(**tdecimate_pass1, output=str(tdec_f))

        with get_render_progress() as pr:
            task = pr.add_task("calculating matches and metrics...", total=ivtc_clip.num_frames)

            for _ in ivtc_clip.frames(close=True):
                pr.update(task, advance=1)

        ivtc_clip = None  # type: ignore
        del ivtc_clip

        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

        _set_paths()

    if decimate != -1:
        clip = clip.tivtc.TFM(**tfm_pass2, input=str(tfm_f))

    if decimate == 0:
        return clip

    return clip.tivtc.TDecimate(
        **tdecimate_pass2, input=str(tdec_f), tfmIn=str(tfm_f), mkvOut=str(timecodes_f)
    )


def tivtc_vfr(
    clip: vs.VideoNode,
    tfm_in: Path | str = f".ivtc/{main_file}_matches.txt",
    tdec_in: Path | str = f".ivtc/{main_file}_metrics.txt",
    timecodes_out: Path | str = f".ivtc/{main_file}_timecodes.txt",
    decimate: int | bool = True, tfm_args: dict[str, Any] | None = None,
    tdecimate_args: dict[str, Any] | None = None, **kwargs: Any
) -> vs.VideoNode:
    """
    Perform TFM and TDecimate on a clip that is supposed to be VFR.

    Includes automatic generation of a metrics/matches/timecodes txt file.

    Dependencies:

    * `TIVTC <https://github.com/dubhater/vapoursynth-tivtc>`_


    :param clip:                Clip to process.
    :param tfm_in:              Location for TFM's matches analysis.
    :param tdec_in:             Location for TDecimate's metrics analysis.
    :param timecodes_out:       Location for TDecimate's timecodes analysis.
    :param decimate:            Whether to perform TDecimate on the clip or returns TFM'd clip only.
                                Set to -1 to use TDecimate without TFM.
    :param tfm_args:            Additional arguments to pass to TFM.
    :param tdecimate_args:      Additional arguments to pass to TDecimate.
    :param kwargs:              Additional kwargs for ``tivtc_2pass``.

    :return:                    IVTC'd VFR clip with external timecode/matches/metrics txt files.

    :raises TypeError:          Invalid ``decimate`` argument is passed.
    """

    return tivtc_2pass(
        clip, (4, 5), decimate, tfm_in, tdec_in, timecodes_out, tfm_args,
        tdecimate_args, None, (None, dict(hybrid=2, vfrDec=1)), **kwargs
    )
