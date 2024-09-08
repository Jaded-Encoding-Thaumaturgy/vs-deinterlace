from __future__ import annotations

from typing import Any

from stgpytools import CustomIntEnum
from vsdenoise import MVTools
from vstools import InvalidFramerateError, check_variable, core, vs

__all__ = [
    'telop_resample'
]


class telop_resample(CustomIntEnum):
    TXT60i_on_24telecined = 0
    TXT60i_on_24duped = 1
    TXT30p_on_24telecined = 2

    def __call__(self, bobbed_clip: vs.VideoNode, pattern: int, **mv_args: Any) -> vs.VideoNode:
        """
        Virtually oversamples the video to 120 fps with motion interpolation on credits only, and decimates to 24 fps.
        Requires manually specifying the 3:2 pulldown pattern (the clip must be split into parts if it changes).

        :param bobbed_clip:             Bobbed clip. Framerate must be 60000/1001.
        :param pattern:                 First frame in the pattern.
        :param mv_args:                 Arguments to pass on to MVTools, used for motion compensation.

        :return:                        Decimated clip with text resampled down to 24p.

        :raises InvalidFramerateError:  Bobbed clip does not have a framerate of 60000/1001 (59.94)
        """

        assert check_variable(bobbed_clip, telop_resample)

        InvalidFramerateError.check(telop_resample, bobbed_clip, (60000, 1001))

        invpos = (5 - pattern * 2 % 5) % 5

        offset = [0, 0, -1, 1, 1][pattern]
        pattern = [0, 1, 0, 0, 1][pattern]
        direction = [-1, -1, 1, 1, 1][pattern]

        ivtc_fps, ivtc_fps_div = (dict(fpsnum=x, fpsden=1001) for x in (24000, 12000))

        pos = []
        assumefps = 0

        interlaced = self in (telop_resample.TXT60i_on_24telecined, telop_resample.TXT60i_on_24duped)
        decimate = self is telop_resample.TXT60i_on_24duped

        cycle = 10 // (1 + interlaced)

        def bb(idx: int, cut: bool = False) -> vs.VideoNode:
            if cut:
                return bobbed_clip[cycle:].std.SelectEvery(cycle, [idx])
            return bobbed_clip.std.SelectEvery(cycle, [idx])

        def intl(clips: list[vs.VideoNode], toreverse: bool, halv: list[int]) -> vs.VideoNode:
            clips = [c[::2] if i in halv else c for i, c in enumerate(clips)]
            if not toreverse:
                clips = list(reversed(clips))
            return core.std.Interleave(clips)

        if interlaced:
            if decimate:
                cleanpos = 4
                pos = [1, 2]

                if invpos > 2:
                    pos = [6, 7]
                    assumefps = 2
                elif invpos > 1:
                    pos = [2, 6]
                    assumefps = 1
            else:
                cleanpos = 1
                pos = [3, 4]

                if invpos > 1:
                    cleanpos = 6
                    assumefps = 1

                if invpos > 3:
                    pos = [4, 8]
                    assumefps = 1

            clean = bobbed_clip.std.SelectEvery(cycle, [cleanpos - invpos])
            jitter = bobbed_clip.std.SelectEvery(cycle, [p - invpos for p in pos])

            if assumefps:
                jitter = core.std.AssumeFPS(
                    bobbed_clip[0] * assumefps + jitter, **(ivtc_fps_div if cleanpos == 6 else ivtc_fps)
                )

            comp = MVTools(jitter, **mv_args).flow_interpolate()

            out = intl([comp, clean], decimate, [0])
            offs = 3 if decimate else 2

            return out[invpos // offs:]

        if pattern == 0:
            c1pos = [0, 2, 7, 5]
            c2pos = [3, 4, 9, 8]

            if offset == -1:
                c1pos = [2, 7, 5, 10]

            if offset == 1:
                c2pos = []
                c2 = core.std.Interleave([bb(4), bb(5), bb(0, True), bb(9)])
        else:
            c1pos = [2, 4, 9, 7]
            c2pos = [0, 1, 6, 5]

            if offset == 1:
                c1pos = []
                c1 = core.std.Interleave([bb(3), bb(5), bb(0, True), bb(8)])

            if offset == -1:
                c2pos = [1, 6, 5, 10]

        if c1pos:
            c1 = bobbed_clip.std.SelectEvery(cycle, [c + offset for c in c1pos])

        if c2pos:
            c2 = bobbed_clip.std.SelectEvery(cycle, [c + offset for c in c2pos])

        if offset == -1:
            c1, c2 = (core.std.AssumeFPS(bobbed_clip[0] + c, **ivtc_fps) for c in (c1, c2))

        fix1 = MVTools(c1, **mv_args).flow_interpolate(time=50 + direction * 25)
        fix2 = MVTools(c2, **mv_args).flow_interpolate()

        return intl([fix1, fix2], pattern == 0, [0, 1])
