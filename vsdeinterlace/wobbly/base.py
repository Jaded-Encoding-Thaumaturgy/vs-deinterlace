from typing import Sequence

from vstools import (CustomValueError, DependencyNotFoundError, FuncExceptT,
                     core, replace_ranges, vs)

from vsdeinterlace.combing import fix_interlaced_fades

from .info import FreezeFrame, InterlacedFade, OrphanField
from .types import Match


class _WobblyProcessBase:
    """A base class for Wobbly processing methods."""

    def _check_plugin_installed(self, plugin: str, func_except: FuncExceptT | None = None) -> None:
        """Check if a plugin is installed."""

        func = func_except or self._check_plugin_installed

        if not hasattr(core, plugin):
            raise DependencyNotFoundError(
                f"Could not find the \"{plugin}\" plugin. Please install it!", func  # type: ignore[arg-type]
            )

    def __apply_fieldmatches(
        self, clip: vs.VideoNode, matches: Sequence[Match],
        func_except: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """Apply fieldmatches to a clip."""

        self._check_plugin_installed('fh', func_except)

        match_clips = dict[str, vs.VideoNode]()

        for match in set(matches):
            match_clips |= {match: clip.std.SetFrameProps(wobbly_match=match)}

        return clip.std.FrameEval(lambda n: match_clips.get(matches[n]))

    def __apply_freezeframes(
        self, clip: vs.VideoNode, freezes: set[FreezeFrame] = [],
        func_except: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """Apply freezeframes to a clip."""

        start_frames = end_frames = replacements = list[int]()
        freeze_props: dict[int, dict[str, int]] = {}

        for freeze in freezes:
            start_frames.append(freeze.start_frame)
            end_frames.append(freeze.end_frame)
            replacements.append(freeze.replacement)

            freeze_props |= {
                freeze.start_frame: {
                    'wobbly_freeze_start': freeze.start_frame,
                    'wobbly_freeze_end': freeze.end_frame,
                    'wobbly_freeze_replacement': freeze.replacement
                }
            }

        try:
            fclip = clip.std.FreezeFrames(start_frames, end_frames, replacements)
        except vs.Error as e:
            raise CustomValueError("Could not freeze frames!", func_except or self.__apply_freezeframes) from e

        return fclip.std.FrameEval(
            lambda n, clip: clip.std.SetFrameProps(freeze_props[n]) if n in freeze_props else clip
        )

    def __deinterlace_orphans(
        self, clip: vs.VideoNode, orphans: Sequence[OrphanField] = [],
        func_except: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """Deinterlace orphaned fields."""

        func = func_except or self.__deinterlace_orphans  # noqa

        b_frames = [o.framenum for o in orphans if o.match == 'b']
        n_frames = [o.framenum for o in orphans if o.match == 'n']
        p_frames = [o.framenum for o in orphans if o.match == 'p']
        u_frames = [o.framenum for o in orphans if o.match == 'u']

        # TODO: implement good deinterlacing aimed at orphan fields.
        # TODO: Try to be smart and freezeframe frames instead if they're literally identical to prev/next frame.
        deint_clip = clip

        out_clip = replace_ranges(deint_clip, deint_clip.std.SetFrameProps(wobbly_orphan_deinterlace='b'), b_frames)
        out_clip = replace_ranges(out_clip, out_clip.std.SetFrameProps(wobbly_orphan_deinterlace='n'), n_frames)
        out_clip = replace_ranges(out_clip, out_clip.std.SetFrameProps(wobbly_orphan_deinterlace='p'), p_frames)
        out_clip = replace_ranges(out_clip, out_clip.std.SetFrameProps(wobbly_orphan_deinterlace='u'), u_frames)

        return out_clip

    def __apply_combed_markers(self, clip: vs.VideoNode, combed_frames: set[int]) -> vs.VideoNode:
        """Apply combed markers to a clip."""

        return replace_ranges(
            clip.std.SetFrameProps(wobbly_combed=0),
            clip.std.SetFrameProps(wobbly_combed=1),
            combed_frames
        )

    def __apply_interlaced_fades(
        self, clip: vs.VideoNode, ifades: set[InterlacedFade] = [],
        func_except: FuncExceptT | None = None
    ) -> vs.VideoNode:
        # TODO: Figure out how to get the right `color` param per frame with an eval.

        func = func_except or self.__apply_interlaced_fades

        return replace_ranges(
            clip.std.SetFrameProps(wobbly_fif=False),
            fix_interlaced_fades(clip, colors=0, planes=0, func=func).std.SetFrameProps(wobbly_fif=True),
            [f.framenum for f in ifades]
        )
