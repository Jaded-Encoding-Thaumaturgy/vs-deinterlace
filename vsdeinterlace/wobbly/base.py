from functools import partial
from typing import Sequence

from vstools import (CustomKeyError, CustomValueError, DependencyNotFoundError,
                     FieldBased, FieldBasedT, FuncExceptT, VSFunction, core,
                     replace_ranges, vs)

from vsdeinterlace.combing import fix_interlaced_fades
from vsdeinterlace.wobbly.info import CustomList

from .info import FreezeFrame, InterlacedFade, OrphanField
from .types import CustomPostFiltering, Match


class _WobblyProcessBase:
    """A base class for Wobbly processing methods."""

    def _check_plugin_installed(self, plugin: str, func_except: FuncExceptT | None = None) -> None:
        """Check if a plugin is installed."""

        func = func_except or self._check_plugin_installed

        if not hasattr(core, plugin):
            raise DependencyNotFoundError(
                f"Could not find the \"{plugin}\" plugin. Please install it!", func  # type: ignore[arg-type]
            )

    def _apply_fieldmatches(
        self, clip: vs.VideoNode, matches: Sequence[Match],
        tff: FieldBasedT | None = None,
        func_except: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """Apply fieldmatches to a clip."""

        self._check_plugin_installed('fh', func_except)

        tff = FieldBased.from_param_or_video(tff, clip)

        clip = clip.fh.FieldHint(None, tff, ''.join(matches))

        match_clips = {match: clip.std.SetFrameProps(wobbly_match=match) for match in set(matches)}

        return clip.std.FrameEval(lambda n: match_clips.get(matches[n]))

    def _apply_freezeframes(
        self, clip: vs.VideoNode, freezes: set[FreezeFrame],
        func_except: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """Apply freezeframes to a clip."""

        start_frames, end_frames, replacements = list[int](), list[int](), list[int]()
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
            raise CustomValueError("Could not freeze frames!", func_except or self._apply_freezeframes) from e

        def _set_props(n: int, clip: vs.VideoNode) -> vs.VideoNode:
            return clip.std.SetFrameProps(**dict(freeze_props[n])) if n in freeze_props else clip

        return fclip.std.FrameEval(partial(_set_props, clip=clip))

    def _deinterlace_orphans(
        self, clip: vs.VideoNode, orphans: Sequence[OrphanField],
        deinterlacing_function: VSFunction = core.resize.Bob,
        func_except: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """Deinterlace orphaned fields."""

        func = func_except or self._deinterlace_orphans  # noqa

        field_order = FieldBased.from_video(clip)

        # TODO: implement good deinterlacing aimed at orphan fields.
        # TODO: Try to be smart and freezeframe frames instead if they're literally identical to prev/next frame.
        # TODO: Figure out what to actually pass here for custom deinterlacing functions.
        deint_np = deinterlacing_function(clip, tff=not field_order.is_tff)[field_order.is_tff::2]  # type:ignore
        deint_bu = deinterlacing_function(clip, tff=field_order.is_tff)[field_order.is_tff::2]  # type:ignore

        frames_by_match: dict[str, list[int]] = {
            k.__args__[0]: [] for k in Match.__args__ if k.__args__[0] != 'c'  # type:ignore
        }

        try:
            for o in orphans:
                frames_by_match[o.match].append(o.framenum)
        except KeyError:
            raise CustomKeyError("Invalid orphan match!", func)

        out_clip = replace_ranges(clip, deint_np, frames_by_match['n'] + frames_by_match['p'])
        out_clip = replace_ranges(out_clip, deint_bu, frames_by_match['b'] + frames_by_match['u'])

        # TODO: Put the match in the frame props.
        out_clip = replace_ranges(
            out_clip, out_clip.std.SetFrameProps(wobbly_orphan_deinterlace=True),
            [f for frames in frames_by_match.values() for f in frames]
        )

        return out_clip

    def _apply_combed_markers(self, clip: vs.VideoNode, combed_frames: set[int]) -> vs.VideoNode:
        """Apply combed markers to a clip."""

        return replace_ranges(
            clip.std.SetFrameProps(wobbly_combed=0),
            clip.std.SetFrameProps(wobbly_combed=1),
            list(combed_frames)
        )

    def _apply_interlaced_fades(
        self, clip: vs.VideoNode, ifades: set[InterlacedFade],
        func_except: FuncExceptT | None = None
    ) -> vs.VideoNode:
        # TODO: Figure out how to get the right `color` param per frame with an eval.

        func = func_except or self._apply_interlaced_fades

        return replace_ranges(
            clip.std.SetFrameProps(wobbly_fif=False),
            fix_interlaced_fades(clip, colors=0, planes=0, func=func).std.SetFrameProps(wobbly_fif=True),
            [f.framenum for f in ifades]
        )

    def _apply_custom_list(
        self, clip: vs.VideoNode, custom_list: list[CustomList], pos: CustomPostFiltering
    ) -> vs.VideoNode:
        """Apply a list of custom functions to a clip based on the pos."""

        for custom in custom_list:
            if custom.position == pos:
                custom_clip = custom.preset.apply_preset(clip)

                custom_clip = custom_clip.std.SetFrameProps(
                    wobbly_custom_list_name=custom.name,
                    wobbly_custom_list_position=str(pos)
                )

                clip = replace_ranges(clip, custom_clip, custom.frames)

        return clip
