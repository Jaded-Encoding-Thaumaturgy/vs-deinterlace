import json
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from typing import Any, Literal

from vstools import (CustomValueError, FieldBased, FileNotExistsError, Keyframes, SPath, SPathLike, Timecodes, core,
                     get_prop, replace_ranges, vs)

from ..combing import fix_telecined_fades
from .types import Types

__all__ = [
    "WobblyParsed", "parse_wobbly"
]


@dataclass
class WobblyParsed:
    """Dataclass representing the contents of a parsed Wobbly file."""

    field_order: FieldBased
    """The field order represented as a FieldBased object."""

    cycle: int
    """Size of the cycle in number of frames."""

    base_fps: Fraction
    """The base framerate of the input clip."""

    trims: list[tuple[int, int]]
    """The trims applied to the clip"""

    matches: list[str]
    """The field matches."""

    combs: list[int]
    """A list of combed frames. Frames with interlaced fades will be excluded."""

    orphans: list[tuple[int, Types.Match]]
    """A list of orphan fields and the type of fieldmatch for that frame."""

    decimations: list[int]
    """A list of decimated frames."""

    scenechanges: Keyframes
    """Scenechanges (sections) represented as a Keyframes object."""

    timecodes: Timecodes
    """The timecodes represented as a Timecode object."""

    interlaced_fades: list[tuple[int, float]]
    """
    A list of tuples containing information about frames marked as being interlaced fades.
    The first item is the frame. The second item is the field difference.
    """

    freezes: list[tuple[int, int, int]]
    """
    A list of tuples representing the start and end ranges of frames to freeze,
    and which frame to replace them with.
    """

    def __post__init__(self) -> None:
        if self.cycle != 5:
            raise CustomValueError("Wobbly currently only supports a cycle of 5!", "WobblyParsed")

    def apply(
        self, clip: vs.VideoNode,
        deint_orphans: bool = False,
        **qtgmc_kwargs: Any
    ) -> vs.VideoNode:
        """
        Apply all the Wobbly processing to a given clip.

        :param clip:                Clip to process.
        :param deint_orphans:       Deinterlace orphan fields. This may restore motion that is
                                    otherwise lost, at a heavy cost of speed and potential issues.
                                    Default: False.
        """
        clip = self.field_order.apply(clip)
        ref = clip

        if self.trims:
            # Can this be done more efficiently?
            clip = core.std.Splice([clip.std.Trim(s, e) for s, e in self.trims])

        if deint_orphans:
            for f in self.orphans:
                self.matches[f] = 'c'  # type:ignore[call-overload]

        clip = clip.fh.FieldHint(None, self.field_order - 1, "".join(self.matches))

        if self.freezes:
            clip = clip.std.FreezeFrames(*zip(*self.freezes))

        clip = self._mark_framerates(clip)

        if self.interlaced_fades:
            clip = self._ftf(clip)

        if self.orphans:
            clip = replace_ranges(
                replace_ranges(
                    clip, clip.std.SetFrameProps(wobbly_orphan_match="n"),
                    [f for f, m in self.orphans if m == "n"]
                ),
                clip.std.SetFrameProps(wobbly_orphan_match="b"),
                [f for f, m in self.orphans if m == "b"]
            )

            if deint_orphans:
                clip = self._deint_orphans(clip, ref, **qtgmc_kwargs)

        if self.combs:
            clip = replace_ranges(
                clip.std.SetFrameProps(_Combed=0),
                clip.std.SetFrameProps(_Combed=1),
                self.combs
            )

        if self.decimations:
            clip = clip.std.DeleteFrames(self.decimations)

        return FieldBased.PROGRESSIVE.apply(clip)

    def _mark_framerates(self, clip: vs.VideoNode) -> vs.VideoNode:
        """Mark the framerates per cycle."""
        framerates = [self.base_fps.numerator / self.cycle * i for i in range(self.cycle, 0, -1)]

        fps_clips = [
            clip.std.AssumeFPS(None, int(fps), self.base_fps.denominator)
            .std.SetFrameProps(
                wobbly_cycle_fps=int(fps // 1000),
                _DurationNum=int(fps),
                _DurationDen=self.base_fps.denominator
            ) for fps in framerates
        ]

        split_decimations = [
            [j for j in range(i * self.cycle, (i + 1) * self.cycle) if j in self.decimations]
            for i in range(0, ceil((max(self.decimations) + 1) / self.cycle))
        ]

        n_split_decimations = len(split_decimations)

        indices = [
            0 if (sd_idx := ceil(n / self.cycle)) >= n_split_decimations
            else len(split_decimations[sd_idx]) for n in range(clip.num_frames)
        ]

        return clip.std.FrameEval(lambda n: fps_clips[indices[n]])

    def _ftf(self, clip: vs.VideoNode) -> vs.VideoNode:
        # TODO: Figure out how to get the right `color` param per frame with an eval.
        # TODO: Check if other ftf works better and implement that

        clip = clip.std.SetFrameProps(wobbly_ftf=False)

        ftf = fix_telecined_fades(clip, colors=0, planes=0) \
            .std.SetFrameProps(wobbly_ftf=True)

        return replace_ranges(clip, ftf, [f for f, _ in self.interlaced_fades])

    def _deint_orphans(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        """Deinterlace orphan frames."""
        from havsfunc import QTGMC  # type:ignore[import]

        clip = clip.std.SetFrameProps(wobbly_deint=False)

        # TODO: test these settings a bit better.
        qtgmc_args: dict[str, Any] = dict(
            TR0=2, TR1=2, TR2=2, Sharpness=0, Lossless=1, InputType=0,
            Rep0=3, Rep1=3, Rep2=2, SourceMatch=3, EdiMode='EEDI3+NNEDI3', EdiQual=2,
            Sbb=3, SubPel=4, SubPelInterp=2, opencl=False, RefineMotion=True,
            Preset='Placebo', MatchPreset='Placebo', MatchPreset2='Placebo'
        ) | kwargs | dict(FPSDivisor=1, TFF=self.field_order.is_tff)

        # TODO: test this further
        deint = clip.std.SetFrameProps(wobbly_deint=True)

        deint_n = QTGMC(deint[1:] + deint[-1], **qtgmc_args)[self.field_order.field::2]
        deint_b = QTGMC(deint, **qtgmc_args)[self.field_order.field::2]

        out = replace_ranges(clip, deint_n, [f for f, m in self.orphans if m == "n"])
        out = replace_ranges(out, deint_b, [f for f, m in self.orphans if m == "b"])

        return out


# TODO: overloads
def parse_wobbly(
    clip: vs.VideoNode,
    file: SPathLike,
) -> WobblyParsed:
    """
    Parse the contents of a Wobbly project file and return a processed clip.

    :param clip:            The clip to process. If a Wobbly file is passed instead,
                            it will load the clip as defined in `input file`.
    """
    wob_file = SPath(file)

    if not wob_file.suffix == ".wob":
        wob_file = wob_file.with_suffix(f"{wob_file.suffix}.wob")

    if not wob_file.exists():
        raise FileNotExistsError(f"Could not find the file \"{wob_file.absolute()}\"", parse_wobbly)

    with open(wob_file, "r") as f:
        data = dict(json.load(f))

    # Get the contents
    input_file = data.get("input file")
    framerate = Fraction(*data.get("input frame rate", [30000, 1001]))
    cycle = dict(data.get("vdecimate parameters", {})).get("cycle", 5)
    order = dict(data.get("vfm parameters", {})).get("order", -1)
    matches = data.get("matches", [])
    combs = data.get("combed frames", [])
    decimations = data.get("decimated frames", [])
    sections = data.get("sections", [])
    fades = data.get("interlaced fades", [])
    # TODO: See if we can somehow hack in 60p support. Probably have users set a prop with `presets`? Maybe in `Wibbly`?
    # presets = data.get("presets", [])
    # custom = data.get("custom lists", [])

    if (idx_path := get_prop(clip, "idx_filepath", bytes, None, b"", parse_wobbly).decode('utf-8')):
        if idx_path != input_file:
            # TODO: throw a proper error later once I have to actually test this properly
            print(f"file mismatch ({idx_path} != {input_file})")

    if bool(len(illegal_chars := set(matches) - {*FieldMatch.__args__})):  # type:ignore[attr-defined]
        raise CustomValueError(f"Illegal characters found in matches {tuple(illegal_chars)}", parse_wobbly)

    fades = [(fade.get("frame"), fade.get("field difference")) for fade in (dict(f) for f in fades)]

    # TODO: timecodes object
    # timecodes = Timecodes(
    #     Timecode(f, (framerate.numerator * (i // cycle)), framerate.denominator)
    #     for i in []
    #     for f in len(matches)
    # )

    # TODO: fix typing
    return WobblyParsed(
        field_order=FieldBased.from_param(order + 1, parse_wobbly) or FieldBased.TFF,
        base_fps=framerate,
        cycle=cycle,
        trims=data.get("trim", []),
        matches=matches,
        combs=list(set(combs) - set([f for f, _ in fades])),
        orphans=[(o, matches[o]) for o in data.get("orphan frames", [])],   # type:ignore[misc]
        decimations=decimations,
        scenechanges=sections,
        timecodes=Timecodes(sections),
        interlaced_fades=fades,
        freezes=[tuple(freeze) for freeze in data.get("frozen frames", [])]  # type:ignore[misc]
    )
