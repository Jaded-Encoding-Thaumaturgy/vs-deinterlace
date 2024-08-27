from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, NamedTuple

from vstools import (CustomValueError, FieldBased, FieldBasedT,
                     FileNotExistsError, FrameRangeN, FuncExceptT,
                     SceneChangeMode, SPath, SPathLike, VSCoreProxy, core, vs)

from .exceptions import InvalidMatchError
from .types import Match, SectionPreset

__all__: list[str] = [
    "WobblyMeta", "WobblyVideo",
    "WibblyConfig", "WibblyConfigSettings",
    "VfmParams", "VDecParams",
    "FreezeFrame",
    "InterlacedFade",
    "OrphanField",
    "Section",
]


@dataclass
class WobblyMeta:
    """Meta information about Wobbly."""

    version: int
    """The Wobbly version used. A value of -1 indicates an unknown version."""

    format_version: int
    """The project formatting version used. A value of -1 indicates an unknown version."""

    author: str | None = None
    """An optional field for external wobbly file authoring functions, such as vsdeinterlace's."""


@dataclass
class WobblyVideo:
    """A class containing information about the clip used inside of wobbly."""

    file_path: SPathLike
    """The path to the input file."""

    source_filter: Callable[[str], vs.VideoNode] | str | VSCoreProxy
    """The source filter used to index the input file."""

    trims: list[tuple[int, int]]
    """The trims applied to the clip"""

    framerate: Fraction
    """The base framerate of the input clip."""

    def __post_init__(self) -> None:
        if isinstance(self.source_filter, str):
            obj = core

            for p in self.source_filter.split('.'):
                obj = getattr(obj, p)

            self.source_filter = obj

    def source(self, func_except: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        Video indexing method.

        Trims are applied to the clip if they exist.

        :param func_except:     Function returned for custom error handling.
                                This should only be set by VS package developers.
        :param kwargs:          Additional arguments to pass to the source filter.

        :return:                The indexed video clip.
        """

        func_except = func_except or self.source

        if not (sfile := SPath(self.file_path)).exists():
            raise FileNotExistsError(f"Could not find the input file, \"{sfile}\"!", func_except)

        src = self.source_filter(sfile.to_str(), **kwargs)  # type:ignore[operator]

        if src.fps != self.framerate:
            src = src.std.AssumeFPS(src, self.framerate.numerator, self.framerate.denominator)

        if not self.trims:
            return src

        return core.std.Splice([src.std.Trim(s, e) for s, e in self.trims])


# TODO: refactor this
class WibblyConfigSettings:
    """Wibbly configuration settings. Please see the plugin documentation for more information."""

    class Crop(NamedTuple):
        left: int = 0
        top: int = 0
        right: int = 0
        bottom: int = 0

        early: bool = True
        """Whether to crop early."""

    class DMetrics(NamedTuple):
        nt: int = 10

    class VMFParams(NamedTuple):
        order: int = 1
        cthresh: int = 9
        mi: int = 80
        blockx: int = 16
        blocky: int = 16
        y0: int = 16
        y1: int = 16
        micmatch: bool = True
        scthresh: float = 12.0
        mchroma: bool = True
        chroma: bool = True

    class VDECParams(NamedTuple):
        blockx: int = 32
        blocky: int = 32
        dupthresh: float = 1.1
        scthresh: float = 15.0
        chroma: bool = True


@dataclass
class WibblyConfig(WibblyConfigSettings):
    """A class representing the configuration for the Wibbly metrics-gathering process."""

    crop: WibblyConfigSettings.Crop | None = WibblyConfigSettings.Crop()
    """The amount the image was cropped on each side."""

    dmetrics: WibblyConfigSettings.DMetrics | None = WibblyConfigSettings.DMetrics()
    """Parameters for DMetrics."""

    vfm: WibblyConfigSettings.VMFParams | None = WibblyConfigSettings.VMFParams()
    """Parameters for VFM."""

    vdec: WibblyConfigSettings.VDECParams | None = WibblyConfigSettings.VDECParams()
    """Parameters for VDecimate."""

    fade_thr: float | None = 0.4 / 255
    """The threshold for detecting fades."""

    sc_mode: SceneChangeMode | None = SceneChangeMode.WWXD
    """Scene detection mode."""


class FrameMetric(NamedTuple):
    """Metric data for single frames."""

    is_combed: bool
    """Whether the frame is combed."""

    is_keyframe: bool
    """Whether the frame is a keyframe."""

    match: Match | None
    """The match type."""

    vfm_mics: list[int] | None
    """VFM mics."""

    mm_dmet: list[int] | None
    """MMetrics."""

    vm_dmet: list[int] | None
    """VMetrics."""

    dec_met: int | None
    """VDecimate metrics."""

    dec_drop: bool
    """Whether the frame was dropped by VDecimate."""

    field_diff: float | None
    """Field difference for interlaced fades."""


@dataclass
class VfmParams:
    """
    VFM parameters used by Wobbly obtained from Wibbly.

    For more information, see the VIVTC documentation.
    """

    order: FieldBasedT = True
    field: int = 2

    mode: int = 1
    mchroma: bool = True
    cthresh: float = 9.0
    chroma: bool = True

    mi: float = 80.0
    blockx: float = 16.0
    blocky: float = 16.0
    y0: float = 16.0
    y1: float = 16.0
    scthresh: float = 12.0
    micmatch: bool = False
    micout: bool = False

    def __post_init__(self) -> None:
        self.order = bool(int(FieldBased.from_param(self.order)) - 1)


@dataclass
class VDecParams:
    """
    VDecimate parameters used by Wobbly obtained from Wibbly.

    For more information, see the VIVTC documentation.
    """

    cycle: int = 5
    chroma: bool = True

    dupthresh: float = 1.1
    scthresh: float = 15.0
    blockx: float = 32.0
    blocky: float = 32.0

    ovr: str | None = None
    dryrun: bool = False


@dataclass
class _HoldsStartEndFrames:
    """A dataclass to store start and end frame data."""

    start_frame: int
    """The first frame of the section."""

    end_frame: int
    """The last frame of the section"""

    def __post_init__(self) -> None:
        if self.start_frame > self.end_frame:
            raise CustomValueError("The start frame must be less than or equal to the end frame!")


@dataclass
class Section(_HoldsStartEndFrames):
    """The sections set in the Wobbly file."""

    presets: list[SectionPreset] = field(default_factory=lambda: [])
    """A list of presets applied to the section of a clip."""

    def __post_init__(self) -> None:
        # TODO: Check if the presets are callable

        super().__post_init__()

    def __str__(self) -> str:
        return f"Section({self.start_frame}, {self.end_frame})"

    @property
    def as_frame_range(self) -> FrameRangeN:
        """The section as a FrameRange."""

        if self.start_frame == self.end_frame:
            return self.start_frame

        return (self.start_frame, self.end_frame)


@dataclass
class FreezeFrame(_HoldsStartEndFrames):
    """Frame ranges to freeze."""

    replacement: int
    """The frame to replace all frames in the range with."""

    def __str__(self) -> str:
        return f"FreezeFrame({self.start_frame}, {self.end_frame}, {self.replacement})"


@dataclass
class _HoldsFrameNum:
    """A dataclass to store frame number data."""

    framenum: int
    """The frame number."""

    def __post_init__(self) -> None:
        if self.framenum < 0:
            raise CustomValueError("Frame number must be greater than or equal to 0!")


@dataclass
class InterlacedFade(_HoldsFrameNum):
    """Information about interlaced fades."""

    field_difference: float
    """The differences between the two fields."""

    def __str__(self) -> str:
        return f"Frame {self.framenum}: {self.field_difference=}"

    def __post_init__(self) -> None:
        self.field_difference = abs(self.field_difference)

        super().__post_init__()


@dataclass
class OrphanField(_HoldsFrameNum):
    """Information about the orphan fields."""

    match: Match
    """The match for the given field."""

    def __str__(self) -> str:
        return f"Frame {self.framenum}: {self.match=} (orphan field)"

    def __post_init__(self) -> None:
        # TODO: Use the Match type here somehow?
        if self.match not in ('b', 'n', 'p', 'u'):
            raise InvalidMatchError(f'Invalid match value for an orphan field ({self.match=})!', OrphanField)

        super().__post_init__()

    @property
    def deinterlace_order(self) -> FieldBased:
        """The fieldorder to deinterlace in to properly deinterlace the orphan field."""

        return FieldBased.TFF if self.match in ('n', 'p') else FieldBased.BFF
