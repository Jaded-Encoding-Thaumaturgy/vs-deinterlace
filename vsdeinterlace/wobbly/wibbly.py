from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from vstools import (CustomValueError, FunctionUtil, Keyframes,
                     SceneChangeMode, SPath, SPathLike, clip_async_render,
                     core, get_prop, get_y, normalize_ranges, vs)

from .types import Types

__all__ = [
    "WibblyConfig", "Wibbly"
]


class Config:
    class Crop(NamedTuple):
        left: int = 0
        top: int = 0
        right: int = 0
        bottom: int = 0
        early: bool = True

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
class WibblyConfig(Config):
    crop: Config.Crop | None = Config.Crop()
    dmetrics: Config.DMetrics | None = Config.DMetrics()
    vfm: Config.VMFParams | None = Config.VMFParams()
    vdec: Config.VDECParams | None = Config.VDECParams()
    fade_thr: float | None = 0.4 / 255
    sc_mode: SceneChangeMode | None = SceneChangeMode.WWXD


class FrameMetric(NamedTuple):
    is_combed: bool
    is_keyframe: bool
    match: Types.Match | None
    vfm_mics: list[int] | None
    mm_dmet: list[int] | None
    vm_dmet: list[int] | None
    dec_met: int | None
    dec_drop: bool
    field_diff: float | None


@dataclass
class Wibbly:
    clip: vs.VideoNode
    config: WibblyConfig = field(default_factory=lambda: WibblyConfig())
    trims: list[tuple[int | None, int | None]] | None = None
    metrics: list[FrameMetric] = field(default_factory=list[FrameMetric])

    def _get_clip(self, display: bool = False) -> vs.VideoNode:
        src, out_props = self.clip, list[str]()

        func = FunctionUtil(src, Wibbly, None, (vs.YUV, vs.GRAY), 8)

        wclip = func.work_clip

        if not display:
            norm_trims = normalize_ranges(func.work_clip, self.trims)

            if len(norm_trims) == 1:
                if (trim := norm_trims.pop()) != (0, wclip.num_frames - 1):
                    wclip = wclip[trim[0]:trim[1] + 1]
            else:
                wclip = core.std.Splice([
                    wclip[start:end + 1] for start, end in norm_trims
                ])

        if (crop := self.config.crop) is not None:
            wclip = wclip.std.CropRel(crop.left, crop.right, crop.top, crop.bottom)

        if (vfm := self.config.vfm) is not None:
            if (dmet := self.config.dmetrics):
                wclip = wclip.dmetrics.DMetrics(vfm.order, vfm.chroma, dmet.nt, vfm.y0, vfm.y1)
                out_props.extend(['MMetrics', 'VMetrics'])

            wclip = wclip.vivtc.VFM(
                vfm.order, int(not vfm.order), 0, vfm.mchroma, vfm.cthresh, vfm.mi, vfm.chroma,
                vfm.blockx, vfm.blocky, vfm.y0, vfm.y1, vfm.scthresh, vfm.micmatch, True, None
            )
            out_props.extend(['VFMMatch', 'VFMMics', 'VFMSceneChange', '_Combed'])

        if self.config.fade_thr is not None:
            separated = get_y(wclip).std.SeparateFields(True)

            even_avg = separated[::2].std.PlaneStats()
            odd_avg = separated[1::2].std.PlaneStats()

            if hasattr(core, 'akarin'):
                wclip = core.akarin.PropExpr(
                    [wclip, even_avg, odd_avg],
                    lambda: {'WibblyFieldDiff': 'y.PlaneStatsAverage z.PlaneStatsAverage - abs'}
                )
            else:
                def _selector(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
                    f_out = f[0].copy()
                    f_out.props.WibblyFieldDiff = abs(f[1].props.PlaneStatsAverage - f[2].props.PlaneStatsAverage)
                    return f_out

                wclip = core.std.ModifyFrame(wclip.std.BlankClip(), [wclip, even_avg, odd_avg], _selector)

            out_props.extend(['WibblyFieldDiff'])

        if (vdec := self.config.vdec) is not None:
            wclip = wclip.vivtc.VDecimate(
                5, vdec.chroma, vdec.dupthresh, vdec.scthresh,
                vdec.blockx, vdec.blocky, None, None, True
            )
            out_props.extend(['VDecimateDrop', 'VDecimateTotalDiff', 'VDecimateMaxBlockDiff'])

        if not display and self.config.sc_mode is not None:
            wclip = Keyframes.to_clip(wclip, mode=self.config.sc_mode, prop_key='WobblySceneChange')

        if display and out_props:
            wclip = wclip.text.FrameProps(out_props)

        return func.return_clip(wclip)

    @property
    def display_clip(self) -> vs.VideoNode:
        return self._get_clip(True)

    @property
    def output_clip(self) -> vs.VideoNode:
        return self._get_clip(False)

    def calculate_metrics(self) -> list[FrameMetric]:
        match_chars = list[Types.Match](['p', 'c', 'n', 'b', 'u'])

        def _to_midx(match: int) -> Types.Match:
            return match_chars[match]

        def _callback(n: int, f: vs.VideoFrame) -> FrameMetric:
            props = f.props

            match = get_prop(props, 'VFMMatch', int, _to_midx, None)

            is_combed = get_prop(props, '_Combed', int, bool, False)

            vfm_mics = get_prop(props, 'VFMMics', list, None, None)

            mm_dmet = get_prop(props, 'MMetrics', list, None, None)
            vm_dmet = get_prop(props, 'VMetrics', list, None, None)

            is_keyframe = get_prop(props, 'WobblySceneChange', int, bool, False)

            dec_met = get_prop(props, 'VDecimateMaxBlockDiff', int, None, None)

            dec_drop = get_prop(props, 'VDecimateDrop', int, bool, False)

            field_diff = get_prop(props, 'WibblyFieldDiff', float, None, 0)

            return FrameMetric(
                is_combed, is_keyframe, match, vfm_mics, mm_dmet, vm_dmet, dec_met, dec_drop, field_diff
            )

        self.metrics = clip_async_render(self.output_clip, None, 'Analyzing clip...', _callback)

        return self.metrics

    def to_file(
        self,
        video_path: SPathLike | None = None,
        out_path: SPathLike | None = None,
        metrics: list[FrameMetric] | None = None
    ) -> SPath:
        """
        Write a wob file to be used in Wobbly.
        """
        from .._metadata import __version__

        if video_path is None:
            try:
                video_path = get_prop(self.clip, "idx_path", bytes).decode()
            except:
                raise CustomValueError("You must pass a path to the video file!", self.to_file)

        video_path = SPath(video_path)

        if out_path is None:
            out_path = video_path.with_suffix(".wob")

        out_path = SPath(out_path)

        metrics = metrics or self.metrics

        if not metrics:
            raise CustomValueError("You must generate metrics before you can write them to a file!", self.to_file)

        mics = []
        mmetrics = []
        matches = []
        combed = []
        dec_drop = []
        dec_metrics = []
        scenechanges = []
        i_fades = []

        for i, metric in enumerate(metrics):
            mics.append(metric.vfm_mics)
            mmetrics.append(metric.mm_dmet)
            matches.append(metric.match)

            if metric.is_combed:
                combed.append(i)

            if metric.dec_drop:
                dec_drop.append(i)

            dec_metrics.append(metric.dec_met)

            if metric.is_keyframe:
                scenechanges.append(i)

            if metric.field_diff >= self.config.fade_thr:
                i_fades.append({"frame": i, "field difference": metric.field_diff})

        width = self.clip.width
        height = self.clip.height

        if self.config.crop:
            width -= self.config.crop.left
            width -= self.config.crop.right

            height -= self.config.crop.top
            height -= self.config.crop.bottom

        vfm_out: dict[str, Any] = {}
        vdec_out: dict[str, Any] = {}

        for k, v in self.config.vfm._asdict().items():
            vfm_out |= {k: float(v) if isinstance(v, bool) else v}

        for k, v in self.config.vdec._asdict().items():
            vdec_out |= {k: float(v) if isinstance(v, bool) else v}

        out_dict = {
            "wobbly version": 6,
            "project format version": 2,
            "generated with": f"vs-deinterlace v{__version__}",  # TODO: get package name directly
            "input file": video_path.as_posix(),
            "input frame rate": [self.clip.fps.numerator, self.clip.fps.denominator],
            "input resolution": [width, height],
            "trim": [] if not self.trims else [[s, e] for s, e in self.trims],  # type:ignore[misc]
            "vfm parameters": vfm_out,
            "vdecimate parameters": vdec_out,
            "mics": mics,
            "mmetrics": mmetrics,
            "matches": matches,
            "original matches": matches,
            "combed frames": combed,
            "decimated frames": dec_drop,
            "decimate metrics": dec_metrics,
            "sections": self._to_sections(scenechanges),
            "source filter": self._guess_idx(video_path),
            "interlaced fades": i_fades
        }

        out_path.touch(exist_ok=True)

        with open("test.wob", "w", encoding="utf-8") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=4)

        return out_path

    def _to_sections(self, scenechanges: list[int]) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []

        if not scenechanges:
            return [dict(start=0, presets=[])]

        for start in scenechanges:
            sections += [dict(start=start, presets=[])]

        return sections

    def _guess_idx(self, in_file: SPath) -> str:
        """Guess the idx based on the filename. Set to (mostly) match Wibbly."""
        match in_file.suffix:
            case ".dgi": return "dgdecodenv.DGSource"
            case ".d2v": return "d2v.Source"
            case ".mp4" | ".m4v" | ".mov": return "lsmas.LibavSMASHSource"
            case _: pass

        return "lsmas.LWLibavSource"
