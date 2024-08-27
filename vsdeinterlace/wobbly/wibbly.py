from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

from vstools import (CustomValueError, FramePropError, FunctionUtil, Keyframes,
                     SPath, SPathLike, clip_async_render, core, get_prop,
                     get_y, normalize_ranges, vs)

from .._metadata import __version__
from .info import FrameMetric, WibblyConfig, WibblyConfigSettings
from .types import Match

__all__ = [
    "Wibbly"
]


@dataclass
class Wibbly:
    """A class representing the Wibbly metrics-gathering process."""

    clip: vs.VideoNode
    config: WibblyConfig = field(default_factory=lambda: WibblyConfig())
    trims: list[tuple[int | None, int | None]] | None = None
    metrics: list[FrameMetric] = field(default_factory=list[FrameMetric])

    # TODO: refactor
    def _get_clip(self, display: bool = False) -> vs.VideoNode:
        src, out_props = self.clip, list[str]()

        func = FunctionUtil(src, Wibbly, None, (vs.YUV, vs.GRAY), 8)

        wclip = cast(vs.VideoNode, func.work_clip)

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
                wclip = core.akarin.PropExpr(  # type:ignore
                    [wclip, even_avg, odd_avg],
                    lambda: {'WibblyFieldDiff': 'y.PlaneStatsAverage z.PlaneStatsAverage - abs'}
                )
            else:
                def _selector(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
                    f_out = f[0].copy()
                    f_out.props.WibblyFieldDiff = abs(
                        f[1].props.PlaneStatsAverage - f[2].props.PlaneStatsAverage  # type:ignore
                    )
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
        """Calculate the Wobbly metrics for the given clip with the given configuration."""

        match_chars = list[Match](['p', 'c', 'n', 'b', 'u'])

        def _to_midx(match: int) -> Match:
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
        self, video_path: SPathLike | None = None, out_path: SPathLike | None = None,
        metrics: list[FrameMetric] | None = None
    ) -> SPath:
        """
        Write a file to be used in Wobbly for further processing.

        :param video_path:      The path to the video file. This must be present for wobbly to load the video.
                                If your source was indexed using `vssource.source`, it will automatically
                                grab the path from the frameprops. If no path can be found, an error is raised.
        :param out_path:        Output location for the Wobbly file. If None, automatically outputs to
                                the `video_path` with the suffix set to ".wob".
        :param metrics:         A list of FrameMetric objects. Does not need to be passed if you used
                                `calculate_metrics`. If no metrics can be found, an error is raised.

        :return:                Path to the output Wobbly file.
        """

        video_path = self._get_video_path(video_path)
        out_path = self._get_out_path(video_path, out_path)

        if not (metrics := metrics or self.metrics):
            raise CustomValueError("You must generate metrics before you can write them to a file!", self.to_file)

        out_dict = self._build_wob_json(video_path, metrics)
        self._write_to_file(out_path, out_dict)

        return out_path

    def all_matches_to_c(self) -> None:
        """Sets all matches to 'c' matches."""

        if not self.metrics:
            raise CustomValueError(
                "You must generate metrics before you can write them to a file!",
                self.all_matches_to_c
            )

        self.metrics = [metric._replace(match='c') for metric in self.metrics]

    def _to_sections(self, scenechanges: list[int]) -> list[dict[str, Any]]:
        if not scenechanges:
            return [dict(start=0, presets=[])]

        return [dict(start=start, presets=[]) for start in scenechanges]

    def _guess_idx(self, in_file: SPath) -> str:
        """Guess the idx based on the filename. Set to (mostly) match Wibbly."""

        match in_file.suffix:
            case ".dgi": return "dgdecodenv.DGSource"
            case ".d2v": return "d2v.Source"
            case ".mp4" | ".m4v" | ".mov": return "lsmas.LibavSMASHSource"
            case _: pass

        return "bs.VideoSource"

    def _get_video_path(self, video_path: SPathLike | None) -> SPath:
        """Retrieve the video path"""

        if video_path is None:
            try:
                return SPath(get_prop(self.clip, "idx_path", bytes).decode())
            except (FramePropError, TypeError):
                raise CustomValueError("You must pass a path to the video file!", self.to_file)
        return SPath(video_path)

    def _get_out_path(self, video_path: SPath, out_path: SPathLike | None) -> SPath:
        """Determine the output path based on video path and provided out_path."""

        return SPath(out_path) if out_path else video_path.with_suffix(".wob")

    def _build_wob_json(self, video_path: SPath, metrics: list[FrameMetric]) -> dict[str, Any]:
        """Build the JSON file to output to a .wob file."""

        vfm_out = self._parse_config(self.config.vfm)
        vdec_out = self._parse_config(self.config.vdec)

        width, height = self._get_resolution()

        mics, mmetrics, matches, combed, dec_drop, dec_metrics, scenechanges, ifades = self._process_metrics(metrics)

        return {
            "wobbly version": 6,
            "project format version": 2,
            "generated with": f"vs-deinterlace v{__version__}",
            "input file": video_path.as_posix(),
            "input frame rate": [self.clip.fps.numerator, self.clip.fps.denominator],
            "input resolution": [width, height],
            "trim": [[s, e] for s, e in self.trims] if self.trims else [],
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
            "interlaced fades": ifades
        }

    def _process_metrics(
        self, metrics: list[FrameMetric]
    ) -> tuple[
        list[list[int] | None], list[list[int] | None], list[Match | None], list[int], list[int],
        list[int | None], list[int], list[dict[str, int | float]]
    ]:
        """Process all the metrics."""

        mics, mmetrics, matches, combed, dec_drop, dec_metrics, scenechanges, ifades = [], [], [], [], [], [], [], []

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

            if self.config.fade_thr is not None:
                assert isinstance(metric.field_diff, float), "Field difference must be a float!"

                if metric.field_diff >= self.config.fade_thr:
                    ifades.append({"frame": i, "field difference": metric.field_diff})

        return mics, mmetrics, matches, combed, dec_drop, dec_metrics, scenechanges, ifades

    def _parse_config(
        self, config: WibblyConfigSettings.VMFParams | WibblyConfigSettings.VDECParams | None
    ) -> dict[str, Any]:
        """Convert the config to a dictionary and handle boolean values."""

        return {
            k: float(v) if isinstance(v, bool) else v
            for k, v in config._asdict().items()
        } if config else {}

    def _get_resolution(self) -> tuple[int, int]:
        """Calculate the resolution and apply cropping."""

        width, height = self.clip.width, self.clip.height

        if self.config.crop:
            width -= self.config.crop.left + self.config.crop.right
            height -= self.config.crop.top + self.config.crop.bottom

        return width, height

    def _write_to_file(self, out_path: SPath, data: dict[str, Any]) -> None:
        """Write the JSON data to a file."""

        out_path.touch(exist_ok=True)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
