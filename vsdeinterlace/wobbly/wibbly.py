import inspect
import json
from typing import Any, Literal, cast

from vskernels import Catrom
from vssource import source
from vstools import (CustomKeyError, CustomNotImplementedError, CustomValueError, DependencyNotFoundError, FieldBased,
                     FieldBasedT, FrameRangesN, FuncExceptT, FunctionUtil, SceneChangeMode, SPath, SPathLike,
                     UnsupportedFieldBasedError, clip_async_render, core, depth, get_prop, get_y, inject_self,
                     merge_clip_props, mod4, vs)

from ..combing import fix_telecined_fades


class Wibbly:
    """A class representing the Wibbly metric collection process."""

    pref_clip: vs.VideoNode
    """Prefiltered clip."""

    out_file: SPath
    """The location of the processed wob file."""

    def __init__(self, clip: vs.VideoNode, out_path: SPathLike | None = None) -> None:
        # TODO: figure out what I wanna put in here
        if out_path is None:
            out_path = self._get_out_spath()

        self.pref_clip = depth(clip, 8)
        self.out_file = SPath(out_path)

    @inject_self
    def process(
        self,
        clip: vs.VideoNode | SPathLike | None = None,
        out_path: SPathLike | None = None,
        trims: FrameRangesN = [],
        tff: bool | FieldBasedT | None = None,
        chroma: bool = True,
        fades_thr: float | Literal[False] = 0.4 / 255,
        colors: float | list[float] = 0,
        field_match: bool = True,
        decimate: bool = True,
        scenechanges: bool = True,
        scenechange_mode: SceneChangeMode = SceneChangeMode.WWXD,
        dmetrics: bool = True,
        dmetrics_nt: int = 10,
        func: FuncExceptT | None = None,
        **kwargs: Any
    ) -> SPath:
        """
        Initiate the metric gathering process and output it to a file.

        The output file is intended to be used in conjunction
        with `Wobbly <https://github.com/Setsugennoao/Wobbly/tree/master>`_,
        where you can apply further IVTC refining.

        DO NOT trim your clip before passing it here! That will throw Wobbly off!

        Unlike Wibbly, this does not handle pre-cropping. This is the user's responsibility.
        When applying prefiltering, be mindful not to filter in a way that makes it difficult to gather metrics.
        This means non-field-aware filtering must be kept to a minimum!

        :param clip:                Clip to gather metrics from.
        :param out_path:            Path to write the Wobbly metrics file to.
        :param trims:               Trims for the clip. This MUST be done through this class!
        :param tff:                 Top-field-first. `False` sets it to Bottom-Field-First.
                                    If `None`, get the field order from the _FieldBased prop.
        :param chroma:              Whether to take chroma into account during processing as well.
                                    This will be slower than running it on just the luma!
        :param fade_thr:            Threshold for when a fade is considered to be an interlaced fade.
                                    If set to `False`, do not check for this value.
        :param colors:              Color offset for the plane average for fix_telecined_fades.
        :param field_match:         Enable fieldmatching metrics gathering.
        :param decimate:            Enable decimation metrics gathering.
        :param scenechanges:        Enable Scenechange metrics gathering.
        :param scenechange_mode:    The scenechange mode to apply to the clip.
        :param dmetrics:            Enable dmetrics metrics gathering.
        :param dmetrics_nt:         The dmetrics `nt` parameter. @@@What does this do?@@@
        :param kwargs:              Keyword arguments to pass on to VFM.

        :return:                    Path to the output .wob file.
        """
        func = func or self.process

        if clip is None:
            clip = self.pref_clip

        if out_path is None:
            out_path = self._get_out_spath()

        if not any((field_match, decimate, scenechanges, dmetrics)):
            raise CustomValueError("You must enable at least one option!", func)

        if (out_file := SPath(out_path)).exists():
            raise CustomValueError(f"The file \"{out_file.absolute()}\" already exists!", func)

        if isinstance(clip, vs.VideoNode):
            clip = cast(vs.VideoNode, clip)

            try:
                wclip = clip
                clip = SPath(get_prop(clip, "idx_filepath", bytes).decode('utf-8'))
            except KeyError:
                raise CustomKeyError("You must pass a filepath to `clip` or index using `vssource!", func)
        else:
            clip = SPath(str(clip))
            wclip = source(clip, 8)

        self._check_plugins_installed(field_match or decimate, dmetrics, scenechanges, scenechange_mode, func)

        vfm_kwargs = dict(micmatch=0, mode=0, micout=True) | kwargs

        if trims:
            if not isinstance(trims, list):
                trims = [trims]  # type:ignore[list-item]

            trimmed_wclip = wclip.std.BlankClip(length=1)

            for start, end in trims:
                trimmed_wclip = trimmed_wclip + wclip[start:end + 1]  # TODO: iirc this can be made faster

            wclip = trimmed_wclip[1:]

        f = FunctionUtil(wclip, func, None if chroma else 0, (vs.GRAY, vs.YUV), 8)

        pclip = FieldBased.ensure_presence(f.work_clip, tff, func)

        if not (p_tff := FieldBased.from_video(pclip)).is_inter:
            raise UnsupportedFieldBasedError("The given clip is progressive! Please set `tff`", func)

        pclip, props = self._prepare_clip(
            pclip, p_tff, fades_thr, colors,
            field_match, decimate, scenechanges,
            scenechange_mode, dmetrics, dmetrics_nt,
            **vfm_kwargs
        )

        metrics = self._render_metrics(pclip, props, fades_thr, func)

        self._write_wob_file(wclip, clip.as_posix(), out_file, trims, metrics)
        self.out_file = out_file

        return self.out_file

    def _get_out_spath(self) -> SPath:
        return SPath(inspect.stack()[3].filename + ".wob")

    def _prepare_clip(
        self, clip: vs.VideoNode, tff: FieldBased,
        fades_thr: float | Literal[False], colors: float | list[float],
        field_match: bool, decimate: bool, scenechanges: bool, scenechange_mode: SceneChangeMode,
        dmetrics: bool, dmetrics_nt: int,
        **kwargs: Any
    ) -> tuple[vs.VideoNode, list[str]]:
        """Run metrics plugins over the work clip and return a list of props to iterate over later."""
        assert clip.format

        fmt = clip.format

        props: list[str] = []

        if field_match and dmetrics:
            clip = Catrom.resample(clip, vs.YUV420P8)  # dmetrics requires a YUV420P8 clip
            clip = clip.dmetrics.DMetrics(nt=dmetrics_nt, y0=kwargs.get("y0", 16), y1=kwargs.get("y1", 16))

            if fmt.color_family is vs.GRAY:
                clip = get_y(clip)

            props += ["MMetrics", "VMetrics"]

        if field_match:
            clip = clip.vivtc.VFM(tff.is_tff, field=not tff.is_tff, **kwargs)
            props += ["VFMMatch", "VFMMics", "VFMSceneChange", "_Combed"]

        if fades_thr:
            # wobbly currently only supports Y.
            planes = 0

            clip = fix_telecined_fades(clip, tff, colors, planes, func=self.process)
            # props += list(f"FtfDiff{i}" for i in normalize_planes(clip))
            props += ["FtfDiff0"]

        if decimate:
            clip = clip.vivtc.VDecimate(dryrun=True)
            props += ["VDecimateDrop", "VDecimateMaxBlockDiff", "VDecimateTotalDiff"]

        if scenechanges:
            clip = scenechange_mode.prepare_clip(clip, max(mod4(clip.height // 4), 120), True)

            if scenechange_mode.is_WWXD:
                props += ["Scenechange"]
            if scenechange_mode.is_SCXVID:
                props += ["_SceneChangePrev"]

        prop_clip = core.std.BlankClip(None, 1, 1, vs.GRAY8, clip.num_frames)
        prop_clip = merge_clip_props(prop_clip, clip)

        return prop_clip, props

    def _check_plugins_installed(
        self,
        vfm: bool,
        dmetrics: bool,
        scenechanges: bool,
        scenechange_mode: SceneChangeMode,
        func: FuncExceptT | None = None
    ) -> None:
        """Check whether all the necessary plugins are installed."""
        plugins = ["akarin"]

        if vfm:
            plugins += ["vivtc"]

        if dmetrics:
            plugins += ["dmetrics"]

        if scenechanges and scenechange_mode.is_WWXD:
            plugins += ["wwxd"]
        if scenechanges and scenechange_mode.is_SCXVID:
            plugins += ["scxvid"]

        if func is None:
            func = inspect.stack()[1].function

        missing_plugins: list[str] = []

        for plugin in plugins:
            if not hasattr(vs.core, plugin):
                missing_plugins += [plugin]

        if not missing_plugins:
            return

        if len(missing_plugins) > 1:
            raise DependencyNotFoundError(func, ", ".join(missing_plugins), "Missing dependencies: '{package}'!")

        raise DependencyNotFoundError(func, missing_plugins[0])

    def _render_metrics(
        self, clip: vs.VideoNode, props: list[str],
        ftf_diff_thr: float, func: FuncExceptT | None = None
    ) -> dict[str, Any]:
        """Render over the clip and gather metrics."""
        func = func or self.process

        matches: list[str] = []
        matches_map = ["p", "c", "n", "b", "u"]

        scenechanges: list[int] = []

        combs: list[int] = []

        mics: list[list[int]] = []
        mmetrics: list[list[int]] = []
        vmetrics: list[list[int]] = []

        vdec_max_block: list[int] = []
        vdec_drop: list[int] = []

        ftf_diff0: list[tuple[int, float]] = []
        # ftf_diff1: list[tuple[int, float]] = []
        # ftf_diff2: list[tuple[int, float]] = []

        def _cb(n: int, f: vs.VideoFrame) -> None:
            for p in props:
                match p:
                    case "VFMMatch": matches.append(matches_map[get_prop(f, p, int, None, 0, func=func)])
                    case "_Combed":
                        if get_prop(f, p, int, None, 0, func=func):
                            combs.append(n)
                    case "Scenechange":
                        if get_prop(f, p, int, func=func):
                            scenechanges.append(n)
                    case "_PrevSceneChange":
                        # TODO: I can't get clips to actually preview with SCXVID, so can't test them. plsfix!
                        raise CustomNotImplementedError(None, func, reason="dev skill issue")
                    case "VFMMics": mics.append(get_prop(f, p, list, list[int], func=func))
                    case "MMetrics": mmetrics.append(get_prop(f, p, list, list[int], func=func))
                    case "VMetrics": vmetrics.append(get_prop(f, p, list, list[int], func=func))
                    case "VDecimateMaxBlockDiff": vdec_max_block.append(get_prop(f, p, int, func=func))
                    case "VDecimateDrop": vdec_drop.append(get_prop(f, p, int, func=func))
                    # TODO: make this way less bad
                    case "FtfDiff0":
                        if not ftf_diff_thr:
                            continue

                        field_diff = abs(
                            get_prop(f, "fbAvg0", float | int, float)  # type:ignore[call-overload]
                            - get_prop(f, "ftAvg0", float | int, float)  # type:ignore[call-overload]
                        )

                        if field_diff > ftf_diff_thr:
                            ftf_diff0.append((n, field_diff))
                    # case "FtfDiff1":
                    #     field_diff = abs(
                    #         get_prop(f, "fbAvg1", float | int, float)  # type:ignore[call-overload]
                    #         - get_prop(f, "ftAvg1", float | int, float)  # type:ignore[call-overload]
                    #     )

                    #     if field_diff > ftf_diff_thr:
                    #         ftf_diff1.append((n, field_diff))
                    # case "FtfDiff2":
                    #     field_diff = abs(
                    #         get_prop(f, "fbAvg2", float | int, float)  # type:ignore[call-overload]
                    #         - get_prop(f, "ftAvg2", float | int, float)  # type:ignore[call-overload]
                    #     )

                    #     if field_diff > ftf_diff_thr:
                    #         ftf_diff2.append((n, field_diff))
                    case _: continue

        clip_async_render(clip, progress="Gathering metrics...", callback=_cb)

        return {
            "matches": matches,
            "scenechanges": scenechanges,
            "combed_frames": combs,
            "mics": mics,
            "mmetrics": mmetrics,
            "vmetrics": vmetrics,
            "vdecimate_max_block_diff": vdec_max_block,
            "vdecimate_drop": vdec_drop,
            "ftf_diff_0": ftf_diff0,
            # "ftf_diff_1": ftf_diff1,
            # "ftf_diff_2": ftf_diff2,
        }

    def _write_wob_file(
        self, work_clip: vs.VideoNode,
        in_file: str, out_file: SPath,
        trims: FrameRangesN, metrics: dict[str, Any]
    ) -> None:
        from .._metadata import __version__

        out_dict = {
            "wobbly version": 6,
            "project format version": 2,
            "generated with": f"vs-deinterlace v{__version__}",  # TODO: get package name directly
            "input file": in_file,
            "input frame rate": [work_clip.fps.numerator, work_clip.fps.denominator],
            "input resolution": [work_clip.width, work_clip.height],
            "trim": [] if not trims else [[s, e] for s, e in trims],  # type:ignore[misc]
            # TODO: VFM params
            # TODO: video_heuristics params
            "mics": metrics.get("mics", []),
            "mmetrics": metrics.get("mmetrics", []),
            "matches": metrics.get("matches", ""),
            "original matches": metrics.get("matches", ""),
            "combed frames": metrics.get("combed_frames", []),
            "decimated frames": metrics.get("vdecimate_drop", []),
            "decimate metrics": metrics.get("vdecimate_max_block_diff", []),
            "sections": self._to_sections(metrics.get("scenechanges", [])),
            "source filter": self._guess_idx(SPath(in_file)),
            "interlaced fades": self._get_fades(metrics.get("ftf_diff_0", []))
        }

        out_file.touch(exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=4)

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
            case "mp4" | "m4v" | "mov": return "lsmas.LibavSMASHSource"
            case _: pass

        return "lsmas.LWLibavSource"

    def _get_fades(self, fades: list[tuple[int, float]]) -> list[dict[str, float | int]]:
        out: list[dict[str, float | int]] = []

        for frame, fade in fades:
            out += [{"frame": frame, "field difference": fade}]

        return out
