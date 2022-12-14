from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from vsexprtools import aka_expr_available
from vstools import (
    CustomValueError, FramesLengthError, FuncExceptT, check_perms, fallback, get_render_progress, replace_ranges, vs
)

__all__ = [
    'get_timecodes',
    'normalize_timecodes', 'normalize_range_timecodes',
    'separate_norm_timecodes',
    'accumulate_norm_timecodes',
    'assume_vfr',
    'generate_timecodes'
]


def get_timecodes(
    clip: vs.VideoNode, timecodes: str | Path, den: int | None = None, func: FuncExceptT | None = None
) -> list[Fraction]:
    func = func or get_timecodes

    file = Path(timecodes).resolve()

    denominator = den or (None if clip.fps_den in {0, 1} else clip.fps_den) or 1001

    version, *_timecodes = file.read_text().splitlines()

    if 'v1' in version:
        def _norm(xd: str) -> Fraction:
            return Fraction(int(denominator * float(xd)), denominator)

        assume = None

        timecodes_d = dict[tuple[int | None, int | None], Fraction]()

        for line in _timecodes:
            if line.startswith('#'):
                continue

            if line.startswith('Assume'):
                assume = _norm(_timecodes[0][7:])
                continue

            starts, ends, _fps = line.split(',')
            timecodes_d[(int(starts), int(ends) + 1)] = _norm(_fps)

        norm_timecodes = normalize_range_timecodes(timecodes_d, clip.num_frames, assume)
    elif 'v2' in version:
        timecodes_l = [float(t) for t in _timecodes if not t.startswith('#')]
        norm_timecodes = [
            Fraction(int(denominator / float(f'{round((x - y) * 100, 4) / 100000:.08f}'[:-1])), denominator)
            for x, y in zip(timecodes_l[1:], timecodes_l[:-1])
        ]
    else:
        raise CustomValueError('timecodes file not supported!', func, timecodes)

    if len(norm_timecodes) != clip.num_frames:
        raise FramesLengthError(
            func, '', 'timecodes file length mismatch with clip\'s length!',
            reason=dict(timecodes=len(norm_timecodes), clip=clip.num_frames)
        )

    return norm_timecodes


def normalize_timecodes(timecodes: list[Fraction]) -> dict[tuple[int, int], Fraction]:
    timecodes_ranges = dict[tuple[int, int], Fraction]()

    last_i = len(timecodes) - 1
    last_fps = (0, timecodes[0])

    for i, fps in enumerate(timecodes[1:], 1):
        start, lfps = last_fps

        if fps != lfps:
            timecodes_ranges[start, i - 1] = lfps
            last_fps = (i, fps)
        elif i == last_i:
            timecodes_ranges[start, i + 1] = fps

    return timecodes_ranges


def normalize_range_timecodes(
    timecodes: dict[tuple[int | None, int | None], Fraction], end: int, assume: Fraction | None = None
) -> list[Fraction]:
    norm_timecodes = [assume] * end if assume else list[Fraction]()

    for (startn, endn), fps in timecodes.items():
        start = max(fallback(startn, 0), 0)
        end = fallback(endn, end)

        if end > len(norm_timecodes):
            norm_timecodes += [fps] * (end - len(norm_timecodes))

        norm_timecodes[start:end + 1] = [fps] * (end - start)

    return norm_timecodes


def separate_norm_timecodes(timecodes: dict[tuple[int, int], Fraction]) -> tuple[
    Fraction, dict[tuple[int, int], Fraction]
]:
    times_count = {k: 0 for k in timecodes.values()}

    for v in timecodes.values():
        times_count[v] += 1

    major_count = max(times_count.values())
    major_time = next(t for t, c in times_count.items() if c == major_count)
    minor_fps = {r: v for r, v in timecodes.items() if v != major_time}

    return major_time, minor_fps


def accumulate_norm_timecodes(timecodes: dict[tuple[int, int], Fraction]) -> tuple[
    Fraction, dict[Fraction, list[tuple[int, int]]]
]:
    major_time, minor_fps = separate_norm_timecodes(timecodes)

    acc_ranges = dict[Fraction, list[tuple[int, int]]]()

    for k, v in minor_fps.items():
        if v not in acc_ranges:
            acc_ranges[v] = []

        acc_ranges[v].append(k)

    return major_time, acc_ranges


def assume_vfr(
    clip: vs.VideoNode, timecodes: str | Path, den: int | None = None, func: FuncExceptT | None = None
) -> vs.VideoNode:
    func = func or assume_vfr

    timecodes_ranges = normalize_timecodes(get_timecodes(clip, timecodes, den, func))

    major_time, minor_fps = accumulate_norm_timecodes(timecodes_ranges)

    assumed_clip = clip.std.AssumeFPS(None, major_time.numerator, major_time.denominator)

    for other_fps, fps_ranges in minor_fps.items():
        assumed_clip = replace_ranges(
            assumed_clip, clip.std.AssumeFPS(None, other_fps.numerator, other_fps.denominator),
            fps_ranges, False, False, False  # type: ignore
        )

    return assumed_clip


def generate_timecodes(clip: vs.VideoNode, out: str | Path, format: int = 2, func: FuncExceptT | None = None) -> None:
    func = func or generate_timecodes

    out_path = Path(out).resolve()

    check_perms(out_path, 'w+', func=func)

    prop_clip = clip

    p = get_render_progress()
    t = p.add_task(f'Getting v{format} timecodes from clip...', total=clip.num_frames)

    if aka_expr_available:
        prop_clip = prop_clip.std.BlankClip(2, 1, vs.GRAY16, keep=True).std.CopyFrameProps(clip)
        prop_clip = prop_clip.akarin.Expr('X 1 = x._DurationNum x._DurationDen ?')

        def _get_fraction(f: vs.VideoFrame) -> Fraction:
            p.update(t, advance=1)
            return Fraction((m := f[0])[0, 0], m[0, 1], _normalize=False)  # type: ignore
    else:
        def _get_fraction(f: vs.VideoFrame) -> Fraction:
            p.update(t, advance=1)
            return Fraction(f.props._DurationNum, f.props._DurationDen, _normalize=False)  # type: ignore

    with p:
        timecodes = [_get_fraction(f) for f in prop_clip.frames(close=True)]

    out_text = [
        f'# timecode format v{format}'
    ]

    if format == 1:
        timecodes_ranges = normalize_timecodes(timecodes)

        major_time, minor_fps = separate_norm_timecodes(timecodes_ranges)

        out_text.append(f'Assume {round(float(major_time), 12)}')

        out_text.extend([
            ','.join(map(str, [*frange, round(float(fps), 12)]))
            for frange, fps in minor_fps.items()
        ])
    elif format == 2:
        acc = 0.0
        for time in timecodes:
            s_acc = str(round(acc / 100, 12) * 100)
            l, i = len(s_acc), s_acc.index('.')
            d = l - i - 1
            if d < 6:
                s_acc += '0' * (6 - d)
            else:
                s_acc = s_acc[:i + 7]

            out_text.append(s_acc)
            acc += (time.denominator * 100) / (time.numerator * 100) * 1000
        out_text.append(str(acc))
    else:
        raise CustomValueError('timecodes format not supported!', func, format)

    out_path.unlink(True)
    out_path.touch()
    out_path.write_text('\n'.join(out_text + ['']))
