"""Generate WebVTT caption files (rolling two-line, Google-style) from
pipeline_data without needing the FastAPI service to be running.

This mirrors the logic in ``api/src/routers/stitch.py`` (functions
``_segments_to_vtt``, ``_format_vtt_time``, ``_compute_speech_offset``)
so that a pipeline run can be completed without spinning up the
container stack.  Useful when:

  * you only need the captions side-car file for an existing dubbed mp4,
  * the API session has gone away (e.g. HPC Slurm job ended), or
  * you want to regenerate captions after editing translations on disk.

Usage
-----
From the project root::

    python scripts/generate_vtt.py

By default it scans ``pipeline_data/api/`` and produces, for every
translation found:

  * ``pipeline_data/api/dubbed_captions/<title>.vtt`` — translated, with
    YouTube-derived speech offset applied (matches the project default).
  * ``pipeline_data/api/dubbed_captions/<title>.original.vtt`` — original
    English captions in the same rolling format (drop-in side-car for the
    source mp4).

Override the data root with ``--data-dir``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# ── identical to api/src/routers/stitch.py ──────────────────────────────────


def _format_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _segments_to_vtt(segments: list[dict]) -> str:
    """Convert transcript segments to rolling two-line WebVTT.

    Each cue shows the current line on top and the previous line on the
    bottom — same format the FastAPI service serves at ``GET
    /api/captions/{video_id}``.
    """
    segs = [s for s in segments if s.get("text", "").strip()]
    if not segs:
        return "WEBVTT\n"

    lines = ["WEBVTT", ""]
    prev_text: str | None = None
    for i, seg in enumerate(segs, 1):
        start = _format_vtt_time(seg["start"])
        end = _format_vtt_time(seg["end"])
        text = seg.get("text", "").strip()
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        if prev_text:
            lines.append(f"{text}\n{prev_text}")
        else:
            lines.append(text)
        lines.append("")
        prev_text = text
    return "\n".join(lines)


def _youtube_captions_to_segments(caption_path: pathlib.Path) -> list[dict]:
    """Parse YouTube line-delimited JSON captions into segment dicts."""
    out: list[dict] = []
    for line in caption_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        seg = json.loads(line)
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)
        duration = seg.get("duration", 0.0)
        if text and duration > 0:
            out.append({"start": start, "end": start + duration, "text": text})
    return out


def _compute_speech_offset(
    yt_path: pathlib.Path, whisper_path: pathlib.Path,
) -> float:
    """Seconds to add to translated segments so subtitles begin on speech."""
    if not yt_path.exists() or not whisper_path.exists():
        return 0.0
    first_line = yt_path.read_text(encoding="utf-8").split("\n", 1)[0].strip()
    if not first_line:
        return 0.0
    yt_start = json.loads(first_line).get("start", 0.0)
    whisper_data = json.loads(whisper_path.read_text(encoding="utf-8"))
    segments = whisper_data.get("segments", [])
    whisper_start = segments[0]["start"] if segments else 0.0
    return yt_start - whisper_start


# ── driver ──────────────────────────────────────────────────────────────────


def generate_for_title(
    title: str,
    translations_dir: pathlib.Path,
    youtube_captions_dir: pathlib.Path,
    transcriptions_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> tuple[pathlib.Path | None, pathlib.Path | None]:
    """Produce the dubbed and original VTT side-cars for *title*.

    Returns the (dubbed, original) paths actually written, or ``None`` for
    any side-car whose source data was missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written_dubbed: pathlib.Path | None = None
    written_original: pathlib.Path | None = None

    translation_json = translations_dir / f"{title}.json"
    if translation_json.exists():
        data = json.loads(translation_json.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
        offset = _compute_speech_offset(
            youtube_captions_dir / f"{title}.txt",
            transcriptions_dir / f"{title}.json",
        )
        if offset > 0:
            segments = [
                {**s, "start": s["start"] + offset, "end": s["end"] + offset}
                for s in segments
            ]
        out = out_dir / f"{title}.vtt"
        out.write_text(_segments_to_vtt(segments), encoding="utf-8")
        written_dubbed = out

    yt_caption_path = youtube_captions_dir / f"{title}.txt"
    if yt_caption_path.exists():
        original_segments = _youtube_captions_to_segments(yt_caption_path)
        out = out_dir / f"{title}.original.vtt"
        out.write_text(_segments_to_vtt(original_segments), encoding="utf-8")
        written_original = out

    return written_dubbed, written_original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("pipeline_data") / "api",
        help="Path to the per-API pipeline_data directory "
             "(default: pipeline_data/api).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Generate captions only for this exact title (no extension). "
             "If omitted, all translations under translations/argos/ are "
             "processed.",
    )
    args = parser.parse_args()

    data_dir: pathlib.Path = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"error: data dir not found: {data_dir}", file=sys.stderr)
        return 1

    translations_dir = data_dir / "translations" / "argos"
    youtube_captions_dir = data_dir / "youtube_captions"
    transcriptions_dir = data_dir / "transcriptions" / "whisper"
    out_dir = data_dir / "dubbed_captions"

    if args.title:
        titles = [args.title]
    else:
        if not translations_dir.exists():
            print(
                f"error: no translations directory at {translations_dir}",
                file=sys.stderr,
            )
            return 1
        titles = sorted(p.stem for p in translations_dir.glob("*.json"))

    if not titles:
        print("nothing to do (no translations found)", file=sys.stderr)
        return 1

    for title in titles:
        dubbed, original = generate_for_title(
            title,
            translations_dir,
            youtube_captions_dir,
            transcriptions_dir,
            out_dir,
        )
        print(f"[{title}]")
        print(f"  dubbed:   {dubbed if dubbed else '(skipped — no translation JSON)'}")
        print(f"  original: {original if original else '(skipped — no YouTube captions)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
