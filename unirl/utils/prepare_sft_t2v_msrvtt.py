"""Build WAN T2V SFT manifests from a local MSR-VTT snapshot.

Expected input layout (the Hugging Face ``friedrichor/MSR-VTT`` release uses
these names after extracting ``MSRVTT_Videos.zip``)::

    <data-root>/
      msrvtt_train_7k.json
      msrvtt_test_1k.json
      MSRVTT_Videos/
        video/video0.mp4
        video/video7020.mp4
        ...

The output is the supervised target-video format consumed by
``VideoDiffusionSupervisedTrackBuilder``::

    {"prompt": <caption>,
     "media": [{"modality": "video", "role": "target", "uri": <mp4>}]}

Example:

  hf download friedrichor/MSR-VTT \
      msrvtt_train_7k.json msrvtt_test_1k.json MSRVTT_Videos.zip \
      --repo-type dataset \
      --local-dir /path/to/MSR-VTT
  unzip /path/to/MSR-VTT/MSRVTT_Videos.zip -d /path/to/MSR-VTT/MSRVTT_Videos
  python -m unirl.utils.prepare_sft_t2v_msrvtt \
      --data-root /path/to/MSR-VTT --out-dir data/sft_msrvtt --max-train-samples 512
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List


def _load_annotations(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(rows).__name__}.")
    return rows


def _find_video(videos_dir: str, video_name: str) -> str:
    candidates = (
        os.path.join(videos_dir, "video", video_name),
        os.path.join(videos_dir, video_name),
        os.path.join(videos_dir, "TrainValVideo", video_name),
        os.path.join(videos_dir, "TestVideo", video_name),
    )
    return next((path for path in candidates if os.path.isfile(path)), candidates[0])


def _convert_rows(
    annotations: Iterable[Dict[str, Any]],
    *,
    videos_dir: str,
    max_samples: int,
    captions_per_video: int,
    seed: int,
    keep_missing: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for item in annotations:
        video_id = str(item.get("video_id") or os.path.splitext(str(item.get("video", "")))[0]).strip()
        video_name = str(item.get("video") or f"{video_id}.mp4").strip()
        raw_captions = item.get("caption", [])
        if isinstance(raw_captions, str):
            raw_captions = [raw_captions]
        captions = [str(c).strip() for c in raw_captions if str(c).strip()]
        if not video_id or not video_name or not captions:
            continue
        video_path = os.path.abspath(_find_video(videos_dir, video_name))
        if not keep_missing and not os.path.isfile(video_path):
            continue
        rng.shuffle(captions)
        for caption_idx, caption in enumerate(captions[:captions_per_video]):
            rows.append(
                {
                    "sample_id": f"msrvtt:{video_id}:{caption_idx}",
                    "prompt": caption,
                    "media": [{"modality": "video", "role": "target", "uri": video_path}],
                    "metadata": {"source": "MSR-VTT", "video_id": video_id},
                }
            )
            if max_samples > 0 and len(rows) >= max_samples:
                return rows
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):6d} rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="MSR-VTT annotations + extracted videos root")
    parser.add_argument("--videos-dir", default=None, help="video directory (default: <data-root>/MSRVTT_Videos)")
    parser.add_argument("--train-annotations", default="msrvtt_train_7k.json")
    parser.add_argument("--val-annotations", default="msrvtt_test_1k.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-train-samples", type=int, default=512, help="0 = all")
    parser.add_argument("--max-val-samples", type=int, default=64, help="0 = all")
    parser.add_argument("--captions-per-video", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-missing", action="store_true", help="keep rows before videos are downloaded")
    args = parser.parse_args()

    if args.captions_per_video < 1:
        raise SystemExit("--captions-per-video must be >= 1")
    videos_dir = args.videos_dir or os.path.join(args.data_root, "MSRVTT_Videos")
    train_annotations = os.path.join(args.data_root, args.train_annotations)
    val_annotations = os.path.join(args.data_root, args.val_annotations)

    train_rows = _convert_rows(
        _load_annotations(train_annotations),
        videos_dir=videos_dir,
        max_samples=args.max_train_samples,
        captions_per_video=args.captions_per_video,
        seed=args.seed,
        keep_missing=args.keep_missing,
    )
    val_rows = _convert_rows(
        _load_annotations(val_annotations),
        videos_dir=videos_dir,
        max_samples=args.max_val_samples,
        captions_per_video=args.captions_per_video,
        seed=args.seed + 1,
        keep_missing=args.keep_missing,
    )
    if not train_rows or not val_rows:
        raise SystemExit(
            f"prepare_sft_t2v_msrvtt: train={len(train_rows)} val={len(val_rows)} usable rows; "
            "check annotation paths and extracted videos (or pass --keep-missing to inspect manifests)."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    _write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_rows)
    _write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val_rows)


if __name__ == "__main__":
    main()
