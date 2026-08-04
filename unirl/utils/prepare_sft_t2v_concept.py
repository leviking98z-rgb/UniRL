"""Build concept-video SFT manifests from paired path and caption files.

The input layout matches small Hugging Face concept datasets such as
``finetrainers/3dgs-dissolve``::

    <data-root>/
      videos.txt
      prompt.txt
      videos/0.mp4
      ...

Each non-empty line in ``videos.txt`` is paired with the same-numbered line in
``prompt.txt``. The output is a deterministic video-level train/validation
split in the format consumed by ``VideoDiffusionSupervisedTrackBuilder``.

Example:

  hf download finetrainers/3dgs-dissolve --repo-type dataset \
      --local-dir /path/to/3dgs-dissolve
  python -m unirl.utils.prepare_sft_t2v_concept \
      --data-root /path/to/3dgs-dissolve \
      --out-dir data/sft_3dgs_dissolve \
      --trigger-token 3DGS_DISSOLVE --val-count 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _read_nonempty_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _with_trigger(caption: str, trigger_token: str) -> str:
    trigger_token = trigger_token.strip()
    caption = caption.strip()
    if not trigger_token:
        return caption
    words = caption.split()
    while words and words[0] == trigger_token:
        words.pop(0)
    return " ".join([trigger_token, *words])


def _load_pairs(
    *,
    data_root: str,
    videos_file: str,
    prompts_file: str,
    trigger_token: str,
    source: str,
    keep_missing: bool,
) -> List[Dict[str, Any]]:
    video_lines = _read_nonempty_lines(os.path.join(data_root, videos_file))
    prompt_lines = _read_nonempty_lines(os.path.join(data_root, prompts_file))
    if len(video_lines) != len(prompt_lines):
        raise ValueError(
            f"paired concept files have different lengths: {videos_file}={len(video_lines)}, "
            f"{prompts_file}={len(prompt_lines)}."
        )

    rows: List[Dict[str, Any]] = []
    for index, (video_ref, caption) in enumerate(zip(video_lines, prompt_lines)):
        video_path = video_ref if os.path.isabs(video_ref) else os.path.join(data_root, video_ref)
        video_path = os.path.abspath(video_path)
        if not keep_missing and not os.path.isfile(video_path):
            raise FileNotFoundError(f"missing concept video at pair {index}: {video_path}")
        video_id = Path(video_ref).stem
        rows.append(
            {
                "sample_id": f"{source}:{video_id}",
                "prompt": _with_trigger(caption, trigger_token),
                "media": [{"modality": "video", "role": "target", "uri": video_path}],
                "metadata": {"source": source, "video_id": video_id},
            }
        )
    return rows


def _split_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    val_count: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if val_count < 1:
        raise ValueError(f"val_count must be >= 1, got {val_count}.")
    if val_count >= len(rows):
        raise ValueError(f"val_count={val_count} must be smaller than dataset size {len(rows)}.")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    val_indices = set(order[:val_count])
    train = [dict(row) for index, row in enumerate(rows) if index not in val_indices]
    val = [dict(row) for index, row in enumerate(rows) if index in val_indices]
    return train, val


def _write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):6d} rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--videos-file", default="videos.txt")
    parser.add_argument("--prompts-file", default="prompt.txt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--trigger-token", default="3DGS_DISSOLVE")
    parser.add_argument("--source", default="3dgs-dissolve")
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-missing", action="store_true", help="write manifests before videos are downloaded")
    args = parser.parse_args()

    rows = _load_pairs(
        data_root=os.path.abspath(args.data_root),
        videos_file=args.videos_file,
        prompts_file=args.prompts_file,
        trigger_token=args.trigger_token,
        source=args.source,
        keep_missing=args.keep_missing,
    )
    train_rows, val_rows = _split_rows(rows, val_count=args.val_count, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    _write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_rows)
    _write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val_rows)


if __name__ == "__main__":
    main()
