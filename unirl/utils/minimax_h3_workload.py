"""CPU-only MiniMax-H3 workload geometry and JSONL telemetry helpers."""

from __future__ import annotations

import fcntl
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

WORKLOAD_TELEMETRY_SCHEMA = "unirl:minimax-h3:workload:v1"
_MINIMAX_H3_SPATIAL_COMPRESSION = 16
_MINIMAX_H3_PATCH_SPATIAL = 2
_MINIMAX_H3_CANVAS_MULTIPLE = 32
_MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
_MINIMAX_H3_MAX_ASPECT_RATIO = 4
_MINIMAX_H3_MAX_PIXELS = 768 * 1344
_MINIMAX_H3_FPS = 24
_MINIMAX_H3_MIN_DURATION = 5.0
_MINIMAX_H3_MAX_DURATION = 15.0
_MINIMAX_H3_FRAMES_PER_CHUNK = 17
_MINIMAX_H3_LATENTS_PER_CHUNK = 5
_MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
_MINIMAX_H3_AUDIO_CHANNELS = 2


@dataclass(frozen=True)
class MiniMaxH3WorkloadGeometry:
    """CPU-only packed-row geometry for one MiniMax-H3 request."""

    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int

    @property
    def rows_per_frame(self) -> int:
        return (self.latent_height // _MINIMAX_H3_PATCH_SPATIAL) * (self.latent_width // _MINIMAX_H3_PATCH_SPATIAL)

    @property
    def num_video_rows(self) -> int:
        return self.num_latent_frames * self.rows_per_frame

    @property
    def num_audio_rows(self) -> int:
        return self.num_audio_latents * _MINIMAX_H3_AUDIO_CHANNELS

    @property
    def base_packed_rows(self) -> int:
        return self.num_video_rows + self.num_audio_rows

    def packed_rows(self, text_tokens: int) -> int:
        tokens = int(text_tokens)
        if tokens < 0:
            raise ValueError(f"text_tokens must be non-negative, got {tokens}")
        return self.base_packed_rows + tokens

    def padded_rows(self, text_tokens: int, sp_size: int = 1) -> int:
        size = int(sp_size)
        if size < 1:
            raise ValueError(f"sp_size must be >= 1, got {size}")
        rows = self.packed_rows(text_tokens)
        return rows + (-rows % size)

    @classmethod
    def resolve(cls, *, height: int, width: int, num_frames: int) -> "MiniMaxH3WorkloadGeometry":
        """Validate one request and calculate its packed-row geometry."""
        h, w, frames = int(height), int(width), int(num_frames)
        if h <= 0 or w <= 0:
            raise ValueError(f"height and width must be positive, got height={h} width={w}")
        if h % _MINIMAX_H3_CANVAS_MULTIPLE or w % _MINIMAX_H3_CANVAS_MULTIPLE:
            raise ValueError(f"height={h} width={w} must both be multiples of {_MINIMAX_H3_CANVAS_MULTIPLE}")
        ratio = w / h
        if not _MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= _MINIMAX_H3_MAX_ASPECT_RATIO:
            raise ValueError(f"aspect ratio {w}:{h} ({ratio:g}) is outside the 1:4..4:1 range")
        if h * w > _MINIMAX_H3_MAX_PIXELS:
            raise ValueError(f"height={h} width={w} exceeds the {_MINIMAX_H3_MAX_PIXELS}-pixel area cap")
        if frames < 1 or frames % _MINIMAX_H3_FRAMES_PER_CHUNK != _MINIMAX_H3_LATENTS_PER_CHUNK:
            raise ValueError(f"num_frames={frames} must be of the form 17n+5")
        duration = frames / _MINIMAX_H3_FPS
        if not _MINIMAX_H3_MIN_DURATION <= duration <= _MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"num_frames={frames} is {duration:.2f}s at {_MINIMAX_H3_FPS} fps, outside the supported "
                f"{_MINIMAX_H3_MIN_DURATION:g}-{_MINIMAX_H3_MAX_DURATION:g}s range"
            )
        latent_frames = (
            frames - _MINIMAX_H3_LATENTS_PER_CHUNK
        ) // _MINIMAX_H3_FRAMES_PER_CHUNK * _MINIMAX_H3_LATENTS_PER_CHUNK + 2
        return cls(
            height=h,
            width=w,
            num_frames=frames,
            num_latent_frames=latent_frames,
            latent_height=h // _MINIMAX_H3_SPATIAL_COMPRESSION,
            latent_width=w // _MINIMAX_H3_SPATIAL_COMPRESSION,
            num_audio_latents=round(frames / _MINIMAX_H3_FPS * _MINIMAX_H3_AUDIO_LATENTS_PER_SECOND),
        )


@dataclass(frozen=True)
class MiniMaxH3WorkloadRecord:
    """One generated sample's dimensions, row counts, placement, and phase times."""

    sample_id: str
    root_id: str
    height: int
    width: int
    num_frames: int
    text_tokens: int
    video_rows: int
    audio_rows: int
    packed_rows: int
    sp_size: int
    padded_rows: int
    dp_rank: int
    sp_rank: int
    text_embed_s: float
    denoise_s: float
    decode_s: float
    total_s: float
    schema: str = WORKLOAD_TELEMETRY_SCHEMA

    @classmethod
    def build(
        cls,
        *,
        sample_id: str,
        root_id: str,
        height: int,
        width: int,
        num_frames: int,
        text_tokens: int,
        sp_size: int = 1,
        dp_rank: int = 0,
        sp_rank: int = 0,
        text_embed_s: float = 0.0,
        denoise_s: float = 0.0,
        decode_s: float = 0.0,
        total_s: float = 0.0,
    ) -> "MiniMaxH3WorkloadRecord":
        """Build a validated record and derive all row-count fields."""
        geometry = MiniMaxH3WorkloadGeometry.resolve(height=height, width=width, num_frames=num_frames)
        tokens = int(text_tokens)
        size = int(sp_size)
        if not str(sample_id):
            raise ValueError("sample_id must be non-empty")
        if not str(root_id):
            raise ValueError("root_id must be non-empty")
        if int(dp_rank) < 0 or int(sp_rank) < 0:
            raise ValueError("dp_rank and sp_rank must be non-negative")
        if size < 1 or int(sp_rank) >= size:
            raise ValueError(f"sp_rank={sp_rank} must be in [0, {size})")
        timings = [float(text_embed_s), float(denoise_s), float(decode_s), float(total_s)]
        if any(not math.isfinite(value) or value < 0 for value in timings):
            raise ValueError(f"phase timings must be finite and non-negative, got {timings}")
        return cls(
            sample_id=str(sample_id),
            root_id=str(root_id),
            height=geometry.height,
            width=geometry.width,
            num_frames=geometry.num_frames,
            text_tokens=tokens,
            video_rows=geometry.num_video_rows,
            audio_rows=geometry.num_audio_rows,
            packed_rows=geometry.packed_rows(tokens),
            sp_size=size,
            padded_rows=geometry.padded_rows(tokens, size),
            dp_rank=int(dp_rank),
            sp_rank=int(sp_rank),
            text_embed_s=timings[0],
            denoise_s=timings[1],
            decode_s=timings[2],
            total_s=timings[3],
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MiniMaxH3WorkloadRecord":
        """Validate one decoded telemetry row."""
        data = dict(value)
        if data.get("schema") != WORKLOAD_TELEMETRY_SCHEMA:
            raise ValueError(f"unsupported workload telemetry schema {data.get('schema')!r}")
        expected = set(cls.__dataclass_fields__)
        extra = sorted(set(data) - expected)
        missing = sorted(expected - set(data))
        if extra or missing:
            raise ValueError(f"workload record fields mismatch: missing={missing} extra={extra}")
        record = cls(**data)
        derived = cls.build(
            sample_id=record.sample_id,
            root_id=record.root_id,
            height=record.height,
            width=record.width,
            num_frames=record.num_frames,
            text_tokens=record.text_tokens,
            sp_size=record.sp_size,
            dp_rank=record.dp_rank,
            sp_rank=record.sp_rank,
            text_embed_s=record.text_embed_s,
            denoise_s=record.denoise_s,
            decode_s=record.decode_s,
            total_s=record.total_s,
        )
        if record != derived:
            raise ValueError("workload record row counts do not match its geometry")
        return record

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation."""
        return asdict(self)


def append_workload_record(path: str | os.PathLike[str], record: MiniMaxH3WorkloadRecord) -> None:
    """Append one JSONL record under an inter-process advisory lock."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    with destination.open("a", encoding="utf-8") as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        try:
            output.write(line)
            output.flush()
        finally:
            fcntl.flock(output.fileno(), fcntl.LOCK_UN)


def read_workload_records(paths: Iterable[str | os.PathLike[str]]) -> list[MiniMaxH3WorkloadRecord]:
    """Read and validate JSONL records from one or more files."""
    records: list[MiniMaxH3WorkloadRecord] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise TypeError(f"expected object, got {type(value).__name__}")
                    records.append(MiniMaxH3WorkloadRecord.from_mapping(value))
                except Exception as exc:
                    raise ValueError(f"invalid workload telemetry at {path}:{line_number}: {exc}") from exc
    return records


__all__ = [
    "WORKLOAD_TELEMETRY_SCHEMA",
    "MiniMaxH3WorkloadGeometry",
    "MiniMaxH3WorkloadRecord",
    "append_workload_record",
    "read_workload_records",
]
