"""Generate matched WAN2.1 base-versus-UniRL-LoRA comparison videos.

The tool loads one WAN pipeline, exports a UniRL training checkpoint to a
Diffusers-compatible LoRA state dict in memory, and renders each prompt twice
with identical initial latents: first with the adapter disabled, then enabled.

Example:

  python -m unirl.tools.generate_wan21_lora_comparison \
      --base Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
      --checkpoint outputs/wan21_t2v_dissolve_lora/checkpoint-2000 \
      --output-dir outputs/wan21_dissolve_comparison
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from unirl.tools._checkpoint import load_training_checkpoint

DEFAULT_PROMPTS = (
    "A ceramic teapot in a dark studio, rendered in a 3D appearance, slowly rotating under cinematic lighting.",
    "3DGS_DISSOLVE A ceramic teapot in a dark studio, rendered in a 3D appearance, "
    "slowly rotating under cinematic lighting.",
    "3DGS_DISSOLVE A friendly robot in a dark studio, rendered in a 3D appearance, "
    "gradually evaporates into a burst of glowing red sparks.",
    "3DGS_DISSOLVE A red sports car in a dark studio, rendered in a 3D appearance, "
    "gradually evaporates into a burst of glowing red sparks.",
)


def _diffusers_lora_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    adapter: str,
) -> Dict[str, torch.Tensor]:
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for suffix in ("lora_A", "lora_B"):
        marker = f".{suffix}.{adapter}.weight"
        for key, value in state_dict.items():
            if not key.endswith(marker):
                continue
            stem = key[: -len(marker)]
            pairs.setdefault(stem, {})[suffix] = value.detach().cpu()
    if not pairs:
        raise SystemExit(f"no LoRA tensors for adapter {adapter!r} in the checkpoint")

    output: Dict[str, torch.Tensor] = {}
    for stem, pair in pairs.items():
        if set(pair) != {"lora_A", "lora_B"}:
            raise SystemExit(f"incomplete LoRA adapter pair for {stem!r}")
        output[f"transformer.{stem}.lora_A.weight"] = pair["lora_A"]
        output[f"transformer.{stem}.lora_B.weight"] = pair["lora_B"]
    return output


def _slug(text: str, index: int) -> str:
    body = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return f"{index:02d}-{body[:52] or 'prompt'}"


def _to_uint8_frames(value: object) -> List[np.ndarray]:
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 4:
        raise ValueError(f"WAN output must be a 4D video, got shape {array.shape}.")
    if array.shape[-1] in (3, 4):
        array = array[..., :3]
    elif array.shape[1] in (3, 4):
        array = np.transpose(array[:, :3], (0, 2, 3, 1))
    else:
        raise ValueError(f"WAN output channel axis is ambiguous for shape {array.shape}.")
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = np.rint(array * 255.0).astype(np.uint8)
    return [frame for frame in array]


def _write_video(path: str, frames: Iterable[np.ndarray], fps: int) -> None:
    import av

    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.pix_fmt = "yuv420p"
    for array in frames:
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        if stream.width == 0:
            stream.width = frame.width
            stream.height = frame.height
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font_size = max(20, image.height // 24)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except OSError:
        font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    text_width = right - left
    text_height = bottom - top
    margin = max(10, image.height // 48)
    draw.rounded_rectangle(
        (margin, margin, margin + text_width + 2 * margin, margin + text_height + 2 * margin),
        radius=max(6, margin // 2),
        fill=(0, 0, 0, 180),
    )
    draw.text(
        (2 * margin, 2 * margin),
        label,
        fill=(255, 255, 255, 255),
        font=font,
        stroke_width=1,
        stroke_fill=(0, 0, 0, 255),
    )
    return np.asarray(image)


def _side_by_side(
    left: List[np.ndarray],
    right: List[np.ndarray],
    *,
    left_label: str = "BASE",
    right_label: str = "LoRA",
) -> List[np.ndarray]:
    if len(left) != len(right):
        raise ValueError(f"base/LoRA frame counts differ: {len(left)} != {len(right)}")
    return [
        np.concatenate([_label_frame(a, left_label), _label_frame(b, right_label)], axis=1) for a, b in zip(left, right)
    ]


def _read_prompts(path: str | None) -> List[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    with open(path, encoding="utf-8") as fh:
        prompts = [line.strip() for line in fh if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="WAN2.1 diffusers checkpoint or local snapshot")
    parser.add_argument("--checkpoint", required=True, help="UniRL checkpoint-<step> directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts-file", default=None, help="one showcase prompt per line")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--adapter", default="default")
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    from diffusers import WanPipeline

    checkpoint = load_training_checkpoint(args.checkpoint)
    recorded = checkpoint.get("lora_config") or {}
    rank = int(recorded.get("rank") or 0)
    alpha = float(recorded.get("alpha") or 0.0)
    if rank < 1 or alpha <= 0:
        raise SystemExit("checkpoint has no usable LoRA rank/alpha metadata")
    if alpha != rank:
        raise SystemExit(
            f"Diffusers infers LoRA alpha from tensor rank for in-memory state dicts; "
            f"checkpoint alpha={alpha:g} != rank={rank}. Export a merged transformer with "
            "`python -m unirl.tools.export_full` or train this showcase with alpha=rank."
        )

    lora_state = _diffusers_lora_state_dict(checkpoint["policy_state_dict"], adapter=args.adapter)
    metadata = {
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": recorded.get("target_modules"),
        "lora_dropout": float(recorded.get("dropout", 0.0)),
        "bias": str(recorded.get("bias", "none")),
        "task_type": str(recorded.get("task_type", "FEATURE_EXTRACTION")),
    }
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    pipe = WanPipeline.from_pretrained(args.base, dtype=dtype)
    pipe.load_lora_weights(lora_state, adapter_name="unirl", low_cpu_mem_usage=True)
    pipe.set_adapters("unirl", adapter_weights=float(args.adapter_scale))
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = _read_prompts(args.prompts_file)
    manifest = {
        "base": args.base,
        "checkpoint": os.path.abspath(args.checkpoint),
        "seed": args.seed,
        "adapter_scale": args.adapter_scale,
        "lora": metadata,
        "prompts": [],
    }
    for index, prompt in enumerate(prompts):
        slug = _slug(prompt, index)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + index)
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=int(pipe.transformer.config.in_channels),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=generator,
        )
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "output_type": "np",
        }

        pipe.disable_lora()
        base_frames = _to_uint8_frames(pipe(**call_kwargs, latents=latents.clone()).frames[0])
        pipe.enable_lora()
        lora_frames = _to_uint8_frames(pipe(**call_kwargs, latents=latents.clone()).frames[0])

        base_path = os.path.join(args.output_dir, f"{slug}-base.mp4")
        lora_path = os.path.join(args.output_dir, f"{slug}-lora.mp4")
        comparison_path = os.path.join(args.output_dir, f"{slug}-base-vs-lora.mp4")
        _write_video(base_path, base_frames, args.fps)
        _write_video(lora_path, lora_frames, args.fps)
        _write_video(comparison_path, _side_by_side(base_frames, lora_frames), args.fps)
        manifest["prompts"].append(
            {
                "prompt": prompt,
                "base": os.path.basename(base_path),
                "lora": os.path.basename(lora_path),
                "comparison": os.path.basename(comparison_path),
            }
        )
        print(comparison_path)

    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
