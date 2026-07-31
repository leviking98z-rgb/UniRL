#!/usr/bin/env python3
"""Validate the recipe-backed claims in ``docs/tracks/video.md``."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "tracks" / "video.md"
CONFIG_ROOT = ROOT / "examples"

SUPPORT_HEADERS = (
    "Runtime",
    "Model / task",
    "Canonical entry",
    "Trainside",
    "SGLang",
    "vLLM-Omni",
    "FastVideo",
    "Default reward",
    "Status",
    "Owners",
    "Verification",
)
REWARD_HEADERS = (
    "Reward",
    "Runtime / owner",
    "Canonical usage",
    "Status",
    "Verification",
)

EXPECTED_CANONICAL_RECIPES = {
    "examples/ar/qwen3_omni_audio_video_gspo_lora_vllm_omni_1x4.yaml",
    "examples/ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x4.yaml",
    "examples/diffusion/hunyuan_video/hunyuan_video_t2v_trainside.yaml",
    "examples/diffusion/hunyuan_video15/hunyuan_video15_t2v_dancegrpo_trainside.yaml",
    "examples/diffusion/ltx2/ltx2_3_t2av_audioreward_trainside.yaml",
    "examples/diffusion/ltx2/ltx2_t2v_trainside.yaml",
    "examples/diffusion/wan21/wan21_i2v.yaml",
    "examples/diffusion/wan21/wan21_t2v.yaml",
    "examples/diffusion/wan22/wan22_i2v.yaml",
    "examples/diffusion/wan22/wan22_t2v_14b.yaml",
    "examples/diffusion/wan22_v2v/wan22_v2v_14b.yaml",
}

ENGINE_TARGETS = {
    "Trainside": "unirl.rollout.engine.trainside.engine.TrainsideRolloutEngine",
    "SGLang": "unirl.rollout.engine.sglang_diffusion.engine.SGLangDiffusionRolloutEngine",
    "vLLM-Omni": "unirl.rollout.engine.vllm_omni.engine.VLLMOmniRolloutEngine",
    "FastVideo": "unirl.rollout.engine.fastvideo.engine.FastVideoRolloutEngine",
}

DEFAULT_REWARD_TARGETS = {
    "MCExactMatch": "unirl.reward.local.mc_exact_match.MCExactMatchRewardScorer",
    "T2AVComposite": "unirl.reward.local.t2av_composite.T2AVCompositeScorer",
    "VideoCLIPDelta": "unirl.reward.local.video_clip_delta.VideoCLIPDeltaScorer",
    "VideoPickScore": "unirl.reward.local.video_pickscore.VideoPickScoreScorer",
}

REWARD_USAGE_TARGETS = {
    "T2AVComposite": "unirl.reward.local.t2av_composite.T2AVCompositeScorer",
    "VideoAlign": "required_rewards: [videoalign]",
    "VideoAlign ReFL": "experimental.refl.reward.videoalign.VideoAlignRewardScorer",
    "VideoCLIPDelta": "unirl.reward.local.video_clip_delta.VideoCLIPDeltaScorer",
    "VideoPickScore": "unirl.reward.local.video_pickscore.VideoPickScoreScorer",
}

ALLOWED_SUPPORT_STATUS = {"Preview", "Supported"}
ALLOWED_REWARD_STATUS = {"Core", "Experimental", "Service"}

_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_COMMIT_RE = re.compile(r"`[0-9a-f]{7,40}`")


def _table(text: str, heading: str, expected_headers: tuple[str, ...], errors: list[str]) -> list[dict[str, str]]:
    lines = text.splitlines()
    try:
        heading_index = lines.index(heading)
    except ValueError:
        errors.append(f"missing heading {heading!r}")
        return []

    table_lines: list[str] = []
    for line in lines[heading_index + 1 :]:
        if not table_lines and not line.startswith("|"):
            continue
        if not line.startswith("|"):
            break
        table_lines.append(line)

    if len(table_lines) < 3:
        errors.append(f"{heading}: missing Markdown table rows")
        return []

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    headers = tuple(cells(table_lines[0]))
    if headers != expected_headers:
        errors.append(f"{heading}: expected headers {expected_headers}, got {headers}")
        return []

    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(table_lines[2:], start=heading_index + 4):
        values = cells(line)
        if len(values) != len(headers):
            errors.append(f"{DOC.relative_to(ROOT)}:{line_number}: expected {len(headers)} cells, got {len(values)}")
            continue
        rows.append(dict(zip(headers, values)))
    return rows


def _links(cell: str) -> list[str]:
    return _LINK_RE.findall(cell)


def _resolve_doc_link(target: str, errors: list[str]) -> Path | None:
    target = target.split("#", 1)[0]
    path = (DOC.parent / target).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        errors.append(f"link escapes repository root: {target}")
        return None
    if not path.is_file():
        errors.append(f"linked recipe does not exist: {path.relative_to(ROOT)}")
        return None
    return path


def _default_entries(path: Path, text: str, errors: list[str]) -> list[Path]:
    lines = text.splitlines()
    entries: list[Path] = []
    in_defaults = False
    for line_number, line in enumerate(lines, start=1):
        if line == "defaults:":
            in_defaults = True
            continue
        if not in_defaults:
            continue
        if line and not line[0].isspace():
            break
        match = re.match(r"^\s*-\s+([^#]+?)\s*$", line)
        if not match:
            continue
        entry = match.group(1).strip()
        if entry == "_self_":
            continue
        if ":" in entry:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number}: mapping-style Hydra defaults are not supported by this guard"
            )
            continue
        if entry.startswith("/"):
            dependency = CONFIG_ROOT / entry.removeprefix("/")
        else:
            dependency = path.parent / entry
        if dependency.suffix not in {".yaml", ".yml"}:
            dependency = dependency.with_suffix(".yaml")
        if not dependency.is_file():
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number}: missing composed config {dependency.relative_to(ROOT)}"
            )
            continue
        entries.append(dependency)
    return entries


def _recipe_closure(path: Path, errors: list[str], seen: set[Path] | None = None) -> str:
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return ""
    seen.add(path)
    text = path.read_text(encoding="utf-8")
    dependencies = _default_entries(path, text, errors)
    return "\n".join([text, *(_recipe_closure(dependency, errors, seen) for dependency in dependencies)])


def _has_target(recipe_text: str, dotted: str) -> bool:
    pattern = rf"^\s*_target_:\s*{re.escape(dotted)}\s*$"
    return re.search(pattern, recipe_text, flags=re.MULTILINE) is not None


def _check_verification(row_name: str, verification: str, errors: list[str]) -> None:
    if _COMMIT_RE.search(verification) is None:
        errors.append(f"{row_name}: verification must include a commit hash")
    if "GPU:" not in verification:
        errors.append(f"{row_name}: verification must state GPU evidence or 'GPU: unrecorded'")


def _check_support_rows(rows: list[dict[str, str]], errors: list[str]) -> None:
    canonical_recipes: set[str] = set()

    for row in rows:
        row_name = f"{row['Runtime']} / {row['Model / task']}"
        canonical_links = _links(row["Canonical entry"])
        if len(canonical_links) != 1:
            errors.append(f"{row_name}: canonical entry must contain exactly one recipe link")
            continue

        canonical_path = _resolve_doc_link(canonical_links[0], errors)
        if canonical_path is None:
            continue
        canonical_rel = canonical_path.relative_to(ROOT).as_posix()
        canonical_recipes.add(canonical_rel)

        engine_paths: set[Path] = set()
        for engine, target in ENGINE_TARGETS.items():
            cell = row[engine]
            links = _links(cell)
            if cell != "—" and not links:
                errors.append(f"{row_name}: {engine} cell must be '—' or link at least one recipe")
            for link in links:
                recipe_path = _resolve_doc_link(link, errors)
                if recipe_path is None:
                    continue
                engine_paths.add(recipe_path)
                recipe_text = _recipe_closure(recipe_path, errors)
                if not _has_target(recipe_text, target):
                    errors.append(
                        f"{row_name}: {recipe_path.relative_to(ROOT)} is listed under {engine} "
                        f"but does not select {target}"
                    )

        if canonical_path not in engine_paths:
            errors.append(f"{row_name}: canonical entry must also appear in one engine column")

        reward = row["Default reward"]
        reward_target = DEFAULT_REWARD_TARGETS.get(reward)
        if reward_target is None:
            errors.append(f"{row_name}: unknown default reward {reward!r}")
        elif not _has_target(_recipe_closure(canonical_path, errors), reward_target):
            errors.append(f"{row_name}: canonical recipe does not select declared reward {reward}")

        if row["Status"] not in ALLOWED_SUPPORT_STATUS:
            errors.append(f"{row_name}: unsupported status {row['Status']!r}")
        if "@" not in row["Owners"]:
            errors.append(f"{row_name}: owners must contain at least one GitHub handle")
        _check_verification(row_name, row["Verification"], errors)

    if canonical_recipes != EXPECTED_CANONICAL_RECIPES:
        errors.append(
            "canonical recipe set drifted: "
            f"missing={sorted(EXPECTED_CANONICAL_RECIPES - canonical_recipes)}, "
            f"unexpected={sorted(canonical_recipes - EXPECTED_CANONICAL_RECIPES)}"
        )


def _check_reward_rows(rows: list[dict[str, str]], errors: list[str]) -> None:
    seen_rewards: set[str] = set()
    for row in rows:
        reward = row["Reward"]
        seen_rewards.add(reward)
        target = REWARD_USAGE_TARGETS.get(reward)
        if target is None:
            errors.append(f"reward ownership: unknown reward {reward!r}")
            continue

        usage_links = _links(row["Canonical usage"])
        if len(usage_links) != 1:
            errors.append(f"{reward}: canonical usage must contain exactly one recipe link")
            continue
        recipe_path = _resolve_doc_link(usage_links[0], errors)
        if recipe_path is not None:
            recipe_text = _recipe_closure(recipe_path, errors)
            if target.startswith("unirl.") or target.startswith("experimental."):
                matches = _has_target(recipe_text, target)
            else:
                matches = target in recipe_text
            if not matches:
                errors.append(f"{reward}: canonical usage does not select {target}")

        if "@" not in row["Runtime / owner"]:
            errors.append(f"{reward}: runtime/owner must contain at least one GitHub handle")
        if row["Status"] not in ALLOWED_REWARD_STATUS:
            errors.append(f"{reward}: unsupported ownership status {row['Status']!r}")
        _check_verification(reward, row["Verification"], errors)

    if seen_rewards != set(REWARD_USAGE_TARGETS):
        errors.append(
            "reward ownership set drifted: "
            f"missing={sorted(set(REWARD_USAGE_TARGETS) - seen_rewards)}, "
            f"unexpected={sorted(seen_rewards - set(REWARD_USAGE_TARGETS))}"
        )


def main() -> int:
    errors: list[str] = []
    text = DOC.read_text(encoding="utf-8")
    support_rows = _table(text, "## Recipe-backed support", SUPPORT_HEADERS, errors)
    reward_rows = _table(text, "## Reward ownership", REWARD_HEADERS, errors)
    _check_support_rows(support_rows, errors)
    _check_reward_rows(reward_rows, errors)

    if errors:
        print("check-video-support-matrix: FAILED")
        for error in errors:
            print(f"  {error}")
        return 1

    print(
        f"check-video-support-matrix: {len(support_rows)} model/task rows and "
        f"{len(reward_rows)} reward owners are consistent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
