#!/usr/bin/env python3
"""Seal the corrected PR37 -> PR38 -> P2 production source into a reproducible bundle."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from harness_lib import (
    EXPECTED_P0_PLANNER_COMMIT,
    EXPECTED_P0_PLANNER_TREE,
    EXPECTED_REBASED_PR38_COMMIT,
    EXPECTED_REBASED_PR38_TREE,
    EXPECTED_REMOTE,
    GateError,
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)

EXPECTED_P2_COMMIT = "48225e5ec732a3c7a06d2e0cfc61a98286a969cd"
EXPECTED_P2_TREE = "548aaff33347954736d0c086baa1b986f5fcb37c"
EXPECTED_P2_PATCH_SHA256 = "f89c68d9996a8d67d16b81bf1daa60795353b5f344a9533df2151d82a3b46d7d"
EXPECTED_P2_PATHS = {
    "examples/diffusion/minimax_h3/minimax_h3_t2va_trainside.yaml",
    "unirl/models/minimax_h3/bundle.py",
    "unirl/models/minimax_h3/config.py",
    "unirl/models/minimax_h3/pipeline.py",
    "unirl/models/minimax_h3/prefetch.py",
    "unirl/models/minimax_h3/text_embed.py",
    "unirl/rollout/engine/trainside/engine.py",
}


def _git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), *args],
        check=False,
        capture_output=True,
        text=text,
        timeout=180,
    )
    if result.returncode:
        stderr = result.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        raise GateError(f"git {' '.join(args)} failed: {str(stderr).strip()}")
    output = result.stdout
    return output.strip() if text else output


def _safe_path(raw: str) -> str:
    path = PurePosixPath(raw)
    if not raw or raw.startswith("/") or ".." in path.parts or "." in path.parts or path.as_posix() != raw:
        raise GateError(f"unsafe/non-canonical Git tree path: {raw!r}")
    return raw


def source_chain_gate(
    repo: Path,
    *,
    planner_commit: str = EXPECTED_P0_PLANNER_COMMIT,
    planner_tree: str = EXPECTED_P0_PLANNER_TREE,
    pr38_commit: str = EXPECTED_REBASED_PR38_COMMIT,
    pr38_tree: str = EXPECTED_REBASED_PR38_TREE,
    p2_commit: str = EXPECTED_P2_COMMIT,
    p2_tree: str = EXPECTED_P2_TREE,
    patch_sha256: str = EXPECTED_P2_PATCH_SHA256,
    require_fork: bool = True,
    expected_paths: set[str] | None = EXPECTED_P2_PATHS,
) -> dict[str, Any]:
    """Prove the exact corrected single-parent source chain before sealing."""
    repo = repo.resolve()
    if not (repo / ".git").exists():
        raise GateError(f"source repository is not a Git worktree: {repo}")

    observed = {
        "planner_tree": str(_git(repo, "rev-parse", f"{planner_commit}^{{tree}}")),
        "pr38_tree": str(_git(repo, "rev-parse", f"{pr38_commit}^{{tree}}")),
        "p2_tree": str(_git(repo, "rev-parse", f"{p2_commit}^{{tree}}")),
    }
    expected = {
        "planner_tree": planner_tree,
        "pr38_tree": pr38_tree,
        "p2_tree": p2_tree,
    }
    if observed != expected:
        raise GateError(f"corrected source tree mismatch: observed={observed} expected={expected}")

    pr38_parents = str(_git(repo, "rev-list", "--parents", "-n", "1", pr38_commit)).split()
    p2_parents = str(_git(repo, "rev-list", "--parents", "-n", "1", p2_commit)).split()
    if pr38_parents != [pr38_commit, planner_commit]:
        raise GateError("PR38 is not a single commit directly on corrected PR37/P0")
    if p2_parents != [p2_commit, pr38_commit]:
        raise GateError("P2 is not a single commit directly on corrected PR38")

    patch = _git(
        repo,
        "diff",
        "--binary",
        "--full-index",
        pr38_commit,
        p2_commit,
        text=False,
    )
    if not isinstance(patch, bytes):
        raise GateError("Git returned a non-binary P2 patch")
    observed_patch = sha256_bytes(patch)
    if observed_patch != patch_sha256:
        raise GateError(f"P2 patch digest mismatch: {observed_patch} != {patch_sha256}")

    changed = {line for line in str(_git(repo, "diff", "--name-only", pr38_commit, p2_commit)).splitlines() if line}
    if expected_paths is not None and changed != expected_paths:
        raise GateError(
            "P2 production path set mismatch: "
            f"missing={sorted(expected_paths - changed)} unexpected={sorted(changed - expected_paths)}"
        )

    fork_remote: str | None = None
    origin_push: str | None = None
    if require_fork:
        fork_remote = str(_git(repo, "config", "--get", "remote.fork.url"))
        origin_push = str(_git(repo, "config", "--get", "remote.origin.pushurl"))
        if fork_remote != EXPECTED_REMOTE or origin_push != "DISABLED_use_fork_remote":
            raise GateError("source sealing requires the user fork and a disabled official push URL")

    return {
        "schema": "unirl-minimax-h3-p2-corrected-source-chain-v1",
        "repo": os.fspath(repo),
        "planner_commit": planner_commit,
        "planner_tree": planner_tree,
        "pr38_commit": pr38_commit,
        "pr38_tree": pr38_tree,
        "p2_commit": p2_commit,
        "p2_tree": p2_tree,
        "p2_patch_sha256": observed_patch,
        "production_paths": sorted(changed),
        "fork_remote": fork_remote,
        "origin_push": origin_push,
    }


def _tree_entries(
    repo: Path,
    commit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = _git(repo, "ls-tree", "-r", "-z", "-l", commit, text=False)
    if not isinstance(raw, bytes):
        raise GateError("Git returned a non-binary tree listing")
    files: list[dict[str, Any]] = []
    symlinks: list[dict[str, Any]] = []
    gitlinks: list[dict[str, Any]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id, raw_size = header.decode().split()
            relative = _safe_path(raw_path.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise GateError("malformed Git tree record") from exc
        if not re.fullmatch(r"[0-9a-f]{40,64}", object_id):
            raise GateError(f"unsupported Git tree entry: {record!r}")
        if mode == "160000" and object_type == "commit":
            if raw_size != "-":
                raise GateError(f"malformed Git gitlink size for {relative}: {raw_size!r}")
            gitlinks.append({"path": relative, "commit": object_id})
            continue
        if object_type != "blob":
            raise GateError(f"unsupported Git tree entry: {record!r}")
        content = _git(repo, "cat-file", "blob", object_id, text=False)
        if not isinstance(content, bytes):
            raise GateError(f"Git returned non-binary blob content for {relative}")
        if mode == "120000":
            try:
                target = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise GateError(f"symlink target is not UTF-8: {relative}") from exc
            target_path = PurePosixPath(target)
            if not target or target.startswith("/") or ".." in target_path.parts or "\x00" in target:
                raise GateError(f"unsafe symlink target in source tree: {relative} -> {target!r}")
            symlinks.append({"path": relative, "target": target})
            continue
        if mode not in {"100644", "100755"}:
            raise GateError(f"unsupported Git file mode {mode} for {relative}")
        size = int(raw_size)
        if size != len(content):
            raise GateError(f"Git tree size mismatch for {relative}: {size} != {len(content)}")
        files.append(
            {
                "path": relative,
                "mode": "0755" if mode == "100755" else "0644",
                "size": size,
                "sha256": sha256_bytes(content),
                "_content": content,
            }
        )
    if not files:
        raise GateError("refusing to seal an empty source tree")
    return files, symlinks, gitlinks


def _full_tree_digest(
    files: list[dict[str, Any]],
    symlinks: list[dict[str, Any]],
    gitlinks: list[dict[str, Any]],
) -> str:
    digest = bytearray()
    for row in sorted(files, key=lambda value: value["path"]):
        digest.extend(f"F\0{row['path']}\0{row['mode']}\0{row['size']}\0{row['sha256']}\n".encode())
    for row in sorted(symlinks, key=lambda value: value["path"]):
        digest.extend(f"L\0{row['path']}\0{row['target']}\n".encode())
    for row in sorted(gitlinks, key=lambda value: value["path"]):
        digest.extend(f"G\0{row['path']}\0{row['commit']}\n".encode())
    return sha256_bytes(bytes(digest))


def _write_archive(
    path: Path,
    files: list[dict[str, Any]],
    symlinks: list[dict[str, Any]],
    gitlinks: list[dict[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
                    for row in sorted(files, key=lambda value: value["path"]):
                        info = tarfile.TarInfo(row["path"])
                        info.size = int(row["size"])
                        info.mode = int(row["mode"], 8)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        archive.addfile(info, io.BytesIO(row["_content"]))
                    for row in sorted(symlinks, key=lambda value: value["path"]):
                        info = tarfile.TarInfo(row["path"])
                        info.type = tarfile.SYMTYPE
                        info.linkname = row["target"]
                        info.size = 0
                        info.mode = stat.S_IMODE(0o777)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        archive.addfile(info)
                    for row in sorted(gitlinks, key=lambda value: value["path"]):
                        # A Git gitlink has no blob payload.  Preserve its
                        # object identity in the manifest and materialize an
                        # empty directory so an uninitialized submodule has
                        # the same usable checkout shape on the GPU node.
                        info = tarfile.TarInfo(row["path"])
                        info.type = tarfile.DIRTYPE
                        info.size = 0
                        info.mode = stat.S_IMODE(0o755)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        archive.addfile(info)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def seal_repository(
    repo: Path,
    output_dir: Path,
    *,
    planner_commit: str = EXPECTED_P0_PLANNER_COMMIT,
    planner_tree: str = EXPECTED_P0_PLANNER_TREE,
    pr38_commit: str = EXPECTED_REBASED_PR38_COMMIT,
    pr38_tree: str = EXPECTED_REBASED_PR38_TREE,
    p2_commit: str = EXPECTED_P2_COMMIT,
    p2_tree: str = EXPECTED_P2_TREE,
    patch_sha256: str = EXPECTED_P2_PATCH_SHA256,
    require_fork: bool = True,
    expected_paths: set[str] | None = EXPECTED_P2_PATHS,
) -> dict[str, Any]:
    """Create a deterministic archive, complete tree manifest, and seal receipt."""
    chain = source_chain_gate(
        repo,
        planner_commit=planner_commit,
        planner_tree=planner_tree,
        pr38_commit=pr38_commit,
        pr38_tree=pr38_tree,
        p2_commit=p2_commit,
        p2_tree=p2_tree,
        patch_sha256=patch_sha256,
        require_fork=require_fork,
        expected_paths=expected_paths,
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"unirl-p2-{p2_commit[:12]}"
    archive_path = output_dir / f"{stem}.tar.gz"
    manifest_path = output_dir / f"{stem}.tree.json"
    receipt_path = output_dir / f"{stem}.seal.json"
    if any(path.exists() for path in (archive_path, manifest_path, receipt_path)):
        raise GateError(f"refusing existing source-seal output below {output_dir}")

    files, symlinks, gitlinks = _tree_entries(repo.resolve(), p2_commit)
    full_tree_sha256 = _full_tree_digest(files, symlinks, gitlinks)
    public_files = [{key: row[key] for key in ("path", "mode", "size", "sha256")} for row in files]
    manifest = {
        "schema": "unirl-sealed-source-tree-v1",
        "head": p2_commit,
        "tree": p2_tree,
        "full_tree_sha256": full_tree_sha256,
        "files": public_files,
        "symlinks": symlinks,
        "gitlinks": gitlinks,
    }
    try:
        _write_archive(archive_path, files, symlinks, gitlinks)
        write_json_atomic(manifest_path, manifest)
        receipt = {
            "schema": "unirl-minimax-h3-p2-source-seal-v1",
            "completed": True,
            "chain": chain,
            "archive": {
                "path": os.fspath(archive_path),
                "sha256": sha256_file(archive_path),
                "bytes": archive_path.stat().st_size,
            },
            "tree_manifest": {
                "path": os.fspath(manifest_path),
                "sha256": sha256_file(manifest_path),
                "bytes": manifest_path.stat().st_size,
            },
            "full_tree_sha256": full_tree_sha256,
            "file_count": len(public_files),
            "symlink_count": len(symlinks),
            "gitlink_count": len(gitlinks),
            "manifest_payload_sha256": sha256_bytes(canonical_json(manifest)),
        }
        write_json_atomic(receipt_path, receipt)
        return {**receipt, "receipt": {"path": os.fspath(receipt_path), "sha256": sha256_file(receipt_path)}}
    except BaseException:
        archive_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = seal_repository(args.repo, args.output_dir)
    except GateError as exc:
        print(json.dumps({"completed": False, "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
