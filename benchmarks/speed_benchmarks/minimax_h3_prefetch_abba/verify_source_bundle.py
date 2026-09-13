#!/usr/bin/env python3
"""Verify or extract the immutable P0+P2 integration source bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from harness_lib import GateError, load_json, sha256_file, validate_p0_contract, write_json_atomic


def full_tree_digest(
    files: list[dict[str, Any]],
    symlinks: list[dict[str, Any]],
    gitlinks: list[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for row in sorted(files, key=lambda value: value["path"]):
        digest.update(f"F\0{row['path']}\0{row['mode']}\0{row['size']}\0{row['sha256']}\n".encode())
    for row in sorted(symlinks, key=lambda value: value["path"]):
        digest.update(f"L\0{row['path']}\0{row['target']}\n".encode())
    for row in sorted(gitlinks, key=lambda value: value["path"]):
        digest.update(f"G\0{row['path']}\0{row['commit']}\n".encode())
    return digest.hexdigest()


def safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise GateError(f"unsafe archive member: {name!r}")
    normalized = path.as_posix()
    if normalized in ("", "."):
        raise GateError(f"unsafe archive member: {name!r}")
    return normalized


def verify_archive(archive: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected_files = {row["path"]: row for row in manifest["files"]}
    expected_links = {row["path"]: row for row in manifest["symlinks"]}
    expected_gitlinks = {row["path"]: row for row in manifest.get("gitlinks", [])}
    observed_files: set[str] = set()
    observed_links: set[str] = set()
    observed_gitlinks: set[str] = set()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            name = safe_member(member.name)
            if member.isdir():
                if name not in expected_gitlinks:
                    raise GateError(f"unexpected directory in source archive: {name}")
                observed_gitlinks.add(name)
                continue
            if member.islnk():
                raise GateError(f"hard links are forbidden in source archive: {name}")
            if member.issym():
                expected = expected_links.get(name)
                if expected is None or member.linkname != expected["target"]:
                    raise GateError(f"unexpected symlink in source archive: {name}")
                observed_links.add(name)
                continue
            if not member.isfile():
                raise GateError(f"unsupported archive entry type: {name}")
            expected = expected_files.get(name)
            if expected is None:
                raise GateError(f"unexpected file in source archive: {name}")
            source = handle.extractfile(member)
            if source is None:
                raise GateError(f"cannot read archive file: {name}")
            digest = hashlib.sha256(source.read()).hexdigest()
            mode = format(stat.S_IMODE(member.mode), "04o")
            if digest != expected["sha256"] or member.size != expected["size"] or mode != expected["mode"]:
                raise GateError(f"archive file identity mismatch: {name}")
            observed_files.add(name)
    if (
        observed_files != set(expected_files)
        or observed_links != set(expected_links)
        or observed_gitlinks != set(expected_gitlinks)
    ):
        raise GateError("archive file/symlink/gitlink set differs from tree manifest")
    observed_tree = full_tree_digest(
        list(expected_files.values()),
        list(expected_links.values()),
        list(expected_gitlinks.values()),
    )
    if observed_tree != manifest["full_tree_sha256"]:
        raise GateError("tree manifest full_tree_sha256 mismatch")
    return {
        "archive_sha256": sha256_file(archive),
        "full_tree_sha256": observed_tree,
        "file_count": len(observed_files),
        "symlink_count": len(observed_links),
        "gitlink_count": len(observed_gitlinks),
    }


def verify_directory(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected_files = {row["path"]: row for row in manifest["files"]}
    expected_links = {row["path"]: row for row in manifest["symlinks"]}
    expected_gitlinks = {row["path"]: row for row in manifest.get("gitlinks", [])}
    observed_files: set[str] = set()
    observed_links: set[str] = set()
    observed_gitlinks: set[str] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == ".p2-source-identity.json":
            continue
        if path.is_symlink():
            expected = expected_links.get(relative)
            if expected is None or os.readlink(path) != expected["target"]:
                raise GateError(f"unexpected/mismatched source symlink: {relative}")
            observed_links.add(relative)
        elif path.is_file():
            expected = expected_files.get(relative)
            if expected is None:
                raise GateError(f"unexpected source file: {relative}")
            mode = format(stat.S_IMODE(path.stat().st_mode), "04o")
            if (
                sha256_file(path) != expected["sha256"]
                or path.stat().st_size != expected["size"]
                or mode != expected["mode"]
            ):
                raise GateError(f"source file identity mismatch: {relative}")
            observed_files.add(relative)
        elif path.is_dir() and relative in expected_gitlinks:
            observed_gitlinks.add(relative)
    if (
        observed_files != set(expected_files)
        or observed_links != set(expected_links)
        or observed_gitlinks != set(expected_gitlinks)
    ):
        raise GateError("extracted source file/symlink/gitlink set differs from tree manifest")
    observed_tree = full_tree_digest(
        list(expected_files.values()),
        list(expected_links.values()),
        list(expected_gitlinks.values()),
    )
    return {
        "full_tree_sha256": observed_tree,
        "file_count": len(observed_files),
        "symlink_count": len(observed_links),
        "gitlink_count": len(observed_gitlinks),
    }


def extract_verified(archive: Path, destination: Path, manifest: Mapping[str, Any]) -> None:
    if destination.exists():
        raise GateError(f"refusing existing extraction destination: {destination}")
    destination.mkdir(parents=True)
    expected_gitlinks = {row["path"] for row in manifest.get("gitlinks", [])}
    try:
        with tarfile.open(archive, "r:*") as handle:
            for member in handle.getmembers():
                name = safe_member(member.name)
                target = destination / name
                if member.isdir():
                    if name not in expected_gitlinks:
                        raise GateError(f"unexpected directory in source archive: {name}")
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.issym():
                    os.symlink(member.linkname, target)
                else:
                    source = handle.extractfile(member)
                    if source is None:
                        raise GateError(f"cannot extract archive file: {name}")
                    with target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    os.chmod(target, stat.S_IMODE(member.mode))
        verify_directory(destination, manifest)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-contract", type=Path, required=True)
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--verify-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.extract_to) == bool(args.verify_dir):
        raise SystemExit("specify exactly one of --extract-to or --verify-dir")
    try:
        p0 = validate_p0_contract(args.p0_contract.resolve())
        archive = Path(p0["source_archive"]["path"])
        manifest = load_json(Path(p0["source_tree_manifest"]["path"]))
        archive_result = verify_archive(archive, manifest)
        root = args.extract_to.resolve() if args.extract_to else args.verify_dir.resolve()
        if args.extract_to:
            extract_verified(archive, root, manifest)
        directory_result = verify_directory(root, manifest)
        payload = {
            "schema": "unirl-minimax-h3-p2-prefetch-source-identity-v1",
            "completed": True,
            "head": p0["source"]["integration_commit"],
            "tree": p0["source"]["integration_tree"],
            "full_tree_sha256": p0["source"]["full_tree_sha256"],
            "p0_contract_sha256": p0["sha256"],
            "archive": archive_result,
            "directory": directory_result,
            "root": os.fspath(root),
        }
        write_json_atomic(args.output, payload)
        if args.extract_to:
            write_json_atomic(root / ".p2-source-identity.json", payload)
        print(json.dumps(payload, sort_keys=True))
    except GateError as exc:
        payload = {"schema": "unirl-minimax-h3-p2-prefetch-source-identity-v1", "completed": False, "error": str(exc)}
        write_json_atomic(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
