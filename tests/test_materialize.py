#!/usr/bin/env python3
"""Materializing an HF snapshot tree into real files before an NFS export.

huggingface_hub lays a snapshot down as relative symlinks into ../../blobs/<sha>.
Exporting that directory over NFS fails on the worker: the export root *is* the
snapshot dir, so every entry resolves outside it. `ls /m` lists the names while
`test -f /m/config.json` returns non-zero.

`cp -a --link` does not fix it — it preserves the symlink, so the copy is just
as dangling. These tests drive the same dereference contract start.sh's
`materialize_dir` implements, and assert the resulting tree is self-contained.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def is_symlink_tree(directory: Path) -> bool:
    """Mirror of start.sh:is_symlink_tree."""
    entries = [p for p in directory.iterdir()]
    return bool(entries) and all(p.is_symlink() for p in entries)


def materialize_dir(src: Path, dst: Path) -> None:
    """Mirror of start.sh:materialize_dir — dereference, hardlink when possible."""
    src = src.resolve()
    dst = dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        resolved = entry.resolve()
        if not resolved.exists():
            continue  # dangling symlink (e.g. a blob not fetched)
        target = dst / entry.name
        if target.exists() and not target.is_symlink():
            continue  # never clobber an operator's real file
        if target.is_symlink() or target.exists():
            target.unlink()
        try:
            os.link(resolved, target)
        except OSError:
            shutil.copy2(resolved, target)


def _make_snapshot(root: Path) -> Path:
    """A hub cache layout: blobs/ beside snapshots/<rev>/, entries are symlinks."""
    blobs = root / "blobs"
    snap = root / "snapshots" / "rev"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (blobs / "aaa").write_text("config-body")
    (blobs / "bbb").write_bytes(b"shard-body")
    (snap / "config.json").symlink_to("../../blobs/aaa")
    (snap / "model-00001-of-00039.safetensors").symlink_to("../../blobs/bbb")
    return snap


def test_is_symlink_tree_detects_hub_snapshot() -> None:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        snap = _make_snapshot(root)
        assert is_symlink_tree(snap), "hub snapshot entries are all symlinks"


def test_is_symlink_tree_rejects_real_files() -> None:
    with TemporaryDirectory() as raw:
        real = Path(raw) / "real"
        real.mkdir()
        (real / "config.json").write_text("{}")
        assert not is_symlink_tree(real)


def test_cp_a_link_leaves_a_dangling_copy() -> None:
    """The bug: --link preserves the symlink, so the copy still dangles."""
    with TemporaryDirectory() as raw:
        root = Path(raw)
        snap = _make_snapshot(root)
        out = root / "cp_out"
        out.mkdir()
        # cp -a --link is what nfs_hardlink_tree does.
        (out / "config.json").symlink_to((snap / "config.json").readlink())
        assert not (out / "config.json").exists(), (
            "a preserved symlink still points outside the export root"
        )


def test_materialize_self_contained() -> None:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        snap = _make_snapshot(root)
        out = root / "materialized" / "model"
        materialize_dir(snap, out)

        assert not (out / "config.json").is_symlink()
        assert (out / "config.json").read_text() == "config-body"
        assert (out / "model-00001-of-00039.safetensors").read_bytes() == b"shard-body"
        assert not any(p.is_symlink() for p in out.iterdir())

        # The real check: the tree survives being exported on its own.
        exported = root / "export_root"
        exported.mkdir()
        shutil.copytree(out, exported / "model", symlinks=True)
        assert (exported / "model" / "config.json").is_file()


def test_materialize_is_idempotent() -> None:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        snap = _make_snapshot(root)
        out = root / "materialized" / "model"
        materialize_dir(snap, out)
        first = {
            p.name: p.stat().st_ino for p in out.iterdir()
        }
        materialize_dir(snap, out)
        second = {
            p.name: p.stat().st_ino for p in out.iterdir()
        }
        assert first == second, "re-running must not disturb the materialized tree"
        assert (out / "config.json").read_text() == "config-body"


def test_materialize_skips_dangling_symlink() -> None:
    """A snapshot symlink with no blob behind it must not become a broken copy."""
    with TemporaryDirectory() as raw:
        root = Path(raw)
        snap = _make_snapshot(root)
        (snap / "model-00099-of-00039.safetensors").symlink_to("../../blobs/missing")
        out = root / "materialized" / "model"
        materialize_dir(snap, out)
        assert (out / "config.json").is_file()
        assert not (out / "model-00099-of-00039.safetensors").exists()


if __name__ == "__main__":
    test_is_symlink_tree_detects_hub_snapshot()
    test_is_symlink_tree_rejects_real_files()
    test_cp_a_link_leaves_a_dangling_copy()
    test_materialize_self_contained()
    test_materialize_is_idempotent()
    test_materialize_skips_dangling_symlink()
    print("test_materialize: ok")
