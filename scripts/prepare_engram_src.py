#!/usr/bin/env python3
"""Build a slim Engram source dir: shards 47+48 + an index of embed tables only.

The native checkpoint is 48 shards / ~476 GiB. Only layers 1 and 14 n-gram
tables are unquantized, and they live in model-00047/00048 (~95 GiB each).
vLLM's DefaultModelLoader glob + index would otherwise pull the whole tree
if we pointed engram_table_dir at the native checkpoint.

Hardlink the two shards (same filesystem, zero extra bytes) and write a
weight_map that lists only *.engram.embed.{weight,scale}.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SHARDS = (
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
)
KEEP_SUFFIXES = (".engram.embed.weight", ".engram.embed.scale")


def slim_weight_map(weight_map: dict[str, str]) -> dict[str, str]:
    keep: dict[str, str] = {}
    for name, shard in weight_map.items():
        if name.endswith(KEEP_SUFFIXES):
            keep[name] = shard
    return keep


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists() or dst.is_symlink():
        try:
            if dst.stat().st_ino == src.stat().st_ino and dst.stat().st_dev == src.stat().st_dev:
                return "exists"
        except OSError:
            pass
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def prepare(src: Path, dst: Path, config_from: Path | None = None) -> dict[str, object]:
    index_path = src / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"missing {index_path}")
    for shard in SHARDS:
        if not (src / shard).is_file():
            raise SystemExit(f"missing Engram shard {src / shard}")

    raw = json.loads(index_path.read_text())
    keep = slim_weight_map(raw.get("weight_map") or {})
    if not keep:
        raise SystemExit(
            f"no {KEEP_SUFFIXES} keys in {index_path} — is ENGRAM_DIR the native V4.1 tree?"
        )
    unexpected = sorted({shard for shard in keep.values() if shard not in SHARDS})
    if unexpected:
        raise SystemExit(f"embed tables not in shards 47/48: {unexpected}")

    dst.mkdir(parents=True, exist_ok=True)
    actions = {}
    for shard in SHARDS:
        actions[shard] = link_or_copy(src / shard, dst / shard)

    slim = {
        "metadata": {
            "total_size": raw.get("metadata", {}).get("total_size"),
            "dsv41_engram_src": "embed-only",
        },
        "weight_map": keep,
    }
    (dst / "model.safetensors.index.json").write_text(json.dumps(slim, indent=2) + "\n")

    # config.json must be present in dst: engram_file_backend._layer_id_from_embeddings
    # reads dst/config.json for text_config.engram_layer_ids /
    # engram_num_embeddings and raises FileNotFoundError otherwise.
    #
    # $ENGRAM_DIR (the native checkpoint tree) usually has no config.json of its
    # own, so fall back to an explicit --config-from source — normally the EXL3
    # model dir, whose config.json carries the same engram fields.
    cfg = src / "config.json"
    if not cfg.is_file() and config_from is not None:
        candidate = config_from / "config.json" if config_from.is_dir() else config_from
        if candidate.is_file():
            cfg = candidate
            actions["config.json-from"] = str(candidate)
    if cfg.is_file():
        shutil.copy2(cfg, dst / "config.json")
        actions["config.json"] = "copy"
    else:
        raise SystemExit(
            f"no config.json in {src} and no --config-from source given; "
            f"engram_file_backend needs table_dir/config.json "
            f"(pass --config-from <EXL3 model dir>)"
        )
    return {"dst": str(dst), "tensors": sorted(keep), "files": actions}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument(
        "--config-from",
        type=Path,
        default=None,
        help="directory (or config.json path) to take config.json from when "
        "--src has none — typically the EXL3 model dir",
    )
    args = parser.parse_args()
    info = prepare(args.src.resolve(), args.dst.resolve(), args.config_from)
    print(
        f"engram-src {info['dst']}: {len(info['tensors'])} embed tensors, "
        + ", ".join(f"{k}={v}" for k, v in info["files"].items()),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
