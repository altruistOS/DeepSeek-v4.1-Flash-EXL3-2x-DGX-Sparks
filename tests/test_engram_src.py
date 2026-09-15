#!/usr/bin/env python3
"""Slim Engram source: only embed tensors from shards 47+48."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_engram_src import KEEP_SUFFIXES, SHARDS, prepare, slim_weight_map  # noqa: E402


def test_slim_weight_map() -> None:
    keep = slim_weight_map(
        {
            "layers.1.engram.embed.weight": SHARDS[0],
            "layers.1.engram.embed.scale": SHARDS[0],
            "layers.1.engram.q_weight": SHARDS[0],
            "layers.1.engram.wkv.weight": SHARDS[0],
            "layers.14.engram.embed.weight": SHARDS[1],
            "layers.14.engram.embed.scale": SHARDS[1],
            "vision.blocks.0.attn.wqkv.weight": "model-00001-of-00048.safetensors",
        }
    )
    assert set(keep) == {
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        "layers.14.engram.embed.weight",
        "layers.14.engram.embed.scale",
    }
    assert keep["layers.1.engram.embed.weight"] == SHARDS[0]
    assert keep["layers.14.engram.embed.scale"] == SHARDS[1]
    assert all(n.endswith(KEEP_SUFFIXES) for n in keep)


def test_prepare_hardlinks(tmp_path: Path | None = None) -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as raw:
        src = Path(raw) / "native"
        dst = Path(raw) / "slim"
        src.mkdir()
        (src / SHARDS[0]).write_bytes(b"47")
        (src / SHARDS[1]).write_bytes(b"48")
        (src / "config.json").write_text("{}\n")
        index = {
            "metadata": {"total_size": 3},
            "weight_map": {
                "layers.1.engram.embed.weight": SHARDS[0],
                "layers.1.engram.embed.scale": SHARDS[0],
                "layers.1.engram.q_weight": SHARDS[0],
                "layers.14.engram.embed.weight": SHARDS[1],
                "layers.14.engram.embed.scale": SHARDS[1],
            },
        }
        (src / "model.safetensors.index.json").write_text(json.dumps(index))
        info = prepare(src, dst)
        assert (dst / SHARDS[0]).read_bytes() == b"47"
        assert (dst / SHARDS[0]).stat().st_ino == (src / SHARDS[0]).stat().st_ino
        slim = json.loads((dst / "model.safetensors.index.json").read_text())
        assert set(slim["weight_map"]) == {
            "layers.1.engram.embed.weight",
            "layers.1.engram.embed.scale",
            "layers.14.engram.embed.weight",
            "layers.14.engram.embed.scale",
        }
        assert "q_weight" not in json.dumps(slim)
        assert info["files"][SHARDS[0]] in ("hardlink", "exists")


def test_config_from_fallback() -> None:
    """$ENGRAM_DIR (native tree) has no config.json; take it from MODEL_HOST.

    engram_file_backend._layer_id_from_embeddings reads table_dir/config.json
    and raises FileNotFoundError when the slim dir has none, which is how the
    engine died on /engram-src/config.json.
    """
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as raw:
        src = Path(raw) / "native"
        model = Path(raw) / "model"
        dst = Path(raw) / "slim"
        src.mkdir()
        model.mkdir()
        (src / SHARDS[0]).write_bytes(b"47")
        (src / SHARDS[1]).write_bytes(b"48")
        cfg = {
            "text_config": {
                "engram_layer_ids": [1, 14],
                "engram_num_embeddings": [384006168, 384016682],
            }
        }
        (model / "config.json").write_text(json.dumps(cfg))
        index = {
            "metadata": {"total_size": 3},
            "weight_map": {
                "layers.1.engram.embed.weight": SHARDS[0],
                "layers.1.engram.embed.scale": SHARDS[0],
                "layers.14.engram.embed.weight": SHARDS[1],
                "layers.14.engram.embed.scale": SHARDS[1],
            },
        }
        (src / "model.safetensors.index.json").write_text(json.dumps(index))

        info = prepare(src, dst, model)
        assert (dst / "config.json").is_file(), "config.json must land in dst"
        got = json.loads((dst / "config.json").read_text())
        assert got["text_config"]["engram_layer_ids"] == [1, 14]
        assert info["files"]["config.json"] == "copy"


def test_config_from_accepts_a_file_path() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as raw:
        src = Path(raw) / "native"
        model = Path(raw) / "model"
        dst = Path(raw) / "slim"
        src.mkdir()
        model.mkdir()
        (src / SHARDS[0]).write_bytes(b"47")
        (src / SHARDS[1]).write_bytes(b"48")
        (model / "config.json").write_text('{"text_config": {"engram_layer_ids": [1]}}')
        index = {
            "metadata": {},
            "weight_map": {
                "layers.1.engram.embed.weight": SHARDS[0],
                "layers.1.engram.embed.scale": SHARDS[0],
            },
        }
        (src / "model.safetensors.index.json").write_text(json.dumps(index))

        prepare(src, dst, model / "config.json")
        assert (dst / "config.json").is_file()


def test_missing_config_raises() -> None:
    """Without a config.json the backend cannot map layers — fail loudly here."""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as raw:
        src = Path(raw) / "native"
        dst = Path(raw) / "slim"
        src.mkdir()
        (src / SHARDS[0]).write_bytes(b"47")
        (src / SHARDS[1]).write_bytes(b"48")
        index = {
            "metadata": {},
            "weight_map": {
                "layers.1.engram.embed.weight": SHARDS[0],
                "layers.1.engram.embed.scale": SHARDS[0],
            },
        }
        (src / "model.safetensors.index.json").write_text(json.dumps(index))

        try:
            prepare(src, dst)
        except SystemExit as exc:
            assert "config.json" in str(exc)
        else:
            raise AssertionError("prepare must fail when no config.json is available")


if __name__ == "__main__":
    test_slim_weight_map()
    test_prepare_hardlinks()
    test_config_from_fallback()
    test_config_from_accepts_a_file_path()
    test_missing_config_raises()
    print("test_engram_src: ok")
