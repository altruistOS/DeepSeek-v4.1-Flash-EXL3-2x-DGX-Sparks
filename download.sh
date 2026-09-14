#!/usr/bin/env bash
# download.sh — fetch the weights this recipe serves, without booting it.
#
# EXL3 weights (39 shards, ~197 GiB) come from HF_MODEL_REPO.
# Engram tables are never quantized and never copied into the EXL3 tree: only
# shards 47+48 of the original 48-shard checkpoint (~95 GiB each) and the index
# are pulled from HF_ENGRAM_REPO.
#
# DOWNLOAD POLICY (this fork, 2026-09-14):
#   * Downloads land in the DEFAULT HF hub cache ($HF_HOME or
#     ~/.cache/huggingface/hub) — the operator's habit. No --local-dir.
#   * $MODEL_HOST / $ENGRAM_DIR become symlinks into the cache snapshot
#     (snapshots/main), so start.sh's `find "$MODEL_HOST" -maxdepth 1` still
#     counts 39/39 shards and config.json: find follows the symlink given on
#     the command line, and the snapshot dir is the only level below it.
#   * The whole run is resumable: hf caches blobs by content hash, re-run
#     after an interruption and finished files are skipped.
#   * An existing real directory (an older --local-dir fetch, or a hand-placed
#     tree) is NEVER replaced by the symlink; the script warns and leaves it.
#   * HF_ENDPOINT (e.g. https://hf-mirror.com) from the environment is used by
#     the hf CLI for the mirror; nothing here overrides it.
#
# Resumable — re-run after an interruption. ./start.sh runs the same fetch
# automatically, so this script is only for staging the download separately.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
[ -f "$SCRIPT_DIR/.env" ] || cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a

MODEL_HOST="${MODEL_HOST:-$SCRIPT_DIR/model}"
ENGRAM_DIR="${ENGRAM_DIR:-$SCRIPT_DIR/engram-src}"
HF_MODEL_REPO="${HF_MODEL_REPO:-Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw}"
HF_ENGRAM_REPO="${HF_ENGRAM_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-39}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

hf_cli() {
    if command -v hf >/dev/null 2>&1; then hf "$@"
    elif command -v huggingface-cli >/dev/null 2>&1; then huggingface-cli "$@"
    else
        echo "need the Hugging Face CLI: pip install -U 'huggingface_hub[hf_transfer]'" >&2
        exit 1
    fi
}

# cache_snapshot_dir REPO_LABEL — the snapshot dir of REPO_LABEL inside the
# default hub cache, the way huggingface_hub lays it down:
#   hub/models--<org>--<repo>/snapshots/<revision>
# The blobs live in hub/blobs and are hardlinked into the snapshot — same
# filesystem, so the hardlinks are valid and start.sh's page-cache-drop and
# prepare_engram_src.py's os.link() keep working.
cache_snapshot_dir() {
    local label="${1//\//--}"
    local hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    [ -d "$hub/models--$label" ] || return 1
    local rev
    rev="$(git -C "$hub/models--$label" rev-parse refs/remotes/origin/main 2>/dev/null \
           || echo main)"
    [ -d "$hub/models--$label/snapshots/$rev" ] && { echo "$hub/models--$label/snapshots/$rev"; return 0; }
    # fall back to the newest snapshot directory that is not incomplete
    local d
    for d in "$hub"/models--"$label"/snapshots/*/; do
        [ -d "$d" ] && printf '%s\n' "${d%/}"
    done | sort | tail -1
}

# link_into TARGET SNAPSHOT — make TARGET a symlink pointing at SNAPSHOT so
# find -maxdepth 1 sees the files one level below. An existing real directory
# (a previous --local-dir fetch) is kept as-is: never replace the operator's
# own tree.
link_into() {
    local target="$1" snap="$2"
    if [ -L "$target" ]; then
        local cur; cur="$(readlink -f "$target")"
        if [ "$cur" = "$(readlink -f "$snap")" ]; then
            echo "        link: $target -> $cur (already in place)"
        else
            ln -sfn "$snap" "$target"
            echo "        link: $target -> $(readlink -f "$target") (retargeted)"
        fi
    elif [ -e "$target" ]; then
        echo "        link: keep $target (real directory present — not replaced by a symlink)"
    else
        ln -sfn "$snap" "$target"
        echo "        link: $target -> $(readlink -f "$target")"
    fi
}

# 1) EXL3 weights → default cache → symlink $MODEL_HOST → snapshots/main
have=$(find "$MODEL_HOST" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]')
echo "EXL3    $MODEL_HOST  ${have:-0}/$EXPECTED_SHARDS shards"
if [ "${have:-0}" -lt "$EXPECTED_SHARDS" ] || [ ! -f "$MODEL_HOST/config.json" ]; then
    echo "fetching $HF_MODEL_REPO → hub cache (default; HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co})"
    hf_cli download "$HF_MODEL_REPO" --max-workers "${HF_MAX_WORKERS:-8}"
    snap="$(cache_snapshot_dir "$HF_MODEL_REPO")" || {
        echo "download finished but no snapshot dir for $HF_MODEL_REPO in the hub cache" >&2; exit 1; }
    link_into "$MODEL_HOST" "$snap"
    have=$(find "$MODEL_HOST" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]')
    echo "EXL3    $MODEL_HOST  ${have:-0}/$EXPECTED_SHARDS shards (after link)"
fi

# 2) Engram shards 47+48 + index → default cache → symlink $ENGRAM_DIR
ENGRAM_FILES=(
    "model-00047-of-00048.safetensors"
    "model-00048-of-00048.safetensors"
    "model.safetensors.index.json"
)
missing=()
for f in "${ENGRAM_FILES[@]}"; do
    [ -f "$ENGRAM_DIR/$f" ] || missing+=("$f")
done
echo "Engram  $ENGRAM_DIR  $(( ${#ENGRAM_FILES[@]} - ${#missing[@]} ))/${#ENGRAM_FILES[@]} files"
if [ "${#missing[@]}" -gt 0 ]; then
    echo "fetching $HF_ENGRAM_REPO (shards 47+48 only) → hub cache (default)"
    inc=()
    for f in "${ENGRAM_FILES[@]}"; do inc+=(--include "$f"); done
    hf_cli download "$HF_ENGRAM_REPO" "${inc[@]}" --max-workers "${HF_MAX_WORKERS:-8}"
    snap="$(cache_snapshot_dir "$HF_ENGRAM_REPO")" || {
        echo "download finished but no snapshot dir for $HF_ENGRAM_REPO in the hub cache" >&2; exit 1; }
    # Guard: the shared cache snapshot also holds the other 46 native shards;
    # vLLM would glob them if pointed at directly. Keep only 47+48+index in a
    # slim real directory next to the cache, hardlinked from the snapshot
    # (same filesystem ⇒ zero extra bytes), then symlink ENGRAM_DIR at it.
    slim="${ENGRAM_DIR}.d"
    mkdir -p "$slim"
    for f in "${ENGRAM_FILES[@]}"; do
        [ -f "$snap/$f" ] || { echo "missing $snap/$f after download" >&2; exit 1; }
        [ -e "$slim/$f" ] || ln "$snap/$f" "$slim/$f" 2>/dev/null || cp -p "$snap/$f" "$slim/$f"
    done
    link_into "$ENGRAM_DIR" "$slim"
fi

# 3) check: the same verification start.sh will run before it boots
have=$(find "$MODEL_HOST" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]')
[ "${have:-0}" -ge "$EXPECTED_SHARDS" ] || { echo "still ${have:-0}/$EXPECTED_SHARDS EXL3 shards" >&2; exit 1; }
for f in "${ENGRAM_FILES[@]}"; do
    [ -f "$ENGRAM_DIR/$f" ] || { echo "missing $ENGRAM_DIR/$f" >&2; exit 1; }
done
echo "complete. ./start.sh will NFS-share both trees to the worker (WEIGHT_SYNC=nfs)."
