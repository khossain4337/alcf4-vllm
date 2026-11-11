#!/bin/bash
# usage: hf_download_model.sh <org/model> [revision]   (run on a login node)
ml add frameworks
set -euo pipefail                      # after `ml`, module scripts often trip on -u

MODEL=${1:?usage: $0 <org/model> [revision]}
REV=${2:-main}
HF_TOKEN=$(cat ~/.hf_token) || { echo "no token" >&2; exit 1; }
export HF_TOKEN
export HF_HOME=${HF_HOME:-/lus/flare/projects/datasets/model-weights}
export HF_DATASETS_CACHE=$HF_HOME HF_MODULES_CACHE=$HF_HOME
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128 HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY

# Python API instead of the `hf` CLI, to avoid the rc=1 traceback
SNAP=$(python - "$MODEL" "$REV" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2],
      allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer.model", "*.tiktoken"]))
EOF
)
echo "snapshot: $SNAP"

# Integrity: LFS blobs are named by their sha256
shopt -s nullglob
files=("$SNAP"/*.safetensors)
[ ${#files[@]} -gt 0 ] || { echo "no .safetensors found" >&2; exit 1; }
printf '%s\0' "${files[@]}" | xargs -0 -P 8 -I{} bash -c \
  'f=$(readlink -f "$1"); [ "$(sha256sum "$f" | cut -d" " -f1)" = "$(basename "$f")" ] \
     && echo "sha OK: $(basename "$1")" || { echo "sha MISMATCH: $1" >&2; exit 255; }' _ {}
echo "sha: all ${#files[@]} shards OK"

# Module transformers may predate the model's config class; the venv's is what runs.
HF_HUB_OFFLINE=1 python -c "import sys; from transformers import AutoConfig, AutoTokenizer as T; AutoConfig.from_pretrained(sys.argv[1]); T.from_pretrained(sys.argv[1]); print('offline OK')" "$MODEL" \
  || echo "WARN: offline config/tokenizer check failed (see above)"
